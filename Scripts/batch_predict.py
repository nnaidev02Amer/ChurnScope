"""Batch churn prediction: read an uploaded customer file, validate every row, predict with the saved
model and threshold, and summarise the results.

Nothing here retrains or changes the model. Prediction reuses churn_pipeline.predict_churn() (saved
0.39 threshold), validation reuses churn_insights.validate_customer(), and risk levels reuse
churn_insights.risk_band(). No customer is silently dropped: rows that cannot be predicted are kept,
marked "Not predicted" and given a reason.
"""
import hashlib
import io
import math
import re
from datetime import datetime

import numpy as np
import pandas as pd

import churn_insights as ci
import churn_pipeline as cp

ACCEPTED_TYPES = ["csv", "xlsx", "xls", "pdf"]
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_ROWS = 10000
MAX_PDF_PAGES = 100
MAX_KEPT_BATCHES = 10          # how many batch runs the session history keeps

LEAVE, STAY, NOT_PREDICTED = "Likely to Leave", "Likely to Stay", "Not predicted"
RESULT_COLUMNS = ["Row", "Churn probability (%)", "Prediction", "Risk level", "Status", "Issue"]
ID_ALIASES = ["customerid", "custid", "customerno", "customernumber", "accountid", "accountnumber", "id", "customer"]
SENIOR_MAP = {"0": "No", "0.0": "No", "no": "No", "n": "No", "false": "No",
              "1": "Yes", "1.0": "Yes", "yes": "Yes", "y": "Yes", "true": "Yes"}


class UploadError(ValueError):
    """The file as a whole cannot be used. The message is written for the user."""


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


FEATURE_BY_NORM = {_norm(f): f for f in cp.FEATURES}


def file_hash(data):
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------- reading files
def _all_blank(series):
    """True if every cell is missing or an empty string (CSV blanks arrive as '', Excel blanks as NaN)."""
    return bool(series.isna().all() or series.astype(str).str.strip().isin(["", "nan", "None"]).all())


def _clean_frame(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [not ((c.startswith("Unnamed:") or c == "") and _all_blank(df[c])) for c in df.columns]]
    df = df.dropna(how="all")
    df = df[~(df.astype(str).apply(lambda col: col.str.strip()).eq("").all(axis=1))]
    return df.reset_index(drop=True)


def _read_csv(data):
    last = None
    for enc in ("utf-8-sig", "utf-8", "cp1252"):
        try:
            return pd.read_csv(io.BytesIO(data), sep=None, engine="python", dtype=str, keep_default_na=False,
                               encoding=enc)
        except UnicodeDecodeError as e:
            last = e
        except (pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            raise UploadError(f"The CSV file could not be read ({e}). Check that it is a normal comma-separated "
                              "file with one header row.") from e
    raise UploadError(f"The CSV file's text encoding is not supported ({last}). Save it as UTF-8 and try again.")


def _read_excel(data, ext):
    engine = "openpyxl" if ext == "xlsx" else "xlrd"
    try:
        __import__(engine)
    except ImportError as e:
        raise UploadError(f"Reading .{ext} files needs the '{engine}' package, which is not installed here. "
                          f"Install it with: pip install {engine}  (or save the file as .xlsx or .csv).") from e
    try:
        return pd.read_excel(io.BytesIO(data), sheet_name=0, engine=engine, dtype=object)
    except Exception as e:  # corrupt or password-protected workbooks raise many different errors
        raise UploadError(f"The Excel file could not be read ({type(e).__name__}). Make sure it is a valid, "
                          "unprotected .xlsx/.xls file with the customer table on the first sheet.") from e


def _read_pdf(data):
    try:
        import pdfplumber
    except ImportError as e:
        raise UploadError("Reading PDF files needs the 'pdfplumber' package, which is not installed here. "
                          "Install it with: pip install pdfplumber  (or upload a CSV/Excel file).") from e
    try:
        rows = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            if len(pdf.pages) > MAX_PDF_PAGES:
                raise UploadError(f"The PDF has {len(pdf.pages)} pages; the limit is {MAX_PDF_PAGES}.")
            for strategy in (None, {"vertical_strategy": "text", "horizontal_strategy": "text"}):
                rows = []
                for page in pdf.pages:
                    for table in (page.extract_tables(strategy) if strategy else page.extract_tables()):
                        rows.extend(table)
                if rows:
                    break
    except UploadError:
        raise
    except Exception as e:
        raise UploadError(f"The PDF could not be read ({type(e).__name__}). It may be damaged or password-protected.") from e
    rows = [[("" if c is None else " ".join(str(c).split())) for c in r] for r in rows if r]
    if len(rows) < 2:
        raise UploadError("No customer table was found in the PDF. PDFs must contain a text-based table (a header row "
                          "and one row per customer); scanned images are not supported. Try a CSV or Excel file.")
    header, body = rows[0], rows[1:]
    body = [r for r in body if r != header]                     # headers repeated on later pages
    width = len(header)
    body = [(r + [""] * width)[:width] for r in body]
    return pd.DataFrame(body, columns=header, dtype=object)


def read_upload(filename, data):
    """Return the uploaded table as a DataFrame (original column names, values as read)."""
    name = str(filename or "")
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ACCEPTED_TYPES:
        raise UploadError(f"'{name}' is not a supported file type. Upload a .csv, .xlsx, .xls or .pdf file.")
    if not data:
        raise UploadError("The file is empty.")
    if len(data) > MAX_FILE_BYTES:
        raise UploadError(f"The file is {len(data) / 1e6:.1f} MB; the limit is {MAX_FILE_BYTES // (1024 * 1024)} MB.")
    raw = {"csv": lambda: _read_csv(data), "xlsx": lambda: _read_excel(data, "xlsx"),
           "xls": lambda: _read_excel(data, "xls"), "pdf": lambda: _read_pdf(data)}[ext]()
    df = _clean_frame(raw)
    if df.shape[1] == 0:
        raise UploadError("The file has no columns.")
    if len(df) == 0:
        raise UploadError("The file has column headings but no customer rows.")
    if len(df) > MAX_ROWS:
        raise UploadError(f"The file has {len(df):,} customers; the limit is {MAX_ROWS:,} per upload. Split it into smaller files.")
    return df


# ------------------------------------------------------------------ matching columns and rows
def map_columns(df):
    """Match the file's headings to the 19 model features (ignoring case, spaces and underscores).
    Returns (feature -> column name, id column or None). Raises UploadError listing missing columns."""
    found, seen = {}, {}
    for col in df.columns:
        n = _norm(col)
        if n in seen:
            raise UploadError(f"The column heading '{col}' appears more than once (also as '{seen[n]}'). "
                              "Each column must be unique.")
        seen[n] = col
        if n in FEATURE_BY_NORM:
            found[FEATURE_BY_NORM[n]] = col
    missing = [f for f in cp.FEATURES if f not in found]
    if missing:
        raise UploadError(f"{len(missing)} of the 19 required columns are missing: {', '.join(missing)}. "
                          f"The file has these columns: {', '.join(map(str, df.columns))}.")
    id_col = next((seen[a] for a in ID_ALIASES if a in seen and seen[a] not in found.values()), None)
    return found, id_col


def _to_number(value, label, problems):
    if value is None or (isinstance(value, float) and math.isnan(value)) or str(value).strip() == "":
        problems.append(f"{label} is empty.")
        return None
    if isinstance(value, (bool, np.bool_)):
        problems.append(f"{label} must be a number (got {value!r}).")
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        x = float(value)
    else:
        s = str(value).strip()
        if "," in s:                                           # accept 1,234.50 only; never guess decimal commas
            if not re.fullmatch(r"\d{1,3}(,\d{3})+(\.\d+)?", s):
                problems.append(f"{label} must be a plain number (got '{value}').")
                return None
            s = s.replace(",", "")
        try:
            x = float(s)
        except ValueError:
            problems.append(f"{label} must be a number (got '{value}').")
            return None
    if not math.isfinite(x):
        problems.append(f"{label} must be a finite number.")
        return None
    return x


def _clean_row(row, cols, allowed, lookups):
    """Return (customer dict or None, problems). Tidies case/spacing and the 0/1 senior-citizen coding.
    `row` is a plain dict (fast); `lookups` maps each categorical feature to {lower-case value: value}."""
    problems, cust = [], {}
    for feat in cp.CATEGORICAL_FEATURES:
        v = row[cols[feat]]
        if v is None or (isinstance(v, float) and math.isnan(v)) or str(v).strip() == "":
            problems.append(f"{feat} is empty.")
            continue
        s = " ".join(str(v).strip().split())
        if feat == "SeniorCitizen" and s.lower() in SENIOR_MAP:
            cust[feat] = SENIOR_MAP[s.lower()]
            continue
        match = lookups[feat].get(s.lower())
        if match is None:
            problems.append(f"{feat}: '{v}' is not recognised (allowed: {', '.join(allowed[feat])}).")
        else:
            cust[feat] = match
    for feat, label in (("tenure", "tenure"), ("MonthlyCharges", "MonthlyCharges"), ("TotalCharges", "TotalCharges")):
        x = _to_number(row[cols[feat]], label, problems)
        if x is not None:
            cust[feat] = x
    return (None, problems) if problems else (cust, [])


# --------------------------------------------------------------------------------- predicting
def predict_batch(bundle, df, allowed, filename=""):
    """Validate every row, predict the valid ones and return a result dict.

    Keys: results (one row per uploaded customer, same order), invalid (rows not predicted and why),
    summary, id_col, filename."""
    cols, id_col = map_columns(df)
    thr = float(bundle["threshold"])
    valid_idx, valid_customers, notes, problems_by_row = [], [], {}, {}
    lookups = {feat: {a.lower(): a for a in allowed[feat]} for feat in cp.CATEGORICAL_FEATURES}
    for i, row in enumerate(df.reset_index(drop=True).to_dict("records")):
        cust, problems = _clean_row(row, cols, allowed, lookups)
        if cust is not None:
            v = ci.validate_customer(cust, allowed)
            if v.ok:
                valid_idx.append(i)
                valid_customers.append(v.customer)
                if v.warnings:
                    notes[i] = "Check: " + " ".join(v.warnings)
                continue
            problems = v.errors
        problems_by_row[i] = " ".join(problems)

    n = len(df)
    prob = np.full(n, np.nan)
    prediction = [NOT_PREDICTED] * n
    risk = [""] * n
    if valid_customers:
        frame = pd.DataFrame(valid_customers)[cp.FEATURES]
        out = cp.predict_churn(bundle, frame, threshold=thr)       # saved model and saved threshold
        p_arr, leave_arr = out["churn_probability"].to_numpy(), out["churn"].to_numpy()
        for j, i in enumerate(valid_idx):
            p = float(p_arr[j])
            prob[i] = p * 100
            prediction[i] = LEAVE if bool(leave_arr[j]) else STAY
            risk[i] = ci.risk_band(p, thr)

    original = df.reset_index(drop=True).copy()
    clash = {c: f"{c} (uploaded)" for c in original.columns if c in RESULT_COLUMNS}
    original = original.rename(columns=clash)
    predicted_rows = set(valid_idx)
    res = pd.DataFrame({"Row": np.arange(2, n + 2),                 # spreadsheet row (row 1 is the header)
                        "Churn probability (%)": np.round(prob, 2), "Prediction": prediction, "Risk level": risk,
                        "Status": ["Predicted" if i in predicted_rows else "Not predicted" for i in range(n)],
                        "Issue": [problems_by_row.get(i, notes.get(i, "")) for i in range(n)]})
    results = pd.concat([res, original], axis=1)
    if id_col is not None:                                          # show the customer ID right after the row number
        idc = clash.get(id_col, id_col)
        results = results[["Row", idc] + [c for c in results.columns if c not in ("Row", idc)]]

    invalid = results[results["Status"] == "Not predicted"]
    invalid = invalid[["Row"] + ([clash.get(id_col, id_col)] if id_col else []) + ["Issue"]].reset_index(drop=True)
    return {"results": results, "invalid": invalid, "id_col": id_col, "filename": filename,
            "summary": summarise(results, thr)}


def summarise(results, threshold):
    done = results[results["Status"] == "Predicted"]
    leave = int((done["Prediction"] == LEAVE).sum())
    stay = int((done["Prediction"] == STAY).sum())
    return {"total": int(len(results)), "predicted": int(len(done)), "invalid": int(len(results) - len(done)),
            "leave": leave, "stay": stay,
            "churn_pct": (leave / len(done) * 100) if len(done) else float("nan"),
            "avg_probability": float(done["Churn probability (%)"].mean()) if len(done) else float("nan"),
            "risk": {k: int((done["Risk level"] == k).sum()) for k in ("Low", "Medium", "High")},
            "threshold": float(threshold)}


# ---------------------------------------------------------------------------------- findings
def batch_findings(summary, results, dataset_rate):
    """A few plain-English sentences about one batch. Descriptive only: they describe the uploaded
    customers and the model's estimates, not causes."""
    s = summary
    if s["predicted"] == 0:
        return ["No customer in this file could be predicted, so there are no findings. See the list of problems."]
    out = []
    pct = s["churn_pct"] / 100
    gap = pct - dataset_rate
    cmp = ("about the same as" if abs(gap) < 0.03 else "higher than" if gap > 0 else "lower than")
    out.append(f"**{s['leave']:,} of {s['predicted']:,} customers ({pct:.0%})** are likely to leave. "
               f"That is {cmp} the {dataset_rate:.1%} of customers who left in the original dataset.")
    leavers = results[results["Prediction"] == LEAVE]
    if len(leavers):
        parts = []
        ccol = next((c for c in leavers.columns if _norm(c) == "contract"), None)
        if ccol is not None:
            top = leavers[ccol].astype(str).str.strip().value_counts()
            parts.append(f"{top.iloc[0] / len(leavers):.0%} of them are on a {str(top.index[0]).lower()} contract")
        tcol = next((c for c in leavers.columns if _norm(c) == "tenure"), None)
        if tcol is not None:
            months = pd.to_numeric(leavers[tcol], errors="coerce")
            new = int((months <= 12).sum())
            if new:
                parts.append(f"{new:,} ({new / len(leavers):.0%}) have been customers for 12 months or less")
        if parts:
            out.append("Among the customers likely to leave, " + " and ".join(parts) + ".")
    else:
        out.append("No customer in this file is above the alert line "
                   f"({s['threshold']:.0%}), so none is labelled likely to leave.")
    line = f"The average chance of leaving across this file is **{s['avg_probability']:.1f}%**."
    if s["invalid"]:
        line += f" **{s['invalid']:,} row(s) could not be predicted** and are listed separately."
    out.append(line)
    return out


# ------------------------------------------------------------------------- export / history
def _safe_cell(v):
    """Stop spreadsheet programs treating text from an uploaded file as a formula."""
    return "'" + v if isinstance(v, str) and v[:1] in ("=", "+", "-", "@") else v


def results_to_csv(results):
    """CSV bytes with every uploaded column and the prediction columns (all rows, nothing dropped)."""
    safe = results.copy()
    for col in safe.columns:
        if not pd.api.types.is_numeric_dtype(safe[col]):     # text columns (object or pandas 'str' dtype)
            safe[col] = safe[col].map(_safe_cell)
    return safe.to_csv(index=False).encode("utf-8")


def make_record(batch_id, filename, batch, digest, when=None):
    s = batch["summary"]
    return {"id": batch_id, "filename": filename, "digest": digest,
            "time": (when or datetime.now()).strftime("%Y-%m-%d %H:%M:%S"),
            **{k: s[k] for k in ("total", "predicted", "invalid", "leave", "stay", "churn_pct", "avg_probability",
                                 "risk", "threshold")},
            "results": batch["results"], "invalid_rows": batch["invalid"], "id_col": batch["id_col"]}


def add_record(records, record, keep=MAX_KEPT_BATCHES):
    """Append a run, keeping only the most recent `keep` (bounded memory). Returns the new list."""
    return (list(records) + [record])[-keep:]


def latest_with_predictions(records):
    return next((r for r in reversed(records) if r["predicted"] > 0), None)


def template_csv():
    """A header-only template plus two example rows, so users can see the expected layout."""
    rows = pd.DataFrame([
        {"customerID": "EX-001", "gender": "Female", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "No",
         "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "DSL", "OnlineSecurity": "No",
         "OnlineBackup": "Yes", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
         "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
         "PaymentMethod": "Electronic check", "MonthlyCharges": 55.0, "TotalCharges": 660.0},
        {"customerID": "EX-002", "gender": "Male", "SeniorCitizen": "No", "Partner": "Yes", "Dependents": "Yes",
         "tenure": 60, "PhoneService": "No", "MultipleLines": "No phone service", "InternetService": "No",
         "OnlineSecurity": "No internet service", "OnlineBackup": "No internet service",
         "DeviceProtection": "No internet service", "TechSupport": "No internet service",
         "StreamingTV": "No internet service", "StreamingMovies": "No internet service", "Contract": "Two year",
         "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 25.0,
         "TotalCharges": 1500.0}])
    return rows.to_csv(index=False).encode("utf-8")
