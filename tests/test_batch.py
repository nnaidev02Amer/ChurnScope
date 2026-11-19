"""Batch prediction logic: reading CSV/Excel/PDF, validation, per-customer results, summaries, export, history."""
import hashlib
import io
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Scripts"))
import _batch_data as bd  # noqa: E402
import batch_predict as bp  # noqa: E402
import churn_insights as ci  # noqa: E402
import churn_pipeline as cp  # noqa: E402

pytestmark = pytest.mark.skipif(not cp.MODEL_PATH.exists(), reason="run Scripts/train_churn_model.py first")


@pytest.fixture(scope="module")
def bundle():
    return cp.load_model()


@pytest.fixture(scope="module")
def allowed(bundle):
    return ci.allowed_categories(bundle)


def run(bundle, allowed, df, name="test.csv"):
    return bp.predict_batch(bundle, bp.read_upload(name, bd.csv_bytes(df)), allowed, name)


class StubPipeline:
    """Always returns the same churn probability, to test the threshold exactly."""
    def __init__(self, p):
        self.p = p

    def predict_proba(self, frame):
        return np.tile([1 - self.p, self.p], (len(frame), 1))


# ------------------------------------------------------------------ predictions are the model's
def test_csv_predictions_equal_the_saved_model_and_threshold(bundle, allowed):
    df = bd.customers(60)
    batch = run(bundle, allowed, df)
    res = batch["results"]
    X, _ = cp.load_and_clean()
    direct = cp.predict_churn(bundle, X.iloc[:60])                  # first 60 customers, same order
    assert len(res) == 60 and (res["Status"] == "Predicted").all()
    assert np.allclose(res["Churn probability (%)"], direct["churn_probability"] * 100, atol=0.006)
    assert (res["Prediction"] == np.where(direct["churn"], bp.LEAVE, bp.STAY)).all()
    for p, band in zip(direct["churn_probability"], res["Risk level"]):
        assert band == ci.risk_band(p, 0.39)                          # existing risk-band logic is reused
    assert batch["summary"]["threshold"] == 0.39


@pytest.mark.parametrize("p,expected", [(0.0, bp.STAY), (0.389, bp.STAY), (0.3899999, bp.STAY), (0.39, bp.LEAVE),
                                        (0.391, bp.LEAVE), (1.0, bp.LEAVE)])
def test_batch_respects_the_039_threshold_boundary(allowed, p, expected):
    stub = {"pipeline": StubPipeline(p), "threshold": 0.39}
    res = run(stub, allowed, bd.customers(5))["results"]
    assert set(res["Prediction"]) == {expected}
    assert set(res["Risk level"]) == {ci.risk_band(p, 0.39)}


def test_multiple_customers_each_get_a_result(bundle, allowed):
    batch = run(bundle, allowed, bd.customers(200))
    res, s = batch["results"], batch["summary"]
    assert s["total"] == 200 and s["predicted"] == 200 and s["invalid"] == 0
    assert s["leave"] + s["stay"] == s["predicted"]
    assert res["Prediction"].isin([bp.LEAVE, bp.STAY]).all() and res["Risk level"].isin(["Low", "Medium", "High"]).all()
    assert s["leave"] == (res["Prediction"] == bp.LEAVE).sum()
    assert s["churn_pct"] == pytest.approx(s["leave"] / 200 * 100)
    assert s["avg_probability"] == pytest.approx(res["Churn probability (%)"].mean())
    assert sum(s["risk"].values()) == 200 and s["risk"]["High"] == s["leave"]   # High band equals the churn decision
    assert list(res["Row"]) == list(range(2, 202))                              # spreadsheet row numbers, nothing missing


def test_customer_id_and_original_details_are_kept(bundle, allowed):
    df = bd.customers(10)
    res = run(bundle, allowed, df)["results"]
    assert list(res.columns[:2]) == ["Row", "customerID"]                       # ID shown first when present
    for col in df.columns:                                                      # every uploaded column, unchanged
        assert list(res[col].astype(str)) == list(df[col].astype(str)), col
    assert "Churn" in res.columns                                               # extra columns are kept too


def test_file_without_id_column_still_works(bundle, allowed):
    res = run(bundle, allowed, bd.customers(5).drop(columns=["customerID"]))
    assert res["summary"]["predicted"] == 5 and res["id_col"] is None and res["results"].columns[0] == "Row"


# ------------------------------------------------------------------------------- file formats
def test_excel_xlsx_gives_the_same_results_as_csv(bundle, allowed):
    df = bd.customers(40)
    from_csv = run(bundle, allowed, df)["results"]
    xl = bp.read_upload("customers.xlsx", bd.xlsx_bytes(df))
    from_xlsx = bp.predict_batch(bundle, xl, allowed, "customers.xlsx")["results"]
    assert len(xl) == 40
    assert np.allclose(from_csv["Churn probability (%)"], from_xlsx["Churn probability (%)"])
    assert list(from_csv["Prediction"]) == list(from_xlsx["Prediction"])


def test_pdf_table_gives_the_same_results_as_csv(bundle, allowed):
    df = bd.customers(8, keep_churn_column=False)
    pdf = bp.read_upload("customers.pdf", bd.pdf_bytes(df))
    assert pdf.shape == (8, 20)
    from_pdf = bp.predict_batch(bundle, pdf, allowed, "customers.pdf")
    from_csv = run(bundle, allowed, df)
    assert from_pdf["summary"]["predicted"] == 8 and from_pdf["summary"]["invalid"] == 0
    assert np.allclose(from_pdf["results"]["Churn probability (%)"], from_csv["results"]["Churn probability (%)"])


def test_xls_is_read_with_the_xlrd_engine(monkeypatch):
    seen = {}

    def fake_read_excel(buf, sheet_name=None, engine=None, dtype=None):
        seen["engine"] = engine
        return bd.customers(3)
    monkeypatch.setattr(pd, "read_excel", fake_read_excel)
    out = bp.read_upload("old_format.xls", b"not-really-xls")
    assert seen["engine"] == "xlrd" and len(out) == 3
    bp.read_upload("new_format.xlsx", b"x")
    assert seen["engine"] == "openpyxl"


@pytest.mark.parametrize("module,filename,hint", [("xlrd", "a.xls", "pip install xlrd"),
                                                  ("pdfplumber", "a.pdf", "pip install pdfplumber"),
                                                  ("openpyxl", "a.xlsx", "pip install openpyxl")])
def test_missing_reader_library_gives_a_clear_message(monkeypatch, module, filename, hint):
    monkeypatch.setitem(sys.modules, module, None)                   # makes "import module" fail
    with pytest.raises(bp.UploadError, match=hint):
        bp.read_upload(filename, b"data")


def test_csv_variants_are_read(bundle, allowed):
    df = bd.customers(5)
    for text in (df.to_csv(index=False, sep=";"), df.to_csv(index=False, sep="\t"), "﻿" + df.to_csv(index=False)):
        out = bp.read_upload("x.csv", text.encode("utf-8"))
        assert out.shape == (5, 21)
    latin = df.copy()
    latin.loc[0, "customerID"] = "café-1"
    out = bp.read_upload("x.csv", latin.to_csv(index=False).encode("cp1252"))
    assert out.loc[0, "customerID"] == "café-1"


def test_headings_are_matched_ignoring_case_spaces_and_underscores(bundle, allowed):
    df = bd.customers(5, keep_churn_column=False).rename(columns={
        "tenure": "TENURE", "MonthlyCharges": "Monthly Charges", "SeniorCitizen": "senior_citizen",
        "PaymentMethod": " payment method ", "customerID": "Customer ID"})
    batch = run(bundle, allowed, df)
    assert batch["summary"]["predicted"] == 5 and batch["id_col"] == "Customer ID"


# ------------------------------------------------------------------------------ invalid files
@pytest.mark.parametrize("name,data,match", [
    ("notes.txt", b"hello", "not a supported file type"),
    ("noextension", b"hello", "not a supported file type"),
    ("customers.docx", b"PK", "not a supported file type"),
    ("empty.csv", b"", "empty"),
    ("headers_only.csv", b"gender,tenure\n", "no customer rows"),
    ("broken.xlsx", b"this is not a workbook", "Excel file could not be read"),
    ("broken.pdf", b"%PDF-1.4 garbage", "PDF could not be read"),
])
def test_unusable_files_are_rejected_with_a_clear_message(name, data, match):
    with pytest.raises(bp.UploadError, match=match):
        bp.read_upload(name, data)


def test_pdf_without_a_table_is_rejected(monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.text(0.1, 0.5, "Just a sentence, no table here.")
    buf = io.BytesIO()
    fig.savefig(buf, format="pdf")
    plt.close(fig)
    with pytest.raises(bp.UploadError, match="No customer table was found"):
        bp.read_upload("letter.pdf", buf.getvalue())


def test_file_size_and_row_limits(monkeypatch):
    monkeypatch.setattr(bp, "MAX_FILE_BYTES", 100)
    with pytest.raises(bp.UploadError, match="limit is"):
        bp.read_upload("big.csv", b"a,b\n" + b"1,2\n" * 100)
    monkeypatch.setattr(bp, "MAX_FILE_BYTES", 10 * 1024 * 1024)
    monkeypatch.setattr(bp, "MAX_ROWS", 5)
    with pytest.raises(bp.UploadError, match="per upload"):
        bp.read_upload("many.csv", bd.csv_bytes(bd.customers(6)))


def test_missing_columns_are_listed_by_name():
    df = bp.read_upload("x.csv", bd.csv_bytes(bd.customers(3).drop(columns=["Contract", "tenure", "TotalCharges"])))
    with pytest.raises(bp.UploadError) as err:
        bp.map_columns(df)
    msg = str(err.value)
    assert "3 of the 19 required columns are missing" in msg and "Contract" in msg and "tenure" in msg and "TotalCharges" in msg


def test_a_file_of_the_wrong_kind_reports_all_19_missing():
    df = bp.read_upload("sales.csv", b"product,price\napple,1\npear,2\n")
    with pytest.raises(bp.UploadError, match="19 of the 19"):
        bp.map_columns(df)


def test_duplicate_column_headings_are_rejected():
    df = pd.DataFrame([[1, 2]], columns=["tenure", "Tenure"])
    with pytest.raises(bp.UploadError, match="more than once"):
        bp.map_columns(df)


def test_blank_lines_and_empty_columns_are_ignored(bundle, allowed):
    # what some spreadsheet exports look like: a trailing comma on every line, then blank lines at the end
    lines = bd.customers(3).to_csv(index=False).strip().splitlines()      # (Windows writes \r\n, hence splitlines)
    empty_row = "," * len(lines[0].split(","))                            # same width as the other lines, all empty
    text = "\n".join(line + "," for line in lines) + "\n" + empty_row + "\n\n\n"
    out = bp.read_upload("x.csv", text.encode())
    assert len(out) == 3 and out.shape[1] == 21 and not any(c.startswith("Unnamed") for c in out.columns)


# ------------------------------------------------------------------------------ invalid rows
def test_bad_rows_are_reported_with_reasons_and_never_dropped(bundle, allowed):
    df = bd.customers(12, keep_churn_column=False)
    df = bd.with_changes(df, 1, Contract="Weekly")                           # unseen category
    df = bd.with_changes(df, 2, tenure=0)                                    # outside the trained range
    df = bd.with_changes(df, 3, tenure=73)
    df = bd.with_changes(df, 4, tenure=12.5)
    df = bd.with_changes(df, 5, MonthlyCharges="abc")                        # not a number
    df = bd.with_changes(df, 6, MonthlyCharges=-5)
    df = bd.with_changes(df, 7, gender="")                                   # empty
    df = bd.with_changes(df, 8, TotalCharges=" ")                            # blank, like the dataset's tenure-0 rows
    batch = run(bundle, allowed, df)
    res, s = batch["results"], batch["summary"]
    assert len(res) == 12 and s["total"] == 12 and s["predicted"] == 4 and s["invalid"] == 8   # nothing dropped
    bad = res[res["Status"] == "Not predicted"]
    assert list(bad["Row"]) == [3, 4, 5, 6, 7, 8, 9, 10]                    # spreadsheet rows of the 8 bad customers
    assert bad["Churn probability (%)"].isna().all() and (bad["Prediction"] == bp.NOT_PREDICTED).all()
    assert (bad["Risk level"] == "").all()
    reasons = dict(zip(bad["Row"], bad["Issue"]))
    assert "Contract" in reasons[3] and "Weekly" in reasons[3]
    assert "tenure" in reasons[4].lower() and "tenure" in reasons[5].lower() and "tenure" in reasons[6].lower()
    assert "MonthlyCharges" in reasons[7] and "MonthlyCharges" in reasons[8].replace("Monthly charges", "MonthlyCharges")
    assert "gender is empty" in reasons[9] and "TotalCharges is empty" in reasons[10]
    assert list(batch["invalid"]["Row"]) == list(bad["Row"]) and list(batch["invalid"].columns) == ["Row", "customerID", "Issue"]
    good = res[res["Status"] == "Predicted"]
    assert len(good) == 4 and good["Churn probability (%)"].between(0, 100).all()


def test_service_inconsistencies_are_caught(bundle, allowed):
    df = bd.customers(6, keep_churn_column=False)
    df = bd.with_changes(df, 0, PhoneService="No", MultipleLines="Yes")
    df = bd.with_changes(df, 1, InternetService="No", TechSupport="Yes")
    res = run(bundle, allowed, df)["results"]
    assert (res["Status"] == "Not predicted").sum() == 2
    assert "Multiple lines" in res.loc[0, "Issue"] and "internet" in res.loc[1, "Issue"].lower()


def test_tidy_up_of_valid_but_untidy_values(bundle, allowed):
    df = bd.customers(8, keep_churn_column=False)
    df = bd.with_changes(df, 0, Contract="  month-to-MONTH ")                # case and spaces
    df = bd.with_changes(df, 1, SeniorCitizen="yes")
    df = bd.with_changes(df, 2, SeniorCitizen=1)                             # the dataset's own 0/1 coding
    df = bd.with_changes(df, 3, TotalCharges="1,234.50")                     # thousands separator
    df = bd.with_changes(df, 4, tenure="12.0")
    res = run(bundle, allowed, df)
    assert res["summary"]["invalid"] == 0, res["invalid"].to_string()
    # decimal-comma numbers are rejected rather than guessed
    bad = run(bundle, allowed, bd.with_changes(bd.customers(3, keep_churn_column=False), 0, TotalCharges="1.234,50"))
    assert bad["results"].loc[0, "Status"] == "Not predicted" and "plain number" in bad["results"].loc[0, "Issue"]


def test_out_of_range_charges_warn_but_still_predict(bundle, allowed):
    df = bd.with_changes(bd.customers(3, keep_churn_column=False), 0, MonthlyCharges=150.0, TotalCharges=900.0)
    res = run(bundle, allowed, df)["results"]
    assert res.loc[0, "Status"] == "Predicted" and res.loc[0, "Issue"].startswith("Check:")


def test_every_row_invalid_is_handled(bundle, allowed):
    df = bd.customers(4, keep_churn_column=False)
    df["Contract"] = "Weekly"
    batch = run(bundle, allowed, df)
    s = batch["summary"]
    assert s["predicted"] == 0 and s["invalid"] == 4 and math.isnan(s["churn_pct"]) and math.isnan(s["avg_probability"])
    assert "no findings" in bp.batch_findings(s, batch["results"], 0.265)[0]
    rec = bp.make_record(1, "bad.csv", batch, "h")
    assert bp.latest_with_predictions([rec]) is None


# ------------------------------------------------------------------------- summary and findings
def test_findings_are_correct_and_plain(bundle, allowed):
    batch = run(bundle, allowed, bd.customers(300, keep_churn_column=False))
    s, res = batch["summary"], batch["results"]
    lines = bp.batch_findings(s, res, 0.2654)
    text = " ".join(lines)
    assert f"{s['leave']:,} of {s['predicted']:,} customers" in text
    assert f"{s['avg_probability']:.1f}%" in text
    leavers = res[res["Prediction"] == bp.LEAVE]
    top = leavers["Contract"].value_counts()
    assert f"{top.iloc[0] / len(leavers):.0%} of them are on a {top.index[0].lower()} contract" in text
    new = int((pd.to_numeric(leavers["tenure"]) <= 12).sum())
    assert f"{new:,} ({new / len(leavers):.0%}) have been customers for 12 months or less" in text
    for forbidden in ("because", "cause", "ROC", "SHAP"):
        assert forbidden not in text
    assert ("higher than" in text) == (s["churn_pct"] / 100 - 0.2654 >= 0.03)


def test_findings_when_nobody_is_likely_to_leave(allowed):
    stub = {"pipeline": StubPipeline(0.05), "threshold": 0.39}
    batch = run(stub, allowed, bd.customers(5, keep_churn_column=False))
    text = " ".join(bp.batch_findings(batch["summary"], batch["results"], 0.265))
    assert "No customer in this file is above the alert line" in text


def test_findings_mention_problem_rows(bundle, allowed):
    df = bd.with_changes(bd.customers(6, keep_churn_column=False), 0, Contract="Weekly")
    batch = run(bundle, allowed, df)
    assert "1 row(s) could not be predicted" in " ".join(bp.batch_findings(batch["summary"], batch["results"], 0.265))


# ------------------------------------------------------------------------------- CSV export
def test_csv_export_has_every_row_and_column(bundle, allowed):
    df = bd.with_changes(bd.customers(15), 3, Contract="Weekly")
    batch = run(bundle, allowed, df)
    out = pd.read_csv(io.BytesIO(bp.results_to_csv(batch["results"])), dtype=str, keep_default_na=False)
    assert len(out) == 15                                                    # the invalid customer is still there
    assert list(out.columns) == list(batch["results"].columns)
    assert out.loc[3, "Status"] == "Not predicted" and "Weekly" in out.loc[3, "Issue"]
    for col in df.columns:                                                   # all original columns present, unchanged
        assert list(out[col]) == list(df[col].astype(str)), col
    assert {"Churn probability (%)", "Prediction", "Risk level", "Status", "Issue"} <= set(out.columns)


def test_csv_export_blocks_spreadsheet_formulas(bundle, allowed):
    df = bd.customers(3)
    df.loc[0, "customerID"] = "=HYPERLINK(\"http://x\")"
    df.loc[1, "customerID"] = "-5+2"
    out = pd.read_csv(io.BytesIO(bp.results_to_csv(run(bundle, allowed, df)["results"])), dtype=str)
    assert out.loc[0, "customerID"].startswith("'=") and out.loc[1, "customerID"].startswith("'-")
    assert out.loc[2, "customerID"] == df.loc[2, "customerID"]


def test_original_columns_named_like_result_columns_do_not_collide(bundle, allowed):
    df = bd.customers(3)
    df["Prediction"] = "mine"
    res = run(bundle, allowed, df)["results"]
    assert "Prediction (uploaded)" in res.columns and set(res["Prediction (uploaded)"]) == {"mine"}
    assert res["Prediction"].isin([bp.LEAVE, bp.STAY]).all()


# -------------------------------------------------------------------------------------- history
def test_history_records_and_bounded_memory(bundle, allowed):
    recs = []
    for i in range(1, 15):
        batch = run(bundle, allowed, bd.customers(3, offset=i))
        recs = bp.add_record(recs, bp.make_record(i, f"f{i}.csv", batch, "d"))
    assert len(recs) == bp.MAX_KEPT_BATCHES and recs[0]["id"] == 5 and recs[-1]["id"] == 14
    r = recs[-1]
    assert {"id", "filename", "time", "total", "predicted", "leave", "stay", "churn_pct", "results"} <= set(r)
    assert r["total"] == 3 and r["filename"] == "f14.csv" and len(r["time"]) == 19
    assert bp.latest_with_predictions(recs)["id"] == 14


def test_latest_skips_runs_with_no_predictions(bundle, allowed):
    good = bp.make_record(1, "good.csv", run(bundle, allowed, bd.customers(4)), "a")
    badf = bd.customers(4)
    badf["Contract"] = "Weekly"
    bad = bp.make_record(2, "bad.csv", run(bundle, allowed, badf), "b")
    assert bp.latest_with_predictions([good, bad])["id"] == 1


def test_template_file_is_valid(bundle, allowed):
    batch = bp.predict_batch(bundle, bp.read_upload("customer_template.csv", bp.template_csv()), allowed)
    assert batch["summary"]["total"] == 2 and batch["summary"]["predicted"] == 2 and batch["summary"]["invalid"] == 0


# ------------------------------------------------------------------------------- integrity
def test_batch_prediction_does_not_touch_data_model_or_metrics(bundle, allowed):
    files = [cp.DATA_PATH, cp.MODEL_PATH, cp.PROJECT_DIR / "reports" / "metrics.json"]
    before = [hashlib.sha256(f.read_bytes()).hexdigest() for f in files]
    run(bundle, allowed, bd.customers(100))
    assert [hashlib.sha256(f.read_bytes()).hexdigest() for f in files] == before
    assert bundle["threshold"] == 0.39
