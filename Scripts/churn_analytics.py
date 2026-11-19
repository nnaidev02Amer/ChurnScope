"""Descriptive customer analytics for ChurnScope (Phase 2).

Everything here is computed from the ORIGINAL dataset (all 7,043 customers) unless a
function says otherwise. Two kinds of numbers are kept strictly apart:

* OBSERVED / ACTUAL: counts, churn rates and revenue taken straight from the records.
* ESTIMATED: model predictions (churn probabilities, "revenue at risk"). Only the saved
  model is used; nothing is retrained and no test label influences the model.

Missing values: 11 customers have a blank TotalCharges (all tenure == 0, i.e. not billed
yet). They stay in every customer and monthly-revenue figure, TotalCharges is left as NaN
(never imputed) and skipped by sums. They are not scored (outside the model's 1-72 month range).
"""
import math

import numpy as np
import pandas as pd

import churn_insights as ci
import churn_pipeline as cp

MIN_N = 30  # groups smaller than this are flagged as unreliable

TENURE_GROUPS = ["0-12 months", "13-24 months", "25-48 months", "49-72 months"]
CHARGE_BANDS = ["under 35", "35 to 70", "70 to 90", "90 and over"]
HIGH_SPENDER_MIN = 90.0   # monthly charges; roughly the top quartile (75th percentile is 89.85)
MULTI_SERVICE_MIN = 5     # number of subscribed services
LONG_TERM_MIN = 49        # months
NEW_MAX = 12              # months

# label -> (column, display order)
DIMENSIONS = {
    "Contract": ("Contract", ["Month-to-month", "One year", "Two year"]),
    "Tenure group": ("tenure_group", TENURE_GROUPS),
    "Internet service": ("InternetService", ["DSL", "Fiber optic", "No"]),
    "Payment method": ("PaymentMethod", ["Electronic check", "Mailed check",
                                         "Bank transfer (automatic)", "Credit card (automatic)"]),
    "Monthly charges": ("charge_band", CHARGE_BANDS),
    "Senior citizen": ("senior", ["No", "Yes"]),
}


def tenure_group(months):
    if months <= 12:
        return TENURE_GROUPS[0]
    if months <= 24:
        return TENURE_GROUPS[1]
    if months <= 48:
        return TENURE_GROUPS[2]
    return TENURE_GROUPS[3]


# ----------------------------------------------------------------------- loading
def service_count(df):
    """Number of subscribed services (0-9): phone, multiple lines, internet, and the six add-ons."""
    n = (df["PhoneService"] == "Yes").astype(int) + (df["MultipleLines"] == "Yes").astype(int)
    n = n + (df["InternetService"] != "No").astype(int)
    for col in ci.ADDON_COLUMNS:
        n = n + (df[col] == "Yes").astype(int)
    return n


def load_customers(path=cp.DATA_PATH):
    """All customers from the original CSV with derived analytics columns. One row per customer."""
    df = pd.read_csv(path)
    if df[cp.ID_COLUMN].duplicated().any():
        raise ValueError("Duplicate customerID found; refusing to double count customers")
    bad = set(df[cp.TARGET].unique()) - {"Yes", "No"}
    if bad:
        raise ValueError(f"Unexpected Churn values: {sorted(bad)}")
    df = df.drop(columns=[cp.ID_COLUMN])  # identifier not needed for analytics
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # blanks -> NaN
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    df["churned"] = df[cp.TARGET].eq("Yes")
    df["senior"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["tenure_group"] = df["tenure"].map(tenure_group)
    df["charge_band"] = df["MonthlyCharges"].map(lambda m: ci.charge_band(m) if pd.notna(m) else np.nan)
    df["service_count"] = service_count(df)
    return df


def scored_customers(bundle, path=cp.DATA_PATH):
    """load_customers() plus the saved model's churn probability for every customer it can score.

    Columns added: p_churn (NaN when not scored), in_test (customer was in the held-out test
    split, so the model never saw it), flagged (probability >= saved threshold), risk_band.
    Scores for non-test customers are in-sample (the model trained on them)."""
    df = load_customers(path)
    X, y = cp.load_and_clean(path)
    _, X_te, _, _ = cp.split_data(X, y)
    pos = df.index[df["tenure"] > 0].to_numpy()          # cleaned rows keep the raw row order
    if len(pos) != len(X):
        raise ValueError("Cleaned data and raw data are out of sync")
    thr = float(bundle["threshold"])
    df["p_churn"] = np.nan
    df.loc[pos, "p_churn"] = bundle["pipeline"].predict_proba(X[cp.FEATURES])[:, 1]
    df["in_test"] = False
    df.loc[pos[X_te.index.to_numpy()], "in_test"] = True
    df["flagged"] = df["p_churn"].ge(thr)
    df["risk_band"] = [ci.risk_band(p, thr) if pd.notna(p) else "Not scored" for p in df["p_churn"]]
    return df


# ----------------------------------------------------------------------- filters
def apply_filters(df, contract=None, internet=None, payment=None, tenure_groups=None,
                  senior=None, monthly_range=None):
    """Keep customers matching every filter. None = no filter; an empty list = nobody."""
    mask = pd.Series(True, index=df.index)
    for col, allowed in (("Contract", contract), ("InternetService", internet),
                         ("PaymentMethod", payment), ("tenure_group", tenure_groups), ("senior", senior)):
        if allowed is not None:
            mask &= df[col].isin(list(allowed))
    if monthly_range is not None:
        lo, hi = monthly_range
        mask &= df["MonthlyCharges"].between(lo, hi)
    return df[mask]


# --------------------------------------------------------------------------- KPIs
def _rate(k, n):
    return float(k) / n if n else float("nan")


def kpis(df):
    """Customer overview. Monthly revenue = sum of MonthlyCharges over the customer records."""
    total = len(df)
    churned = int(df["churned"].sum())
    monthly = float(df["MonthlyCharges"].sum())
    monthly_churned = float(df.loc[df["churned"], "MonthlyCharges"].sum())
    return {
        "customers": total,
        "churned": churned,
        "active": total - churned,
        "churn_rate": _rate(churned, total),
        "avg_monthly_charges": float(df["MonthlyCharges"].mean()) if total else float("nan"),
        "monthly_revenue": monthly,
        "monthly_revenue_active": monthly - monthly_churned,
        "monthly_revenue_churned": monthly_churned,
        "churned_revenue_share": monthly_churned / monthly if monthly else float("nan"),
    }


# ----------------------------------------------------------------- churn analysis
def wilson_interval(k, n, z=1.96):
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def churn_by(df, column, order=None):
    """Observed churn per group: customers, churned, rate, 95% CI, share, monthly revenue."""
    rows = []
    groups = [g for g in (order or sorted(df[column].dropna().unique())) if (df[column] == g).any()]
    total = len(df)
    for g in groups:
        sub = df[df[column] == g]
        n, k = len(sub), int(sub["churned"].sum())
        lo, hi = wilson_interval(k, n)
        rows.append({"group": g, "customers": n, "churned": k, "churn_rate": _rate(k, n),
                     "ci_low": lo, "ci_high": hi, "share": _rate(n, total),
                     "monthly_revenue": float(sub["MonthlyCharges"].sum()),
                     "small_group": n < MIN_N})
    return pd.DataFrame(rows, columns=["group", "customers", "churned", "churn_rate", "ci_low", "ci_high",
                                       "share", "monthly_revenue", "small_group"])


def cross_churn(df, col_a, col_b, order_a=None, order_b=None):
    """Observed churn for every combination of two dimensions (mutually exclusive cells)."""
    rows = []
    for a in (order_a or sorted(df[col_a].dropna().unique())):
        for b in (order_b or sorted(df[col_b].dropna().unique())):
            sub = df[(df[col_a] == a) & (df[col_b] == b)]
            n, k = len(sub), int(sub["churned"].sum())
            rows.append({"a": a, "b": b, "customers": n, "churned": k, "churn_rate": _rate(k, n),
                         "small_group": n < MIN_N})
    return pd.DataFrame(rows, columns=["a", "b", "customers", "churned", "churn_rate", "small_group"])


# ------------------------------------------------------------------- segmentation
# Segments OVERLAP (a customer can be new AND month-to-month AND a high spender), so their
# counts and revenue must never be added together.
SEGMENTS = {
    "New customers": (f"tenure of {NEW_MAX} months or less", lambda d: d["tenure"] <= NEW_MAX),
    "Long-term customers": (f"tenure of {LONG_TERM_MIN} months or more", lambda d: d["tenure"] >= LONG_TERM_MIN),
    "High monthly spenders": (f"monthly charges of {HIGH_SPENDER_MIN:.0f} or more",
                              lambda d: d["MonthlyCharges"] >= HIGH_SPENDER_MIN),
    "Month-to-month customers": ("month-to-month contract", lambda d: d["Contract"] == "Month-to-month"),
    "Multi-service customers": (f"{MULTI_SERVICE_MIN} or more subscribed services",
                                lambda d: d["service_count"] >= MULTI_SERVICE_MIN),
}


def segment_summary(df):
    """One row per segment plus an 'All customers' baseline. Segments overlap; do not add rows."""
    rows = []
    base_rate = _rate(int(df["churned"].sum()), len(df))
    entries = [("All customers", "every customer in the current selection", pd.Series(True, index=df.index))]
    entries += [(name, text, fn(df)) for name, (text, fn) in SEGMENTS.items()]
    for name, text, mask in entries:
        sub = df[mask]
        n, k = len(sub), int(sub["churned"].sum())
        rate = _rate(k, n)
        lo, hi = wilson_interval(k, n)
        rows.append({"segment": name, "definition": text, "customers": n,
                     "share_of_customers": _rate(n, len(df)), "churned": k, "churn_rate": rate,
                     "ci_low": lo, "ci_high": hi,
                     "vs_all_pp": (rate - base_rate) * 100 if n and not math.isnan(base_rate) else float("nan"),
                     "avg_monthly_charges": float(sub["MonthlyCharges"].mean()) if n else float("nan"),
                     "monthly_revenue": float(sub["MonthlyCharges"].sum()),
                     "small_group": n < MIN_N})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------- revenue
def revenue_actual(df):
    """ACTUAL historical figures from the records (no model involved).

    monthly_* use MonthlyCharges (the current monthly bill). billed_to_date uses TotalCharges
    (what customers have been charged so far); blank values are skipped, not imputed."""
    k = kpis(df)
    return {
        "monthly_revenue": k["monthly_revenue"],
        "monthly_revenue_active": k["monthly_revenue_active"],
        "monthly_revenue_churned": k["monthly_revenue_churned"],
        "churned_revenue_share": k["churned_revenue_share"],
        "billed_to_date": float(df["TotalCharges"].sum()),
        "billed_to_date_churned": float(df.loc[df["churned"], "TotalCharges"].sum()),
        "total_charges_missing": int(df["TotalCharges"].isna().sum()),
    }


def revenue_by_group(df, column, order=None):
    """ACTUAL monthly revenue per group, split into active and churned customers (sums to the total)."""
    rows = []
    for g in [g for g in (order or sorted(df[column].dropna().unique())) if (df[column] == g).any()]:
        sub = df[df[column] == g]
        churned = float(sub.loc[sub["churned"], "MonthlyCharges"].sum())
        total = float(sub["MonthlyCharges"].sum())
        rows.append({"group": g, "monthly_revenue": total, "active": total - churned, "churned": churned,
                     "churned_share": churned / total if total else float("nan")})
    return pd.DataFrame(rows, columns=["group", "monthly_revenue", "active", "churned", "churned_share"])


def _active_scored(df, scope):
    active = df[~df["churned"] & df["p_churn"].notna()]
    if scope == "test":
        active = active[active["in_test"]]
    elif scope != "all":
        raise ValueError("scope must be 'all' or 'test'")
    return active


def revenue_at_risk(df, scope="all"):
    """ESTIMATED exposure among ACTIVE customers the model can score.

    flagged_monthly: monthly revenue of active customers the model predicts will churn
        (probability >= saved threshold). An exposure figure, not a forecast.
    expected_monthly_loss: sum of probability x monthly charge (probability-weighted); only as
        reliable as the model's probabilities (a one-month, roughly calibrated estimate).
    annualised_exposure: flagged_monthly x 12, if every flagged customer left and nobody replaced them.
    scope 'test' uses only held-out customers (never seen in training); 'all' includes in-sample scores."""
    active = _active_scored(df, scope)
    flagged = active[active["flagged"]]
    active_monthly = float(active["MonthlyCharges"].sum())
    not_scored = df[~df["churned"] & df["p_churn"].isna()]
    return {
        "scope": scope,
        "active_scored": len(active),
        "active_monthly": active_monthly,
        "flagged_customers": len(flagged),
        "flagged_share_of_customers": _rate(len(flagged), len(active)),
        "flagged_monthly": float(flagged["MonthlyCharges"].sum()),
        "flagged_share_of_revenue": float(flagged["MonthlyCharges"].sum()) / active_monthly if active_monthly else float("nan"),
        "annualised_exposure": 12.0 * float(flagged["MonthlyCharges"].sum()),
        "expected_monthly_loss": float((active["p_churn"] * active["MonthlyCharges"]).sum()),
        "not_scored_customers": len(not_scored),
        "not_scored_monthly": float(not_scored["MonthlyCharges"].sum()),
    }


def at_risk_by_group(df, column, order=None, scope="all"):
    """ESTIMATED flagged monthly revenue per group next to the group's total active monthly revenue."""
    active = _active_scored(df, scope)
    rows = []
    for g in [g for g in (order or sorted(active[column].dropna().unique())) if (active[column] == g).any()]:
        sub = active[active[column] == g]
        f = sub[sub["flagged"]]
        rows.append({"group": g, "active_customers": len(sub), "flagged_customers": len(f),
                     "active_monthly": float(sub["MonthlyCharges"].sum()),
                     "flagged_monthly": float(f["MonthlyCharges"].sum()),
                     "expected_monthly_loss": float((sub["p_churn"] * sub["MonthlyCharges"]).sum())})
    return pd.DataFrame(rows, columns=["group", "active_customers", "flagged_customers", "active_monthly",
                                       "flagged_monthly", "expected_monthly_loss"])


def revenue_backtest(df):
    """Check the revenue-at-risk idea on HELD-OUT customers whose outcomes are known.

    Uses test-split labels for EVALUATION only (nothing is tuned). Compares the monthly revenue the
    model flagged with the monthly revenue that actually churned."""
    test = df[df["in_test"]]
    churned = test[test["churned"]]
    flagged = test[test["flagged"]]
    caught = flagged[flagged["churned"]]
    actual = float(churned["MonthlyCharges"].sum())
    flagged_total = float(flagged["MonthlyCharges"].sum())
    return {
        "test_customers": len(test),
        "actual_churned_monthly": actual,
        "flagged_monthly": flagged_total,
        "caught_monthly": float(caught["MonthlyCharges"].sum()),
        "revenue_recall": float(caught["MonthlyCharges"].sum()) / actual if actual else float("nan"),
        "revenue_precision": float(caught["MonthlyCharges"].sum()) / flagged_total if flagged_total else float("nan"),
        "mean_predicted_probability": float(test["p_churn"].mean()) if len(test) else float("nan"),
        "observed_churn_rate": float(test["churned"].mean()) if len(test) else float("nan"),
    }
