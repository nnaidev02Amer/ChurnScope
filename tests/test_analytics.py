"""Phase 2 analytics: calculations checked against the raw CSV (read with the stdlib csv module,
independently of pandas), segment definitions, filters, revenue accounting, scoring and integrity."""
import csv
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Scripts"))
import churn_analytics as ca  # noqa: E402
import churn_insights as ci  # noqa: E402
import churn_pipeline as cp  # noqa: E402

needs_model = pytest.mark.skipif(not cp.MODEL_PATH.exists(), reason="run Scripts/train_churn_model.py first")
ADDONS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


@pytest.fixture(scope="module")
def raw():
    with open(cp.DATA_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def df():
    return ca.load_customers()


@pytest.fixture(scope="module")
def scored():
    return ca.scored_customers(cp.load_model())


def one_row(**changes):
    """A single real customer row with some fields overridden (for boundary tests)."""
    row = ca.load_customers().iloc[[0]].copy()
    for k, v in changes.items():
        row[k] = v
    row["tenure_group"] = row["tenure"].map(ca.tenure_group)
    row["service_count"] = ca.service_count(row)
    return row


# ----------------------------------------------------------------- loading / missing values
def test_load_counts_every_customer_once(df, raw):
    assert len(df) == len(raw) == 7043
    assert "customerID" not in df.columns                       # identifier is not carried into analytics
    assert df["churned"].sum() == sum(r["Churn"] == "Yes" for r in raw) == 1869


def test_blank_total_charges_become_nan_not_zero(df, raw):
    blanks = [r for r in raw if r["TotalCharges"].strip() == ""]
    assert len(blanks) == 11 and all(r["tenure"] == "0" for r in blanks)
    assert df["TotalCharges"].isna().sum() == 11
    assert df.loc[df["tenure"] == 0, "TotalCharges"].isna().all()   # never silently filled with 0
    assert df["MonthlyCharges"].notna().all()


def test_loader_rejects_duplicates_and_bad_labels(tmp_path):
    rows = list(csv.DictReader(open(cp.DATA_PATH, newline="", encoding="utf-8")))
    dup = tmp_path / "dup.csv"
    with open(dup, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows + [rows[0]])
    with pytest.raises(ValueError, match="Duplicate"):
        ca.load_customers(dup)
    bad = tmp_path / "bad.csv"
    rows[0]["Churn"] = "Maybe"
    with open(bad, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    with pytest.raises(ValueError, match="Churn"):
        ca.load_customers(bad)


def test_sums_skip_missing_values():
    t = pd.DataFrame({"churned": [True, False, False], "MonthlyCharges": [10.0, np.nan, 30.0],
                      "TotalCharges": [100.0, np.nan, 50.0]})
    k = ca.kpis(t)
    assert k["monthly_revenue"] == 40.0 and k["monthly_revenue_churned"] == 10.0 and k["monthly_revenue_active"] == 30.0
    assert k["avg_monthly_charges"] == 20.0                      # mean over the 2 non-missing values
    rev = ca.revenue_actual(t)
    assert rev["billed_to_date"] == 150.0 and rev["total_charges_missing"] == 1


# -------------------------------------------------------------------------- overview KPIs
def test_kpis_match_raw_csv(df, raw):
    k = ca.kpis(df)
    churned = [r for r in raw if r["Churn"] == "Yes"]
    monthly = sum(float(r["MonthlyCharges"]) for r in raw)
    assert (k["customers"], k["churned"], k["active"]) == (7043, 1869, 5174)
    assert k["churn_rate"] == pytest.approx(1869 / 7043)
    assert k["avg_monthly_charges"] == pytest.approx(monthly / 7043)
    assert k["monthly_revenue"] == pytest.approx(monthly)
    assert k["monthly_revenue_churned"] == pytest.approx(sum(float(r["MonthlyCharges"]) for r in churned))
    assert k["monthly_revenue_active"] + k["monthly_revenue_churned"] == pytest.approx(k["monthly_revenue"])
    assert round(k["monthly_revenue"], 2) == 456116.60 and round(k["churn_rate"] * 100, 2) == 26.54


def test_kpis_on_empty_selection_do_not_crash(df):
    k = ca.kpis(df.iloc[0:0])
    assert k["customers"] == 0 and math.isnan(k["churn_rate"]) and k["monthly_revenue"] == 0


# ------------------------------------------------------------------- churn analysis
@pytest.mark.parametrize("label", list(ca.DIMENSIONS))
def test_churn_by_partitions_customers_exactly(df, label):
    col, order = ca.DIMENSIONS[label]
    t = ca.churn_by(df, col, order)
    assert t["customers"].sum() == len(df)                      # every customer in exactly one group
    assert t["churned"].sum() == df["churned"].sum()
    assert t["monthly_revenue"].sum() == pytest.approx(df["MonthlyCharges"].sum())
    assert t["share"].sum() == pytest.approx(1.0)
    assert (t["churn_rate"] == t["churned"] / t["customers"]).all()
    assert (t["ci_low"] <= t["churn_rate"]).all() and (t["churn_rate"] <= t["ci_high"]).all()


def test_churn_rates_match_independent_calculation(df, raw):
    def rate(pred):
        grp = [r for r in raw if pred(r)]
        return len(grp), sum(r["Churn"] == "Yes" for r in grp) / len(grp)
    t = ca.churn_by(df, "Contract", ca.DIMENSIONS["Contract"][1]).set_index("group")
    for name in ("Month-to-month", "One year", "Two year"):
        n, r = rate(lambda x, name=name: x["Contract"] == name)
        assert t.loc[name, "customers"] == n and t.loc[name, "churn_rate"] == pytest.approx(r)
    assert round(t.loc["Month-to-month", "churn_rate"] * 100, 1) == 42.7
    s = ca.churn_by(df, "senior", ["No", "Yes"]).set_index("group")
    n, r = rate(lambda x: x["SeniorCitizen"] == "1")
    assert s.loc["Yes", "customers"] == n and s.loc["Yes", "churn_rate"] == pytest.approx(r)
    pay = ca.churn_by(df, "PaymentMethod", ca.DIMENSIONS["Payment method"][1]).set_index("group")
    n, r = rate(lambda x: x["PaymentMethod"] == "Electronic check")
    assert pay.loc["Electronic check", "churn_rate"] == pytest.approx(r)
    assert round(pay.loc["Electronic check", "churn_rate"] * 100, 1) == 45.3


def test_tenure_groups_boundaries_and_zero_tenure():
    assert [ca.tenure_group(m) for m in (0, 1, 12, 13, 24, 25, 48, 49, 72)] == [
        "0-12 months", "0-12 months", "0-12 months", "13-24 months", "13-24 months", "25-48 months",
        "25-48 months", "49-72 months", "49-72 months"]


def test_wilson_interval_known_values():
    lo, hi = ca.wilson_interval(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    lo, hi = ca.wilson_interval(5, 10)
    assert (lo, hi) == pytest.approx((0.2366, 0.7634), abs=1e-3)
    assert all(math.isnan(v) for v in ca.wilson_interval(0, 0))
    lo, hi = ca.wilson_interval(10, 10)
    assert hi == 1.0 and lo < 1.0


def test_small_groups_are_flagged(df):
    tiny = df.iloc[:10]
    t = ca.churn_by(tiny, "Contract", ca.DIMENSIONS["Contract"][1])
    assert t["small_group"].all()
    assert not ca.churn_by(df, "Contract", ca.DIMENSIONS["Contract"][1])["small_group"].any()


def test_cross_churn_cells_are_mutually_exclusive(df):
    (ca_, oa), (cb, ob) = ca.DIMENSIONS["Contract"], ca.DIMENSIONS["Tenure group"]
    c = ca.cross_churn(df, ca_, cb, oa, ob)
    assert c["customers"].sum() == len(df) and c["churned"].sum() == df["churned"].sum()
    cell = c[(c["a"] == "Two year") & (c["b"] == "0-12 months")].iloc[0]
    assert cell["customers"] == ((df["Contract"] == "Two year") & (df["tenure"] <= 12)).sum()


# ---------------------------------------------------------------------- segmentation
def test_service_count_matches_independent_count(df, raw):
    expected = []
    for r in raw:
        n = (r["PhoneService"] == "Yes") + (r["MultipleLines"] == "Yes") + (r["InternetService"] != "No")
        n += sum(r[c] == "Yes" for c in ADDONS)
        expected.append(int(n))
    assert df["service_count"].tolist() == expected
    assert df["service_count"].min() >= 0 and df["service_count"].max() <= 9


@pytest.mark.parametrize("seg,field,inside,outside", [
    ("New customers", "tenure", 12, 13),
    ("Long-term customers", "tenure", 49, 48),
    ("High monthly spenders", "MonthlyCharges", 90.0, 89.99),
    ("Month-to-month customers", "Contract", "Month-to-month", "One year"),
])
def test_segment_boundaries(seg, field, inside, outside):
    fn = ca.SEGMENTS[seg][1]
    assert bool(fn(one_row(**{field: inside})).iloc[0]) is True
    assert bool(fn(one_row(**{field: outside})).iloc[0]) is False


def test_multi_service_boundary():
    fn = ca.SEGMENTS["Multi-service customers"][1]
    base = dict(PhoneService="Yes", MultipleLines="No", InternetService="DSL", **{c: "No" for c in ADDONS})
    four = one_row(**{**base, "OnlineSecurity": "Yes", "OnlineBackup": "Yes"})            # phone + internet + 2 = 4
    five = one_row(**{**base, "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "TechSupport": "Yes"})
    assert four["service_count"].iloc[0] == 4 and not fn(four).iloc[0]
    assert five["service_count"].iloc[0] == 5 and fn(five).iloc[0]


def test_segment_summary_matches_independent_counts(df, raw):
    s = ca.segment_summary(df).set_index("segment")
    n = lambda pred: sum(1 for r in raw if pred(r))  # noqa: E731
    ch = lambda pred: sum(1 for r in raw if pred(r) and r["Churn"] == "Yes")  # noqa: E731
    checks = {"New customers": lambda r: int(r["tenure"]) <= 12,
              "Long-term customers": lambda r: int(r["tenure"]) >= 49,
              "High monthly spenders": lambda r: float(r["MonthlyCharges"]) >= 90,
              "Month-to-month customers": lambda r: r["Contract"] == "Month-to-month"}
    for name, pred in checks.items():
        assert s.loc[name, "customers"] == n(pred) and s.loc[name, "churned"] == ch(pred), name
        assert s.loc[name, "churn_rate"] == pytest.approx(ch(pred) / n(pred))
        assert s.loc[name, "monthly_revenue"] == pytest.approx(sum(float(r["MonthlyCharges"]) for r in raw if pred(r)))
    assert s.loc["All customers", "customers"] == 7043 and s.loc["All customers", "vs_all_pp"] == 0
    assert s.loc["New customers", "churn_rate"] > s.loc["All customers", "churn_rate"] > s.loc["Long-term customers", "churn_rate"]


def test_segments_overlap_so_rows_must_not_be_summed(df):
    s = ca.segment_summary(df)
    seg_total = s[s["segment"] != "All customers"]["customers"].sum()
    assert seg_total > len(df)                                   # overlapping memberships
    both = ca.SEGMENTS["New customers"][1](df) & ca.SEGMENTS["Month-to-month customers"][1](df)
    assert both.sum() > 0                                        # a real customer sits in two segments


def test_segment_summary_handles_empty_selection(df):
    s = ca.segment_summary(df.iloc[0:0])
    assert (s["customers"] == 0).all() and s["churn_rate"].isna().all()


# --------------------------------------------------------------------------- filters
def test_filters(df):
    assert len(ca.apply_filters(df)) == len(df)
    m = ca.apply_filters(df, contract=["Month-to-month"])
    assert len(m) == 3875 and (m["Contract"] == "Month-to-month").all()
    assert ca.apply_filters(df, contract=[]).empty                                  # empty selection = nobody
    combo = ca.apply_filters(df, contract=["Month-to-month"], internet=["Fiber optic"], senior=["Yes"],
                             monthly_range=(70, 100))
    manual = df[(df["Contract"] == "Month-to-month") & (df["InternetService"] == "Fiber optic")
                & (df["SeniorCitizen"] == 1) & df["MonthlyCharges"].between(70, 100)]
    assert combo.index.equals(manual.index)
    edge = ca.apply_filters(df, monthly_range=(df["MonthlyCharges"].min(), df["MonthlyCharges"].max()))
    assert len(edge) == len(df)                                                     # bounds inclusive


# ----------------------------------------------------------------------- revenue accounting
def test_actual_revenue_adds_up(df, raw):
    r = ca.revenue_actual(df)
    assert r["monthly_revenue_active"] + r["monthly_revenue_churned"] == pytest.approx(r["monthly_revenue"])
    assert r["billed_to_date"] == pytest.approx(sum(float(x["TotalCharges"]) for x in raw if x["TotalCharges"].strip()))
    assert round(r["billed_to_date"], 2) == 16056168.70 and r["total_charges_missing"] == 11
    assert r["churned_revenue_share"] == pytest.approx(r["monthly_revenue_churned"] / r["monthly_revenue"])
    for label in ca.DIMENSIONS:
        col, order = ca.DIMENSIONS[label]
        g = ca.revenue_by_group(df, col, order)
        assert g["monthly_revenue"].sum() == pytest.approx(r["monthly_revenue"])      # nothing double counted
        assert (g["active"] + g["churned"]).sum() == pytest.approx(r["monthly_revenue"])
        assert g["churned"].sum() == pytest.approx(r["monthly_revenue_churned"])


# --------------------------------------------------------------- scoring / estimated revenue
@needs_model
def test_scored_frame_is_consistent_with_the_model_and_split(scored):
    X, y = cp.load_and_clean()
    _, X_te, _, y_te = cp.split_data(X, y)
    assert len(scored) == 7043 and scored["p_churn"].notna().sum() == 7032 and scored["in_test"].sum() == 2110
    unscored = scored[scored["p_churn"].isna()]
    assert len(unscored) == 11 and (unscored["tenure"] == 0).all()
    assert (unscored["risk_band"] == "Not scored").all() and not unscored["flagged"].any()
    assert not unscored["churned"].any()                                              # all 11 are active
    proba = cp.load_model()["pipeline"].predict_proba(X[cp.FEATURES])[:, 1]          # features only, no labels
    assert np.allclose(scored.loc[scored["p_churn"].notna(), "p_churn"].to_numpy(), proba)
    test = scored[scored["in_test"]]
    assert sorted(test["churned"].astype(int).tolist()) == sorted(y_te.tolist())     # same customers as the model's test set
    assert (test["flagged"] == (test["p_churn"] >= 0.39)).all()


@needs_model
def test_flagged_counts_match_verified_metrics(scored):
    m = ci.load_metrics()["test_tuned_threshold"]["confusion_matrix"]
    t = scored[scored["in_test"]]
    assert int((t["flagged"] & t["churned"]).sum()) == m["tp"] and int((t["flagged"] & ~t["churned"]).sum()) == m["fp"]
    assert int((~t["flagged"] & t["churned"]).sum()) == m["fn"]


@needs_model
def test_revenue_at_risk_counts_only_active_customers_once(scored):
    r = ca.revenue_at_risk(scored, "all")
    active = scored[~scored["churned"] & scored["p_churn"].notna()]
    assert r["active_scored"] == len(active) == 5163 and r["not_scored_customers"] == 11
    flagged = active[active["p_churn"] >= 0.39]
    assert r["flagged_customers"] == len(flagged)
    assert r["flagged_monthly"] == pytest.approx(flagged["MonthlyCharges"].sum())
    assert r["flagged_monthly"] <= r["active_monthly"]
    assert r["annualised_exposure"] == pytest.approx(12 * r["flagged_monthly"])
    assert r["expected_monthly_loss"] == pytest.approx((active["p_churn"] * active["MonthlyCharges"]).sum())
    assert r["not_scored_monthly"] == pytest.approx(455.6)
    # churned customers never contribute: revenue at risk is separate from revenue already lost
    churned_flagged = scored[scored["churned"] & scored["flagged"]]["MonthlyCharges"].sum()
    assert churned_flagged > 0 and r["flagged_monthly"] == pytest.approx(flagged["MonthlyCharges"].sum())


@needs_model
def test_test_scope_is_a_subset_of_all_scope(scored):
    a, t = ca.revenue_at_risk(scored, "all"), ca.revenue_at_risk(scored, "test")
    assert t["active_scored"] == 1549 and t["flagged_customers"] <= a["flagged_customers"]
    assert t["flagged_monthly"] <= a["flagged_monthly"]
    with pytest.raises(ValueError):
        ca.revenue_at_risk(scored, "bogus")


@needs_model
@pytest.mark.parametrize("label", list(ca.DIMENSIONS))
def test_at_risk_by_group_adds_up_to_total(scored, label):
    col, order = ca.DIMENSIONS[label]
    for scope in ("all", "test"):
        g, total = ca.at_risk_by_group(scored, col, order, scope), ca.revenue_at_risk(scored, scope)
        assert g["flagged_monthly"].sum() == pytest.approx(total["flagged_monthly"])
        assert g["active_monthly"].sum() == pytest.approx(total["active_monthly"])
        assert g["expected_monthly_loss"].sum() == pytest.approx(total["expected_monthly_loss"])


@needs_model
def test_backtest_uses_held_out_customers_and_is_consistent(scored):
    bt = ca.revenue_backtest(scored)
    X, y = cp.load_and_clean()
    _, X_te, _, y_te = cp.split_data(X, y)
    raw_df = pd.read_csv(cp.DATA_PATH)
    pos = raw_df.index[raw_df["tenure"] > 0].to_numpy()[X_te.index.to_numpy()]
    churned_rev = raw_df.loc[pos][y_te.to_numpy() == 1]["MonthlyCharges"].sum()      # independent route
    assert bt["test_customers"] == 2110 and bt["actual_churned_monthly"] == pytest.approx(churned_rev)
    assert 0 < bt["revenue_recall"] <= 1 and 0 < bt["revenue_precision"] <= 1
    assert bt["caught_monthly"] <= min(bt["actual_churned_monthly"], bt["flagged_monthly"])
    assert bt["observed_churn_rate"] == pytest.approx(y_te.mean())


@needs_model
def test_analytics_do_not_modify_data_model_or_metrics():
    files = [cp.DATA_PATH, cp.MODEL_PATH, cp.PROJECT_DIR / "reports" / "metrics.json"]
    before = [hashlib.sha256(f.read_bytes()).hexdigest() for f in files]
    s = ca.scored_customers(cp.load_model())
    ca.revenue_at_risk(s); ca.revenue_backtest(s); ca.segment_summary(s)
    assert [hashlib.sha256(f.read_bytes()).hexdigest() for f in files] == before
