"""Tests for the app logic: validation, threshold, risk bands, observations, suggestions."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Scripts"))
import churn_insights as ci  # noqa: E402
import churn_pipeline as cp  # noqa: E402

pytestmark = pytest.mark.skipif(not cp.MODEL_PATH.exists(), reason="run Scripts/train_churn_model.py first")

RISKY = {"tenure": 2, "MonthlyCharges": 85.0, "TotalCharges": 170.0, "gender": "Male", "SeniorCitizen": "No",
         "Partner": "No", "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
         "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
         "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
         "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check"}
LOYAL = {**RISKY, "tenure": 62, "MonthlyCharges": 68.0, "TotalCharges": 4216.0, "Dependents": "Yes",
         "OnlineSecurity": "Yes", "TechSupport": "Yes", "Contract": "Two year", "InternetService": "DSL",
         "PaymentMethod": "Bank transfer (automatic)"}


@pytest.fixture(scope="module")
def bundle():
    return cp.load_model()


@pytest.fixture(scope="module")
def allowed(bundle):
    return ci.allowed_categories(bundle)


@pytest.fixture(scope="module")
def split():
    X, y = cp.load_and_clean()
    return cp.split_data(X, y)


class StubPipeline:
    """Returns a fixed churn probability so the threshold can be tested exactly."""
    def __init__(self, p):
        self.p = p

    def predict_proba(self, frame):
        return np.tile([1 - self.p, self.p], (len(frame), 1))


def test_saved_threshold_is_039(bundle):
    assert bundle["threshold"] == 0.39


@pytest.mark.parametrize("p,expected", [(0.0, False), (0.389, False), (0.3899999, False),
                                         (0.39, True), (0.391, True), (0.5, True), (1.0, True)])
def test_threshold_boundary_is_respected(p, expected):
    stub = {"pipeline": StubPipeline(p), "threshold": 0.39}
    out = cp.predict_churn(stub, RISKY)
    assert bool(out.loc[0, "churn"]) is expected
    assert out.loc[0, "churn_probability"] == pytest.approx(p)


def test_threshold_override_and_bare_pipeline_default():
    assert not cp.predict_churn({"pipeline": StubPipeline(0.45), "threshold": 0.5}, RISKY).loc[0, "churn"]
    assert cp.predict_churn({"pipeline": StubPipeline(0.45), "threshold": 0.5}, RISKY, threshold=0.4).loc[0, "churn"]
    assert not cp.predict_churn(StubPipeline(0.45), RISKY).loc[0, "churn"]  # bare pipeline -> 0.5


def test_real_model_decision_matches_threshold(bundle, split):
    _, X_te, _, _ = split
    out = cp.predict_churn(bundle, X_te)
    assert (out["churn"].to_numpy() == (out["churn_probability"].to_numpy() >= 0.39)).all()


@pytest.mark.parametrize("p,band", [(0.0, "Low"), (0.199, "Low"), (0.20, "Medium"), (0.389, "Medium"),
                                    (0.39, "High"), (0.99, "High")])
def test_risk_band_cutoffs(p, band):
    assert ci.risk_band(p, 0.39) == band


def test_high_band_equals_churn_decision(bundle, split):
    _, X_te, _, _ = split
    out = cp.predict_churn(bundle, X_te)
    bands = [ci.risk_band(p, 0.39) for p in out["churn_probability"]]
    assert [b == "High" for b in bands] == out["churn"].tolist()


def test_band_calibration_is_consistent_with_verified_metrics(bundle, split):
    _, X_te, _, y_te = split
    calib = ci.band_calibration(bundle, X_te, y_te).set_index("Band")
    m = ci.load_metrics()["test_tuned_threshold"]["confusion_matrix"]
    assert calib["Customers"].sum() == len(y_te)
    assert calib.loc["High", "Customers"] == m["tp"] + m["fp"]          # flagged customers
    rates = calib["Observed churn rate"]
    assert rates["Low"] < rates["Medium"] < rates["High"]                # bands are ordered


def test_allowed_categories_come_from_model(allowed):
    assert allowed["Contract"] == ["Month-to-month", "One year", "Two year"]
    assert set(allowed) == set(cp.CATEGORICAL_FEATURES)


def test_valid_customer_passes_and_scores(bundle, allowed):
    v = ci.validate_customer(RISKY, allowed)
    assert v.ok and not v.warnings
    out = cp.predict_churn(bundle, v.customer)
    assert 0 <= out.loc[0, "churn_probability"] <= 1


@pytest.mark.parametrize("field,value", [("Contract", "Weekly"), ("PaymentMethod", "Crypto"), ("gender", ""),
                                         ("InternetService", "5G"), ("SeniorCitizen", "Maybe")])
def test_unseen_categories_are_rejected(allowed, field, value):
    v = ci.validate_customer({**RISKY, field: value}, allowed)
    assert not v.ok and any(field in e for e in v.errors)


@pytest.mark.parametrize("field,value", [("tenure", 0), ("tenure", 73), ("tenure", 1.5), ("tenure", "abc"),
                                         ("tenure", float("nan")), ("tenure", None), ("MonthlyCharges", 0),
                                         ("MonthlyCharges", -5), ("MonthlyCharges", "x"),
                                         ("MonthlyCharges", float("inf")), ("TotalCharges", -1),
                                         ("TotalCharges", True)])
def test_invalid_numbers_are_rejected(allowed, field, value):
    assert not ci.validate_customer({**RISKY, field: value}, allowed).ok


def test_missing_and_unknown_fields(allowed):
    partial = {k: v for k, v in RISKY.items() if k != "Contract"}
    assert not ci.validate_customer(partial, allowed).ok
    assert not ci.validate_customer({**RISKY, "customerID": "7590-VHVEG"}, allowed).ok


def test_service_dependencies(allowed):
    assert not ci.validate_customer({**RISKY, "PhoneService": "No"}, allowed).ok            # lines != 'No phone service'
    assert not ci.validate_customer({**RISKY, "MultipleLines": "No phone service"}, allowed).ok
    assert not ci.validate_customer({**RISKY, "InternetService": "No"}, allowed).ok         # add-ons not 'No internet service'
    no_net = {**RISKY, "InternetService": "No", **{c: "No internet service" for c in ci.ADDON_COLUMNS}}
    assert ci.validate_customer(no_net, allowed).ok
    assert not ci.validate_customer({**RISKY, "TechSupport": "No internet service"}, allowed).ok


def test_warnings_do_not_block(allowed):
    v = ci.validate_customer({**RISKY, "MonthlyCharges": 150.0, "TotalCharges": 1000.0}, allowed)
    assert v.ok and len(v.warnings) == 2   # out-of-range monthly charge, total far from tenure x monthly


def test_extreme_but_valid_inputs_still_score(bundle, allowed):
    for t, m in ((1, 18.25), (72, 118.75)):
        v = ci.validate_customer({**RISKY, "tenure": t, "MonthlyCharges": m, "TotalCharges": t * m}, allowed)
        assert v.ok
        assert 0 <= cp.predict_churn(bundle, v.customer).loc[0, "churn_probability"] <= 1


def test_explanations_are_descriptive_and_grounded(split):
    X_tr, _, y_tr, _ = split
    stats = ci.build_reference_stats(X_tr, y_tr)
    assert stats["overall"] == pytest.approx(y_tr.mean())
    exp = ci.explain_risk(RISKY, stats)
    factors = {i["factor"] for i in exp["risk"]}
    assert "Contract" in factors or "Tenure" in factors
    for i in exp["risk"]:
        assert i["rate"] / stats["overall"] == pytest.approx(i["lift"]) and i["lift"] >= ci.RISK_LIFT and i["n"] >= ci.MIN_GROUP
    loyal = ci.explain_risk(LOYAL, stats)
    assert any(i["value"] == "Two year" for i in loyal["protective"])
    assert not loyal["risk"] or all(i["factor"] != "Contract" for i in loyal["risk"])


def test_retention_suggestions_follow_band_and_traits():
    high = ci.retention_suggestions(RISKY, "High")
    titles = [i["title"] for i in high["ideas"]]
    assert any("contract" in t.lower() for t in titles) and "High risk" in high["headline"]
    low = ci.retention_suggestions(LOYAL, "Low")
    assert len(low["ideas"]) <= 2 and "no intervention" in low["headline"].lower()
    assert all(i["title"] and i["detail"] for b in ("Low", "Medium", "High") for i in ci.retention_suggestions(RISKY, b)["ideas"])
