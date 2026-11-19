"""Fast correctness tests for the churn pipeline (no long training). Run: python -m pytest tests"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Scripts"))
import churn_pipeline as cp  # noqa: E402


@pytest.fixture(scope="module")
def data():
    X, y = cp.load_and_clean()
    return X, y, cp.split_data(X, y)


def test_cleaning(data):
    X, y, _ = data
    assert len(X) == 7032 and len(y) == 7032          # 11 tenure==0 rows removed
    assert list(X.columns) == cp.FEATURES             # customerID and target not in features
    assert not X.isnull().any().any()
    assert set(y.unique()) == {0, 1}
    assert X["TotalCharges"].dtype.kind == "f"


def test_split_is_stratified_disjoint_and_seeded(data):
    X, y, (X_tr, X_te, y_tr, y_te) = data
    assert len(X_tr) + len(X_te) == len(X)
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert abs(y_tr.mean() - y.mean()) < 0.002 and abs(y_te.mean() - y.mean()) < 0.002
    again = cp.split_data(X, y)
    assert list(again[0].index) == list(X_tr.index)   # same seed -> same split


def test_preprocessing_is_fitted_on_training_data_only(data):
    _, _, (X_tr, X_te, y_tr, _) = data
    pipe = cp.make_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    scaler = pipe.named_steps["prep"].named_transformers_["num"]
    assert np.allclose(scaler.mean_, X_tr[cp.NUMERIC_FEATURES].mean().values)
    assert not np.allclose(scaler.mean_, X_te[cp.NUMERIC_FEATURES].mean().values)
    # transformed TEST data is NOT re-centred on itself (the original notebook's bug)
    test_scaled = pipe.named_steps["prep"].transform(X_te)[:, : len(cp.NUMERIC_FEATURES)]
    assert abs(test_scaled.mean(axis=0)).max() > 1e-3


def test_unseen_category_does_not_crash(data):
    _, _, (X_tr, X_te, y_tr, _) = data
    pipe = cp.make_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    row = X_te.iloc[[0]].copy()
    row["PaymentMethod"] = "Crypto"
    assert pipe.predict_proba(row).shape == (1, 2)


def test_same_seed_gives_identical_predictions(data):
    _, _, (X_tr, X_te, y_tr, _) = data
    models = [cp.base_models()["Gradient Boosting"], cp.base_models()["Random Forest"]]
    for m in models:
        a = cp.make_pipeline(clone(m)).fit(X_tr, y_tr).predict_proba(X_te)
        b = cp.make_pipeline(clone(m)).fit(X_tr, y_tr).predict_proba(X_te)
        # exact for boosting; the multi-threaded forest differs only by ~1e-16 rounding
        assert np.allclose(a, b, rtol=0, atol=1e-12)


def test_all_models_seeded_and_voting_is_soft():
    for name, m in cp.base_models().items():
        params = m.get_params()
        if "random_state" in params:
            assert params["random_state"] == cp.SEED, name
    assert cp.base_models()["Voting (GB+LR+Ada)"].voting == "soft"


def test_evaluate_and_predict_churn(data):
    _, _, (X_tr, X_te, y_tr, y_te) = data
    pipe = cp.make_pipeline(LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
    m = cp.evaluate(pipe, X_te, y_te)
    cm = m["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y_te)
    assert 0.8 < m["roc_auc"] < 0.9
    out = cp.predict_churn(pipe, X_te.iloc[0].to_dict())
    assert 0 <= out.loc[0, "churn_probability"] <= 1


def test_predict_churn_uses_saved_threshold_by_default(data):
    if not cp.MODEL_PATH.exists():
        pytest.skip("run Scripts/train_churn_model.py first")
    bundle = cp.load_model()
    _, _, (_, X_te, _, _) = data
    rows = X_te.head(300)
    default = cp.predict_churn(bundle, rows)
    proba = default["churn_probability"].to_numpy()
    assert np.array_equal(default["churn"].to_numpy(), proba >= bundle["threshold"])
    assert default["churn"].sum() > cp.predict_churn(bundle, rows, threshold=0.5)["churn"].sum()
    # a bare pipeline (no saved threshold) falls back to 0.5
    bare = cp.predict_churn(bundle["pipeline"], rows)
    assert np.array_equal(bare["churn"].to_numpy(), proba >= cp.DEFAULT_THRESHOLD)


def test_saved_model_round_trip(data):
    model_file = cp.PROJECT_DIR / "models" / "churn_pipeline.joblib"
    if not model_file.exists():
        pytest.skip("run Scripts/train_churn_model.py first")
    import joblib
    bundle = joblib.load(model_file)
    _, _, (_, X_te, _, _) = data
    proba = bundle["pipeline"].predict_proba(X_te)[:, 1]
    assert proba.shape == (len(X_te),) and 0 <= bundle["threshold"] <= 1
