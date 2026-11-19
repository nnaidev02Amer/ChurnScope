"""ChurnScope: a simple local Streamlit app for the telecom churn model.

    python -m streamlit run app.py        (or double-click run_app.bat)

Four pages: Home Dashboard, Predict Customer Churn, Understand Customer Churn, How This Project Works.
Uses the saved model bundle (models/churn_pipeline.joblib) and verified metrics (reports/metrics.json).
It never retrains anything. Academic prototype.
"""
import sys
from pathlib import Path

import pandas as pd
import sklearn
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "Scripts"))
import churn_analytics as ca  # noqa: E402
import churn_insights as ci  # noqa: E402
import churn_pipeline as cp  # noqa: E402
import page_home  # noqa: E402
import page_how  # noqa: E402
import page_predict  # noqa: E402
import page_understand  # noqa: E402
from ui_common import (CSS, LOGO, PAGE_HOME, PAGE_HOW, PAGE_PREDICT, PAGE_UNDERSTAND, PAGES)  # noqa: E402

st.set_page_config(page_title="ChurnScope | Telecom customer churn", page_icon="📡", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

METRICS_PATH = cp.PROJECT_DIR / "reports" / "metrics.json"
COMPARISON_PATH = cp.PROJECT_DIR / "reports" / "model_comparison_cv.csv"

if not cp.MODEL_PATH.exists() or not METRICS_PATH.exists():
    st.error("The trained model or its metrics were not found. Run `python Scripts/train_churn_model.py` "
             "once to create `models/churn_pipeline.joblib` and `reports/metrics.json`, then reload.")
    st.stop()


# --------------------------------------------------------------- cached, read-only data
@st.cache_resource
def get_bundle():
    return cp.load_model()


@st.cache_data
def get_customers():
    """All customers from the original CSV (observed data only)."""
    return ca.load_customers()


@st.cache_data
def get_reference():
    """How often training customers with each trait churned (used for the 'what stands out' notes)."""
    X, y = cp.load_and_clean()
    X_tr, _, y_tr, _ = cp.split_data(X, y)
    return ci.build_reference_stats(X_tr, y_tr)


@st.cache_data
def get_calibration():
    """Average predicted vs observed churn on the held-out test customers (reads data only)."""
    X, y = cp.load_and_clean()
    _, X_te, _, y_te = cp.split_data(X, y)
    proba = get_bundle()["pipeline"].predict_proba(X_te[cp.FEATURES])[:, 1]
    return {"mean_predicted": float(proba.mean()), "observed": float(y_te.mean())}


@st.cache_data
def get_comparison():
    return pd.read_csv(COMPARISON_PATH) if COMPARISON_PATH.exists() else None


bundle = get_bundle()
ctx = {"bundle": bundle, "allowed": ci.allowed_categories(bundle), "customers": get_customers(),
       "stats": get_reference(), "metrics": ci.load_metrics(), "comparison": get_comparison(),
       "calibration": get_calibration()}

# --------------------------------------------------------------------- sidebar navigation
with st.sidebar:
    st.markdown(f'<div class="cs-brand">{LOGO}<div><b>ChurnScope</b><span>Telecom customer churn</span></div></div>',
                unsafe_allow_html=True)
    st.session_state.setdefault("nav", PAGE_HOME)
    st.radio("Go to", PAGES, key="nav", label_visibility="collapsed")
    st.markdown('<div class="cs-side-note">Academic final-year project.<br>Predictions are estimates, not guarantees.</div>',
                unsafe_allow_html=True)

if bundle.get("sklearn_version") != sklearn.__version__:
    st.warning(f"The model was saved with scikit-learn {bundle.get('sklearn_version')} but this environment "
               f"has {sklearn.__version__}. Predictions may differ; re-run training if results look off.")

# ---------------------------------------------------------------------------- router
{PAGE_HOME: page_home, PAGE_PREDICT: page_predict, PAGE_UNDERSTAND: page_understand, PAGE_HOW: page_how}[
    st.session_state["nav"]].render(ctx)
