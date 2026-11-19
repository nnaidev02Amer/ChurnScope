"""Content tests for Home, Understand and How-it-works. Figures are checked against the raw CSV
(stdlib csv, independent of pandas) and reports/metrics.json."""
import csv
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Scripts"))
import churn_analytics as ca  # noqa: E402
import churn_pipeline as cp  # noqa: E402
import page_home  # noqa: E402
import page_understand  # noqa: E402
from ui_common import PAGE_HOME, PAGE_HOW, PAGE_PREDICT, PAGE_UNDERSTAND  # noqa: E402

pytestmark = pytest.mark.skipif(not cp.MODEL_PATH.exists(), reason="run Scripts/train_churn_model.py first")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture(scope="module")
def raw():
    with open(cp.DATA_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@pytest.fixture(scope="module")
def metrics():
    return json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))


def open_page(page):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    if page != PAGE_HOME:
        at.sidebar.radio(key="nav").set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def text(at):
    return "\n".join([m.value for m in at.markdown] + [c.value for c in at.caption])


def rate(raw, pred):
    g = [r for r in raw if pred(r)]
    return sum(r["Churn"] == "Yes" for r in g) / len(g)


# --------------------------------------------------------------------------- Home
def test_home_shows_only_the_four_numbers_and_three_charts():
    at = open_page(PAGE_HOME)
    t = text(at)
    for expected in ("7,043", "1,869", "5,174", "26.5%", "Customers who left", "Still using the service", "Total customers",
                     "Overall churn rate", "What is customer churn?"):
        assert expected in t, expected
    assert len(at.get("plotly_chart")) == 3
    assert not any(w in t for w in ("Revenue", "Segment", "filter", "ROC-AUC", "interval"))   # no clutter or jargon


def test_home_figures_match_the_raw_csv(raw):
    f = page_home.home_figures(ca.load_customers())
    left = [r for r in raw if r["Churn"] == "Yes"]
    stayed = [r for r in raw if r["Churn"] == "No"]
    assert (f["total"], f["left"], f["stayed"]) == (len(raw), len(left), len(stayed)) == (7043, 1869, 5174)
    assert f["rate"] == pytest.approx(len(left) / len(raw))
    assert f["avg_bill_left"] == pytest.approx(sum(float(r["MonthlyCharges"]) for r in left) / len(left))
    assert f["avg_bill_stayed"] == pytest.approx(sum(float(r["MonthlyCharges"]) for r in stayed) / len(stayed))
    assert f["avg_bill_left"] > f["avg_bill_stayed"]                  # the caption on the page says exactly this
    for name, n in f["by_contract"].items():
        assert n == sum(r["Contract"] == name for r in raw)
    assert max(f["by_contract"], key=f["by_contract"].get) == "Month-to-month"   # caption claim


def test_home_buttons_open_the_other_sections():
    at = open_page(PAGE_HOME)
    at.button(key="home_to_predict").click().run()
    assert not at.exception and at.session_state["nav"] == PAGE_PREDICT and "<h1>Predict Customer Churn</h1>" in text(at)
    at = open_page(PAGE_HOME)
    at.button(key="home_to_how").click().run()
    assert at.session_state["nav"] == PAGE_HOW and "<h1>How This Project Works</h1>" in text(at)


# ----------------------------------------------------------------------- Understand
def test_understand_has_exactly_three_analyses_each_explained():
    at = open_page(PAGE_UNDERSTAND)
    t = text(at)
    assert len(at.get("plotly_chart")) == 3
    assert t.count("What this chart shows:") == 3 and t.count("Takeaway:") == 3
    for title in ("1. Churn by contract type", "2. Churn by how long the customer has stayed (tenure)",
                  "3. Churn by internet service"):
        assert title in t
    assert "not proof of what causes" in t
    assert not any(w in t for w in ("Revenue", "Segment", "heatmap", "95%"))


def test_takeaway_sentences_match_the_data(raw):
    tk = page_understand.takeaways(ca.load_customers())
    m2m = rate(raw, lambda r: r["Contract"] == "Month-to-month")
    two = rate(raw, lambda r: r["Contract"] == "Two year")
    assert f"{m2m:.0%}" in tk["contract"] and f"{two:.0%}" in tk["contract"] and "month-to-month" in tk["contract"]
    first = rate(raw, lambda r: int(r["tenure"]) <= 12)
    last = rate(raw, lambda r: int(r["tenure"]) >= 49)
    assert f"{first:.0%}" in tk["tenure"] and f"{last:.0%}" in tk["tenure"]
    fiber = rate(raw, lambda r: r["InternetService"] == "Fiber optic")
    none = rate(raw, lambda r: r["InternetService"] == "No")
    assert "fiber optic" in tk["internet"] and f"{fiber:.0%}" in tk["internet"] and f"{none:.0%}" in tk["internet"]
    assert "no internet service" in tk["internet"]
    for s in tk.values():                                           # claims of cause must not appear
        assert "because" not in s and "causes" not in s


def test_understand_page_displays_the_takeaways():
    at = open_page(PAGE_UNDERSTAND)
    t = text(at)
    for s in page_understand.takeaways(ca.load_customers()).values():
        assert s in t


def test_chart_rates_equal_dataset_rates(raw):
    r = page_understand.rates(ca.load_customers(), "Contract")
    for name in ("Month-to-month", "One year", "Two year"):
        assert r[name][0] == pytest.approx(rate(raw, lambda x, n=name: x["Contract"] == n))
        assert r[name][1] == sum(x["Contract"] == name for x in raw)


# ---------------------------------------------------------------------------- How it works
def test_how_page_has_the_workflow_and_all_ten_topics():
    at = open_page(PAGE_HOW)
    t = text(at)
    order = ["Customer Data", "Data Preprocessing", "ML Model", "Churn Probability", "Prediction", "Retention Suggestion"]
    positions = [t.index(f"<b>{n}</b>") for n in order]
    assert positions == sorted(positions)                           # workflow nodes appear in the required order
    assert t.count("&rarr;") == 5
    topics = ["What does customer churn mean?", "Why do telecom companies want to predict churn?", "What dataset is used?",
              "What customer details does the model use?", "How is the data prepared (preprocessing)?",
              "Why were several machine-learning algorithms compared?", "How does the final model predict churn?",
              "What does the 39% threshold mean?", "How well does the model work? (evaluation in simple words)",
              "What are the limitations?"]
    idx = [t.index(x) for x in topics]
    assert idx == sorted(idx)
    assert len(re.findall(r'<div class="num">(\d+)</div>', t)) == 10


def test_how_page_numbers_come_from_verified_metrics(metrics):
    t = text(open_page(PAGE_HOW))
    op, dflt = metrics["test_tuned_threshold"], metrics["test_default_threshold"]
    cm = op["confusion_matrix"]
    for expected in (f"{op['accuracy']:.1%}", f"{op['recall']:.1%}", f"{op['precision']:.1%}", f"{op['roc_auc']:.3f}",
                     f"{op['f1']:.2f}", f"{metrics['majority_baseline_accuracy_test']:.1%}", f"{cm['tn']:,}", f"{cm['fp']:,}",
                     f"{cm['fn']:,}", f"{cm['tp']:,}", f"{metrics['n_train']:,}", f"{metrics['n_test']:,}",
                     f"{metrics['n_rows_after_cleaning']:,}", f"{dflt['recall']:.0%}"):
        assert expected in t, expected
    assert "77.5%" in t and "72.0%" in t and "56.0%" in t and "0.841" in t     # the published headline results


def test_how_page_explains_metrics_in_plain_language():
    t = text(open_page(PAGE_HOW))
    assert "of customers who actually left, how many did the model identify?" in t
    assert "of customers predicted to leave, how many actually left?" in t
    assert "Precision" in t and "Recall" in t and "Accuracy" in t           # technical terms kept, each explained
    assert "alert line" in t and "39%" in t and "19 details" in t


def test_how_page_lists_all_ten_algorithms_and_the_limitations():
    t = text(open_page(PAGE_HOW))
    for name in ("Logistic Regression", "Naive Bayes", "Decision Tree", "Random Forest", "AdaBoost", "Gradient Boosting",
                 "K-Nearest Neighbours", "Support Vector Machine (linear)", "Support Vector Machine (curved)", "Voting team"):
        assert name in t, name
    for expected in ("One public dataset", "estimates, not guarantees", "Probabilities run a little high",
                     "Patterns are not causes", "Not for real decisions"):
        assert expected in t, expected
    assert re.search(r"average predicted chance of leaving was \d+\.\d%, while \d+\.\d% really left", t)


def test_calibration_figures_are_measured_not_invented():
    X, y = cp.load_and_clean()
    _, X_te, _, y_te = cp.split_data(X, y)
    proba = cp.load_model()["pipeline"].predict_proba(X_te[cp.FEATURES])[:, 1]
    t = text(open_page(PAGE_HOW))
    assert f"{proba.mean():.1%}" in t and f"{y_te.mean():.1%}" in t


def test_no_page_mentions_removed_features():
    for page in (PAGE_HOME, PAGE_PREDICT, PAGE_UNDERSTAND, PAGE_HOW):
        t = text(open_page(page))
        for gone in ("Revenue insights", "Customer segments", "Customer filters", "Compare two dimensions", "Reset filters"):
            assert gone not in t, (page, gone)
