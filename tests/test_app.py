"""Headless tests of the simplified app: navigation and the Predict page (the model is unchanged)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Scripts"))
import churn_pipeline as cp  # noqa: E402
from ui_common import PAGE_HOME, PAGE_HOW, PAGE_PREDICT, PAGE_UNDERSTAND, PAGES  # noqa: E402

pytestmark = pytest.mark.skipif(not cp.MODEL_PATH.exists(), reason="run Scripts/train_churn_model.py first")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


def new_app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def go(at, page):
    at.sidebar.radio(key="nav").set_value(page).run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def predict_page():
    return go(new_app(), PAGE_PREDICT)


def predict(at):
    at.button(key="predict").click().run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def md(at):
    return "\n".join(m.value for m in at.markdown)


# --------------------------------------------------------------------- navigation
def test_exactly_four_sections_and_home_is_default():
    at = new_app()
    assert PAGES == [PAGE_HOME, PAGE_PREDICT, PAGE_UNDERSTAND, PAGE_HOW]
    radio = at.sidebar.radio(key="nav")
    assert list(radio.options) == PAGES and len(radio.options) == 4
    assert radio.value == PAGE_HOME
    assert len(at.tabs) == 0                                     # the old seven-tab layout is gone
    assert "Home Dashboard" in md(at)


@pytest.mark.parametrize("page,title", [(PAGE_HOME, "Home Dashboard"), (PAGE_PREDICT, "Predict Customer Churn"),
                                        (PAGE_UNDERSTAND, "Understand Customer Churn"), (PAGE_HOW, "How This Project Works")])
def test_every_section_opens_without_errors(page, title):
    at = go(new_app(), page)
    assert f"<h1>{title}</h1>" in md(at) and not at.error


# --------------------------------------------------------- prediction (unchanged ML behaviour)
def test_default_prediction_matches_model_and_threshold():
    at = predict(predict_page())
    res = at.session_state["result"]
    expected = cp.predict_churn(cp.load_model(), res["customer"]).iloc[0]
    assert res["p"] == pytest.approx(expected["churn_probability"])
    assert res["churn"] == (res["p"] >= 0.39) and res["thr"] == 0.39
    assert len(at.session_state["history"]) == 1


def test_result_is_explained_in_plain_english():
    at = predict(predict_page())
    res = at.session_state["result"]
    text = md(at)
    assert "Chance this customer may leave" in text and f"{res['p']:.0%}" in text
    assert ("Likely to leave" if res["churn"] else "Likely to stay") in text
    assert "alert line" in text and "39%" in text
    assert "Ideas to keep this customer" in text and "What stands out about this customer" in text


def test_presets_give_expected_decisions():
    at = predict_page()
    at.button(key="preset_New fibre customer").click().run()
    predict(at)
    risky = at.session_state["result"]
    assert risky["churn"] and "Likely to leave" in md(at)
    at.button(key="preset_Long-term loyal customer").click().run()
    predict(at)
    loyal = at.session_state["result"]
    assert not loyal["churn"] and "Likely to stay" in md(at)
    assert risky["p"] > 0.39 > loyal["p"] and len(at.session_state["history"]) == 2


def test_one_prominent_predict_button():
    at = predict_page()
    assert len([b for b in at.button if b.key == "predict"]) == 1


def test_service_dependencies_are_enforced_in_the_form():
    at = predict_page()
    at.selectbox(key="f_PhoneService").select("No").run()
    assert at.session_state["f_MultipleLines"] == "No phone service"
    at.selectbox(key="f_InternetService").select("No").run()
    assert all(at.session_state["f_" + c] == "No internet service"
               for c in ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"))
    predict(at)
    assert not at.error and at.session_state["result"] is not None
    at.selectbox(key="f_InternetService").select("DSL").run()
    assert at.session_state["f_TechSupport"] == "No"
    at.selectbox(key="f_PhoneService").select("Yes").run()
    assert at.session_state["f_MultipleLines"] == "No"
    predict(at)
    assert not at.error


def test_manual_total_charges_warning_does_not_block():
    at = predict_page()
    at.checkbox(key="f_total_auto").uncheck().run()
    at.number_input(key="f_TotalCharges").set_value(5.0).run()
    predict(at)
    assert at.warning and at.session_state["result"] is not None


def test_stale_inputs_are_flagged():
    at = predict(predict_page())
    at.number_input(key="f_tenure").set_value(60).run()
    assert "You changed the form since this result" in md(at)


def test_history_label_is_sanitised_and_clearable():
    at = predict_page()
    at.text_input(key="f_label").set_value("=HYPERLINK(1)").run()
    predict(at)
    assert at.session_state["history"][0]["label"].startswith("'=")        # no CSV formula injection
    at.button(key="clear").click().run()
    assert at.session_state["history"] == [] and at.session_state["result"] is None


def test_history_lives_on_the_predict_page_and_shows_latest_five():
    at = predict_page()
    for _ in range(7):
        predict(at)
    assert len(at.session_state["history"]) == 7
    table = next(d.value for d in at.dataframe if "Chance of leaving (%)" in d.value.columns)
    assert len(table) == 5 and list(table["#"]) == [3, 4, 5, 6, 7]


def test_widgets_clamp_out_of_range_numbers():
    at = predict_page()
    at.number_input(key="f_tenure").set_value(0).run()
    assert 1 <= at.session_state["f_tenure"] <= 72
    at.number_input(key="f_tenure").set_value(500).run()
    assert 1 <= at.session_state["f_tenure"] <= 72
    predict(at)
    assert 1 <= at.session_state["result"]["customer"]["tenure"] <= 72


def test_validation_errors_block_prediction_and_are_shown(monkeypatch):
    # The form cannot produce an invalid profile (widgets only offer known values and clamp numbers),
    # so inject a failing validator to test the blocking/error path. Rules are covered in test_insights.py.
    import churn_insights as ci
    monkeypatch.setattr(ci, "validate_customer",
                        lambda customer, allowed: ci.Validation(errors=["Tenure must be a whole number from 1 to 72."]))
    at = predict_page()
    at.button(key="predict").click().run()
    assert not at.exception
    assert at.session_state["result"] is None and at.session_state["history"] == []
    assert any("Tenure" in e.value for e in at.error)


def test_form_entries_and_result_survive_leaving_the_page():
    at = predict_page()
    at.number_input(key="f_tenure").set_value(40).run()
    at.selectbox(key="f_Contract").select("Two year").run()
    predict(at)
    p = at.session_state["result"]["p"]
    go(at, PAGE_HOME)
    go(at, PAGE_PREDICT)
    assert at.session_state["f_tenure"] == 40 and at.session_state["f_Contract"] == "Two year"
    assert at.session_state["result"]["p"] == p and len(at.session_state["history"]) == 1
    assert f"{p:.0%}" in md(at)
