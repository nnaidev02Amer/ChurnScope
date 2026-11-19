"""Batch prediction in the running app: Predict page, Home Dashboard updates and History (headless).

Streamlit's AppTest cannot simulate the file-upload widget, so these tests replace st.file_uploader with a
stand-in that returns an UploadedFile-like object. Everything after the widget is the real app code."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "Scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import _batch_data as bd  # noqa: E402
import batch_predict as bp  # noqa: E402
import churn_insights as ci  # noqa: E402
import churn_pipeline as cp  # noqa: E402
import page_batch as pb  # noqa: E402
from ui_common import PAGE_HOME, PAGE_PREDICT  # noqa: E402

pytestmark = pytest.mark.skipif(not cp.MODEL_PATH.exists(), reason="run Scripts/train_churn_model.py first")
AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


class FakeUpload:
    def __init__(self, name, data):
        self.name, self._data, self.size = name, data, len(data)

    def getvalue(self):
        return self._data

    def read(self):
        return self._data


@pytest.fixture
def upload(monkeypatch):
    from streamlit.delta_generator import DeltaGenerator          # st.file_uploader and col.file_uploader both land here
    holder = {"file": None}
    monkeypatch.setattr(DeltaGenerator, "file_uploader", lambda self, label, type=None, key=None, **kw: holder["file"])

    def choose(name=None, data=None):
        holder["file"] = FakeUpload(name, data) if name else None
    return choose


@pytest.fixture(scope="module")
def bundle():
    return cp.load_model()


def new_app():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def ok(at):
    assert not at.exception, [e.value for e in at.exception]
    return at


def go(at, page):
    at.sidebar.radio(key="nav").set_value(page).run()
    return ok(at)


def mode(at, view):
    at.radio(key="predict_mode").set_value(view).run()
    return ok(at)


def text(at):
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info]
    parts += [w.value for w in at.warning] + [e.value for e in at.error] + [s.value for s in at.success]
    return "\n".join(parts)


def results_table(at):
    return next((d.value for d in at.dataframe if {"Churn probability (%)", "Prediction"} <= set(d.value.columns)), None)


def predict_file(at, upload, name, data):
    """Open the batch view, 'upload' a file and press Predict Churn."""
    go(at, PAGE_PREDICT)
    mode(at, pb.MODE_BATCH)
    upload(name, data)
    at.run()
    ok(at)
    at.button(key="batch_predict").click().run()
    return ok(at)


def expected(df, bundle, name="f.csv"):
    allowed = ci.allowed_categories(bundle)
    return bp.predict_batch(bundle, bp.read_upload(name, bd.csv_bytes(df)), allowed, name)


# ----------------------------------------------------------------------------- predict page
def test_predict_page_offers_single_batch_and_history_views():
    at = go(new_app(), PAGE_PREDICT)
    radio = at.radio(key="predict_mode")
    assert list(radio.options) == pb.MODES and radio.value == pb.MODE_SINGLE
    assert at.number_input(key="f_tenure") is not None and at.button(key="predict")       # single prediction preserved
    mode(at, pb.MODE_BATCH)
    t = text(at)
    assert "Predict many customers at once" in t and "CSV, Excel (.xlsx/.xls) or PDF" in t and "19 customer columns" in t
    assert "Upload a file to get started" in t and not [b for b in at.button if b.key == "batch_predict"]
    mode(at, pb.MODE_HISTORY)
    assert "No batch predictions yet" in text(at) and "No single-customer predictions yet" in text(at)


def test_csv_upload_is_validated_then_predicted(upload, bundle):
    df = bd.customers(50)
    at = go(new_app(), PAGE_PREDICT)
    mode(at, pb.MODE_BATCH)
    upload("customers.csv", bd.csv_bytes(df))
    at.run()
    ok(at)
    assert "50 customers" in text(at) and "all 19 required columns were found" in text(at)
    assert at.session_state["batches"] == []                                       # nothing predicted until the button is pressed
    at.button(key="batch_predict").click().run()
    ok(at)
    rec = at.session_state["batches"][0]
    exp = expected(df, bundle, "customers.csv")["summary"]
    assert (rec["total"], rec["predicted"], rec["leave"], rec["stay"], rec["invalid"]) == (
        50, 50, exp["leave"], exp["stay"], 0)
    t = text(at)
    for label in ("Total customers", "Successfully predicted", "Likely to leave", "Likely to stay", "customers.csv"):
        assert label in t, label
    assert any("50 customers predicted" in s.value or "50 customers" in s.value for s in at.success)
    table = results_table(at)
    assert len(table) == 50 and {"Row", "customerID", "Churn probability (%)", "Prediction", "Risk level"} <= set(table.columns)
    assert at.session_state["current_batch"] == rec["id"]


def test_batch_decisions_use_the_saved_threshold(upload, bundle):
    at = predict_file(new_app(), upload, "c.csv", bd.csv_bytes(bd.customers(80)))
    res = at.session_state["batches"][0]["results"]
    p = res["Churn probability (%)"] / 100
    assert ((p >= 0.39) == (res["Prediction"] == bp.LEAVE)).all()
    assert (res["Risk level"] == "High").equals(res["Prediction"] == bp.LEAVE)


def test_view_filters_show_leavers_stayers_and_problem_rows(upload, bundle):
    df = bd.with_changes(bd.customers(30, keep_churn_column=False), 4, Contract="Weekly")
    at = predict_file(new_app(), upload, "c.csv", bd.csv_bytes(df))
    s = at.session_state["batches"][0]
    assert len(results_table(at)) == 30
    for choice, n in (("Likely to leave", s["leave"]), ("Likely to stay", s["stay"]), ("Not predicted", 1),
                      ("All customers", 30)):
        at.radio(key="batch_filter").set_value(choice).run()
        ok(at)
        assert len(results_table(at)) == n, choice


def test_excel_and_pdf_uploads_give_the_same_answers_as_csv(upload, bundle):
    df = bd.customers(12, keep_churn_column=False)
    exp = expected(df, bundle)["summary"]
    for name, data in (("c.xlsx", bd.xlsx_bytes(df)), ("c.pdf", bd.pdf_bytes(df))):
        at = predict_file(new_app(), upload, name, data)
        s = at.session_state["batches"][0]
        assert (s["total"], s["leave"], s["stay"]) == (12, exp["leave"], exp["stay"]), name
        assert s["filename"] == name


def test_missing_columns_block_the_prediction_with_a_clear_message(upload):
    df = bd.customers(5).drop(columns=["Contract", "tenure"])
    at = go(new_app(), PAGE_PREDICT)
    mode(at, pb.MODE_BATCH)
    upload("c.csv", bd.csv_bytes(df))
    at.run()
    ok(at)
    msg = " ".join(e.value for e in at.error)
    assert "2 of the 19 required columns are missing" in msg and "Contract" in msg and "tenure" in msg
    assert not [b for b in at.button if b.key == "batch_predict"] and at.session_state["batches"] == []


@pytest.mark.parametrize("name,data,expect", [("notes.txt", b"hello", "not a supported file type"),
                                              ("empty.csv", b"", "empty"),
                                              ("broken.xlsx", b"nope", "Excel file could not be read"),
                                              ("broken.pdf", b"%PDF-1.4 x", "PDF could not be read"),
                                              ("headers.csv", b"gender,tenure\n", "no customer rows")])
def test_unusable_files_show_an_error_and_no_button(upload, name, data, expect):
    at = go(new_app(), PAGE_PREDICT)
    mode(at, pb.MODE_BATCH)
    upload(name, data)
    at.run()
    ok(at)
    assert any(expect in e.value for e in at.error) and not [b for b in at.button if b.key == "batch_predict"]


def test_invalid_rows_are_reported_with_reasons_and_kept(upload, bundle):
    df = bd.customers(20, keep_churn_column=False)
    df = bd.with_changes(df, 0, Contract="Weekly")
    df = bd.with_changes(df, 5, tenure=-3)
    df = bd.with_changes(df, 9, gender="")
    at = predict_file(new_app(), upload, "c.csv", bd.csv_bytes(df))
    s = at.session_state["batches"][0]
    assert (s["total"], s["predicted"], s["invalid"]) == (20, 17, 3)
    assert any("3 customer row(s) could not be predicted" in w.value for w in at.warning)
    problems = next(d.value for d in at.dataframe if list(d.value.columns) == ["Row", "customerID", "Issue"])
    assert list(problems["Row"]) == [2, 7, 11] and "Weekly" in problems.loc[0, "Issue"]
    assert len(results_table(at)) == 20                                             # nothing silently discarded


def test_a_file_where_every_row_is_invalid_is_handled(upload):
    df = bd.customers(5, keep_churn_column=False)
    df["Contract"] = "Weekly"
    at = predict_file(new_app(), upload, "bad.csv", bd.csv_bytes(df))
    assert any("None of the customers could be predicted" in e.value for e in at.error)
    assert at.session_state["batches"][0]["predicted"] == 0


def test_choosing_a_new_file_does_not_run_it_automatically(upload):
    at = predict_file(new_app(), upload, "a.csv", bd.csv_bytes(bd.customers(10)))
    upload("b.csv", bd.csv_bytes(bd.customers(15, offset=10)))
    at.run()
    ok(at)
    assert len(at.session_state["batches"]) == 1 and "You selected a different file" in text(at)


# ----------------------------------------------------------------------------- Home Dashboard
def test_home_updates_automatically_after_a_batch(upload, bundle):
    at = new_app()
    assert "No batch has been predicted yet" in text(at) and len(at.get("plotly_chart")) == 3
    df = bd.customers(60)
    predict_file(at, upload, "week1.csv", bd.csv_bytes(df))
    go(at, PAGE_HOME)                                                  # no manual refresh anywhere
    s = expected(df, bundle)["summary"]
    t = text(at)
    for expected_text in ("Latest Batch Prediction", "Model estimates", "week1.csv", "Uploaded customers", "Likely to leave",
                          "Likely to stay", "Predicted churn rate", "Avg chance of leaving", "60",
                          f"{s['leave']:,}", f"{s['stay']:,}", f"{s['churn_pct']:.1f}%", f"{s['avg_probability']:.1f}%"):
        assert expected_text in t, expected_text
    assert len(at.get("plotly_chart")) == 4                            # donut for the batch + the 3 dataset charts
    assert "What this batch shows" in t and f"{s['leave']:,} of {s['predicted']:,} customers" in t


def test_home_keeps_original_dataset_statistics_separate_and_unchanged(upload):
    at = new_app()
    before = text(at)
    predict_file(at, upload, "x.csv", bd.csv_bytes(bd.customers(40)))
    go(at, PAGE_HOME)
    t = text(at)
    for fact in ("7,043", "1,869", "5,174", "26.5%", "Original dataset: customers who already left or stayed", "Recorded data",
                 "Uploading a file never changes them"):
        assert fact in t, fact
    assert t.index("Latest Batch Prediction") < t.index("Original dataset")           # two distinct sections
    for fact in ("7,043", "1,869", "5,174"):
        assert fact in before


def test_home_always_shows_the_latest_successful_batch(upload, bundle):
    at = new_app()
    predict_file(at, upload, "first.csv", bd.csv_bytes(bd.customers(20)))
    predict_file(at, upload, "second.csv", bd.csv_bytes(bd.customers(35, offset=20)))
    go(at, PAGE_HOME)
    s2 = expected(bd.customers(35, offset=20), bundle)["summary"]
    assert "second.csv" in text(at) and "first.csv" not in text(at) and f"{s2['churn_pct']:.1f}%" in text(at)
    bad = bd.customers(4, keep_churn_column=False)
    bad["Contract"] = "Weekly"
    predict_file(at, upload, "broken.csv", bd.csv_bytes(bad))
    go(at, PAGE_HOME)
    assert "second.csv" in text(at) and "broken.csv" not in text(at)           # a failed run never replaces the summary


def test_home_button_opens_that_batch_in_history(upload):
    at = new_app()
    predict_file(at, upload, "a.csv", bd.csv_bytes(bd.customers(10)))
    go(at, PAGE_HOME)
    at.button(key="home_to_history").click().run()
    ok(at)
    assert at.session_state["nav"] == PAGE_PREDICT and at.radio(key="predict_mode").value == pb.MODE_HISTORY
    assert at.selectbox(key="hist_batch").value == at.session_state["batches"][0]["id"]


def test_home_upload_shortcut_opens_the_batch_view():
    at = new_app()
    at.button(key="home_to_batch").click().run()
    ok(at)
    assert at.session_state["nav"] == PAGE_PREDICT and at.radio(key="predict_mode").value == pb.MODE_BATCH


def test_batch_page_button_goes_to_the_home_dashboard(upload):
    at = predict_file(new_app(), upload, "a.csv", bd.csv_bytes(bd.customers(10)))
    at.button(key="batch_to_home").click().run()
    ok(at)
    assert at.session_state["nav"] == PAGE_HOME and "Latest Batch Prediction" in text(at)


# ----------------------------------------------------------------------------------- History
def test_history_lists_batches_and_single_predictions(upload, bundle):
    at = new_app()
    go(at, PAGE_PREDICT)
    at.button(key="predict").click().run()                                         # one single-customer prediction
    ok(at)
    d1, d2 = bd.customers(25), bd.customers(40, offset=25)
    predict_file(at, upload, "one.csv", bd.csv_bytes(d1))
    predict_file(at, upload, "two.xlsx", bd.xlsx_bytes(d2))
    mode(at, pb.MODE_HISTORY)
    runs = next(d.value for d in at.dataframe if "Likely to leave" in d.value.columns and "File" in d.value.columns)
    assert list(runs["File"]) == ["two.xlsx", "one.csv"] and list(runs["Uploaded"]) == [40, 25]       # newest first
    assert set(runs.columns) == {"Run", "File", "Date and time", "Uploaded", "Predicted", "Likely to leave",
                                 "Likely to stay", "Churn %"}
    s2 = expected(d2, bundle)["summary"]
    assert runs.loc[0, "Likely to leave"] == s2["leave"] and runs.loc[0, "Churn %"] == f"{s2['churn_pct']:.1f}%"
    single = next(d.value for d in at.dataframe if "Chance of leaving (%)" in d.value.columns and "tenure" in d.value.columns)
    assert len(single) == 1
    assert [b for b in at.get("download_button")]                                    # downloads are offered


def test_a_previous_batch_can_be_opened_and_its_results_inspected(upload, bundle):
    at = new_app()
    d1, d2 = bd.customers(25), bd.customers(40, offset=25)
    predict_file(at, upload, "one.csv", bd.csv_bytes(d1))
    predict_file(at, upload, "two.csv", bd.csv_bytes(d2))
    mode(at, pb.MODE_HISTORY)
    first_id = at.session_state["batches"][0]["id"]
    assert at.selectbox(key="hist_batch").value == at.session_state["batches"][1]["id"]      # newest selected by default
    at.selectbox(key="hist_batch").select(first_id).run()
    ok(at)
    assert "one.csv" in text(at) and len(results_table(at)) == 25
    exp = expected(d1, bundle)
    assert list(results_table(at)["Prediction"]) == list(exp["results"]["Prediction"])
    at.radio(key="hist_filter").set_value("Likely to leave").run()
    ok(at)
    assert len(results_table(at)) == exp["summary"]["leave"]


def test_download_of_an_earlier_batch_contains_every_row_and_column(upload):
    df = bd.with_changes(bd.customers(18), 2, Contract="Weekly")
    at = predict_file(new_app(), upload, "old.csv", bd.csv_bytes(df))
    predict_file(at, upload, "newer.csv", bd.csv_bytes(bd.customers(5, offset=30)))
    mode(at, pb.MODE_HISTORY)
    at.selectbox(key="hist_batch").select(at.session_state["batches"][0]["id"]).run()
    ok(at)
    buttons = [b for b in at.get("download_button") if "Download all results" in b.proto.label]
    assert buttons, "the earlier batch must offer a download button"
    import io
    import pandas as pd
    rec = at.session_state["batches"][0]                                            # the earlier run selected above
    out = pd.read_csv(io.BytesIO(bp.results_to_csv(rec["results"])), dtype=str, keep_default_na=False)
    assert len(out) == 18 and list(out.columns) == list(rec["results"].columns)        # the bytes the button serves
    assert out.loc[2, "Status"] == "Not predicted"


def test_history_survives_navigation_and_reruns(upload):
    at = predict_file(new_app(), upload, "keep.csv", bd.csv_bytes(bd.customers(15)))
    n = len(at.session_state["batches"])
    for page in (PAGE_HOME, "🎓  How This Project Works", "📊  Understand Customer Churn", PAGE_PREDICT):
        go(at, page)
    at.run()
    assert len(at.session_state["batches"]) == n == 1
    mode(at, pb.MODE_HISTORY)
    assert "keep.csv" in text(at) or any("keep.csv" in str(d.value.to_dict()) for d in at.dataframe)
    go(at, PAGE_HOME)
    assert "keep.csv" in text(at)                                                     # Home still has the latest summary


def test_clearing_batch_history_resets_the_dashboard(upload):
    at = predict_file(new_app(), upload, "gone.csv", bd.csv_bytes(bd.customers(15)))
    mode(at, pb.MODE_HISTORY)
    at.button(key="clear_batches").click().run()
    ok(at)
    assert at.session_state["batches"] == []
    go(at, PAGE_HOME)
    assert "No batch has been predicted yet" in text(at) and "gone.csv" not in text(at)


def test_batch_and_single_histories_are_kept_separate(upload):
    at = predict_file(new_app(), upload, "b.csv", bd.csv_bytes(bd.customers(10)))
    assert len(at.session_state["history"]) == 0                                       # a batch is not a single prediction
    mode(at, pb.MODE_SINGLE)
    at.button(key="predict").click().run()
    ok(at)
    assert len(at.session_state["history"]) == 1 and len(at.session_state["batches"]) == 1


def test_the_history_view_and_chosen_mode_are_remembered_when_leaving_the_page(upload):
    at = predict_file(new_app(), upload, "m.csv", bd.csv_bytes(bd.customers(8)))
    go(at, PAGE_HOME)
    go(at, PAGE_PREDICT)
    assert at.radio(key="predict_mode").value == pb.MODE_BATCH
    assert "m.csv" in text(at)                                                         # last batch results still shown
