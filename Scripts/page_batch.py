"""Batch prediction and prediction history views (used by the Predict Customer Churn page).

Run history lives in st.session_state, so it survives page navigation and Streamlit reruns for as long
as the browser tab stays open. Nothing is written to disk.
"""
import pandas as pd
import streamlit as st

import batch_predict as bp
from ui_common import GREEN, PAGE_HOME, PAGE_PREDICT, RED, callout, card, esc

MODE_SINGLE, MODE_BATCH, MODE_HISTORY = "Single customer", "Batch prediction (upload a file)", "History"
MODES = [MODE_SINGLE, MODE_BATCH, MODE_HISTORY]
FILTERS = ["All customers", "Likely to leave", "Likely to stay", "Not predicted"]
FILTER_MASK = {"Likely to leave": lambda r: r["Prediction"] == bp.LEAVE,
               "Likely to stay": lambda r: r["Prediction"] == bp.STAY,
               "Not predicted": lambda r: r["Status"] == "Not predicted"}


def batches():
    return st.session_state.setdefault("batches", [])


def latest_batch():
    """The most recent run that produced at least one prediction (drives the Home Dashboard)."""
    return bp.latest_with_predictions(st.session_state.get("batches", []))


def by_id(batch_id):
    return next((r for r in batches() if r["id"] == batch_id), None)


def show_in_history(batch_id):
    """Callback: open the Predict page's History view on one batch run."""
    st.session_state["nav"] = PAGE_PREDICT
    st.session_state["predict_mode"] = MODE_HISTORY
    st.session_state["hist_batch"] = batch_id


def clear_batches():
    """Callback: forget every batch run (the Home Dashboard goes back to its 'no batch yet' state)."""
    st.session_state["batches"] = []
    st.session_state.pop("current_batch", None)


def go_home():
    st.session_state["nav"] = PAGE_HOME


def go_batch():
    """Callback: open the Predict page's batch upload view."""
    st.session_state["nav"] = PAGE_PREDICT
    st.session_state["predict_mode"] = MODE_BATCH


def filtered(results, choice):
    return results if choice == "All customers" else results[results.apply(FILTER_MASK[choice], axis=1)]


def show_batch(rec, prefix):
    """Summary cards, problem rows, a filterable customer table and a CSV download for one batch run."""
    res = rec["results"]
    st.markdown(f"##### Results for **{esc(rec['filename'])}** &middot; run {esc(rec['time'])}", unsafe_allow_html=True)
    c = st.columns(4)
    c[0].markdown(card("Total customers", f"{rec['total']:,}", "rows in the uploaded file", size="1.9rem"), unsafe_allow_html=True)
    c[1].markdown(card("Successfully predicted", f"{rec['predicted']:,}",
                       f"{rec['invalid']:,} could not be predicted" if rec["invalid"] else "every customer predicted", GREEN,
                       "1.9rem"), unsafe_allow_html=True)
    c[2].markdown(card("Likely to leave", f"{rec['leave']:,}", "at or above the 39% alert line", RED, "1.9rem"),
                  unsafe_allow_html=True)
    c[3].markdown(card("Likely to stay", f"{rec['stay']:,}", "below the 39% alert line", GREEN, "1.9rem"),
                  unsafe_allow_html=True)

    if rec["invalid"]:
        st.warning(f"{rec['invalid']:,} customer row(s) could not be predicted. They are kept in the table and in the "
                   "CSV download, marked 'Not predicted', with the reason.")
        with st.expander(f"See the {rec['invalid']:,} row(s) with problems"):
            st.dataframe(rec["invalid_rows"], hide_index=True, width="stretch")

    choice = st.radio("Show", FILTERS, horizontal=True, key=f"{prefix}_filter")
    view = filtered(res, choice)
    st.caption(f"Showing {len(view):,} of {len(res):,} customers. Churn probability is the model's estimate of the chance "
               "that the customer may leave; the prediction applies the 39% alert line.")
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "Churn probability (%)": st.column_config.ProgressColumn("Churn probability (%)", min_value=0, max_value=100,
                                                                 format="%.1f")})
    st.download_button("Download all results as CSV", bp.results_to_csv(res),
                       file_name=f"churn_predictions_{rec['filename'].rsplit('.', 1)[0]}.csv", mime="text/csv",
                       key=f"{prefix}_download", help="Every uploaded column plus the prediction columns, all rows.")


def render_batch(ctx):
    bundle, allowed = ctx["bundle"], ctx["allowed"]
    st.markdown(callout(
        "<b class='k'>Predict many customers at once.</b> Upload a <b>CSV, Excel (.xlsx/.xls) or PDF</b> file with one row per "
        "customer. It needs the model's <b>19 customer columns</b> (the same details as the single-customer form: gender, "
        "SeniorCitizen, Partner, Dependents, tenure, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, "
        "DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod, "
        "MonthlyCharges, TotalCharges). A customer ID column is optional, and extra columns are kept in the results. "
        "PDFs must contain a text table, not a scanned picture."), unsafe_allow_html=True)

    left, right = st.columns([3, 1])
    up = left.file_uploader("Upload a customer file", type=bp.ACCEPTED_TYPES, key="batch_file",
                            help=f"Up to {bp.MAX_FILE_BYTES // (1024 * 1024)} MB and {bp.MAX_ROWS:,} customers per file.")
    right.write("")
    right.download_button("Download a template", bp.template_csv(), file_name="customer_template.csv", mime="text/csv",
                          key="batch_template", help="A CSV with the right columns and two example customers.",
                          width="stretch")

    if up is not None:
        data, name = up.getvalue(), up.name
        try:
            df = bp.read_upload(name, data)
            bp.map_columns(df)                         # raises UploadError if any of the 19 columns is missing
        except bp.UploadError as e:
            st.error(str(e))
        else:
            st.success(f"✔ **{len(df):,} customers** detected in **{esc(name)}**, and all 19 required columns were found.")
            with st.expander("Preview the first 5 rows"):
                st.dataframe(df.head(5), hide_index=True, width="stretch")
            if st.button("🎯  Predict Churn", type="primary", key="batch_predict", width="stretch"):
                with st.spinner(f"Predicting churn for {len(df):,} customers..."):
                    batch = bp.predict_batch(bundle, df, allowed, name)
                seq = st.session_state.get("batch_seq", 0) + 1
                st.session_state["batch_seq"] = seq
                rec = bp.make_record(seq, name, batch, bp.file_hash(data))
                st.session_state["batches"] = bp.add_record(batches(), rec)
                st.session_state["current_batch"] = seq
                if rec["predicted"]:
                    st.success(f"Done: {rec['predicted']:,} customers predicted. The Home Dashboard now shows a summary "
                               "of this batch.")
                else:
                    st.error("None of the customers could be predicted. See the list of problems below.")

    rec = by_id(st.session_state.get("current_batch"))
    if rec is None:
        if up is None:
            st.caption("Upload a file to get started. Results will appear here.")
        return
    if up is not None and bp.file_hash(up.getvalue()) != rec["digest"]:
        st.info("You selected a different file. Press **Predict Churn** to run it; the results below are from the previous run.")
    st.divider()
    show_batch(rec, "batch")
    st.button("📊  See this on the Home Dashboard", key="batch_to_home", on_click=go_home)


def render_history(ctx):
    recs = batches()
    st.markdown("#### Batch predictions")
    if not recs:
        st.markdown(callout("No batch predictions yet. Upload a customer file in <b>Batch prediction</b> and each run "
                            "will be listed here."), unsafe_allow_html=True)
    else:
        table = pd.DataFrame([{"Run": r["id"], "File": r["filename"], "Date and time": r["time"],
                               "Uploaded": r["total"], "Predicted": r["predicted"], "Likely to leave": r["leave"],
                               "Likely to stay": r["stay"],
                               "Churn %": "n/a" if r["predicted"] == 0 else f"{r['churn_pct']:.1f}%"} for r in reversed(recs)])
        st.dataframe(table, hide_index=True, width="stretch")
        ids = [r["id"] for r in reversed(recs)]
        if st.session_state.get("hist_batch") not in ids:
            st.session_state["hist_batch"] = ids[0]
        labels = {r["id"]: f"Run {r['id']}: {r['filename']} ({r['time']})" for r in recs}
        chosen = st.selectbox("Open a previous run to see its customers", ids, key="hist_batch",
                              format_func=lambda i: labels.get(i, f"Run {i}"))
        st.write("")
        show_batch(by_id(chosen), "hist")
        st.button("Clear batch history", key="clear_batches", on_click=clear_batches)

    st.divider()
    st.markdown("#### Single-customer predictions")
    hist = st.session_state.get("history", [])
    if not hist:
        st.markdown(callout("No single-customer predictions yet. Use <b>Single customer</b> to make one."),
                    unsafe_allow_html=True)
    else:
        rows = [{"#": i, "Time": r["time"], "Name": r["label"], "Chance of leaving (%)": round(r["p"] * 100, 1),
                 "Decision": "Likely to leave" if r["churn"] else "Likely to stay", **r["customer"]}
                for i, r in enumerate(hist, 1)]
        full = pd.DataFrame(rows)
        st.dataframe(full, hide_index=True, width="stretch")
        st.download_button("Download single-customer predictions (CSV)", bp.results_to_csv(full),
                           file_name="churnscope_single_predictions.csv", mime="text/csv", key="single_download")
    st.caption("History is kept while this page stays open (it is not saved to disk), so download anything you want to keep. "
               f"The {bp.MAX_KEPT_BATCHES} most recent batch runs are remembered.")
