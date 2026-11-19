"""Page B: Predict Customer Churn.

Uses the saved model bundle exactly as before: churn_pipeline.predict_churn() with the saved 0.39
threshold, churn_insights for validation, observations and suggestions. Only the presentation is new.
"""
from datetime import datetime

import pandas as pd
import streamlit as st

import churn_insights as ci
import churn_pipeline as cp
import page_batch
from ui_common import GREEN, RED, callout, esc, meter, page_header

DEFAULT = {
    "tenure": 24, "MonthlyCharges": 60.0, "TotalCharges": 1500.0, "gender": "Female", "SeniorCitizen": "No",
    "Partner": "Yes", "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtection": "No",
    "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "One year",
    "PaperlessBilling": "Yes", "PaymentMethod": "Mailed check",
}
PRESETS = {
    "Typical customer": (DEFAULT, "an average customer"),
    "New fibre customer": ({**DEFAULT, "tenure": 2, "MonthlyCharges": 85.0, "Partner": "No", "InternetService": "Fiber optic",
                            "OnlineBackup": "No", "Contract": "Month-to-month", "PaymentMethod": "Electronic check"},
                           "joined recently, high bill"),
    "Long-term loyal customer": ({**DEFAULT, "tenure": 62, "MonthlyCharges": 68.0, "Dependents": "Yes", "OnlineSecurity": "Yes",
                                  "TechSupport": "Yes", "Contract": "Two year", "PaymentMethod": "Bank transfer (automatic)"},
                                 "stayed for years"),
}
ADDONS = {"OnlineSecurity": "Online security", "OnlineBackup": "Online backup", "DeviceProtection": "Device protection",
          "TechSupport": "Tech support", "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming movies"}
FACTOR_PLAIN = {"Tenure": "Time as a customer", "Payment method": "How they pay", "Contract": "Contract type",
                "Online security": "Online security add-on", "Tech support": "Tech support add-on",
                "Internet service": "Internet service", "Monthly charges": "Monthly bill",
                "Paperless billing": "Paperless billing", "Senior citizen": "Senior citizen",
                "Has partner": "Has a partner", "Has dependents": "Has dependents"}
FORM_KEYS = ["f_" + k for k in DEFAULT] + ["f_total_auto", "f_label"]


def init_state():
    s = st.session_state
    saved = s.get("_saved_form", {})
    for k, v in DEFAULT.items():
        key = "f_" + k
        if key not in s:
            s[key] = saved.get(key, v)         # keep the user's entries when they leave and come back
    for key, v in (("f_total_auto", True), ("f_label", "")):
        if key not in s:
            s[key] = saved.get(key, v)
    s.setdefault("history", [])
    s.setdefault("result", None)


def remember_form():
    st.session_state["_saved_form"] = {k: st.session_state[k] for k in FORM_KEYS if k in st.session_state}


def apply_preset(name):
    for k, v in PRESETS[name][0].items():
        st.session_state["f_" + k] = v
    st.session_state["f_total_auto"] = True


def current_customer():
    s = st.session_state
    c = {f: s["f_" + f] for f in cp.FEATURES if f != "TotalCharges"}
    c["TotalCharges"] = (float(s["f_tenure"]) * float(s["f_MonthlyCharges"]) if s["f_total_auto"]
                         else float(s["f_TotalCharges"]))
    return {f: c[f] for f in cp.FEATURES}


def safe_label(text):
    """Trim, and neutralise spreadsheet formula injection in the exported CSV."""
    text = " ".join(str(text).split())[:40]
    return "'" + text if text[:1] in ("=", "+", "-", "@") else text


def run_prediction(ctx):
    cust = current_customer()
    v = ci.validate_customer(cust, ctx["allowed"])
    st.session_state["last_validation"] = v
    if not v.ok:
        return
    out = cp.predict_churn(ctx["bundle"], v.customer).iloc[0]          # saved model + saved 0.39 threshold
    p, thr = float(out["churn_probability"]), float(ctx["bundle"]["threshold"])
    result = {"customer": v.customer, "p": p, "churn": bool(out["churn"]), "band": ci.risk_band(p, thr), "thr": thr,
              "label": safe_label(st.session_state["f_label"]) or f"Customer {len(st.session_state.history) + 1}",
              "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    st.session_state["result"] = result
    st.session_state.history.append(result)


def explanation(p, thr, churn):
    if churn:
        return (f"Our model estimates a <b>{p:.0%}</b> chance that this customer may leave (the churn probability). "
                f"That is above our <b>{thr:.0%}</b> alert line, so we label the customer <b>likely to leave</b>. "
                f"The alert line is set below 50% on purpose: it is better to check on a few customers who would have "
                f"stayed than to miss many who would have left.")
    return (f"Our model estimates a <b>{p:.0%}</b> chance that this customer may leave (the churn probability). "
            f"That is below our <b>{thr:.0%}</b> alert line, so we label the customer <b>likely to stay</b>. "
            f"This is an estimate, not a promise: customers below the line can still leave.")


def form(ctx):
    allowed = ctx["allowed"]
    st.markdown("##### Start with an example, or enter your own customer")
    cols = st.columns(3)
    for col, (name, (_, desc)) in zip(cols, PRESETS.items()):
        col.button(name, on_click=apply_preset, args=(name,), width="stretch", key="preset_" + name, help=desc.capitalize())
    st.caption("Examples fill in every field below; you can then change anything.")

    with st.container(border=True):
        st.markdown("**1. About the customer**")
        a = st.columns(4)
        a[0].selectbox("Gender", allowed["gender"], key="f_gender")
        a[1].selectbox("Senior citizen", allowed["SeniorCitizen"], key="f_SeniorCitizen",
                       help="Whether the customer is flagged as a senior citizen in the dataset.")
        a[2].selectbox("Has a partner", allowed["Partner"], key="f_Partner")
        a[3].selectbox("Has dependents", allowed["Dependents"], key="f_Dependents", help="For example children.")

    with st.container(border=True):
        st.markdown("**2. Their account**")
        b = st.columns(3)
        b[0].number_input("Months as a customer (tenure)", min_value=1, max_value=72, step=1, key="f_tenure",
                          help="How long the customer has stayed with the company. The model knows 1 to 72 months.")
        b[1].number_input("Monthly bill", min_value=1.0, max_value=200.0, step=0.05, format="%.2f", key="f_MonthlyCharges",
                          help="What the customer pays each month (no currency is stated in the dataset).")
        b[2].selectbox("Contract type", allowed["Contract"], key="f_Contract",
                       help="Month-to-month customers can leave at any time.")
        c = st.columns(2)
        c[0].selectbox("How they pay", allowed["PaymentMethod"], key="f_PaymentMethod")
        c[1].selectbox("Paperless billing", allowed["PaperlessBilling"], key="f_PaperlessBilling")
        st.checkbox("Work out the total charged so far automatically (months x monthly bill)", key="f_total_auto",
                    help="In the dataset, total charges are on average within about 3% of this estimate.")
        if st.session_state["f_total_auto"]:
            est = st.session_state["f_tenure"] * st.session_state["f_MonthlyCharges"]
            st.caption(f"Total charged so far: {est:,.2f}")
        else:
            st.number_input("Total charged so far", min_value=0.0, max_value=20000.0, step=10.0, format="%.2f",
                            key="f_TotalCharges")

    with st.container(border=True):
        st.markdown("**3. Services they use**")
        s1, s2, s3 = st.columns(3)
        s1.selectbox("Phone service", allowed["PhoneService"], key="f_PhoneService")
        no_phone = st.session_state["f_PhoneService"] == "No"
        if no_phone:
            st.session_state["f_MultipleLines"] = "No phone service"
        elif st.session_state["f_MultipleLines"] == "No phone service":
            st.session_state["f_MultipleLines"] = "No"
        s2.selectbox("Multiple phone lines", ["No phone service"] if no_phone else ["No", "Yes"], key="f_MultipleLines",
                     disabled=no_phone, help="Only possible if the customer has phone service.")
        s3.selectbox("Internet service", allowed["InternetService"], key="f_InternetService")
        no_net = st.session_state["f_InternetService"] == "No"
        for addon in ADDONS:
            if no_net:
                st.session_state["f_" + addon] = "No internet service"
            elif st.session_state["f_" + addon] == "No internet service":
                st.session_state["f_" + addon] = "No"
        ac = st.columns(3)
        for i, (addon, label) in enumerate(ADDONS.items()):
            ac[i % 3].selectbox(label, ["No internet service"] if no_net else ["No", "Yes"], key="f_" + addon, disabled=no_net)
        if no_net:
            st.caption("Internet add-ons are not available without internet service.")

    st.text_input("Name this prediction (optional)", key="f_label", max_chars=40,
                  placeholder="e.g. Example A. Please avoid real names or personal data.")


def result_panel(ctx):
    res = st.session_state["result"]
    st.markdown("#### Result")
    if res is None:
        st.markdown(callout("Fill in the form (or pick an example), then press <b>Predict</b>. "
                            "The result appears here."), unsafe_allow_html=True)
        return
    p, thr, churn = res["p"], res["thr"], res["churn"]
    stale = res["customer"] != ci.validate_customer(current_customer(), ctx["allowed"]).customer
    if stale:
        st.markdown(callout("You changed the form since this result. Press <b>Predict</b> again to update it.", "warn"),
                    unsafe_allow_html=True)
    color = RED if churn else GREEN
    st.markdown(f'<div class="cs-card"><div class="lbl">Chance this customer may leave (churn probability)</div>'
                f'<div class="big" style="color:{color};font-size:3rem">{p:.0%}</div>'
                f'{meter(p, thr, color)}'
                f'<div style="margin-top:14px"><span class="lbl">Decision:</span> '
                f'<span class="cs-pill {"leave" if churn else "stay"}">{"Likely to leave" if churn else "Likely to stay"}</span></div>'
                f'</div>', unsafe_allow_html=True)
    st.markdown(callout(explanation(p, thr, churn), "warn" if churn else "tip"), unsafe_allow_html=True)

    exp = ci.explain_risk(res["customer"], ctx["stats"])
    st.markdown("##### What stands out about this customer")
    if exp["risk"]:
        for it in exp["risk"][:3]:
            st.markdown(f"- **{esc(FACTOR_PLAIN.get(it['factor'], it['factor']))}: {esc(it['value'])}.** "
                        f"{it['rate']:.0%} of similar customers left (the average is {exp['overall']:.0%}).")
    else:
        st.markdown("- Nothing in this profile stands out as unusually high-risk.")
    for it in exp["protective"][:2]:
        st.markdown(f"- Good sign, **{esc(FACTOR_PLAIN.get(it['factor'], it['factor']))}: {esc(it['value'])}.** "
                    f"Only {it['rate']:.0%} of similar customers left.")
    st.caption("These are patterns in past customers. They do not prove why this customer might leave.")

    sug = ci.retention_suggestions(res["customer"], res["band"])
    st.markdown("##### Ideas to keep this customer")
    st.markdown(f"_{sug['headline']}_")
    for it in sug["ideas"][:3]:
        st.markdown(f"- **{esc(it['title'])}.** {esc(it['detail'])}")
    st.caption("These are rule-based ideas from the customer's details, not predictions.")


def history_section():
    hist = st.session_state.history
    with st.expander(f"Recent predictions this session ({len(hist)})", expanded=False):
        if not hist:
            st.write("Nothing yet. Each prediction you make is listed here, and you can download them as a CSV file. "
                     "The list is kept only while this page stays open.")
            return
        rows = [{"#": i, "Time": r["time"], "Name": r["label"], "Chance of leaving (%)": round(r["p"] * 100, 1),
                 "Decision": "Likely to leave" if r["churn"] else "Likely to stay", "Alert line (%)": round(r["thr"] * 100, 1),
                 **r["customer"]} for i, r in enumerate(hist, 1)]
        full = pd.DataFrame(rows)
        st.dataframe(full[["#", "Time", "Name", "Chance of leaving (%)", "Decision"]].tail(5), hide_index=True, width="stretch")
        st.caption("Showing the latest 5. The CSV download contains every prediction with all customer details.")
        c1, c2, _ = st.columns([1, 1, 2])
        c1.download_button("Download CSV", full.to_csv(index=False).encode("utf-8"), file_name="churnscope_predictions.csv",
                           mime="text/csv", width="stretch", key="download")
        if c2.button("Clear history", width="stretch", key="clear"):
            st.session_state.history = []
            st.session_state["result"] = None
            st.rerun()


def render(ctx):
    init_state()
    s = st.session_state
    st.markdown(page_header("Predict Customer Churn",
                            "Predict one customer, or upload a file to predict many customers at once."),
                unsafe_allow_html=True)
    if "predict_mode" not in s:                      # remember the chosen view when the user leaves and comes back
        s["predict_mode"] = s.get("_saved_mode", page_batch.MODE_SINGLE)
    st.radio("How do you want to predict?", page_batch.MODES, horizontal=True, key="predict_mode")
    s["_saved_mode"] = s["predict_mode"]
    if s["predict_mode"] == page_batch.MODE_BATCH:
        page_batch.render_batch(ctx)
    elif s["predict_mode"] == page_batch.MODE_HISTORY:
        page_batch.render_history(ctx)
    else:
        render_single(ctx)


def render_single(ctx):
    """The original single-customer prediction page (unchanged behaviour)."""
    left, right = st.columns([3, 2], gap="large")
    with left:
        form(ctx)
    with right:
        st.button("🎯  Predict", type="primary", key="predict", on_click=run_prediction, args=(ctx,), width="stretch")
        v = st.session_state.get("last_validation")
        if v is not None:
            for msg in v.errors:
                st.error(msg)
            for msg in v.warnings:
                st.warning(msg)
        result_panel(ctx)
    history_section()
    st.caption("Predictions are estimates from an academic prototype, not guarantees.")
    remember_form()
