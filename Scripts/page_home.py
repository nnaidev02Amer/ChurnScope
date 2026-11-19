"""Page A: Home Dashboard. Four numbers, three simple charts, one short explanation."""
import plotly.graph_objects as go
import streamlit as st

import batch_predict as bp
import churn_analytics as ca
import page_batch
from ui_common import BLUE, GREEN, NAVY, PAGE_HOW, PAGE_PREDICT, RED, TEXT, callout, card, page_header, style_fig

CFG = {"displayModeBar": False}


def go_to(page):
    st.session_state["nav"] = page


def home_figures(df):
    """The numbers shown on the page (also used by tests)."""
    k = ca.kpis(df)
    stayed, left = df[~df["churned"]], df[df["churned"]]
    return {"total": k["customers"], "left": k["churned"], "stayed": k["active"], "rate": k["churn_rate"],
            "avg_bill_stayed": float(stayed["MonthlyCharges"].mean()), "avg_bill_left": float(left["MonthlyCharges"].mean()),
            "by_contract": df["Contract"].value_counts().reindex(ca.DIMENSIONS["Contract"][1]).to_dict()}


def latest_batch_section(f):
    """Summary of the most recent uploaded batch. These are MODEL ESTIMATES for the uploaded customers,
    kept clearly apart from the recorded statistics of the original dataset below."""
    rec = page_batch.latest_batch()
    st.markdown('<div class="cs-section"><span class="cs-badge est">Model estimates</span>Latest Batch Prediction</div>',
                unsafe_allow_html=True)
    if rec is None:
        st.markdown(callout("<b class='k'>No batch has been predicted yet.</b> Upload a file of customers on the "
                            "<b>Predict Customer Churn</b> page and a summary of the results will appear here automatically."),
                    unsafe_allow_html=True)
        st.button("📁  Upload a customer file", on_click=page_batch.go_batch, key="home_to_batch")
        return
    st.caption(f"From your upload **{rec['filename']}**, predicted {rec['time']}. These are predictions for the customers "
               "in that file, not recorded outcomes.")
    c = st.columns(5)
    items = [("Uploaded customers", f"{rec['total']:,}", "rows in the file", TEXT),
             ("Likely to leave", f"{rec['leave']:,}", "above alert line", RED),
             ("Likely to stay", f"{rec['stay']:,}", "below alert line", GREEN),
             ("Predicted churn rate", f"{rec['churn_pct']:.1f}%", "of those predicted", BLUE),
             ("Avg chance of leaving", f"{rec['avg_probability']:.1f}%", "per customer", NAVY)]
    for col, (a, b, d, color) in zip(c, items):
        col.markdown(card(a, b, d, color, "1.7rem", est=True), unsafe_allow_html=True)

    left, right = st.columns([2, 3])
    with left:
        fig = go.Figure(go.Pie(labels=["Likely to leave", "Likely to stay"], values=[rec["leave"], rec["stay"]], hole=0.6,
                               marker=dict(colors=[RED, GREEN], line=dict(color="white", width=2)), sort=False,
                               textinfo="value", textfont=dict(size=15, color="white"),
                               hovertemplate="%{label}: %{value:,} customers<extra></extra>"))
        fig.update_layout(title="Predicted customers in this batch", legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"))
        st.plotly_chart(style_fig(fig, 300), width="stretch", theme=None, config=CFG)
    with right:
        st.markdown("**What this batch shows**")
        res = rec["results"]
        for line in bp.batch_findings(rec, res, f["rate"]):
            st.markdown(f"- {line}")
        st.caption("These are the model's estimates for the uploaded customers; they describe patterns and do not prove "
                   "why a customer might leave.")
        st.button("🗂️  Open the customer-by-customer results", on_click=page_batch.show_in_history, args=(rec["id"],),
                  key="home_to_history")


def render(ctx):
    df = ctx["customers"]
    f = home_figures(df)
    st.markdown(page_header("Home Dashboard", "A quick look at your latest predictions and the customers behind this project."),
                unsafe_allow_html=True)

    st.markdown(callout(
        "<b class='k'>What is customer churn?</b> Churn simply means a customer <b>stops using a company's service and "
        "leaves</b>. Phone and internet companies watch churn closely because keeping a customer is generally cheaper "
        "than finding a new one. This project studies which customers left, and predicts which customers "
        "may leave next, so the company can try to keep them."), unsafe_allow_html=True)

    latest_batch_section(f)
    st.divider()
    st.markdown('<div class="cs-section"><span class="cs-badge actual">Recorded data</span>Original dataset: customers who '
                'already left or stayed</div>', unsafe_allow_html=True)
    st.caption("These figures come from the project's original dataset of past customers. Uploading a file never changes them.")

    cols = st.columns(4)
    cols[0].markdown(card("Total customers", f"{f['total']:,}", "in the dataset"), unsafe_allow_html=True)
    cols[1].markdown(card("Customers who left", f"{f['left']:,}", "also called churned customers", RED),
                     unsafe_allow_html=True)
    cols[2].markdown(card("Still using the service", f"{f['stayed']:,}", "customers who stayed", GREEN),
                     unsafe_allow_html=True)
    cols[3].markdown(card("Overall churn rate", f"{f['rate']:.1%}", f"about {f['rate'] * 100:.0f} in every 100 left", BLUE),
                     unsafe_allow_html=True)

    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig = go.Figure(go.Pie(labels=["Still with us", "Left"], values=[f["stayed"], f["left"]], hole=0.62,
                               marker=dict(colors=[BLUE, RED], line=dict(color="white", width=2)), sort=False,
                               textinfo="percent", textfont=dict(size=15, color="white"),
                               hovertemplate="%{label}: %{value:,} customers<extra></extra>"))
        fig.add_annotation(text=f"<b>{f['rate']:.1%}</b><br>left", showarrow=False, font=dict(size=20, color=NAVY))
        fig.update_layout(title="How many customers left?", showlegend=True,
                          legend=dict(orientation="h", y=-0.05, x=0.5, xanchor="center"))
        st.plotly_chart(style_fig(fig, 330), width="stretch", theme=None, config=CFG)
        st.caption("Roughly one customer in four left.")
    with c2:
        fig = go.Figure(go.Bar(x=["Stayed", "Left"],
                               y=[f["avg_bill_stayed"], f["avg_bill_left"]], marker_color=[BLUE, RED], width=0.55,
                               text=[f"{f['avg_bill_stayed']:.2f}", f"{f['avg_bill_left']:.2f}"], textposition="outside",
                               textfont=dict(size=15), hovertemplate="%{x}: %{y:.2f} per month<extra></extra>"))
        fig.update_yaxes(range=[0, f["avg_bill_left"] * 1.2], title="average monthly bill")
        fig.update_layout(title="Average monthly bill")
        st.plotly_chart(style_fig(fig, 330), width="stretch", theme=None, config=CFG)
        st.caption("Customers who left were paying more per month on average.")
    with c3:
        names = list(f["by_contract"])
        fig = go.Figure(go.Bar(x=names, y=list(f["by_contract"].values()), marker_color=BLUE, width=0.55,
                               text=[f"{v:,}" for v in f["by_contract"].values()], textposition="outside",
                               textfont=dict(size=15), hovertemplate="%{x}: %{y:,} customers<extra></extra>"))
        fig.update_xaxes(ticktext=[n.replace("Month-to-month", "Month-to-<br>month") for n in names], tickvals=names)
        fig.update_yaxes(range=[0, max(f["by_contract"].values()) * 1.18], title="customers")
        fig.update_layout(title="Customers by contract type")
        st.plotly_chart(style_fig(fig, 330), width="stretch", theme=None, config=CFG)
        st.caption("Most customers are on a month-to-month contract, so they can leave at any time.")

    st.write("")
    b1, b2, _ = st.columns([1.2, 1.2, 2])
    b1.button("🎯  Try a prediction", on_click=go_to, args=(PAGE_PREDICT,), type="primary", width="stretch",
              key="home_to_predict")
    b2.button("🎓  How does it work?", on_click=go_to, args=(PAGE_HOW,), width="stretch", key="home_to_how")
    st.caption(f"Source: public telecom customer dataset ({f['total']:,} customers). "
               "The monthly bill has no currency stated in the dataset.")
