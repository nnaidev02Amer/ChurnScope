"""Page C: Understand Customer Churn. Exactly three simple analyses, each with an explanation and a takeaway."""
import plotly.graph_objects as go
import streamlit as st

import churn_analytics as ca
from ui_common import BLUE, MUTED, NAVY, RED, callout, esc, page_header, style_fig

CFG = {"displayModeBar": False}
TENURE_LABELS = {"0-12 months": "First year<br>(0-12 months)", "13-24 months": "Second year<br>(13-24 months)",
                 "25-48 months": "Years 3-4<br>(25-48 months)", "49-72 months": "Years 5-6<br>(49-72 months)"}
TENURE_PLAIN = {"0-12 months": "in their first year", "13-24 months": "in their second year",
                "25-48 months": "in years 3 and 4", "49-72 months": "in years 5 and 6"}


def rates(df, dimension):
    """{group: (rate, customers)} in display order for one of the dimensions."""
    col, order = ca.DIMENSIONS[dimension]
    t = ca.churn_by(df, col, order)
    return {r.group: (float(r.churn_rate), int(r.customers)) for r in t.itertuples()}


def takeaways(df):
    """The three takeaway sentences, computed from the data (also used by tests)."""
    c, t, i = rates(df, "Contract"), rates(df, "Tenure group"), rates(df, "Internet service")
    hi_c, lo_c = max(c, key=lambda g: c[g][0]), min(c, key=lambda g: c[g][0])
    first, last = "0-12 months", "49-72 months"
    hi_i, lo_i = max(i, key=lambda g: i[g][0]), min(i, key=lambda g: i[g][0])
    net = {"DSL": "DSL internet", "Fiber optic": "fiber optic internet", "No": "no internet service"}
    return {
        "contract": (f"Customers on a {hi_c.lower()} contract left {c[hi_c][0]:.0%} of the time, compared with only "
                     f"{c[lo_c][0]:.0%} of customers on a {lo_c.lower()} contract. Customers who can leave at any time often do."),
        "tenure": (f"Newer customers are the most likely to leave: {t[first][0]:.0%} of customers in their first year left, "
                   f"but only {t[last][0]:.0%} of customers who had stayed 4 to 6 years. The longer someone stays, the "
                   "less likely they are to leave."),
        "internet": (f"Customers with {net[hi_i]} left most often ({i[hi_i][0]:.0%}), while customers with {net[lo_i]} left "
                     f"least often ({i[lo_i][0]:.0%})."),
    }


def bar(df, dimension, title, label_fn=None):
    r = rates(df, dimension)
    overall = float(df["churned"].mean())
    groups = list(r)
    top = max(groups, key=lambda g: r[g][0])
    fig = go.Figure(go.Bar(
        x=list(range(len(groups))), y=[r[g][0] for g in groups], width=0.55,
        marker_color=[RED if g == top else BLUE for g in groups],
        text=[f"{r[g][0]:.0%}" for g in groups], textposition="outside", textfont=dict(size=17, color=NAVY),
        customdata=[[r[g][1], round(r[g][0] * r[g][1])] for g in groups],
        hovertext=[str(g) for g in groups],
        hovertemplate="%{hovertext}<br>%{y:.1%} left (%{customdata[1]:,} of %{customdata[0]:,} customers)<extra></extra>"))
    fig.add_hline(y=overall, line_dash="dot", line_color=MUTED, line_width=1.5,
                  annotation_text=f"all customers: {overall:.0%}", annotation_position="top right",
                  annotation_font=dict(color=MUTED, size=13))
    fig.update_xaxes(tickvals=list(range(len(groups))),
                     ticktext=[f"{(label_fn or (lambda g: g))(g)}<br>{r[g][1]:,} customers" for g in groups],
                     tickfont=dict(size=13))
    fig.update_yaxes(range=[0, max(0.6, max(v[0] for v in r.values()) * 1.2)], tickformat=".0%", title="customers who left")
    fig.update_layout(title=title)
    return fig


def section(title, what_it_shows, fig, takeaway):
    st.markdown(f"### {title}")
    st.markdown(f"**What this chart shows:** {what_it_shows}")
    st.plotly_chart(style_fig(fig, 380), width="stretch", theme=None, config=CFG)
    st.markdown(callout(f"<b class='k'>Takeaway:</b> {esc(takeaway)}", "tip"), unsafe_allow_html=True)


def render(ctx):
    df = ctx["customers"]
    st.markdown(page_header("Understand Customer Churn",
                            "Three simple patterns in the data: who tends to leave, and when."), unsafe_allow_html=True)
    st.markdown(callout("Each bar shows <b>the share of customers in that group who left</b>. The dotted line is the average "
                        "for all customers, so bars above the line are groups that leave more than usual. "
                        "These are <b>patterns in past data, not proof of what causes customers to leave</b>."),
                unsafe_allow_html=True)
    t = takeaways(df)
    section("1. Churn by contract type",
            "out of every 100 customers on each type of contract, how many left. A month-to-month contract can be ended "
            "at any time; one-year and two-year contracts commit the customer for longer.",
            bar(df, "Contract", "Customers who left, by contract type"), t["contract"])
    section("2. Churn by how long the customer has stayed (tenure)",
            "how many customers left, grouped by the number of months they had been with the company.",
            bar(df, "Tenure group", "Customers who left, by time as a customer", lambda g: TENURE_LABELS[g]), t["tenure"])
    section("3. Churn by internet service",
            "how many customers left, depending on the type of internet service they have (DSL, fiber optic) or none.",
            bar(df, "Internet service", "Customers who left, by internet service", lambda g: {"No": "No internet"}.get(g, g)),
            t["internet"])
