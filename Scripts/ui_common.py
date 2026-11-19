"""Shared look and feel for ChurnScope: navy / blue / white identity and small HTML helpers."""
import html

# ---- palette (all text colours meet WCAG AA contrast on their backgrounds)
NAVY = "#0B1F3A"      # sidebar, headings
BLUE = "#1F6FEB"      # primary accent
SKY = "#E8F1FF"       # light blue tint
BG = "#F4F7FB"        # page background
CARD = "#FFFFFF"
BORDER = "#D9E2EF"
TEXT = "#0B1F3A"
MUTED = "#4A5A75"     # secondary text (7:1 on white)
GREEN, AMBER, RED = "#16845B", "#B26A00", "#C23B47"   # darker tones: readable text on white
GREEN_BG, AMBER_BG, RED_BG = "#E6F6EE", "#FFF3DC", "#FDEBED"

PAGE_HOME = "🏠  Home Dashboard"
PAGE_PREDICT = "🎯  Predict Customer Churn"
PAGE_UNDERSTAND = "📊  Understand Customer Churn"
PAGE_HOW = "🎓  How This Project Works"
PAGES = [PAGE_HOME, PAGE_PREDICT, PAGE_UNDERSTAND, PAGE_HOW]

CSS = f"""
<style>
.block-container {{ padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1180px; animation: cs-fade .35s ease-out; }}
@keyframes cs-fade {{ from {{ opacity: 0; transform: translateY(6px); }} to {{ opacity: 1; transform: none; }} }}
@media (prefers-reduced-motion: reduce) {{ .block-container {{ animation: none; }} * {{ transition: none !important; }} }}
footer {{ visibility: hidden; }}
h1, h2, h3 {{ color: {NAVY}; letter-spacing: -0.2px; }}

/* sidebar */
[data-testid="stSidebar"] {{ background: {NAVY}; }}
[data-testid="stSidebar"] * {{ color: #EAF1FB; }}
[data-testid="stSidebar"] [role="radiogroup"] {{ gap: 6px; }}
[data-testid="stSidebar"] [role="radiogroup"] label {{ width: 100%; padding: 11px 14px; border-radius: 10px;
  border: 1px solid transparent; transition: background .15s, border-color .15s; cursor: pointer; }}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {{ display: none; }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: rgba(255,255,255,.08); }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{ background: {BLUE}; border-color: rgba(255,255,255,.35);
  font-weight: 700; }}
[data-testid="stSidebar"] [role="radiogroup"] label p {{ font-size: 1.02rem; }}
.cs-brand {{ display: flex; align-items: center; gap: 12px; padding: 6px 4px 18px 4px; }}
.cs-brand b {{ font-size: 1.45rem; color: #FFFFFF; letter-spacing: .3px; }}
.cs-brand span {{ display: block; font-size: .78rem; color: #A9BCD9; margin-top: -2px; }}
.cs-side-note {{ font-size: .8rem; color: #A9BCD9; line-height: 1.45; border-top: 1px solid rgba(255,255,255,.14);
  padding-top: 12px; margin-top: 18px; }}

/* page header */
.cs-header {{ margin-bottom: 18px; }}
.cs-header h1 {{ margin: 0; font-size: 2rem; }}
.cs-header p {{ margin: 4px 0 0 0; color: {MUTED}; font-size: 1.05rem; }}
.cs-header .bar {{ width: 56px; height: 4px; border-radius: 4px; background: {BLUE}; margin-top: 12px; }}

/* cards */
.cs-card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 14px; padding: 18px 20px; height: 100%;
  box-shadow: 0 1px 2px rgba(11,31,58,.05); }}
.cs-card .lbl {{ color: {MUTED}; font-size: .9rem; font-weight: 600; }}
.cs-card .big {{ font-size: 2.2rem; font-weight: 800; line-height: 1.15; margin-top: 4px; color: {NAVY}; }}
.cs-card .sub {{ color: {MUTED}; font-size: .88rem; margin-top: 4px; }}
.cs-step {{ display: flex; gap: 14px; align-items: flex-start; background: {CARD}; border: 1px solid {BORDER};
  border-radius: 14px; padding: 18px 20px; margin: 10px 0; }}
.cs-step .num {{ flex: 0 0 38px; height: 38px; border-radius: 50%; background: {BLUE}; color: #fff; font-weight: 800;
  display: flex; align-items: center; justify-content: center; font-size: 1.05rem; }}
.cs-step h3 {{ margin: 0 0 4px 0; font-size: 1.15rem; }}
.cs-step p {{ margin: 4px 0; color: {TEXT}; line-height: 1.55; }}

/* call-outs */
.cs-call {{ border-radius: 12px; padding: 12px 16px; margin: 10px 0; line-height: 1.55; border: 1px solid {BORDER};
  background: {SKY}; border-left: 5px solid {BLUE}; color: {TEXT}; }}
.cs-call.tip {{ background: {GREEN_BG}; border-left-color: {GREEN}; }}
.cs-call.warn {{ background: {AMBER_BG}; border-left-color: {AMBER}; }}
.cs-call b.k {{ color: {NAVY}; }}

/* pills and verdicts */
.cs-pill {{ display: inline-block; padding: 5px 16px; border-radius: 999px; font-weight: 800; font-size: 1.1rem; }}
.cs-pill.leave {{ background: {RED_BG}; color: {RED}; border: 1px solid {RED}; }}
.cs-pill.stay {{ background: {GREEN_BG}; color: {GREEN}; border: 1px solid {GREEN}; }}

/* probability meter */
.cs-meter {{ position: relative; height: 16px; background: #E3EAF4; border-radius: 999px; margin: 30px 0 4px 0; }}
.cs-meter .fill {{ height: 100%; border-radius: 999px; transition: width .6s ease; }}
.cs-meter .line {{ position: absolute; top: -8px; bottom: -8px; width: 3px; background: {NAVY}; border-radius: 2px; }}
.cs-meter .tag {{ position: absolute; top: -30px; transform: translateX(-50%); font-size: .8rem; font-weight: 700;
  color: {NAVY}; white-space: nowrap; }}
.cs-meter-scale {{ display: flex; justify-content: space-between; font-size: .78rem; color: {MUTED}; }}

/* workflow diagram */
.cs-flow {{ display: flex; flex-wrap: nowrap; align-items: stretch; justify-content: center; gap: 4px; margin: 12px 0 6px 0; }}
.cs-flow .node {{ flex: 1 1 0; min-width: 0; background: {CARD}; border: 2px solid {BLUE}; border-radius: 14px;
  padding: 12px 6px; text-align: center; }}
.cs-flow .node b {{ display: block; color: {NAVY}; font-size: .92rem; line-height: 1.25; }}
.cs-flow .node span {{ display: block; color: {MUTED}; font-size: .78rem; margin-top: 4px; line-height: 1.3; }}
.cs-flow .arrow {{ align-self: center; color: {BLUE}; font-size: 1.3rem; font-weight: 800; flex: 0 0 auto; }}
@media (max-width: 900px) {{
  .cs-flow {{ flex-direction: column; align-items: stretch; }}
  .cs-flow .arrow {{ transform: rotate(90deg); text-align: center; line-height: 1; }}
}}
.cs-flow .node.end {{ background: {BLUE}; }}
.cs-flow .node.end b, .cs-flow .node.end span {{ color: #fff; }}

/* 2x2 result grid */
.cs-grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 8px 0; }}
.cs-grid2 div {{ border-radius: 12px; padding: 14px; text-align: center; border: 1px solid {BORDER}; }}
.cs-grid2 b {{ display: block; font-size: 1.6rem; color: {NAVY}; }}
.cs-grid2 span {{ font-size: .88rem; color: {TEXT}; }}
.cs-grid2 .ok {{ background: {GREEN_BG}; }} .cs-grid2 .bad {{ background: {RED_BG}; }} .cs-grid2 .mid {{ background: {AMBER_BG}; }}

/* view switch: horizontal radios in the main area look like pill buttons */
[data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] {{ gap: 8px; flex-wrap: wrap; }}
[data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] label {{ background: {CARD}; border: 1px solid {BORDER};
  border-radius: 999px; padding: 6px 16px; cursor: pointer; transition: background .15s, border-color .15s; }}
[data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] label > div:first-child {{ display: none; }}
[data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] label:hover {{ border-color: {BLUE}; }}
[data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {{ background: {BLUE};
  border-color: {BLUE}; }}
[data-testid="stMain"] [data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p {{ color: #fff; font-weight: 700; }}

/* labels that separate real data from model estimates */
.cs-badge {{ display: inline-block; padding: 3px 11px; border-radius: 6px; font-size: .78rem; font-weight: 800;
  letter-spacing: .6px; text-transform: uppercase; margin-right: 8px; vertical-align: middle; }}
.cs-badge.est {{ background: {AMBER_BG}; color: {AMBER}; border: 1px dashed {AMBER}; }}
.cs-badge.actual {{ background: {SKY}; color: {BLUE}; border: 1px solid {BLUE}; }}
.cs-section {{ font-size: 1.35rem; font-weight: 800; color: {NAVY}; margin: 8px 0 4px 0; }}
.cs-card.est {{ border: 2px dashed {AMBER}; background: #FFFDF7; }}

/* buttons */
.stButton > button {{ border-radius: 10px; font-weight: 700; transition: transform .1s, box-shadow .15s; }}
.stButton > button:hover {{ box-shadow: 0 3px 10px rgba(31,111,235,.25); }}
.stButton > button:active {{ transform: translateY(1px); }}
.stButton > button[kind="primary"] {{ background: {BLUE}; border-color: {BLUE}; font-size: 1.15rem; padding: .7rem 1rem; }}
.stDownloadButton > button {{ border-radius: 10px; font-weight: 700; }}
</style>
"""

LOGO = """<svg width="44" height="44" viewBox="0 0 58 58" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="ChurnScope logo">
<rect x="6" y="36" width="8" height="14" rx="3" fill="#8DB8FF"/><rect x="18" y="28" width="8" height="22" rx="3" fill="#5C9BFF"/>
<rect x="30" y="19" width="8" height="31" rx="3" fill="#3D86FF"/><rect x="42" y="9" width="8" height="41" rx="3" fill="#FFFFFF" opacity=".85"/>
<path d="M8 22 A22 22 0 0 1 30 4" fill="none" stroke="#FFFFFF" stroke-width="2.5" stroke-linecap="round" opacity=".9"/>
<path d="M8 14 A14 14 0 0 1 22 2" fill="none" stroke="#8DB8FF" stroke-width="2.5" stroke-linecap="round"/></svg>"""


def esc(x):
    return html.escape(str(x))


def page_header(title, subtitle):
    return (f'<div class="cs-header"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p><div class="bar"></div></div>')


def card(label, big, sub="", color=TEXT, size="2.2rem", est=False):
    """est=True gives the dashed 'estimate' look used for model predictions (vs solid for recorded data)."""
    return (f'<div class="cs-card{" est" if est else ""}"><div class="lbl">{esc(label)}</div>'
            f'<div class="big" style="color:{color};font-size:{size}">{esc(big)}</div>'
            f'<div class="sub">{esc(sub)}</div></div>')


def callout(body_html, kind="info"):
    """Highlighted note. body_html may contain simple <b> tags; pass already-escaped text."""
    return f'<div class="cs-call {kind}">{body_html}</div>'


def step(number, title, body_html):
    return (f'<div class="cs-step"><div class="num">{number}</div><div><h3>{esc(title)}</h3>{body_html}</div></div>')


def meter(probability, threshold, color):
    """Plain HTML bar: filled to the probability, with a marker at the alert line."""
    return (f'<div class="cs-meter"><div class="fill" style="width:{probability * 100:.1f}%;background:{color}"></div>'
            f'<div class="line" style="left:{threshold * 100:.1f}%"></div>'
            f'<div class="tag" style="left:{threshold * 100:.1f}%">alert line {threshold:.0%}</div></div>'
            f'<div class="cs-meter-scale"><span>0%</span><span>50%</span><span>100%</span></div>')


def style_fig(fig, height):
    """Light, readable Plotly theme."""
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=10, r=10, t=46, b=10), font=dict(color=TEXT, size=14),
                      title_font=dict(size=16, color=NAVY))
    fig.update_xaxes(linecolor=BORDER, gridcolor="#E7EDF6", tickangle=0)
    fig.update_yaxes(linecolor=BORDER, gridcolor="#E7EDF6")
    return fig
