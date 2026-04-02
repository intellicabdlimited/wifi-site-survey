import streamlit as st

CUSTOM_CSS = """
<style>
#MainMenu, footer { visibility: hidden; }
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(139,92,246,0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(56,189,248,0.12), transparent 22%),
        linear-gradient(180deg, #0b1020 0%, #0f172a 100%);
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1020 0%, #11172b 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}
.hero-card {
    background: linear-gradient(135deg, rgba(17,24,39,0.88) 0%, rgba(30,41,59,0.86) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 24px 26px;
    box-shadow: 0 20px 45px rgba(2,6,23,0.35);
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 0.35rem;
}
.hero-subtitle {
    color: rgba(226,232,240,0.78);
    font-size: 0.98rem;
}
.badge {
    display: inline-block;
    padding: 4px 11px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    margin-right: 6px;
    margin-top: 8px;
    border: 1px solid transparent;
}
.badge-blue { background: rgba(56,189,248,0.14); color: #7dd3fc; border-color: rgba(56,189,248,0.30); }
.badge-purple { background: rgba(167,139,250,0.14); color: #c4b5fd; border-color: rgba(167,139,250,0.30); }
.badge-green { background: rgba(52,211,153,0.14); color: #86efac; border-color: rgba(52,211,153,0.30); }
.badge-orange { background: rgba(251,191,36,0.14); color: #fcd34d; border-color: rgba(251,191,36,0.30); }
.section-wrap {
    margin-top: 0.35rem;
    margin-bottom: 0.65rem;
}
.section-title {
    color: #f8fafc;
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 0.1rem;
}
.section-subtitle {
    color: rgba(226,232,240,0.68);
    font-size: 0.88rem;
}
div[data-testid="metric-container"] {
    background: rgba(15,23,42,0.84);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 14px 16px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
}
div[data-testid="metric-container"] label {
    color: rgba(203,213,225,0.70) !important;
    font-size: 12px !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    color: #f8fafc !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] {
    font-size: 12px !important;
}
div[data-testid="stPlotlyChart"] {
    background: rgba(15,23,42,0.76) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 18px !important;
    padding: 6px !important;
    box-shadow: 0 10px 30px rgba(2,6,23,0.22);
}
.empty-card {
    background: rgba(15,23,42,0.72);
    border: 1px dashed rgba(255,255,255,0.14);
    border-radius: 18px;
    padding: 18px 18px;
    color: rgba(226,232,240,0.8);
}
small.code-note, .code-note {
    color: rgba(191,219,254,0.9);
}
</style>
"""


def inject() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def badge(text: str, kind: str = "purple") -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def hero(title: str, subtitle: str, badges: list[str]) -> None:
    badge_html = "".join(badges)
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
            <div>{badge_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-wrap">
            <div class="section-title">{title}</div>
            {f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )
