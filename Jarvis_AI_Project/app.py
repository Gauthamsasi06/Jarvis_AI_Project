"""
JARVIS AI - "An AI that reverse-engineers reality from data."
Owner: Frontend Dev (UI/UX) + Product Lead (flow, export)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import tempfile, os

from backend.pipeline import (
    load_csv, basic_profile, correlation_matrix, top_correlations,
    detect_anomalies, build_data_summary, compute_health_score, trend_summary,
)
from backend.agent import get_insights, ask_agent, DEFAULT_MODELS, MODEL_PRESETS, test_connection
from backend.report import build_pdf_report
from backend.sample_data import sales_sample, hr_sample

# --------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------
st.set_page_config(page_title="JARVIS AI", page_icon="🧠", layout="wide", initial_sidebar_state="expanded")

BASE_PALETTE = ["#00F0FF", "#B026FF", "#FF2E9F", "#FFD400", "#39FF6A", "#FF6B35", "#4D7CFF", "#FF3EC9"]
ACCENT_PRESETS = {
    "Cyan Pulse": "#00F0FF",
    "Ultraviolet": "#B026FF",
    "Magenta Pop": "#FF2E9F",
    "Solar Gold": "#FFD400",
    "Emerald Volt": "#39FF6A",
}

PROVIDER_LABELS = {
    "Anthropic (Claude)": "anthropic",
    "Groq (Llama 3 / Mixtral — free, open-source)": "groq",
    "Ollama (local, fully open-source, no key)": "ollama",
    "OpenRouter (open-source models)": "openrouter",
    "Together AI (open-source models)": "together",
}

# --------------------------------------------------------------------------
# SIDEBAR - PART 1: config + theme + upload (must run before CSS is built)
# --------------------------------------------------------------------------
with st.sidebar:
    with st.expander("🔑 AI Configuration", expanded=False):
        provider_label = st.selectbox("AI Provider", list(PROVIDER_LABELS.keys()))
        provider = PROVIDER_LABELS[provider_label]

        preset_models = MODEL_PRESETS.get(provider, []) + ["Custom..."]
        default_model = DEFAULT_MODELS[provider]
        default_idx = preset_models.index(default_model) if default_model in preset_models else 0
        model_choice = st.selectbox("Model", preset_models, index=default_idx)
        model = st.text_input("Custom model name", value=default_model) if model_choice == "Custom..." else model_choice

        if provider == "ollama":
            st.caption("Running locally — make sure `ollama serve` is running and you've pulled the model, e.g. `ollama pull llama3.1`.")
            api_key = ""
        else:
            api_key = st.text_input(f"{provider_label.split(' (')[0]} API Key", type="password")

        if st.button("🔌 Test Connection", use_container_width=True):
            if provider != "ollama" and not api_key:
                st.error("Enter an API key first.")
            else:
                with st.spinner("Testing..."):
                    ok, msg = test_connection(provider, api_key, model)
                st.success(f"Connected — model replied: \"{msg}\"") if ok else st.error(f"Connection failed: {msg}")

    st.markdown("---")
    st.markdown("### 🎨 Dashboard Theme")
    accent_label = st.selectbox("Accent color", list(ACCENT_PRESETS.keys()))
    ACCENT = ACCENT_PRESETS[accent_label]
    st.markdown("---")

PALETTE = [ACCENT] + [c for c in BASE_PALETTE if c != ACCENT]
px.defaults.template = "plotly_dark"
px.defaults.color_discrete_sequence = PALETTE

# --------------------------------------------------------------------------
# THEME CSS (built after ACCENT is known)
# --------------------------------------------------------------------------
NEON_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@600;800&family=Inter:wght@400;600;800&display=swap');

html, body, [class*="css"]  {{ font-family: 'Inter', sans-serif; }}

.stApp {{
    background: radial-gradient(circle at 15% 0%, #171233 0%, #0B0F19 45%, #05070d 100%);
}}

/* ---------- HERO ---------- */
.jarvis-hero {{ text-align: center; padding: 28px 10px 18px 10px; }}
.jarvis-title {{
    font-family: 'Orbitron', sans-serif;
    font-size: 3.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, {ACCENT}, {ACCENT}CC, {PALETTE[1]}, {ACCENT});
    background-size: 300% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shine 6s linear infinite;
    margin-bottom: 0;
}}
@keyframes shine {{ to {{ background-position: 300% center; }} }}
@keyframes popIn {{ from {{ opacity: 0; transform: scale(0.9); }} to {{ opacity: 1; transform: scale(1); }} }}
@keyframes fadeSlideUp {{ from {{ opacity: 0; transform: translateY(14px); }} to {{ opacity: 1; transform: translateY(0); }} }}
@keyframes pulseGlow {{ 0%, 100% {{ box-shadow: 0 0 10px var(--glow-color, transparent); }} 50% {{ box-shadow: 0 0 24px var(--glow-color, transparent); }} }}
.pulse-glow {{ animation: pulseGlow 2.2s ease-in-out infinite; }}
.jarvis-subtitle {{ color: #9aa4c0; font-style: italic; margin-top: 4px; font-size: 1.05rem; }}
.jarvis-badges {{ margin-top: 12px; }}
.badge {{
    display: inline-block; padding: 5px 14px; border-radius: 999px;
    font-size: 0.72rem; font-weight: 800; letter-spacing: 0.04em;
    text-transform: uppercase; margin: 0 4px; border: 1px solid;
    animation: popIn 0.5s cubic-bezier(.2,.8,.3,1.2) backwards;
}}
.badge1 {{ color: {PALETTE[0]}; border-color: {PALETTE[0]}; animation-delay: .05s; }}
.badge2 {{ color: {PALETTE[1]}; border-color: {PALETTE[1]}; animation-delay: .12s; }}
.badge3 {{ color: {PALETTE[2]}; border-color: {PALETTE[2]}; animation-delay: .19s; }}

/* ---------- DROPZONE ---------- */
[data-testid="stFileUploaderDropzone"] {{
    border: 2px dashed {ACCENT}80 !important; border-radius: 16px !important;
    background: rgba(255,255,255,0.02) !important;
    transition: border-color 0.2s ease, background 0.2s ease;
}}
[data-testid="stFileUploaderDropzone"]:hover {{
    border-color: {ACCENT} !important; background: {ACCENT}10 !important;
}}

/* ---------- BUTTONS ---------- */
.stButton>button {{
    transition: transform 0.12s ease, box-shadow 0.12s ease;
}}
.stButton>button:hover {{
    transform: translateY(-2px); box-shadow: 0 6px 18px {ACCENT}40;
}}
.stButton>button:active {{ transform: translateY(0) scale(0.97); }}

/* ---------- FILTER BANNER ---------- */
.filter-banner {{
    display: inline-block; padding: 6px 16px; border-radius: 999px;
    background: linear-gradient(90deg, {ACCENT}22, transparent);
    border: 1px solid {ACCENT}55; color: {ACCENT}; font-weight: 700; font-size: 0.85rem;
    margin-bottom: 10px;
}}

/* ---------- KPI CARDS ---------- */
.kpi-card {{
    border-radius: 18px; padding: 18px 14px 14px 14px;
    background: linear-gradient(160deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
    backdrop-filter: blur(6px); border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 0 22px var(--glow), inset 0 0 20px rgba(255,255,255,0.02);
    text-align: center; transition: transform 0.15s ease;
    animation: popIn 0.45s cubic-bezier(.2,.8,.3,1.2) backwards;
}}
.kpi-card:hover {{ transform: translateY(-4px); }}
.kpi-icon {{ font-size: 1.4rem; margin-bottom: 2px; }}
.kpi-value {{ font-size: 1.9rem; font-weight: 800; color: var(--glow); font-family: 'Orbitron', sans-serif; }}
.kpi-label {{ color: #9aa4c0; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; margin-top: 2px; }}

/* ---------- SECTION HEADERS ---------- */
.jarvis-section {{
    font-family: 'Orbitron', sans-serif; font-size: 1.15rem; font-weight: 700;
    color: #E8ECF4; border-left: 4px solid {ACCENT}; padding-left: 10px;
    margin: 22px 0 12px 0;
}}

/* ---------- GLASS PANEL ---------- */
.glass-panel {{
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px; padding: 18px 20px;
}}

/* ---------- INSIGHT CARDS ---------- */
.insight-card {{
    background: linear-gradient(135deg, {ACCENT}14, rgba(176,38,255,0.05));
    border: 1px solid {ACCENT}55; border-radius: 14px; padding: 14px 18px;
    color: #E8ECF4; margin-bottom: 10px; font-size: 0.95rem; line-height: 1.5;
    animation: fadeSlideUp 0.4s ease backwards;
}}

/* ---------- SIDEBAR PANELS ---------- */
.sidebar-panel {{
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 10px 14px; margin-bottom: 8px; font-size: 0.85rem;
}}

/* ---------- TABS ---------- */
.stTabs [data-baseweb="tab-list"] {{ gap: 6px; }}
.stTabs [data-baseweb="tab"] {{
    background: rgba(255,255,255,0.03); border-radius: 10px 10px 0 0;
    padding: 10px 18px; font-weight: 700;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(90deg, {ACCENT}26, rgba(176,38,255,0.15));
    color: {ACCENT} !important;
}}

.stButton>button {{ border-radius: 10px; font-weight: 700; border: 1px solid {ACCENT}66; }}
</style>
"""
st.markdown(NEON_CSS, unsafe_allow_html=True)

st.markdown(
    """<div class="jarvis-hero">
        <div class="jarvis-title">⚡ JARVIS AI</div>
        <div class="jarvis-subtitle">"An AI that reverse-engineers reality from data."</div>
        <div class="jarvis-badges">
            <span class="badge badge1">AUTONOMOUS AGENT</span>
            <span class="badge badge2">LIVE CODE EXECUTION</span>
            <span class="badge badge3">REAL-TIME INSIGHTS</span>
        </div>
    </div>""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# HERO UPLOAD ZONE + SAMPLE DATASETS (shown only until data is loaded)
# --------------------------------------------------------------------------
if "df_source" not in st.session_state:
    st.session_state.df_source = None

if st.session_state.df_source is None:
    st.markdown('<div class="jarvis-section" style="text-align:center; border:none;">Get started</div>', unsafe_allow_html=True)
    hero_upload = st.file_uploader("📤 Drag & drop a CSV here, or click to browse", type=["csv"], key="hero_uploader")
    if hero_upload is not None:
        try:
            parsed_df = load_csv(hero_upload)
            if parsed_df.empty or len(parsed_df.columns) == 0:
                st.error("That CSV appears to be empty. Please upload a file with data in it.")
            else:
                st.session_state.df_source = ("upload", (parsed_df, hero_upload.name, hero_upload.size))
                st.rerun()
        except Exception as e:
            st.error(f"Could not read that CSV: {e}")

    st.markdown("<p style='text-align:center;color:#9aa4c0;'>— or try a sample dataset —</p>", unsafe_allow_html=True)
    c1, c2, _ = st.columns([1, 1, 2])
    with c1:
        if st.button("📊 Load Sample Sales Data", use_container_width=True):
            st.session_state.df_source = ("sample_sales", None)
            st.rerun()
    with c2:
        if st.button("🧑‍💼 Load Sample HR Data", use_container_width=True):
            st.session_state.df_source = ("sample_hr", None)
            st.rerun()
    st.stop()

kind, payload = st.session_state.df_source
if kind == "upload":
    raw_df, source_name, source_size = payload
elif kind == "sample_sales":
    raw_df = sales_sample()
    source_name, source_size = "sample_sales_data.csv", None
else:
    raw_df = hr_sample()
    source_name, source_size = "sample_hr_data.csv", None

with st.sidebar:
    st.markdown("---")
    if st.button("🔄 Load a different dataset"):
        st.session_state.df_source = None
        st.rerun()

# --------------------------------------------------------------------------
# SIDEBAR - PART 2: dataset info + live filters (needs raw_df, so runs after load)
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.markdown("### 📁 Dataset Info")
    size_label = f"{source_size/1024:.1f} KB" if source_size else "Sample dataset"
    st.markdown(
        f"""<div class="sidebar-panel">
        📄 <b>{source_name}</b><br>
        💾 {size_label} &nbsp;|&nbsp; 📐 {raw_df.shape[0]:,} × {raw_df.shape[1]}
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("### 🧹 Filters")
    filtered_df = raw_df.copy()
    cat_cols_all = raw_df.select_dtypes(exclude=np.number).columns.tolist()
    num_cols_all = raw_df.select_dtypes(include=np.number).columns.tolist()

    with st.expander("Filter by category", expanded=False):
        if cat_cols_all:
            for c in cat_cols_all[:5]:
                options = sorted(raw_df[c].dropna().astype(str).unique().tolist())
                if 1 < len(options) <= 50:
                    selected = st.multiselect(c, options, default=options, key=f"filt_cat_{c}")
                    if selected and len(selected) < len(options):
                        filtered_df = filtered_df[filtered_df[c].astype(str).isin(selected)]
        else:
            st.caption("No categorical columns to filter.")

    with st.expander("Filter by range", expanded=False):
        if num_cols_all:
            for c in num_cols_all[:5]:
                col_min, col_max = float(raw_df[c].min()), float(raw_df[c].max())
                if col_min < col_max:
                    rng = st.slider(c, col_min, col_max, (col_min, col_max), key=f"filt_num_{c}")
                    if rng != (col_min, col_max):
                        filtered_df = filtered_df[(filtered_df[c] >= rng[0]) & (filtered_df[c] <= rng[1])]
        else:
            st.caption("No numeric columns to filter.")

    if st.button("↺ Reset all filters"):
        for k in list(st.session_state.keys()):
            if k.startswith("filt_"):
                del st.session_state[k]
        st.rerun()

    st.markdown("---")
    st.markdown("### 📥 Data Export")
    st.download_button(
        "Download filtered CSV",
        filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="jarvis_filtered_data.csv",
        mime="text/csv",
    )

df = filtered_df
if len(df) < len(raw_df):
    st.markdown(
        f'<div class="filter-banner">🔎 Showing {len(df):,} of {len(raw_df):,} rows after filters</div>',
        unsafe_allow_html=True,
    )

profile = basic_profile(df)
health = compute_health_score(df)
anomalies_df = detect_anomalies(df)

VERDICT_COLOR = {"Excellent": "#39FF6A", "Good": "#00F0FF", "Needs cleanup": "#FFD400", "Poor": "#FF2E9F"}
verdict_color = VERDICT_COLOR.get(health["verdict"], "#00F0FF")

# --------------------------------------------------------------------------
# QUICK SUMMARY METRICS + DTYPE BREAKDOWN
# --------------------------------------------------------------------------
missing_cells = int(round(profile["missing_pct"] / 100 * profile["rows"] * profile["columns"]))
mem_bytes = int(df.memory_usage(deep=True).sum())
mem_label = f"{mem_bytes/1024:.1f} KB" if mem_bytes < 1024*1024 else f"{mem_bytes/1024/1024:.2f} MB"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Rows", f'{profile["rows"]:,}')
m2.metric("Total Columns", profile["columns"])
m3.metric("Memory Usage", mem_label)
m4.metric("Missing Values", f'{missing_cells:,}')

with st.expander("🔍 Column types & missing values (click to expand)"):
    dtype_rows = []
    for c in df.columns:
        kind_label = "🔢 Numeric" if c in profile["numeric_cols"] else "🏷️ Categorical"
        miss_pct = round(df[c].isna().mean() * 100, 2)
        dtype_rows.append({"Column": c, "Detected Type": kind_label, "Missing %": miss_pct})
    st.dataframe(pd.DataFrame(dtype_rows), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# KPI ROW
# --------------------------------------------------------------------------
kpis = [
    ("📐", "Rows", f'{profile["rows"]:,}', "#00F0FF", False),
    ("🧬", "Columns", profile["columns"], "#B026FF", False),
    ("🔢", "Numeric Cols", len(profile["numeric_cols"]), "#4D7CFF", False),
    ("🏷️", "Categorical Cols", len(profile["categorical_cols"]), "#FF6B35", False),
    ("🕳️", "Missing %", f'{profile["missing_pct"]}%', "#FF2E9F", False),
    ("📎", "Duplicates", profile["duplicate_rows"], "#FFD400", False),
    ("🚨", "Anomalies", len(anomalies_df), "#FF3EC9", False),
    ("💯", "Health Score", f'{health["score"]}', verdict_color, True),
]
cols = st.columns(8)
for i, (col, (icon, label, value, color, glow)) in enumerate(zip(cols, kpis)):
    with col:
        glow_cls = "pulse-glow" if glow else ""
        st.markdown(
            f"""<div class="kpi-card {glow_cls}" style="--glow:{color}40; --glow-color:{color}80; border-color:{color}55; animation-delay:{i*0.04}s;">
                    <div class="kpi-icon">{icon}</div>
                    <div class="kpi-value" style="color:{color};">{value}</div>
                    <div class="kpi-label">{label}</div>
                </div>""",
            unsafe_allow_html=True,
        )

st.write("")
st.markdown('<div class="jarvis-section">⚡ Quick Insights (no typing needed)</div>', unsafe_allow_html=True)
qc1, qc2, qc3 = st.columns(3)
if "quick_result" not in st.session_state:
    st.session_state.quick_result = None
if qc1.button("📈 Generate Summary Statistics", use_container_width=True):
    st.session_state.quick_result = ("stats", df[profile["numeric_cols"]].describe().round(3) if profile["numeric_cols"] else None)
if qc2.button("🚨 Detect Outliers", use_container_width=True):
    st.session_state.quick_result = ("outliers", len(anomalies_df))
if qc3.button("🔗 Show Correlation Matrix", use_container_width=True):
    st.session_state.quick_result = ("corr", top_correlations(df, n=5))

if st.session_state.quick_result:
    kind, payload = st.session_state.quick_result
    if kind == "stats":
        st.dataframe(payload, use_container_width=True) if payload is not None else st.warning("No numeric columns to summarize.")
    elif kind == "outliers":
        st.error(f"{payload} anomalous rows flagged out of {profile['rows']}. See the 🚨 Anomalies tab for the full list.") if payload else st.info("No anomalies detected.")
    elif kind == "corr":
        if payload:
            for c in payload:
                st.markdown(f'<div class="insight-card">🔗 &nbsp; <b>{c["a"]}</b> vs <b>{c["b"]}</b>: r = {c["corr"]}</div>', unsafe_allow_html=True)
        else:
            st.warning("Need at least 2 numeric columns for correlations.")

st.write("")
tabs = st.tabs(["📊 Overview", "🩺 Data Health", "🔬 Deep Dive", "🧠 AI Insights", "💬 Ask JARVIS", "🚨 Anomalies", "📄 Export Report"])

# --------------------------------------------------------------------------
# TAB 1 - OVERVIEW
# --------------------------------------------------------------------------
with tabs[0]:
    st.markdown('<div class="jarvis-section">Data Preview</div>', unsafe_allow_html=True)
    st.caption("Click a column header to sort, or use the search icon in the table toolbar to filter rows.")
    st.dataframe(df.head(200), use_container_width=True, height=260)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown('<div class="jarvis-section">Correlation Heatmap</div>', unsafe_allow_html=True)
        corr = correlation_matrix(df)
        if not corr.empty:
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="Turbo", aspect="auto")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns for a correlation heatmap.")
    with c2:
        st.markdown('<div class="jarvis-section">Column Mix</div>', unsafe_allow_html=True)
        mix = pd.DataFrame({
            "Type": ["Numeric", "Categorical"],
            "Count": [len(profile["numeric_cols"]), len(profile["categorical_cols"])],
        })
        fig_pie = px.pie(mix, names="Type", values="Count", hole=0.55,
                          color_discrete_sequence=[ACCENT, "#FF2E9F"])
        fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", showlegend=True, margin=dict(t=10, b=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if profile["numeric_cols"]:
            st.markdown('<div class="jarvis-section">Distribution Explorer</div>', unsafe_allow_html=True)
            col_choice = st.selectbox("Numeric column", profile["numeric_cols"], key="dist_col")
            fig2 = px.histogram(df, x=col_choice, nbins=30, color_discrete_sequence=[ACCENT], marginal="box")
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10))
            st.plotly_chart(fig2, use_container_width=True)
    with c4:
        if profile["categorical_cols"]:
            st.markdown('<div class="jarvis-section">Category Breakdown</div>', unsafe_allow_html=True)
            cat_choice = st.selectbox("Categorical column", profile["categorical_cols"], key="cat_col")
            counts = df[cat_choice].value_counts().head(10).reset_index()
            counts.columns = [cat_choice, "count"]
            fig3 = px.bar(counts, x=cat_choice, y="count", color=cat_choice, color_discrete_sequence=PALETTE)
            fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, margin=dict(t=10))
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No categorical columns detected in this dataset.")

# --------------------------------------------------------------------------
# TAB 2 - DATA HEALTH
# --------------------------------------------------------------------------
with tabs[1]:
    hc1, hc2 = st.columns([1, 2])
    with hc1:
        st.markdown('<div class="jarvis-section">Data Health Score</div>', unsafe_allow_html=True)
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health["score"],
            number={"suffix": " / 100", "font": {"color": verdict_color, "size": 40}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#9aa4c0"},
                "bar": {"color": verdict_color},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 40], "color": "rgba(255,46,159,0.25)"},
                    {"range": [40, 65], "color": "rgba(255,212,0,0.25)"},
                    {"range": [65, 85], "color": "rgba(0,240,255,0.2)"},
                    {"range": [85, 100], "color": "rgba(57,255,106,0.25)"},
                ],
            },
        ))
        gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E8ECF4"}, margin=dict(t=20, b=10))
        st.plotly_chart(gauge, use_container_width=True)
        st.markdown(
            f'<div class="glass-panel" style="text-align:center;">'
            f'<span style="color:{verdict_color};font-weight:800;font-size:1.1rem;">{health["verdict"]}</span>'
            f'</div>', unsafe_allow_html=True,
        )
    with hc2:
        st.markdown('<div class="jarvis-section">Quality Breakdown</div>', unsafe_allow_html=True)
        breakdown = pd.DataFrame({
            "Metric": ["Missing %", "Duplicate %", "Outlier %"],
            "Value": [health["missing_pct"], health["duplicate_pct"], health["outlier_pct"]],
        })
        fig_bd = px.bar(breakdown, x="Metric", y="Value", color="Metric",
                         color_discrete_sequence=["#FF2E9F", "#FFD400", "#FF6B35"], text="Value")
        fig_bd.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_bd.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig_bd, use_container_width=True)

        st.markdown('<div class="jarvis-section">Missing Values Per Column</div>', unsafe_allow_html=True)
        miss = (df.isna().mean() * 100).round(2)
        miss = miss[miss > 0].sort_values(ascending=False)
        if not miss.empty:
            miss_df = miss.reset_index()
            miss_df.columns = ["Column", "Missing %"]
            fig_miss = px.bar(miss_df, x="Missing %", y="Column", orientation="h",
                               color="Missing %", color_continuous_scale="Sunsetdark")
            fig_miss.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10))
            st.plotly_chart(fig_miss, use_container_width=True)
        else:
            st.success("No missing values in this dataset. 🎉")

# --------------------------------------------------------------------------
# TAB 3 - DEEP DIVE
# --------------------------------------------------------------------------
with tabs[2]:
    st.markdown('<div class="jarvis-section">Relationship Explorer (Scatter Matrix)</div>', unsafe_allow_html=True)
    if len(profile["numeric_cols"]) >= 2:
        default_cols = profile["numeric_cols"][:4]
        chosen = st.multiselect("Numeric columns to compare", profile["numeric_cols"], default=default_cols, max_selections=5)
        color_by = st.selectbox("Color by (optional)", ["None"] + profile["categorical_cols"])
        if len(chosen) >= 2:
            fig_sm = px.scatter_matrix(
                df, dimensions=chosen,
                color=None if color_by == "None" else color_by,
                color_discrete_sequence=PALETTE,
            )
            fig_sm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10))
            fig_sm.update_traces(diagonal_visible=False, showupperhalf=False, marker=dict(size=4, opacity=0.7))
            st.plotly_chart(fig_sm, use_container_width=True)
        else:
            st.info("Pick at least 2 numeric columns.")
    else:
        st.warning("Need at least 2 numeric columns for the scatter matrix.")

    c5, c6 = st.columns(2)
    with c5:
        if profile["numeric_cols"] and profile["categorical_cols"]:
            st.markdown('<div class="jarvis-section">Box Plot by Category</div>', unsafe_allow_html=True)
            num_c = st.selectbox("Numeric", profile["numeric_cols"], key="box_num")
            cat_c = st.selectbox("Grouped by", profile["categorical_cols"], key="box_cat")
            top_cats = df[cat_c].value_counts().head(8).index
            fig_box = px.box(df[df[cat_c].isin(top_cats)], x=cat_c, y=num_c, color=cat_c,
                              color_discrete_sequence=PALETTE)
            fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False, margin=dict(t=10))
            st.plotly_chart(fig_box, use_container_width=True)
    with c6:
        if profile["categorical_cols"]:
            st.markdown('<div class="jarvis-section">Category Composition (Treemap)</div>', unsafe_allow_html=True)
            tree_cat = st.selectbox("Column", profile["categorical_cols"], key="tree_cat")
            tcounts = df[tree_cat].value_counts().head(15).reset_index()
            tcounts.columns = [tree_cat, "count"]
            fig_tree = px.treemap(tcounts, path=[tree_cat], values="count", color="count",
                                   color_continuous_scale="Plasma")
            fig_tree.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=10, l=0, r=0, b=0))
            st.plotly_chart(fig_tree, use_container_width=True)

# --------------------------------------------------------------------------
# TAB 4 - AI INSIGHTS
# --------------------------------------------------------------------------
with tabs[3]:
    st.markdown('<div class="jarvis-section">Reverse-Engineered Insights</div>', unsafe_allow_html=True)
    if st.button("⚡ Generate Insights", type="primary"):
        if provider != "ollama" and not api_key:
            st.error(f"Add your {provider_label.split(' (')[0]} API key in the sidebar first.")
        else:
            with st.status("Running JARVIS AI insight agent...", expanded=True) as status:
                st.write("📥 Reading statistical profile...")
                summary = build_data_summary(df)
                st.write("🧠 Asking the model to reverse-engineer insights...")
                try:
                    insights = get_insights(provider, api_key, model, summary)
                    st.session_state["insights"] = insights
                    st.session_state["summary"] = summary
                    st.write("✅ Rendering insight cards...")
                    status.update(label="Insights ready", state="complete")
                except Exception as e:
                    status.update(label="Failed", state="error")
                    st.error(f"Provider error: {e}")

    if "insights" in st.session_state:
        bullets = [b.strip("- ").strip() for b in st.session_state["insights"].split("\n") if b.strip()]
        icons_cycle = ["🔗", "📈", "⚠️", "🔍", "💡", "🎯"]
        for i, b in enumerate(bullets):
            icon = icons_cycle[i % len(icons_cycle)]
            st.markdown(f'<div class="insight-card">{icon} &nbsp; {b}</div>', unsafe_allow_html=True)
        with st.expander("Raw statistical summary fed to the model"):
            st.text(st.session_state["summary"])

# --------------------------------------------------------------------------
# TAB 5 - ASK JARVIS (agentic code generation)
# --------------------------------------------------------------------------
with tabs[4]:
    st.markdown('<div class="jarvis-section">Ask JARVIS a question about your data</div>', unsafe_allow_html=True)
    st.caption("JARVIS writes and runs pandas/plotly code live against your dataset.")
    question = st.text_input("e.g. 'Which column has the strongest effect on the target?'")
    if st.button("Run Agent"):
        if provider != "ollama" and not api_key:
            st.error(f"Add your {provider_label.split(' (')[0]} API key in the sidebar first.")
        elif not question:
            st.warning("Type a question first.")
        else:
            with st.status("Running JARVIS AI agent...", expanded=True) as status:
                st.write("📥 Reading CSV schema...")
                st.write("🧠 Generating Python code...")
                result, code, log = ask_agent(provider, api_key, model, df, question)
                st.write("⚙️ Executing analysis...")
                st.write("📊 Rendering visuals...")
                status.update(label="Done", state="complete")
            with st.expander("Show generated code", expanded=False):
                st.code(code, language="python")
            if log:
                st.text(log)
            if hasattr(result, "to_html"):
                st.plotly_chart(result, use_container_width=True)
            elif isinstance(result, pd.DataFrame):
                st.dataframe(result, use_container_width=True)
            elif result is not None:
                st.success(str(result))

# --------------------------------------------------------------------------
# TAB 6 - ANOMALIES
# --------------------------------------------------------------------------
with tabs[5]:
    st.markdown('<div class="jarvis-section">Anomaly Detection (Isolation Forest)</div>', unsafe_allow_html=True)
    if anomalies_df.empty:
        st.info("No anomalies detected (or not enough numeric data).")
    else:
        st.error(f"{len(anomalies_df)} anomalous rows flagged out of {profile['rows']}.")
        if len(profile["numeric_cols"]) >= 2:
            x_col, y_col = profile["numeric_cols"][:2]
            plot_df = df.select_dtypes(include=np.number).copy()
            plot_df["status"] = "Normal"
            plot_df.loc[anomalies_df.index, "status"] = "Anomaly"
            fig_an = px.scatter(plot_df, x=x_col, y=y_col, color="status",
                                 color_discrete_map={"Normal": "#4D7CFF", "Anomaly": "#FF2E9F"}, opacity=0.75)
            fig_an.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10))
            st.plotly_chart(fig_an, use_container_width=True)
        st.dataframe(anomalies_df, use_container_width=True)

# --------------------------------------------------------------------------
# TAB 7 - EXPORT REPORT
# --------------------------------------------------------------------------
with tabs[6]:
    st.markdown('<div class="jarvis-section">Export PDF Report</div>', unsafe_allow_html=True)
    if st.button("Build PDF Report"):
        insights_text = st.session_state.get("insights", "Generate insights in the AI Insights tab first for a fuller report.")
        corr_pairs = top_correlations(df, n=5)
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
        build_pdf_report(tmp_path, profile, insights_text, corr_pairs, len(anomalies_df), health)
        with open(tmp_path, "rb") as f:
            pdf_bytes = f.read()
        os.unlink(tmp_path)
        st.success("Report built!")
        st.download_button("⬇️ Download Report", pdf_bytes, file_name="jarvis_ai_report.pdf", mime="application/pdf")
