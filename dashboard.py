"""
dashboard.py

Smart Adaptive Environment Monitoring System — Streamlit Dashboard
Run with:  streamlit run dashboard.py
"""

import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Path setup ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from modules.data_preprocessing import DataPreprocessor, FEATURE_COLS, LABEL_COL, NAME_COL, LABEL_NAMES
from modules.fuzzy_logic         import FuzzyDecisionMaker

# ── Constants ────────────────────────────────────────────────────────────────
DATASET     = os.path.join(BASE_DIR, "data", "iot_sensor_dataset.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR  = os.path.join(BASE_DIR, "models")

COLORS  = {0: "#4e9af1", 1: "#f5a623", 2: "#e74c3c"}
LABEL_COLOR_MAP = {"normal": "#4e9af1", "warning": "#f5a623", "dangerous": "#e74c3c"}

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SAEMS Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
        color: #e6edf3;
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #161b22 0%, #1c2128 100%);
        border-right: 1px solid #30363d;
    }
    /* Cards */
    .metric-card {
        background: linear-gradient(135deg, #1c2128, #21262d);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }
    .metric-value {
        font-size: 2.4rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #8b949e;
        margin-top: 4px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    /* Status badges */
    .badge-safe      { color: #3fb950; }
    .badge-moderate  { color: #f5a623; }
    .badge-dangerous { color: #f85149; }
    .badge-critical  { color: #ff0000; }
    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #58a6ff;
        border-bottom: 1px solid #30363d;
        padding-bottom: 0.4rem;
        margin: 1.2rem 0 0.8rem;
    }
    /* Actuator rows */
    .actuator-row {
        background: #21262d;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 1rem;
        border: 1px solid #30363d;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #8b949e;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom-color: #58a6ff !important;
    }
    /* Plot areas */
    [data-testid="stImage"] img {
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    /* Headings */
    h1, h2, h3 { color: #e6edf3 !important; }
    /* Slider label */
    label { color: #c9d1d9 !important; }
    /* DataFrame */
    [data-testid="stDataFrame"] { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ── Cache data loading ────────────────────────────────────────────────────────
@st.cache_data
def load_dataset():
    pre = DataPreprocessor()
    df_raw   = pre.load_data(DATASET)
    df_clean = pre.clean_data(df_raw)
    return df_clean

@st.cache_resource
def get_fuzzy():
    f = FuzzyDecisionMaker()
    f.define_rules()
    return f

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌿 SAEMS")
    st.markdown("*Smart Adaptive Environment Monitoring*")
    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Overview", "🎛️ Live Fuzzy Inference", "📊 Visualisations", "🔍 Dataset Explorer"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Dataset**")
    st.markdown("`iot_sensor_dataset.csv`")
    st.markdown("**Features**")
    st.markdown("Temperature · Humidity · Gas · Light")
    st.markdown("**Labels**")
    st.markdown("🟢 Normal · 🟡 Warning · 🔴 Dangerous")

# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading dataset…"):
    df = load_dataset()

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 🌿 SAEMS — Smart Adaptive Environment Monitoring System")
    st.markdown("Real-time environmental intelligence powered by Fuzzy Logic · PCA · SOM · ART2 · RBF · GA")
    st.divider()

    # ── KPI cards ─────────────────────────────────────────────────────────────
    total   = len(df)
    n_norm  = (df[LABEL_COL] == 0).sum()
    n_warn  = (df[LABEL_COL] == 1).sum()
    n_dang  = (df[LABEL_COL] == 2).sum()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:#58a6ff;">{total:,}</div>
            <div class="metric-label">Total Samples</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value badge-safe">{n_norm:,}</div>
            <div class="metric-label">Normal</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value badge-moderate">{n_warn:,}</div>
            <div class="metric-label">Warning</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value badge-dangerous">{n_dang:,}</div>
            <div class="metric-label">Dangerous</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Feature statistics ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">📈 Feature Statistics</div>', unsafe_allow_html=True)
    stats = df[FEATURE_COLS].describe().T[["mean","std","min","max"]]
    stats.columns = ["Mean", "Std Dev", "Min", "Max"]
    st.dataframe(stats.style.format("{:.2f}").background_gradient(cmap="Blues", axis=1), use_container_width=True)

    # ── Two mini charts ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Quick Insights</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#161b22")
        ax.set_facecolor("#0d1117")
        labels_ord = ["normal", "warning", "dangerous"]
        counts     = [n_norm, n_warn, n_dang]
        colors     = ["#3fb950", "#f5a623", "#f85149"]
        bars = ax.bar(labels_ord, counts, color=colors, width=0.5, edgecolor="none")
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 60,
                    f"{cnt:,}", ha="center", va="bottom", color="#e6edf3", fontsize=9)
        ax.set_title("Label Distribution", color="#e6edf3", pad=10)
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#30363d")
        ax.set_ylabel("Count", color="#8b949e")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_b:
        fig2, ax2 = plt.subplots(figsize=(5, 3.5), facecolor="#161b22")
        ax2.set_facecolor("#0d1117")
        feat_means = df.groupby(NAME_COL)[FEATURE_COLS].mean()
        feat_means_norm = (feat_means - feat_means.min()) / (feat_means.max() - feat_means.min() + 1e-9)
        x = np.arange(len(FEATURE_COLS))
        w = 0.25
        row_colors = ["#3fb950", "#f5a623", "#f85149"]
        for i, (lbl, rc) in enumerate(zip(["normal","warning","dangerous"], row_colors)):
            if lbl in feat_means_norm.index:
                ax2.bar(x + i*w, feat_means_norm.loc[lbl], w, label=lbl.capitalize(), color=rc, alpha=0.85)
        ax2.set_xticks(x + w)
        ax2.set_xticklabels([c.replace("_"," ").title() for c in FEATURE_COLS], color="#8b949e", fontsize=8)
        ax2.tick_params(colors="#8b949e")
        ax2.spines[:].set_color("#30363d")
        ax2.set_title("Normalised Feature Means by Label", color="#e6edf3", pad=10)
        ax2.legend(fontsize=8, labelcolor="#c9d1d9", facecolor="#21262d", edgecolor="#30363d")
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    # ── Pipeline modules summary ──────────────────────────────────────────
    st.markdown('<div class="section-header">⚙️ Pipeline Modules</div>', unsafe_allow_html=True)
    modules = [
        ("🔵", "PCA",              "Visualisation component"),
        ("🟠", "Isolation Forest", "Unsupervised anomaly detection (5% contamination)"),
        ("🟢", "RBF Network",      "3-class event classifier (30 centres, σ=1.0)"),
        ("🟣", "SOM (10×10)",      "Topology-preserving clustering (2000 iterations)"),
        ("🔴", "ART2",             "Adaptive resonance — online category formation"),
        ("🟡", "Fuzzy Logic",      "Rule-based actuator control (17 rules)"),
        ("⚪", "Genetic Algorithm", "RBF weight optimisation (pop=40, gen=60)"),
    ]
    cols = st.columns(2)
    for i, (icon, name, desc) in enumerate(modules):
        with cols[i % 2]:
            st.markdown(f"""
            <div class="actuator-row">
                <span style="font-size:1.4rem">{icon}</span>
                <div>
                    <div style="font-weight:600;color:#e6edf3">{name}</div>
                    <div style="font-size:0.82rem;color:#8b949e">{desc}</div>
                </div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE FUZZY INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎛️ Live Fuzzy Inference":
    st.markdown("# 🎛️ Live Fuzzy Inference")
    st.markdown("Adjust the sensor sliders and see real-time actuator decisions from the fuzzy control system.")
    st.divider()

    fuzzy = get_fuzzy()

    # ── Sensor inputs ──────────────────────────────────────────────────────
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        temp  = st.slider("🌡️ Temperature (°C)",    10,  50, 25, 1)
        gas   = st.slider("💨 Gas Level (ppm)",       0, 1000, 200, 5)
    with col_s2:
        hum   = st.slider("💧 Humidity (%)",          15,  95, 55, 1)
        light = st.slider("💡 Light (0=dark, 1=lit)", 0.0, 1.0, 1.0, 0.01)

    sensors = {"temperature": temp, "humidity": hum, "gas_level": gas, "light": light}
    result  = fuzzy.decide(sensors)

    fan   = result["fan_speed"]
    alarm = result["alarm_level"]
    vent  = result["vent_angle"]
    risk  = result["risk_level"]

    # ── Determine risk colour ──────────────────────────────────────────────
    if risk < 25:
        risk_color, risk_text = "#3fb950", "SAFE"
    elif risk < 55:
        risk_color, risk_text = "#f5a623", "MODERATE"
    elif risk < 85:
        risk_color, risk_text = "#f85149", "DANGEROUS"
    else:
        risk_color, risk_text = "#ff0000", "CRITICAL"

    st.divider()

    # ── Overall risk badge ────────────────────────────────────────────────
    st.markdown(f"""
    <div style="text-align:center; margin: 1rem 0 1.5rem;">
        <div style="font-size:0.9rem;color:#8b949e;letter-spacing:0.08em;text-transform:uppercase;">
            Overall Risk Score
        </div>
        <div style="font-size:4rem;font-weight:800;color:{risk_color};line-height:1.1;">
            {risk:.0f}
        </div>
        <div style="font-size:1.2rem;font-weight:600;color:{risk_color};letter-spacing:0.1em;">
            {risk_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Actuator cards ─────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        fan_color = "#3fb950" if fan < 30 else ("#f5a623" if fan < 70 else "#f85149")
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:2rem;">🌀</div>
            <div class="metric-value" style="color:{fan_color};">{fan:.0f}%</div>
            <div class="metric-label">Fan Speed</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        al_color = "#3fb950" if alarm < 15 else ("#f5a623" if alarm < 65 else "#f85149")
        alarm_label = "OFF" if alarm < 15 else ("WARNING" if alarm < 65 else "CRITICAL")
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:2rem;">🚨</div>
            <div class="metric-value" style="color:{al_color};">{alarm:.0f}</div>
            <div class="metric-label">Alarm — {alarm_label}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        vent_label = "CLOSED" if vent < 40 else ("PARTIAL" if vent < 140 else "OPEN")
        vent_color = "#3fb950" if vent < 40 else ("#f5a623" if vent < 140 else "#58a6ff")
        st.markdown(f"""
        <div class="metric-card">
            <div style="font-size:2rem;">🪟</div>
            <div class="metric-value" style="color:{vent_color};">{vent:.0f}°</div>
            <div class="metric-label">Vent Angle — {vent_label}</div>
        </div>""", unsafe_allow_html=True)

    st.divider()

    # ── Gauge chart ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Actuator Overview</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), facecolor="#0d1117")
    metrics = [
        ("Fan Speed", fan,   100, "#3fb950", "%"),
        ("Alarm",     alarm, 100, "#f5a623", ""),
        ("Vent Angle",vent,  180, "#58a6ff", "°"),
        ("Risk Level",risk,  100, risk_color, ""),
    ]
    for ax, (name, val, mx, col, unit) in zip(axes, metrics):
        ax.set_facecolor("#161b22")
        pct = val / mx
        theta = np.linspace(0, 2*np.pi, 300)
        # Background ring
        ax.barh(0, 2*np.pi, color="#30363d", height=0.4, left=0)
        # Value arc
        ax.barh(0, 2*np.pi*pct, color=col, height=0.4, left=0, alpha=0.9)
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(-0.5, 0.5)
        ax.axis("off")
        ax.text(np.pi, 0, f"{val:.0f}{unit}", ha="center", va="center",
                fontsize=16, fontweight="bold", color=col)
        ax.set_title(name, color="#c9d1d9", fontsize=10, pad=6)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Test presets ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⚡ Quick Presets</div>', unsafe_allow_html=True)
    presets = {
        "🟢 Normal"    : {"temperature": 22, "humidity": 55, "gas_level": 150, "light": 1},
        "🟡 Warning"   : {"temperature": 32, "humidity": 78, "gas_level": 420, "light": 0},
        "🔴 Dangerous" : {"temperature": 45, "humidity": 20, "gas_level": 800, "light": 0},
    }
    rows = []
    for label, s in presets.items():
        d = fuzzy.decide(s)
        txt = fuzzy.get_decision_text(d)
        rows.append({
            "Scenario": label,
            "Temp (°C)": s["temperature"],
            "Humidity (%)": s["humidity"],
            "Gas (ppm)": s["gas_level"],
            "Light": s["light"],
            "Decision": txt,
        })
    st.dataframe(pd.DataFrame(rows).set_index("Scenario"), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — VISUALISATIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Visualisations":
    st.markdown("# 📊 Result Visualisations")
    st.markdown("All plots generated by the training pipeline.")
    st.divider()

    plots = {
        "PCA Projection":          "pca_plot.png",
        "Anomaly Detection":       "anomaly_detection.png",
        "SOM Clusters":            "som_clusters.png",
        "Label Distribution":      "label_distribution.png",
        "Correlation Matrix":      "correlation_matrix.png",
        "Temporal Trends":         "temporal_trends.png",
        "Feature Boxplots":        "boxplots.png",
        "Pairplot":                "pairplot.png",
        "GA Convergence":          "ga_convergence.png",
    }

    available = {k: v for k, v in plots.items()
                 if os.path.isfile(os.path.join(RESULTS_DIR, v))}

    if not available:
        st.warning("No result plots found. Run `main.py` first to generate them.")
    else:
        selected = st.selectbox("Select plot", list(available.keys()))
        path = os.path.join(RESULTS_DIR, available[selected])
        st.image(path, caption=selected, use_container_width=True)

        st.divider()
        st.markdown('<div class="section-header">🖼️ All Plots</div>', unsafe_allow_html=True)

        cols = st.columns(3)
        for i, (name, fname) in enumerate(available.items()):
            p = os.path.join(RESULTS_DIR, fname)
            with cols[i % 3]:
                st.image(p, caption=name, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DATASET EXPLORER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Dataset Explorer":
    st.markdown("# 🔍 Dataset Explorer")
    st.markdown("Filter, sort and analyse the raw sensor dataset.")
    st.divider()

    # ── Filters ────────────────────────────────────────────────────────────
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        label_filter = st.multiselect(
            "Label", ["normal", "warning", "dangerous"],
            default=["normal", "warning", "dangerous"]
        )
    with col_f2:
        temp_range = st.slider("Temperature range (°C)",
                               float(df["temperature"].min()),
                               float(df["temperature"].max()),
                               (float(df["temperature"].min()), float(df["temperature"].max())))
    with col_f3:
        gas_range = st.slider("Gas level range (ppm)",
                              float(df["gas_level"].min()),
                              float(df["gas_level"].max()),
                              (float(df["gas_level"].min()), float(df["gas_level"].max())))

    mask = (
        df[NAME_COL].isin(label_filter) &
        df["temperature"].between(*temp_range) &
        df["gas_level"].between(*gas_range)
    )
    df_filtered = df[mask].reset_index(drop=True)

    st.markdown(f"**{len(df_filtered):,}** rows match the current filters.")

    # ── Sample table ───────────────────────────────────────────────────────
    st.dataframe(
        df_filtered.head(500).style.map(
            lambda v: "background-color:#1a3a1a;" if v == "normal"
                 else ("background-color:#3a2e00;" if v == "warning"
                 else ("background-color:#3a0a0a;" if v == "dangerous" else "")),
            subset=[NAME_COL]
        ),
        use_container_width=True,
        height=360,
    )

    # ── Distribution mini-plots ────────────────────────────────────────────
    st.markdown('<div class="section-header">📉 Feature Distributions (filtered)</div>', unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), facecolor="#0d1117")
    palette_map = {"normal": "#3fb950", "warning": "#f5a623", "dangerous": "#f85149"}
    for ax, col in zip(axes, FEATURE_COLS):
        ax.set_facecolor("#161b22")
        for lbl, clr in palette_map.items():
            sub = df_filtered[df_filtered[NAME_COL] == lbl][col]
            if len(sub):
                sub.plot.kde(ax=ax, color=clr, label=lbl.capitalize(), linewidth=2)
        ax.set_title(col.replace("_", " ").title(), color="#c9d1d9", fontsize=10)
        ax.tick_params(colors="#8b949e")
        ax.spines[:].set_color("#30363d")
        ax.set_xlabel("")
        if col == FEATURE_COLS[0]:
            ax.legend(fontsize=8, labelcolor="#c9d1d9",
                      facecolor="#21262d", edgecolor="#30363d")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── Correlation heatmap ────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔥 Correlation Heatmap (filtered)</div>', unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(6, 4.5), facecolor="#0d1117")
    ax2.set_facecolor("#0d1117")
    corr = df_filtered[FEATURE_COLS].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, linewidths=0.5, ax=ax2,
                cbar_kws={"shrink": 0.8})
    ax2.set_title("Feature Correlation", color="#e6edf3", pad=10)
    ax2.tick_params(colors="#c9d1d9")
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#484f58;font-size:0.8rem;'>"
    "SAEMS · Smart Adaptive Environment Monitoring System · Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
