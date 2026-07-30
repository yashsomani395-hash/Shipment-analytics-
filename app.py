"""
app.py
------
Streamlit dashboard for the Shipment Analytics project.

Run:  python -m streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    run_full_analysis,
    q1_region_performance,
    q2_freight_vs_distance,
    q3_customer_delays,
    q5_kpi_recommendation,
)
from utils import (
    CARRIER_COLORS,
    PALETTE,
    PLOTLY_TEMPLATE,
    REGION_COLORS,
    fmt_currency,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Shipment Analytics",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_PATH = "data/shipments.csv"

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* Background */
    .stApp { background: #0F172A; }
    .block-container { padding: 1.5rem 2rem; max-width: 100%; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #1E293B !important;
        border-right: 1px solid #334155;
    }
    [data-testid="stSidebar"] * { color: #CBD5E1 !important; }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1E293B 0%, #243148 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        transition: transform .2s, box-shadow .2s;
        position: relative;
        overflow: hidden;
    }
    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(79, 142, 247, 0.2);
    }
    .kpi-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 4px; height: 100%;
        border-radius: 16px 0 0 16px;
    }
    .kpi-good::before  { background: #4FC78E; }
    .kpi-warn::before  { background: #F7994F; }
    .kpi-bad::before   { background: #F74F4F; }
    .kpi-blue::before  { background: #4F8EF7; }
    .kpi-purple::before{ background: #C084FC; }
    .kpi-cyan::before  { background: #22D3EE; }

    .kpi-label {
        font-size: 0.72rem; font-weight: 600;
        letter-spacing: .08em; text-transform: uppercase;
        color: #94A3B8; margin-bottom: 0.3rem;
    }
    .kpi-value {
        font-size: 2rem; font-weight: 700; color: #F1F5F9;
        line-height: 1.1;
    }
    .kpi-delta {
        font-size: 0.78rem; margin-top: 0.3rem; color: #64748B;
    }
    .kpi-icon { font-size: 1.8rem; float: right; opacity: 0.35; }

    /* Section headers */
    .section-title {
        font-size: 1.15rem; font-weight: 700; color: #E2E8F0;
        border-left: 4px solid #4F8EF7;
        padding-left: 0.75rem; margin: 1.5rem 0 1rem;
    }

    /* Tab styling */
    [data-testid="stTabs"] button {
        font-weight: 600; color: #94A3B8;
        border-radius: 8px 8px 0 0;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #4F8EF7 !important;
        border-bottom: 2px solid #4F8EF7 !important;
    }

    /* Divider */
    hr { border-color: #334155; margin: 1.5rem 0; }

    /* Metric delta tweak */
    [data-testid="metric-container"] {
        background: #1E293B; border-radius: 12px; padding: 1rem;
        border: 1px solid #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Data loading (cached) ──────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner="Loading & analysing data…")
def load_data(path: str) -> dict:
    return run_full_analysis(path)


# ── Helper: plotly chart defaults ─────────────────────────────────────────────

def _fig_defaults(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#CBD5E1"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=height,
        legend=dict(
            bgcolor="rgba(30,41,59,0.8)",
            bordercolor="#334155",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(gridcolor="#1E293B", showgrid=True)
    fig.update_yaxes(gridcolor="#1E293B", showgrid=True)
    return fig


# ── Matplotlib-free colour scale helper ───────────────────────────────────────

def _color_scale(
    styler,
    subset: str,
    low_color: str = "#4FC78E",
    high_color: str = "#F74F4F",
    reverse: bool = False,
) :
    """
    Apply a cell background gradient without requiring matplotlib.
    Linearly interpolates between low_color and high_color based on
    the normalised value within the column.
    """
    def _hex_to_rgb(h: str) -> tuple:
        h = h.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    def _lerp(a: tuple, b: tuple, t: float) -> str:
        r = int(a[0] + (b[0] - a[0]) * t)
        g = int(a[1] + (b[1] - a[1]) * t)
        b_ = int(a[2] + (b[2] - a[2]) * t)
        return f"#{r:02x}{g:02x}{b_:02x}"

    lo_rgb = _hex_to_rgb(low_color)
    hi_rgb = _hex_to_rgb(high_color)

    col_data = styler.data[subset]
    col_min  = col_data.min()
    col_max  = col_data.max()
    rng      = col_max - col_min if col_max != col_min else 1

    def _style_cell(val):
        try:
            t = float((val - col_min) / rng)
            if reverse:
                t = 1 - t
            bg = _lerp(lo_rgb, hi_rgb, t)
            # Pick readable text colour
            brightness = (int(bg[1:3], 16) * 299 +
                          int(bg[3:5], 16) * 587 +
                          int(bg[5:7], 16) * 114) / 1000
            fg = "#0F172A" if brightness > 128 else "#F1F5F9"
            return f"background-color: {bg}; color: {fg};"
        except Exception:
            return ""

    _apply = getattr(styler, "map", None) or styler.applymap
    return _apply(_style_cell, subset=[subset])


# ── KPI card renderer ──────────────────────────────────────────────────────────

def kpi_card(label: str, value: str, icon: str = "📦",
             css_class: str = "kpi-blue", delta: str = "") -> None:
    st.markdown(
        f"""
        <div class="kpi-card {css_class}">
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-delta">{delta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    region_col  = schema.get("region")
    carrier_col = schema.get("carrier")
    cust_col    = schema.get("customer")
    ship_col    = schema.get("ship_date")

    st.sidebar.markdown("## 🔧 Filters")
    st.sidebar.markdown("---")

    # Region
    all_regions = sorted(df[region_col].dropna().unique()) if region_col else []
    selected_regions = st.sidebar.multiselect(
        "Region", all_regions, default=all_regions
    )

    # Carrier
    all_carriers = sorted(df[carrier_col].dropna().unique()) if carrier_col else []
    selected_carriers = st.sidebar.multiselect(
        "Carrier", all_carriers, default=all_carriers
    )

    # Customer search
    st.sidebar.markdown("**Customer Search**")
    cust_query = st.sidebar.text_input("", placeholder="Type to filter…")

    # Date range
    if ship_col and pd.api.types.is_datetime64_any_dtype(df[ship_col]):
        min_d = df[ship_col].min().date()
        max_d = df[ship_col].max().date()
        date_range = st.sidebar.date_input(
            "Ship Date Range", value=(min_d, max_d),
            min_value=min_d, max_value=max_d
        )
    else:
        date_range = None

    # Status filter
    status_col = schema.get("shipment_status")
    if status_col:
        all_statuses = sorted(df[status_col].dropna().unique())
        selected_statuses = st.sidebar.multiselect(
            "Shipment Status", all_statuses, default=all_statuses
        )
    else:
        selected_statuses = []

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<small style='color:#64748B'>Data refreshes every 5 min · "
        "Built with Streamlit</small>",
        unsafe_allow_html=True,
    )

    # ── Apply filters ──────────────────────────────────────────────────────────
    mask = pd.Series([True] * len(df), index=df.index)

    if selected_regions and region_col:
        mask &= df[region_col].isin(selected_regions)

    if selected_carriers and carrier_col:
        mask &= df[carrier_col].isin(selected_carriers)

    if cust_query and cust_col:
        mask &= df[cust_col].str.contains(cust_query, case=False, na=False)

    if date_range and ship_col and len(date_range) == 2:
        mask &= (df[ship_col].dt.date >= date_range[0]) & \
                (df[ship_col].dt.date <= date_range[1])

    if selected_statuses and status_col:
        mask &= df[status_col].isin(selected_statuses)

    return df[mask].copy()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════

def render_overview(df: pd.DataFrame, schema: dict) -> None:
    region_col  = schema.get("region")
    carrier_col = schema.get("carrier")
    cust_col    = schema.get("customer")
    cost_col    = schema.get("freight_cost")

    # ── KPI Row ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Key Performance Indicators</div>',
                unsafe_allow_html=True)

    total     = len(df)
    on_time   = df["on_time"].mean() * 100 if "on_time" in df.columns else 0
    avg_delay = df["delivery_delay_hours"].mean() if "delivery_delay_hours" in df.columns else 0
    avg_cost  = df[cost_col].mean() if cost_col and cost_col in df.columns else 0
    n_cust    = df[cust_col].nunique() if cust_col else 0
    n_carr    = df[carrier_col].nunique() if carrier_col else 0

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: kpi_card("Total Shipments",  f"{total:,}",         "📦", "kpi-blue")
    with c2: kpi_card("On-Time %",        f"{on_time:.1f}%",    "✅",
                      "kpi-good" if on_time >= 80 else "kpi-bad")
    with c3: kpi_card("Avg Delay (hrs)",  f"{avg_delay:.1f}",   "⏱️",
                      "kpi-good" if avg_delay <= 12 else "kpi-warn")
    with c4: kpi_card("Avg Freight Cost", fmt_currency(avg_cost),"💰", "kpi-cyan")
    with c5: kpi_card("Total Customers",  f"{n_cust:,}",        "👥", "kpi-purple")
    with c6: kpi_card("Total Carriers",   f"{n_carr}",          "🚚", "kpi-blue")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Shipment volume over time ──────────────────────────────────────────────
    ship_col = schema.get("ship_date")
    if ship_col and "ship_month_name" in df.columns:
        st.markdown('<div class="section-title">📅 Monthly Shipment Volume & On-Time Rate</div>',
                    unsafe_allow_html=True)
        monthly = (
            df.groupby("ship_month_name")
            .agg(
                total = ("on_time", "count"),
                on_time_pct = ("on_time", lambda x: x.mean() * 100),
            )
            .reset_index()
        )
        # Sort chronologically
        monthly["_sort"] = pd.to_datetime(monthly["ship_month_name"], format="%b %Y", errors="coerce")
        monthly = monthly.sort_values("_sort")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=monthly["ship_month_name"], y=monthly["total"],
            name="Shipments", marker_color=PALETTE["primary"], opacity=0.8,
        ))
        fig.add_trace(go.Scatter(
            x=monthly["ship_month_name"], y=monthly["on_time_pct"],
            name="On-Time %", mode="lines+markers",
            line=dict(color=PALETTE["success"], width=2.5),
            yaxis="y2",
        ))
        fig.update_layout(
            yaxis=dict(title="Shipment Count"),
            yaxis2=dict(title="On-Time %", overlaying="y", side="right",
                        range=[0, 100]),
            legend=dict(orientation="h", y=1.1),
        )
        fig = _fig_defaults(fig, 380)
        st.plotly_chart(fig, use_container_width=True)

    # ── Delay distribution ─────────────────────────────────────────────────────
    if "delay_bucket" in df.columns:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-title">⏳ Delay Distribution</div>',
                        unsafe_allow_html=True)
            bucket_counts = (
                df["delay_bucket"].value_counts().reindex(
                    ["On-Time", "≤1 day late", "1–3 days late",
                     "3–7 days late", ">7 days late"]
                ).reset_index()
            )
            bucket_counts.columns = ["Bucket", "Count"]
            fig2 = px.bar(
                bucket_counts, x="Bucket", y="Count",
                color="Bucket",
                color_discrete_sequence=[
                    PALETTE["success"], PALETTE["warning"],
                    PALETTE["secondary"], PALETTE["danger"], "#7C3AED",
                ],
                title="Shipments by Delay Bucket",
            )
            fig2 = _fig_defaults(fig2)
            st.plotly_chart(fig2, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-title">🚚 Shipments by Carrier</div>',
                        unsafe_allow_html=True)
            if carrier_col:
                carr_counts = df[carrier_col].value_counts().reset_index()
                carr_counts.columns = ["Carrier", "Count"]
                fig3 = px.pie(
                    carr_counts, names="Carrier", values="Count",
                    color="Carrier",
                    color_discrete_map=CARRIER_COLORS,
                    hole=0.45,
                    title="Carrier Share",
                )
                fig3 = _fig_defaults(fig3)
                st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – REGION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def render_region(df: pd.DataFrame, results_q1: dict, schema: dict) -> None:
    region_col = schema.get("region")
    agg = results_q1["region_summary"]
    worst = results_q1["worst_region"]

    st.markdown('<div class="section-title">🗺️ Region Performance Overview</div>',
                unsafe_allow_html=True)

    # ── On-Time % bar ─────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            agg.sort_values("on_time_pct"),
            x="on_time_pct", y=region_col,
            orientation="h",
            color="on_time_pct",
            color_continuous_scale=["#F74F4F", "#F7994F", "#4FC78E"],
            title="On-Time Delivery % by Region",
            labels={"on_time_pct": "On-Time %"},
            text="on_time_pct",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_coloraxes(showscale=False)
        fig = _fig_defaults(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            agg.sort_values("avg_delay_hours", ascending=False),
            x=region_col, y="avg_delay_hours",
            color=region_col,
            color_discrete_map=REGION_COLORS,
            title="Average Delay (hours) by Region",
            text="avg_delay_hours",
        )
        fig2.update_traces(texttemplate="%{text:.1f}h", textposition="outside")
        fig2 = _fig_defaults(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    # ── Late shipment count ───────────────────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.bar(
            agg.sort_values("late_count", ascending=False),
            x=region_col, y="late_count",
            color=region_col,
            color_discrete_map=REGION_COLORS,
            title="Late Shipment Count by Region",
            text="late_count",
        )
        fig3.update_traces(texttemplate="%{text}", textposition="outside")
        fig3 = _fig_defaults(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.bar(
            agg.sort_values("total_shipments", ascending=False),
            x=region_col, y="total_shipments",
            color=region_col,
            color_discrete_map=REGION_COLORS,
            title="Total Shipments by Region",
            text="total_shipments",
        )
        fig4.update_traces(texttemplate="%{text}", textposition="outside")
        fig4 = _fig_defaults(fig4)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Root cause for worst region ───────────────────────────────────────────
    st.markdown(f'<div class="section-title">🔍 Root Cause: {worst} Region</div>',
                unsafe_allow_html=True)

    rc = results_q1["root_causes"]
    carrier_col = schema.get("carrier")

    if "carrier_delay_in_worst_region" in rc:
        col5, col6 = st.columns(2)
        with col5:
            cdf = rc["carrier_delay_in_worst_region"]
            fig5 = px.bar(
                cdf.sort_values("avg_delay", ascending=False),
                x=carrier_col, y="avg_delay",
                color=carrier_col,
                color_discrete_map=CARRIER_COLORS,
                title=f"Carrier Avg Delay in {worst} Region (hrs)",
                text="avg_delay",
            )
            fig5.update_traces(texttemplate="%{text:.1f}h", textposition="outside")
            fig5 = _fig_defaults(fig5)
            st.plotly_chart(fig5, use_container_width=True)

        with col6:
            d_worst  = rc.get("avg_distance_worst", 0)
            d_others = rc.get("avg_distance_others", 0)
            comp_df = pd.DataFrame({
                "Group": [worst, "Other Regions"],
                "Avg Distance (km)": [d_worst, d_others],
            })
            fig6 = px.bar(
                comp_df, x="Group", y="Avg Distance (km)",
                color="Group",
                color_discrete_sequence=[PALETTE["danger"], PALETTE["primary"]],
                title="Avg Distance: Worst vs Other Regions",
                text="Avg Distance (km)",
            )
            fig6.update_traces(texttemplate="%{text:.0f} km", textposition="outside")
            fig6 = _fig_defaults(fig6)
            st.plotly_chart(fig6, use_container_width=True)

    if "warehouse_delay_in_worst_region" in rc:
        wdf = rc["warehouse_delay_in_worst_region"]
        fig7 = px.bar(
            wdf.sort_values("avg_delay", ascending=False),
            x=schema.get("warehouse"), y="avg_delay",
            title=f"Warehouse Avg Delay in {worst} (hrs)",
            color="avg_delay",
            color_continuous_scale=["#4FC78E", "#F7994F", "#F74F4F"],
            text="avg_delay",
        )
        fig7.update_traces(texttemplate="%{text:.1f}h", textposition="outside")
        fig7.update_coloraxes(showscale=False)
        fig7 = _fig_defaults(fig7, 320)
        st.plotly_chart(fig7, use_container_width=True)

    # ── Summary table ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📋 Region Summary Table</div>',
                unsafe_allow_html=True)
    display_cols = [region_col, "total_shipments", "on_time_pct",
                    "late_count", "avg_delay_hours"]
    display_cols = [c for c in display_cols if c in agg.columns]
    st.dataframe(
        agg[display_cols].rename(columns={
            region_col:        "Region",
            "total_shipments": "Total Shipments",
            "on_time_pct":     "On-Time %",
            "late_count":      "Late Shipments",
            "avg_delay_hours": "Avg Delay (hrs)",
        }).style.format({"On-Time %": "{:.1f}%", "Avg Delay (hrs)": "{:.1f}"}),
        use_container_width=True, hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – CARRIER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def render_carrier(df: pd.DataFrame, results_q2: dict, schema: dict) -> None:
    carrier_col = schema.get("carrier")
    cost_col    = schema.get("freight_cost")

    st.markdown('<div class="section-title">🚛 Carrier Performance Dashboard</div>',
                unsafe_allow_html=True)

    carrier_agg = (
        df.groupby(carrier_col)
        .agg(
            total_shipments   = (carrier_col, "count"),
            on_time_pct       = ("on_time", lambda x: x.mean() * 100),
            avg_delay_hours   = ("delivery_delay_hours", "mean"),
            avg_cost          = (cost_col, "mean"),
            late_count        = ("is_late", "sum"),
        )
        .reset_index()
        .round(2)
    )

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            carrier_agg.sort_values("on_time_pct"),
            x=carrier_col, y="on_time_pct",
            color=carrier_col, color_discrete_map=CARRIER_COLORS,
            title="On-Time % by Carrier",
            text="on_time_pct",
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.add_hline(y=80, line_dash="dash", line_color=PALETTE["warning"],
                      annotation_text="80% target")
        fig = _fig_defaults(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = px.bar(
            carrier_agg.sort_values("avg_delay_hours", ascending=False),
            x=carrier_col, y="avg_delay_hours",
            color=carrier_col, color_discrete_map=CARRIER_COLORS,
            title="Average Delay (hrs) by Carrier",
            text="avg_delay_hours",
        )
        fig2.update_traces(texttemplate="%{text:.1f}h", textposition="outside")
        fig2 = _fig_defaults(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.bar(
            carrier_agg.sort_values("avg_cost", ascending=False),
            x=carrier_col, y="avg_cost",
            color=carrier_col, color_discrete_map=CARRIER_COLORS,
            title="Average Freight Cost by Carrier ($)",
            text="avg_cost",
        )
        fig3.update_traces(texttemplate="$%{text:.0f}", textposition="outside")
        fig3 = _fig_defaults(fig3)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Carrier deviation from expected cost (Q2 results)
        dev = results_q2["carrier_deviation"]
        fig4 = px.bar(
            dev.sort_values("mean_residual", ascending=False),
            x=carrier_col, y="mean_residual",
            color="mean_residual",
            color_continuous_scale=["#4FC78E", "#F7994F", "#F74F4F"],
            title="Avg Cost Deviation vs Expected ($)",
            text="mean_residual",
        )
        fig4.update_traces(texttemplate="$%{text:.1f}", textposition="outside")
        fig4.update_coloraxes(showscale=False)
        fig4 = _fig_defaults(fig4)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Carrier ranking scorecard ──────────────────────────────────────────────
    st.markdown('<div class="section-title">🏆 Carrier Scorecard</div>',
                unsafe_allow_html=True)
    scorecard = carrier_agg.copy()
    scorecard["rank"] = scorecard["on_time_pct"].rank(ascending=False).astype(int)
    scorecard = scorecard.sort_values("rank")


    _styled = _color_scale(
        scorecard[[carrier_col, "rank", "total_shipments", "on_time_pct",
                   "avg_delay_hours", "avg_cost", "late_count"]]
        .rename(columns={
            carrier_col:        "Carrier",
            "rank":             "Rank",
            "total_shipments":  "Shipments",
            "on_time_pct":      "On-Time %",
            "avg_delay_hours":  "Avg Delay (hrs)",
            "avg_cost":         "Avg Cost ($)",
            "late_count":       "Late Deliveries",
        })
        .style.format({
            "On-Time %":      "{:.1f}%",
            "Avg Delay (hrs)":"{:.1f}",
            "Avg Cost ($)":   "${:.0f}",
        }),
        subset="On-Time %",
        low_color="#F74F4F",
        high_color="#4FC78E",
    )
    st.dataframe(_styled, use_container_width=True, hide_index=True)



# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – CUSTOMER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def render_customer(df: pd.DataFrame, results_q3: dict, schema: dict) -> None:
    cust_col = schema.get("customer")

    st.markdown('<div class="section-title">👤 Top Delayed Customers</div>',
                unsafe_allow_html=True)

    top = results_q3["top_delayed"]

    fig = px.bar(
        top.head(15).sort_values("avg_delay_hours", ascending=True),
        x="avg_delay_hours", y=cust_col,
        orientation="h",
        color="avg_delay_hours",
        color_continuous_scale=["#F7994F", "#F74F4F"],
        title="Top 15 Customers by Average Delay (hrs)",
        text="avg_delay_hours",
    )
    fig.update_traces(texttemplate="%{text:.1f}h", textposition="outside")
    fig.update_coloraxes(showscale=False)
    fig = _fig_defaults(fig, 500)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        delay_carrier = results_q3["delay_by_carrier"]
        if not delay_carrier.empty:
            carrier_col = schema.get("carrier")
            fig2 = px.bar(
                delay_carrier, x=carrier_col, y="delivery_delay_hours",
                color=carrier_col, color_discrete_map=CARRIER_COLORS,
                title="Avg Delay (Top-5 Delayed Customers) by Carrier",
            )
            fig2 = _fig_defaults(fig2)
            st.plotly_chart(fig2, use_container_width=True)

    with col2:
        delay_region = results_q3["delay_by_region"]
        if not delay_region.empty:
            region_col = schema.get("region")
            fig3 = px.bar(
                delay_region, x=region_col, y="delivery_delay_hours",
                color=region_col, color_discrete_map=REGION_COLORS,
                title="Avg Delay (Top-5 Delayed Customers) by Region",
            )
            fig3 = _fig_defaults(fig3)
            st.plotly_chart(fig3, use_container_width=True)

    # ── Searchable table ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🔎 Customer Delay Table</div>',
                unsafe_allow_html=True)
    search = st.text_input("Filter customers:", placeholder="Type customer name…")
    all_cust = results_q3["customer_summary"]
    if search:
        all_cust = all_cust[
            all_cust[cust_col].str.contains(search, case=False, na=False)
        ]

    _cust_styled = _color_scale(
        all_cust[[cust_col, "total_shipments", "late_count",
                  "avg_delay_hours", "late_pct"]]
        .rename(columns={
            cust_col:          "Customer",
            "total_shipments": "Shipments",
            "late_count":      "Late Deliveries",
            "avg_delay_hours": "Avg Delay (hrs)",
            "late_pct":        "Late %",
        })
        .style.format({"Avg Delay (hrs)": "{:.1f}", "Late %": "{:.1f}%"}),
        subset="Late %",
        low_color="#4FC78E",
        high_color="#F74F4F",
    )
    st.dataframe(_cust_styled, use_container_width=True, hide_index=True, height=400)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – COST ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def render_cost(df: pd.DataFrame, results_q2: dict, schema: dict) -> None:
    dist_col    = schema.get("distance_km")
    cost_col    = schema.get("freight_cost")
    carrier_col = schema.get("carrier")

    st.markdown('<div class="section-title">💰 Freight Cost vs Distance</div>',
                unsafe_allow_html=True)

    r2 = results_q2["r2"]
    corr = results_q2["correlation"]
    slope = results_q2["slope"]
    intercept = results_q2["intercept"]

    # KPI strip
    c1, c2, c3 = st.columns(3)
    with c1: kpi_card("Pearson Correlation", f"{corr:.3f}", "📈", "kpi-blue")
    with c2: kpi_card("R² (Linear)",         f"{r2:.3f}",  "📐", "kpi-cyan")
    with c3: kpi_card("Rate ($/km)",          f"${slope:.3f}", "💵", "kpi-purple")

    st.markdown("<br>", unsafe_allow_html=True)

    # Sample for scatter (cap at 2000 pts for performance)
    plot_df = df[[dist_col, cost_col, carrier_col]].dropna().sample(
        min(2000, len(df)), random_state=42
    )

    # Regression line
    x_line = np.linspace(plot_df[dist_col].min(), plot_df[dist_col].max(), 200)
    y_line = slope * x_line + intercept

    fig = px.scatter(
        plot_df, x=dist_col, y=cost_col,
        color=carrier_col, color_discrete_map=CARRIER_COLORS,
        opacity=0.55,
        title=f"Distance vs Freight Cost  (r={corr:.3f}, R²={r2:.3f})",
        labels={dist_col: "Distance (km)", cost_col: "Freight Cost ($)"},
    )
    fig.add_trace(go.Scatter(
        x=x_line, y=y_line,
        mode="lines", name="Regression",
        line=dict(color="white", width=2, dash="dash"),
    ))
    fig = _fig_defaults(fig, 480)
    st.plotly_chart(fig, use_container_width=True)

    # ── Carrier deviation ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">📊 Carrier Pricing Deviation</div>',
                unsafe_allow_html=True)

    dev = results_q2["carrier_deviation"]
    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.bar(
            dev.sort_values("mean_residual", ascending=False),
            x=carrier_col, y="mean_residual",
            color=carrier_col, color_discrete_map=CARRIER_COLORS,
            title="Mean Residual (Actual – Expected Cost, $)",
            text="mean_residual",
        )
        fig2.update_traces(texttemplate="$%{text:.1f}", textposition="outside")
        fig2.add_hline(y=0, line_color="white", line_dash="dot")
        fig2 = _fig_defaults(fig2)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        # Cost per km by carrier
        if "cost_per_km" in df.columns:
            cpk = (
                df.groupby(carrier_col)["cost_per_km"]
                .mean().reset_index()
                .rename(columns={"cost_per_km": "avg_cost_per_km"})
            )
            fig3 = px.bar(
                cpk.sort_values("avg_cost_per_km", ascending=False),
                x=carrier_col, y="avg_cost_per_km",
                color=carrier_col, color_discrete_map=CARRIER_COLORS,
                title="Average Cost per KM by Carrier ($/km)",
                text="avg_cost_per_km",
            )
            fig3.update_traces(texttemplate="$%{text:.3f}", textposition="outside")
            fig3 = _fig_defaults(fig3)
            st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(
        dev[[carrier_col, "shipments", "mean_residual", "mean_abs_residual"]]
        .rename(columns={
            carrier_col:          "Carrier",
            "shipments":          "Shipments",
            "mean_residual":      "Avg Overcharge ($)",
            "mean_abs_residual":  "Avg Absolute Deviation ($)",
        })
        .style.format({
            "Avg Overcharge ($)":         "${:.2f}",
            "Avg Absolute Deviation ($)": "${:.2f}",
        }),
        use_container_width=True, hide_index=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 – WEEKLY TRENDS
# ══════════════════════════════════════════════════════════════════════════════

def render_trends(df: pd.DataFrame, results_q5: dict, schema: dict) -> None:
    cost_col = schema.get("freight_cost")

    st.markdown('<div class="section-title">📈 Weekly KPI Trends</div>',
                unsafe_allow_html=True)

    kpi_df = results_q5["kpi_series"].copy()
    if kpi_df.empty:
        st.warning("No weekly data available.")
        return

    kpi_df["week_label"] = kpi_df["ship_year"].astype(str) + "-W" + kpi_df["ship_week"].astype(str).str.zfill(2)

    # On-time rate trend
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=kpi_df["week_label"], y=kpi_df["on_time_pct"],
        name="On-Time %", mode="lines",
        line=dict(color=PALETTE["neutral"], width=1.5),
        opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=kpi_df["week_label"], y=kpi_df["on_time_pct_ma4"],
        name="4-week MA", mode="lines",
        line=dict(color=PALETTE["primary"], width=3),
    ))
    fig.add_hrect(y0=0, y1=80, fillcolor="rgba(247,79,79,0.08)",
                  line_width=0, annotation_text="Below 80% target")
    fig.update_layout(
        title="Weekly On-Time Delivery Rate (WOTDR) — Recommended KPI",
        yaxis_title="On-Time %",
        xaxis_title="Week",
        xaxis_tickangle=-45,
    )
    fig = _fig_defaults(fig, 420)
    st.plotly_chart(fig, use_container_width=True)

    # ── Volume & avg delay ─────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.bar(
            kpi_df, x="week_label", y="total_shipments",
            title="Weekly Shipment Volume",
            color_discrete_sequence=[PALETTE["secondary"]],
        )
        fig2.update_layout(xaxis_tickangle=-45)
        fig2 = _fig_defaults(fig2, 340)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        fig3 = px.line(
            kpi_df, x="week_label", y="avg_delay_hours",
            title="Weekly Average Delay (hrs)",
            color_discrete_sequence=[PALETTE["danger"]],
            markers=True,
        )
        fig3.update_layout(xaxis_tickangle=-45)
        fig3 = _fig_defaults(fig3, 340)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Recommendation callout ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">💡 KPI Recommendation</div>',
                unsafe_allow_html=True)
    rec = results_q5["recommendation"]
    lines = rec.split("\n")
    title = lines[0]
    body  = "\n".join(lines[1:])

    st.markdown(
        f"""
        <div style="background:#1E293B;border:1px solid #4F8EF7;border-left:4px solid #4F8EF7;
                    border-radius:12px;padding:1.4rem 1.6rem;margin-top:1rem;">
          <div style="font-size:1rem;font-weight:700;color:#4F8EF7;margin-bottom:.6rem;">
            🎯 {title}
          </div>
          <div style="color:#CBD5E1;font-size:0.9rem;line-height:1.6;">
            {body.replace(chr(10),'<br>')}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 – DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════

def render_data_quality(results_q4: dict, exploration: dict) -> None:
    st.markdown('<div class="section-title">🔍 Data Quality Report</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: kpi_card("Original Rows",   f"{results_q4['rows_original']:,}",  "📄", "kpi-blue")
    with c2: kpi_card("Rows After Clean",f"{results_q4['rows_cleaned']:,}",   "✅", "kpi-good")
    with c3: kpi_card("Rows Removed",    f"{results_q4['rows_removed']:,}",   "🗑️", "kpi-warn")
    with c4: kpi_card("Issues Found",    f"{results_q4['total_issues']:,}",   "⚠️", "kpi-bad")

    st.markdown("<br>", unsafe_allow_html=True)

    # Issues table
    st.markdown('<div class="section-title">📋 Issue Log</div>',
                unsafe_allow_html=True)
    issues_df = results_q4["issues_df"]
    if not issues_df.empty:
        st.dataframe(
            issues_df.rename(columns={
                "issue":      "Issue",
                "count":      "Records Affected",
                "action":     "Action Taken",
                "assumption": "Assumption / Note",
            }).style.format({"Records Affected": "{:,}"}),
            use_container_width=True, hide_index=True,
        )

    # Null heatmap
    st.markdown('<div class="section-title">🔥 Missing Value Heatmap</div>',
                unsafe_allow_html=True)
    null_pct = exploration.get("null_pct", {})
    null_df  = pd.DataFrame({
        "Column": list(null_pct.keys()),
        "Missing %": list(null_pct.values()),
    }).sort_values("Missing %", ascending=False)

    fig = px.bar(
        null_df, x="Column", y="Missing %",
        color="Missing %",
        color_continuous_scale=["#4FC78E", "#F7994F", "#F74F4F"],
        title="Missing Value % per Column (pre-cleaning)",
    )
    fig.update_coloraxes(showscale=False)
    fig = _fig_defaults(fig, 320)
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 8 – RAW DATA
# ══════════════════════════════════════════════════════════════════════════════

def render_raw_data(df: pd.DataFrame) -> None:
    st.markdown('<div class="section-title">📁 Cleaned Dataset Preview</div>',
                unsafe_allow_html=True)
    st.markdown(
        f"Showing **{len(df):,}** rows × **{len(df.columns)}** columns "
        f"after cleaning & feature engineering."
    )

    # Column selector
    all_cols = list(df.columns)
    selected_cols = st.multiselect(
        "Select columns to display:", all_cols, default=all_cols[:10]
    )
    if selected_cols:
        st.dataframe(df[selected_cols].head(500), use_container_width=True)

    # Download
    st.download_button(
        label="⬇️ Download Cleaned CSV",
        data=df.to_csv(index=False).encode(),
        file_name="shipments_cleaned.csv",
        mime="text/csv",
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Header ─────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:1rem;padding:0.5rem 0 1.5rem;">
          <div style="font-size:2.8rem;">📦</div>
          <div>
            <div style="font-size:1.9rem;font-weight:800;color:#F1F5F9;line-height:1.1;">
              Shipment Analytics
            </div>
            <div style="font-size:0.85rem;color:#64748B;margin-top:0.2rem;">
              Operational Intelligence Dashboard · Real-time Insights
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Load data ──────────────────────────────────────────────────────────────
    if not Path(DATA_PATH).exists():
        st.error(
            f"❌ Dataset not found at `{DATA_PATH}`. "
            "Run `python generate_data.py` to create the dataset."
        )
        st.stop()

    results = load_data(DATA_PATH)
    df_full = results["df"]
    schema  = results["schema"]

    # ── Sidebar filters → filtered DataFrame ───────────────────────────────────
    df = render_sidebar(df_full, schema)

    if len(df) == 0:
        st.warning("No data matches the current filters. Please adjust your selections.")
        st.stop()

    # Re-run analysis on the sidebar-filtered subset so all tabs reflect
    # the user's current filter selections. Q4 always uses the full
    # cleaning report (pre-filter) because it describes raw data issues.
    res_q1 = q1_region_performance(df, schema)
    res_q2 = q2_freight_vs_distance(df, schema)
    res_q3 = q3_customer_delays(df, schema)
    res_q5 = q5_kpi_recommendation(df, schema)

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "📊 Overview",
        "🗺️ Region",
        "🚛 Carrier",
        "👤 Customer",
        "💰 Cost",
        "📈 Weekly Trends",
        "🔍 Data Quality",
        "📁 Raw Data",
    ])

    with tabs[0]: render_overview(df, schema)
    with tabs[1]: render_region(df, res_q1, schema)
    with tabs[2]: render_carrier(df, res_q2, schema)
    with tabs[3]: render_customer(df, res_q3, schema)
    with tabs[4]: render_cost(df, res_q2, schema)
    with tabs[5]: render_trends(df, res_q5, schema)
    with tabs[6]: render_data_quality(results["q4"], results["exploration"])
    with tabs[7]: render_raw_data(df)


if __name__ == "__main__":
    main()
