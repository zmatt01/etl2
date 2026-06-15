"""
app.py — Cross-Channel Marketing Analytics Dashboard
-----------------------------------------------------
Run with:
    streamlit run app.py -- --data path/to/cc_marketing.csv

Or drop cc_marketing.csv alongside this file and run:
    streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from functions import run as etl_run
from functions import (
    compute_ltv_cac, build_channel_summary, build_funnel_summary,
    build_campaign_summary, build_cross_channel_matrix, build_time_series
)

st.set_page_config(
    page_title="Marketing Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 1.6rem; }
  .section-header { font-size: 1.05rem; font-weight: 600; margin-top: 0.5rem; color: #4A4A6A; }
  .kpi-note { font-size: 0.78rem; color: #888; margin-top: -0.4rem; }
</style>
""", unsafe_allow_html=True)

CHANNEL_COLORS = {
    "Email":    "#636EFA",
    "In-Store": "#EF553B",
    "Mobile":   "#00CC96",
    "Social":   "#AB63FA",
    "Web":      "#FFA15A",
}
STAGE_ORDER = ["Awareness", "Consideration", "Conversion", "Retention"]


@st.cache_data(show_spinner="Loading data…")
def load_data(path: str) -> dict:
    return etl_run(path)


def render_sidebar(raw: pd.DataFrame) -> dict:
    st.sidebar.title("Filters")

    channels = sorted(raw["channel_type"].unique())
    sel_channels = st.sidebar.multiselect("Channel", channels, default=channels)

    sel_stages = st.sidebar.multiselect("Funnel Stage", STAGE_ORDER, default=STAGE_ORDER)

    campaigns = sorted(raw["campaign_id"].unique())
    sel_campaigns = st.sidebar.multiselect("Campaign", campaigns, default=campaigns)

    devices = sorted(raw["device_type"].unique())
    sel_devices = st.sidebar.multiselect("Device", devices, default=devices)

    date_min = raw["interaction_timestamp"].min().date()
    date_max = raw["interaction_timestamp"].max().date()
    date_range = st.sidebar.date_input("Date range", value=(date_min, date_max),
                                       min_value=date_min, max_value=date_max)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**KPI Definitions**\n"
        "- **CPC** = Cost / Clicks\n"
        "- **CPM** = Cost / Impressions × 1 000\n"
        "- **CPA** = Cost / Conversions\n"
        "- **ROI** = (Revenue − Cost) / Cost × 100\n"
        "- **LTV:CAC** = Avg CLV / Total Spend per Customer"
    )

    return dict(channels=sel_channels, stages=sel_stages,
                campaigns=sel_campaigns, devices=sel_devices,
                date_range=date_range)


def apply_filters(raw: pd.DataFrame, f: dict) -> pd.DataFrame:
    df = raw.copy()
    df = df[df["channel_type"].isin(f["channels"])]
    df = df[df["consumer_stage"].isin(f["stages"])]
    df = df[df["campaign_id"].isin(f["campaigns"])]
    df = df[df["device_type"].isin(f["devices"])]
    if len(f["date_range"]) == 2:
        s, e = f["date_range"]
        df = df[(df["interaction_timestamp"].dt.date >= s) &
                (df["interaction_timestamp"].dt.date <= e)]
    return df


def render_kpi_row(df: pd.DataFrame):
    spend = df["marketing_cost"].sum()
    clicks = df["clicks"].sum()
    impressions = df["impressions"].sum()
    conversions = df["conversion_flag"].sum()
    revenue = df["conversion_value"].sum()

    cpc = spend / clicks          if clicks > 0          else 0
    cpm = spend / impressions * 1000 if impressions > 0  else 0
    cpa = spend / conversions     if conversions > 0     else 0
    roi = (revenue - spend) / spend * 100 if spend > 0  else 0

    ltv_df = compute_ltv_cac(df)
    avg_ltv_cac = ltv_df["ltv_cac_ratio"].median()

    c = st.columns(8)
    c[0].metric("Total Spend",     f"${spend:,.2f}")
    c[1].metric("Revenue",         f"${revenue:,.0f}")
    c[2].metric("Conversions",     f"{int(conversions):,}")
    c[3].metric("CPC",             f"${cpc:.2f}")
    c[4].metric("CPM",             f"${cpm:.2f}")
    c[5].metric("CPA",             f"${cpa:.2f}")
    c[6].metric("ROI",             f"{roi:.1f}%")
    c[7].metric("Median LTV:CAC",  f"{avg_ltv_cac:.1f}×")

    st.markdown(
        '<p class="kpi-note">ROI = (Revenue − Spend) / Spend × 100 &nbsp;|&nbsp; '
        'Source ROI column replaced with calculated values &nbsp;|&nbsp; '
        'LTV:CAC = median across customers in current selection</p>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Cross-Channel
# ─────────────────────────────────────────────────────────────────────────────
def tab_cross_channel(df: pd.DataFrame):
    ch  = build_channel_summary(df)
    mat = build_cross_channel_matrix(df)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Spend vs Revenue by Channel</p>', unsafe_allow_html=True)
        fig = go.Figure()
        for val, label, color in [("spend","Spend","#636EFA"),("conversion_value","Revenue","#00CC96")]:
            fig.add_trace(go.Bar(name=label, x=ch["channel_type"], y=ch[val], marker_color=color))
        fig.update_layout(barmode="group", height=320, margin=dict(t=10,b=30),
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">ROI by Channel (%)</p>', unsafe_allow_html=True)
        ch_s = ch.sort_values("roi")
        fig = go.Figure(go.Bar(
            x=ch_s["roi"], y=ch_s["channel_type"], orientation="h",
            marker_color=[CHANNEL_COLORS.get(c,"#888") for c in ch_s["channel_type"]],
            text=ch_s["roi"].apply(lambda v: f"{v:.0f}%"), textposition="outside"
        ))
        fig.update_layout(height=320, margin=dict(t=10,b=30,l=80), xaxis_title="ROI (%)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Cost Efficiency Metrics by Channel</p>', unsafe_allow_html=True)
    metrics = {"cpc":"CPC ($)","cpm":"CPM ($)","cpa":"CPA ($)"}
    fig = make_subplots(rows=1, cols=3, subplot_titles=list(metrics.values()))
    for i, (col_name, _) in enumerate(metrics.items(), 1):
        ch_s = ch.sort_values(col_name)
        fig.add_trace(go.Bar(x=ch_s["channel_type"], y=ch_s[col_name],
                             marker_color=[CHANNEL_COLORS.get(c,"#888") for c in ch_s["channel_type"]],
                             showlegend=False), row=1, col=i)
    fig.update_layout(height=300, margin=dict(t=40,b=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Cross-Channel Influence Heatmap (Channel × Funnel Stage)</p>', unsafe_allow_html=True)
    st.markdown('<p class="kpi-note">Avg cross_channel_influence weight per channel-stage pair. Source probabilistic score (0–1).</p>', unsafe_allow_html=True)
    stage_cols = [s for s in STAGE_ORDER if s in mat.columns]
    mat = mat[stage_cols]
    fig = go.Figure(go.Heatmap(
        z=mat.values, x=mat.columns.tolist(), y=mat.index.tolist(),
        colorscale="Blues", text=mat.values.round(3),
        texttemplate="%{text}", colorbar=dict(title="Influence"),
    ))
    fig.update_layout(height=280, margin=dict(t=10,b=30))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Attribution Weight vs Effectiveness Score by Channel</p>', unsafe_allow_html=True)
    st.markdown('<p class="kpi-note">Probabilistic scores from source data (0–1, random in this dataset). Bubble size = spend.</p>', unsafe_allow_html=True)
    fig = px.scatter(ch, x="avg_attribution", y="avg_effectiveness",
                     color="channel_type", size="spend",
                     color_discrete_map=CHANNEL_COLORS,
                     labels={"avg_attribution":"Avg Attribution Weight",
                             "avg_effectiveness":"Avg Effectiveness Score",
                             "channel_type":"Channel"},
                     height=350)
    fig.update_traces(marker=dict(line=dict(width=1, color="white")))
    fig.update_layout(margin=dict(t=10,b=30))
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Full Funnel
# ─────────────────────────────────────────────────────────────────────────────
def tab_full_funnel(df: pd.DataFrame):
    fn = build_funnel_summary(df)
    fn["consumer_stage"] = pd.Categorical(fn["consumer_stage"], categories=STAGE_ORDER, ordered=True)
    fn = fn.sort_values("consumer_stage")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Touchpoint Volume by Funnel Stage</p>', unsafe_allow_html=True)
        fig = go.Figure(go.Funnel(
            y=fn["consumer_stage"].astype(str), x=fn["touchpoints"],
            textinfo="value+percent initial",
            marker=dict(color=["#636EFA","#AB63FA","#EF553B","#00CC96"]),
        ))
        fig.update_layout(height=340, margin=dict(t=10,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">CPA & Conversion Rate by Stage</p>', unsafe_allow_html=True)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=fn["consumer_stage"].astype(str), y=fn["cpa"],
                             name="CPA ($)", marker_color="#636EFA"), secondary_y=False)
        fig.add_trace(go.Scatter(x=fn["consumer_stage"].astype(str), y=fn["cvr"]*100,
                                 name="CVR (%)", mode="lines+markers",
                                 line=dict(color="#EF553B", width=2), marker=dict(size=8)),
                      secondary_y=True)
        fig.update_yaxes(title_text="CPA ($)", secondary_y=False)
        fig.update_yaxes(title_text="CVR (%)", secondary_y=True)
        fig.update_layout(height=340, margin=dict(t=10,b=20), legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Spend vs Revenue by Funnel Stage</p>', unsafe_allow_html=True)
    fig = go.Figure()
    for val, label, color in [("spend","Spend","#636EFA"),("conversion_value","Revenue","#00CC96")]:
        fig.add_trace(go.Bar(name=label, x=fn["consumer_stage"].astype(str),
                             y=fn[val], marker_color=color))
    fig.update_layout(barmode="group", height=300, margin=dict(t=10,b=20),
                      legend=dict(orientation="h",y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Conversion Rate — Channel × Funnel Stage</p>', unsafe_allow_html=True)
    pivot = df.pivot_table(index="channel_type", columns="consumer_stage",
                           values="conversion_flag", aggfunc="mean")
    stage_cols = [s for s in STAGE_ORDER if s in pivot.columns]
    pivot = pivot[stage_cols]
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale="RdYlGn",
        text=(pivot.values * 100).round(1), texttemplate="%{text}%",
        colorbar=dict(title="CVR"),
    ))
    fig.update_layout(height=280, margin=dict(t=10,b=30))
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Campaigns
# ─────────────────────────────────────────────────────────────────────────────
def tab_campaigns(df: pd.DataFrame):
    camp = build_campaign_summary(df)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">ROI by Campaign</p>', unsafe_allow_html=True)
        fig = px.bar(camp.sort_values("roi"), x="roi", y="campaign_id",
                     orientation="h", color="roi",
                     color_continuous_scale="RdYlGn",
                     labels={"roi":"ROI (%)","campaign_id":"Campaign"},
                     height=450)
        fig.update_layout(margin=dict(t=10,b=20), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">CPA vs CVR by Campaign</p>', unsafe_allow_html=True)
        fig = px.scatter(camp, x="cpa", y="cvr", color="roi", size="spend",
                         hover_name="campaign_id", color_continuous_scale="RdYlGn",
                         labels={"cpa":"CPA ($)","cvr":"CVR","roi":"ROI (%)"},
                         height=450)
        fig.update_layout(margin=dict(t=10,b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Campaign Summary Table</p>', unsafe_allow_html=True)
    display_cols = ["campaign_id","spend","conversion_value","conversions",
                    "cpc","cpm","cpa","roi","cvr",
                    "avg_effectiveness","avg_attribution","unique_customers"]
    fmt = {"spend":"${:,.2f}","conversion_value":"${:,.2f}","cpc":"${:.2f}","cpm":"${:.2f}",
           "cpa":"${:.2f}","roi":"{:.1f}%","cvr":"{:.1%}",
           "avg_effectiveness":"{:.3f}","avg_attribution":"{:.3f}"}
    st.dataframe(camp[display_cols].style.format(fmt), use_container_width=True, height=420)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — LTV:CAC
# ─────────────────────────────────────────────────────────────────────────────
def tab_ltv_cac(df: pd.DataFrame):
    ltv = compute_ltv_cac(df)

    c = st.columns(3)
    c[0].metric("Median LTV:CAC", f"{ltv['ltv_cac_ratio'].median():.1f}×")
    c[1].metric("Avg LTV (CLV)",  f"${ltv['ltv'].mean():,.0f}")
    c[2].metric("Avg CAC",        f"${ltv['cac'].mean():,.2f}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">LTV:CAC Distribution</p>', unsafe_allow_html=True)
        fig = px.histogram(ltv, x="ltv_cac_ratio", nbins=40,
                           color_discrete_sequence=["#636EFA"],
                           labels={"ltv_cac_ratio":"LTV:CAC Ratio"}, height=320)
        fig.add_vline(x=3, line_dash="dash", line_color="orange",
                      annotation_text="3× benchmark", annotation_position="top right")
        fig.update_layout(margin=dict(t=10,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">LTV vs CAC per Customer</p>', unsafe_allow_html=True)
        fig = px.scatter(ltv, x="cac", y="ltv", color="ltv_cac_ratio",
                         size="total_conversions", color_continuous_scale="RdYlGn",
                         labels={"cac":"CAC ($)","ltv":"LTV ($)",
                                 "ltv_cac_ratio":"LTV:CAC","total_conversions":"Conversions"},
                         height=320)
        fig.update_layout(margin=dict(t=10,b=20))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">LTV:CAC by Prior Purchase Count</p>', unsafe_allow_html=True)
    st.markdown('<p class="kpi-note">Cohort proxy: customers grouped by prior_purchase_count at first touchpoint.</p>', unsafe_allow_html=True)
    merged = df.merge(ltv[["customer_id","ltv_cac_ratio"]], on="customer_id")
    cohort = merged.groupby("prior_purchase_count").agg(
        avg_ltv_cac=("ltv_cac_ratio","mean"),
        customers=("customer_id","nunique"),
    ).reset_index()
    fig = px.bar(cohort, x="prior_purchase_count", y="avg_ltv_cac",
                 color="avg_ltv_cac", color_continuous_scale="Blues",
                 labels={"prior_purchase_count":"Prior Purchases","avg_ltv_cac":"Avg LTV:CAC"},
                 height=300)
    fig.update_layout(margin=dict(t=10,b=20), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Time Trends
# ─────────────────────────────────────────────────────────────────────────────
def tab_time_trends(df: pd.DataFrame):
    ts = build_time_series(df)

    st.markdown('<p class="section-header">Weekly Spend, Revenue & ROI</p>', unsafe_allow_html=True)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for val, color in [("spend","#636EFA"),("conversion_value","#00CC96")]:
        fig.add_trace(go.Scatter(x=ts["week"], y=ts[val], name=val.title(),
                                 mode="lines", line=dict(color=color, width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=ts["week"], y=ts["roi"], name="ROI (%)",
                             mode="lines+markers", line=dict(color="#FFA15A",dash="dot"),
                             marker=dict(size=5)), secondary_y=True)
    fig.update_yaxes(title_text="$", secondary_y=False)
    fig.update_yaxes(title_text="ROI (%)", secondary_y=True)
    fig.update_layout(height=360, margin=dict(t=10,b=20), legend=dict(orientation="h",y=1.1))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="section-header">Weekly CPC & CPM</p>', unsafe_allow_html=True)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(x=ts["week"], y=ts["cpc"], name="CPC ($)",
                                 line=dict(color="#636EFA")), secondary_y=False)
        fig.add_trace(go.Scatter(x=ts["week"], y=ts["cpm"], name="CPM ($)",
                                 line=dict(color="#AB63FA")), secondary_y=True)
        fig.update_yaxes(title_text="CPC ($)", secondary_y=False)
        fig.update_yaxes(title_text="CPM ($)", secondary_y=True)
        fig.update_layout(height=300, margin=dict(t=10,b=20), legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<p class="section-header">Weekly CPA & Conversions</p>', unsafe_allow_html=True)
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=ts["week"], y=ts["conversions"], name="Conversions",
                             marker_color="#EF553B"), secondary_y=False)
        fig.add_trace(go.Scatter(x=ts["week"], y=ts["cpa"], name="CPA ($)",
                                 line=dict(color="#FFA15A", width=2)), secondary_y=True)
        fig.update_yaxes(title_text="Conversions", secondary_y=False)
        fig.update_yaxes(title_text="CPA ($)", secondary_y=True)
        fig.update_layout(height=300, margin=dict(t=10,b=20), legend=dict(orientation="h",y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<p class="section-header">Weekly Spend by Channel</p>', unsafe_allow_html=True)
    ch_ts = df.groupby(["week","channel_type"])["marketing_cost"].sum().reset_index()
    fig = px.area(ch_ts, x="week", y="marketing_cost", color="channel_type",
                  color_discrete_map=CHANNEL_COLORS,
                  labels={"marketing_cost":"Spend ($)","week":"Week","channel_type":"Channel"},
                  height=320)
    fig.update_layout(margin=dict(t=10,b=20), legend=dict(orientation="h",y=1.1))
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.title("📊 Cross-Channel Marketing Analytics")
    st.markdown("Full-funnel performance across channels, campaigns, and the customer lifecycle.")

    DATA_PATH = "cc_marketing.csv"

    with st.sidebar:
        uploaded = st.file_uploader("Upload cc_marketing.csv", type="csv")

    if uploaded:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            tmp.write(uploaded.read())
            DATA_PATH = tmp.name

    try:
        data = load_data(DATA_PATH)
    except FileNotFoundError:
        st.warning("No data file found. Upload cc_marketing.csv using the sidebar.")
        st.stop()

    raw = data["raw"]
    filters = render_sidebar(raw)
    filtered = apply_filters(raw, filters)

    if filtered.empty:
        st.warning("No data matches the current filters.")
        st.stop()

    st.markdown("---")
    render_kpi_row(filtered)
    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔀 Cross-Channel",
        "🔽 Full Funnel",
        "📋 Campaigns",
        "💰 LTV:CAC",
        "📈 Time Trends",
    ])
    with tab1: tab_cross_channel(filtered)
    with tab2: tab_full_funnel(filtered)
    with tab3: tab_campaigns(filtered)
    with tab4: tab_ltv_cac(filtered)
    with tab5: tab_time_trends(filtered)


if __name__ == "__main__":
    main()
