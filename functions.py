"""
Ingests cc_marketing.csv and produces a cleaned DataFrame with computed KPIs for use in Streamlit visualization

cc_marketing.csv contains anonymised, synthetic data representing a multi-touch marketing campaign across various channels, devices, and consumer stages. Each row corresponds to a single touchpoint interaction between a customer and a marketing channel.

Computed metrics:

CPC             = marketing_cost / clicks
CPM             = (marketing_cost / impressions) * 1000
ROI             = (conversion_value - marketing_cost) / marketing_cost * 100
CPA             = marketing_cost / conversion_flag
LTV             = mean customer_lifetime_value per customer_id
CAC             = sum of marketing_cost per customer_id across all their touchpoints
LTV_CAC_ratio   = LTV / CAC

Probabilistic / influence scores kept as-is for overlay visuals
"""


import pandas as pd
import numpy as np

### INGEST ###

# Reads raw CSV file and parses the specified date column into datetime
def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["interaction_timestamp"])
    return df

### CLEAN ###

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardise string columns
    for col in ["channel_type", "consumer_stage", "device_type",
                "campaign_id", "customer_id", "touchpoint_id"]:
        df[col] = df[col].str.strip()

    # Set the placeholder -1 ROI to NaN for non-converted rows
    df["roi"] = df["roi"].replace(-1.0, np.nan)

    # Replace 0 with null to avoid division-by-zero for impressions / clicks
    df["impressions"] = df["impressions"].replace(0, np.nan)
    df["clicks"]      = df["clicks"].replace(0, np.nan)

    # Generate different time windows for aggregation and time-based analysis
    df["date"]          = df["interaction_timestamp"].dt.date
    df["week"]          = df["interaction_timestamp"].dt.to_period("W").astype(str)
    df["month"]         = df["interaction_timestamp"].dt.to_period("M").astype(str)
    df["hour_of_day"]   = df["interaction_timestamp"].dt.hour
    df["day_of_week"]   = df["interaction_timestamp"].dt.day_name()

    return df


# Build separate dataframes for each dashboard table to keep Streamlit code clean and focused on visuals

### LTV:CAC TABLE ###
def compute_ltv_cac(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a customer-level summary DataFrame with LTV, CAC, and ratio.
    """
    cust = df.groupby("customer_id").agg(
        ltv=("customer_lifetime_value", "mean"),
        cac=("marketing_cost", "sum"),
        total_conversions=("conversion_flag", "sum"),
        total_revenue=("conversion_value", "sum"),
        touchpoints=("touchpoint_id", "count"),
    ).reset_index()

    cust["ltv_cac_ratio"] = cust["ltv"] / cust["cac"]
    return cust

### AGGREGATE SUMMARY TABLES ###
# CHANNEL #
def build_channel_summary(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("channel_type").agg(
        spend=("marketing_cost", "sum"),
        clicks=("clicks", "sum"),
        impressions=("impressions", "sum"),
        conversions=("conversion_flag", "sum"),
        revenue=("conversion_value", "sum"),
        avg_engagement=("engagement_score", "mean"),
        avg_sentiment=("sentiment_score", "mean"),
        avg_attribution=("attribution_weight", "mean"),
        avg_cross_channel=("cross_channel_influence", "mean"),
        avg_effectiveness=("effectiveness_score", "mean"),
        touchpoints=("touchpoint_id", "count"),
    ).reset_index()

    grp["cpc"] = grp["spend"] / grp["clicks"]
    grp["cpm"] = (grp["spend"] / grp["impressions"]) * 1_000
    grp["cpa"] = grp["spend"] / grp["conversions"].replace(0, np.nan)
    grp["roi"] = (grp["revenue"] - grp["spend"]) / grp["spend"] * 100
    grp["ctr"] = grp["clicks"] / grp["impressions"]
    grp["cvr"] = grp["conversions"] / grp["touchpoints"]  # conversion rate
    return grp

# FUNNEL LAYER #
def build_funnel_summary(df: pd.DataFrame) -> pd.DataFrame:
    stage_order = ["Awareness", "Consideration", "Conversion", "Retention"]
    grp = df.groupby("consumer_stage").agg(
        touchpoints=("touchpoint_id", "count"),
        spend=("marketing_cost", "sum"),
        conversions=("conversion_flag", "sum"),
        revenue=("conversion_value", "sum"),
        avg_session=("session_duration_sec", "mean"),
        avg_engagement=("engagement_score", "mean"),
        avg_sentiment=("sentiment_score", "mean"),
        avg_effectiveness=("effectiveness_score", "mean"),
    ).reset_index()

    grp["cpa"] = grp["spend"] / grp["conversions"].replace(0, np.nan)
    grp["cvr"] = grp["conversions"] / grp["touchpoints"]
    grp["stage_order"] = grp["consumer_stage"].map(
        {s: i for i, s in enumerate(stage_order)}
    )
    grp = grp.sort_values("stage_order").drop(columns="stage_order")
    return grp

# CAMPAIGN ID #
def build_campaign_summary(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("campaign_id").agg(
        spend=("marketing_cost", "sum"),
        clicks=("clicks", "sum"),
        impressions=("impressions", "sum"),
        conversions=("conversion_flag", "sum"),
        revenue=("conversion_value", "sum"),
        touchpoints=("touchpoint_id", "count"),
        unique_customers=("customer_id", "nunique"),
        avg_effectiveness=("effectiveness_score", "mean"),
        avg_attribution=("attribution_weight", "mean"),
    ).reset_index()

    grp["cpc"] = grp["spend"] / grp["clicks"]
    grp["cpm"] = (grp["spend"] / grp["impressions"]) * 1_000
    grp["cpa"] = grp["spend"] / grp["conversions"].replace(0, np.nan)
    grp["roi"] = (grp["revenue"] - grp["spend"]) / grp["spend"] * 100
    grp["cvr"] = grp["conversions"] / grp["touchpoints"]
    return grp.sort_values("roi", ascending=False)

# CROSS-CHANNEL INFLUENCE #
def build_cross_channel_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot: avg cross_channel_influence for each channel × consumer_stage combo.
    Useful for a heatmap showing where each channel carries the most
    cross-channel weight in the funnel.
    """
    pivot = df.pivot_table(
        index="channel_type",
        columns="consumer_stage",
        values="cross_channel_influence",
        aggfunc="mean",
    )
    return pivot

# TIME SERIES #
def build_time_series(df: pd.DataFrame) -> pd.DataFrame:
    ts = df.groupby("week").agg(
        spend=("marketing_cost", "sum"),
        clicks=("clicks", "sum"),
        impressions=("impressions", "sum"),
        conversions=("conversion_flag", "sum"),
        revenue=("conversion_value", "sum"),
    ).reset_index()
    ts["roi"]  = (ts["revenue"] - ts["spend"]) / ts["spend"] * 100
    ts["cpc"]  = ts["spend"] / ts["clicks"]
    ts["cpm"]  = (ts["spend"] / ts["impressions"]) * 1_000
    ts["cpa"]  = ts["spend"] / ts["conversions"].replace(0, np.nan)
    return ts


### MASTER FUNCTION ###
def run(path: str) -> dict:
    """
    End-to-end ETL. Returns a dict of DataFrames consumed by the dashboard.
    """
    raw       = load_raw(path)
    cleaned   = clean(raw)
    #enriched  = compute_touchpoint_kpis(cleaned)

    return {
        "raw":             cleaned,                          # full touchpoint-level data
        "ltv_cac":         compute_ltv_cac(cleaned),         # customer-level
        "channel":         build_channel_summary(cleaned),   # by channel
        "funnel":          build_funnel_summary(cleaned),    # by funnel stage
        "campaign":        build_campaign_summary(cleaned),  # by campaign
        "cross_channel":   build_cross_channel_matrix(cleaned),  # pivot heatmap
        "time_series":     build_time_series(cleaned),       # weekly trends
    }