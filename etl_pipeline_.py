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

### BUILD DATAFRAMES FOR APP ###
# Computes touchpoint-level KPIs and returns the enriched DataFrame