"""
analysis.py
-----------
Core analytics engine for the Shipment Analytics project.

Phases:
  1. Schema detection & exploration
  2. Data cleaning
  3. Feature engineering
  4. Business analysis (Q1–Q5)

All results are returned as plain Python dicts / DataFrames so they can be
consumed by both the Streamlit dashboard (app.py) and the CLI report generator.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from utils import (
    describe_df,
    find_column,
    flag_outliers,
    get_logger,
    iqr_bounds,
    safe_divide,
)

warnings.filterwarnings("ignore")
logger = get_logger("analysis")

# ══════════════════════════════════════════════════════════════════════════════
# 1. SCHEMA DETECTION & EXPLORATION
# ══════════════════════════════════════════════════════════════════════════════

# Column-name aliases: canonical_name → list of possible CSV column names
COLUMN_ALIASES: dict[str, list[str]] = {
    "shipment_id":       ["shipment_id", "id", "shipment_no", "order_id"],
    "ship_date":         ["ship_date", "shipped_date", "dispatch_date", "shipment_date"],
    "expected_delivery": ["expected_delivery", "expected_delivery_date", "eta", "due_date"],
    "actual_delivery":   ["actual_delivery", "actual_delivery_date", "delivery_date", "delivered_date"],
    "region":            ["region", "area", "zone", "territory"],
    "carrier":           ["carrier", "carrier_name", "shipping_carrier", "shipper"],
    "warehouse":         ["warehouse", "warehouse_id", "origin_warehouse", "hub"],
    "customer":          ["customer", "customer_name", "client", "account"],
    "distance_km":       ["distance_km", "distance", "miles", "distance_miles", "km"],
    "freight_cost":      ["freight_cost", "cost", "shipping_cost", "charge", "freight_charge"],
    "weight_kg":         ["weight_kg", "weight", "weight_lbs", "mass"],
    "shipment_status":   ["shipment_status", "status", "delivery_status", "state"],
}


def detect_schema(df: pd.DataFrame) -> dict[str, str | None]:
    """
    Auto-detect canonical column names from the raw DataFrame.
    Returns a mapping {canonical_name: actual_column_name_or_None}.
    """
    schema: dict[str, str | None] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        found = find_column(df, aliases)
        schema[canonical] = found
        if found:
            logger.info("Schema: %-20s → %s", canonical, found)
        else:
            logger.warning("Schema: %-20s → NOT FOUND", canonical)
    return schema


def explore(df: pd.DataFrame) -> dict[str, Any]:
    """Run Phase 1 exploration on the raw DataFrame."""
    logger.info("── Phase 1: Exploration ──────────────────────────────────────")
    summary = describe_df(df)
    logger.info(
        "Rows: %d | Cols: %d | Duplicates: %d | Nulls: %d",
        summary["rows"],
        summary["cols"],
        summary["duplicate_rows"],
        sum(summary["null_counts"].values()),
    )
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA CLEANING
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_text_col(series: pd.Series, valid_values: list[str] | None = None) -> pd.Series:
    """
    Strip whitespace, title-case a text column, and optionally map to closest
    valid value using simple matching.
    """
    cleaned = series.astype(str).str.strip()

    if valid_values:
        valid_lower = {v.lower(): v for v in valid_values}

        def _map(val: str) -> str:
            if val in ("nan", "none", ""):
                return val
            low = val.lower().replace(" ", "").replace("-", "")
            for k, v in valid_lower.items():
                if k.replace(" ", "").replace("-", "") == low:
                    return v
            # Partial match
            for k, v in valid_lower.items():
                if k in low or low in k:
                    return v
            return val.title()

        return cleaned.map(_map)

    return cleaned.str.title()


def clean(
    df: pd.DataFrame,
    schema: dict[str, str | None],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Phase 2: Clean the raw DataFrame.

    Returns
    -------
    cleaned_df  : cleaned DataFrame
    report      : dict describing every issue found and action taken
    """
    logger.info("── Phase 2: Cleaning ─────────────────────────────────────────")
    df = df.copy()
    report: dict[str, Any] = {
        "issues": [],
        "rows_removed": 0,
        "rows_original": len(df),
    }

    def _issue(name: str, count: int, action: str, assumption: str = ""):
        report["issues"].append({
            "issue":      name,
            "count":      count,
            "action":     action,
            "assumption": assumption,
        })
        logger.info("  %-35s | count=%-5d | %s", name, count, action)

    # ── 2.1 Remove fully-duplicate rows ───────────────────────────────────────
    dup_rows = int(df.duplicated().sum())
    df = df.drop_duplicates()
    _issue("Duplicate rows", dup_rows, "Dropped exact duplicates")

    # ── 2.2 Duplicate Shipment IDs (same ID, different data) ──────────────────
    sid_col = schema.get("shipment_id")
    if sid_col:
        dup_ids = int(df.duplicated(subset=[sid_col]).sum())
        # Keep first occurrence
        df = df.drop_duplicates(subset=[sid_col], keep="first")
        _issue("Duplicate shipment IDs (non-identical rows)", dup_ids,
               "Kept first occurrence; dropped subsequent")

    # ── 2.3 Normalise Region ──────────────────────────────────────────────────
    region_col = schema.get("region")
    valid_regions = ["North", "South", "East", "West", "Central"]
    if region_col:
        before = df[region_col].nunique()
        df[region_col] = _normalize_text_col(df[region_col], valid_regions)
        after = df[region_col].nunique()
        _issue("Inconsistent region casing/spelling", before - after,
               f"Normalised to {valid_regions}")

    # ── 2.4 Normalise Carrier ─────────────────────────────────────────────────
    carrier_col = schema.get("carrier")
    valid_carriers = ["FastShip", "QuickMove", "ReliableCo", "SpeedEx", "GlobalFreight"]
    if carrier_col:
        before = df[carrier_col].nunique()
        df[carrier_col] = _normalize_text_col(df[carrier_col], valid_carriers)
        after = df[carrier_col].nunique()
        _issue("Inconsistent carrier casing/spelling", before - after,
               f"Normalised to {valid_carriers}")

    # ── 2.5 Parse date columns ────────────────────────────────────────────────
    date_cols = [
        schema.get("ship_date"),
        schema.get("expected_delivery"),
        schema.get("actual_delivery"),
    ]
    for col in date_cols:
        if col and col in df.columns:
            original_nulls = df[col].isnull().sum()
            df[col] = pd.to_datetime(df[col], errors="coerce")
            new_nulls = df[col].isnull().sum()
            invalid = int(new_nulls - original_nulls)
            if invalid:
                _issue(f"Invalid dates in {col}", invalid,
                       "Coerced to NaT; rows flagged but retained",
                       "Kept rows because other fields are valid")

    # ── 2.6 Impossible delivery dates (actual < ship_date) ────────────────────
    ship_col   = schema.get("ship_date")
    actual_col = schema.get("actual_delivery")
    if ship_col and actual_col and ship_col in df.columns and actual_col in df.columns:
        mask = (
            df[actual_col].notna()
            & df[ship_col].notna()
            & (df[actual_col] < df[ship_col])
        )
        impossible = int(mask.sum())
        df.loc[mask, actual_col] = pd.NaT
        _issue("Impossible delivery dates (actual < ship_date)", impossible,
               "Set actual_delivery to NaT; rows flagged",
               "Ship date is authoritative; actual delivery must be ≥ ship date")

    # ── 2.7 Negative distance ─────────────────────────────────────────────────
    dist_col = schema.get("distance_km")
    if dist_col and dist_col in df.columns:
        neg_dist = int((df[dist_col] < 0).sum())
        df.loc[df[dist_col] < 0, dist_col] = np.nan
        _issue("Negative distance_km", neg_dist,
               "Set to NaN; distance cannot be negative")

    # ── 2.8 Negative freight cost ──────────────────────────────────────────────
    cost_col = schema.get("freight_cost")
    if cost_col and cost_col in df.columns:
        neg_cost = int((df[cost_col] < 0).sum())
        df.loc[df[cost_col] < 0, cost_col] = np.nan
        _issue("Negative freight_cost", neg_cost,
               "Set to NaN; costs cannot be negative")

        # ── 2.9 Zero freight cost ──────────────────────────────────────────────
        zero_cost = int((df[cost_col] == 0).sum())
        df.loc[df[cost_col] == 0, cost_col] = np.nan
        _issue("Zero freight_cost", zero_cost,
               "Set to NaN; assumed data entry errors",
               "A shipment of zero cost is implausible; flagged for review")

    # ── 2.10 Empty customer strings ────────────────────────────────────────────
    cust_col = schema.get("customer")
    if cust_col and cust_col in df.columns:
        # Treat empty string / whitespace as null
        df[cust_col] = df[cust_col].astype(str).str.strip().replace({"": np.nan, "nan": np.nan})
        empty_cust = int(df[cust_col].isnull().sum())
        _issue("Missing / empty customer name", empty_cust,
               "Retained rows; customer marked as 'Unknown'",
               "Customer field missing does not invalidate shipment record")
        df[cust_col] = df[cust_col].fillna("Unknown")

    # ── 2.11 Impute missing numeric fields ────────────────────────────────────
    numeric_impute = {
        schema.get("weight_kg"):    "median",
        schema.get("distance_km"):  "median",
        schema.get("freight_cost"): "median",
    }
    for col, method in numeric_impute.items():
        if col and col in df.columns:
            null_count = int(df[col].isnull().sum())
            if null_count:
                fill_val = df[col].median() if method == "median" else df[col].mean()
                df[col] = df[col].fillna(fill_val)
                _issue(f"Missing {col}", null_count,
                       f"Imputed with column {method} ({fill_val:.2f})",
                       f"Median imputation minimises the effect of outliers")

    # ── 2.12 Outlier detection (flag, not remove) ─────────────────────────────
    for col in [dist_col, cost_col, schema.get("weight_kg")]:
        if col and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            lo, hi = iqr_bounds(df[col].dropna())
            n_out = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n_out:
                _issue(f"Outliers in {col}", n_out,
                       f"Flagged via IQR fences [{lo:.1f}, {hi:.1f}]; not removed",
                       "Outliers retained; removal requires domain confirmation")

    report["rows_cleaned"] = len(df)
    report["rows_removed"] = report["rows_original"] - len(df)
    logger.info(
        "Cleaning complete. %d rows retained (removed %d).",
        len(df), report["rows_removed"]
    )
    return df, report


# ══════════════════════════════════════════════════════════════════════════════
# 3. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def engineer_features(
    df: pd.DataFrame,
    schema: dict[str, str | None],
) -> pd.DataFrame:
    """
    Phase 3: Derive new columns from existing fields.

    All new columns are prefixed with '_' to distinguish from raw data.
    Only columns that can be computed from available schema are created.
    """
    logger.info("── Phase 3: Feature Engineering ──────────────────────────────")
    df = df.copy()

    ship_col     = schema.get("ship_date")
    expected_col = schema.get("expected_delivery")
    actual_col   = schema.get("actual_delivery")
    cost_col     = schema.get("freight_cost")
    dist_col     = schema.get("distance_km")
    weight_col   = schema.get("weight_kg")

    # ── 3.1 Delivery delay (hours) ────────────────────────────────────────────
    if actual_col and expected_col:
        df["delivery_delay_hours"] = (
            (df[actual_col] - df[expected_col])
            .dt.total_seconds()
            .div(3600)
            .round(2)
        )
        logger.info("  Created: delivery_delay_hours")

        # ── 3.2 On-time flag ──────────────────────────────────────────────────
        df["on_time"] = df["delivery_delay_hours"] <= 0
        logger.info("  Created: on_time (boolean)")

        # ── 3.3 Late shipment indicator ───────────────────────────────────────
        df["is_late"] = ~df["on_time"]
        logger.info("  Created: is_late (boolean)")

    # ── 3.4 Delivery duration (days) ──────────────────────────────────────────
    if ship_col and actual_col:
        df["delivery_duration_days"] = (
            (df[actual_col] - df[ship_col])
            .dt.total_seconds()
            .div(86400)
            .round(2)
        )
        logger.info("  Created: delivery_duration_days")

    # ── 3.5 Cost per KM ───────────────────────────────────────────────────────
    if cost_col and dist_col:
        df["cost_per_km"] = np.where(
            df[dist_col] > 0,
            (df[cost_col] / df[dist_col]).round(4),
            np.nan,
        )
        logger.info("  Created: cost_per_km")

    # ── 3.6 Cost per KG ───────────────────────────────────────────────────────
    if cost_col and weight_col:
        df["cost_per_kg"] = np.where(
            df[weight_col] > 0,
            (df[cost_col] / df[weight_col]).round(4),
            np.nan,
        )
        logger.info("  Created: cost_per_kg")

    # ── 3.7 Temporal features ─────────────────────────────────────────────────
    if ship_col:
        df["ship_week"]    = df[ship_col].dt.isocalendar().week.astype(int)
        df["ship_month"]   = df[ship_col].dt.month
        df["ship_quarter"] = df[ship_col].dt.quarter
        df["ship_year"]    = df[ship_col].dt.year
        df["ship_month_name"] = df[ship_col].dt.strftime("%b %Y")
        logger.info("  Created: ship_week, ship_month, ship_quarter, ship_year, ship_month_name")

    # ── 3.8 Delay bucket (for visualisation) ─────────────────────────────────
    if "delivery_delay_hours" in df.columns:
        bins   = [-np.inf, 0, 24, 72, 168, np.inf]
        labels = ["On-Time", "≤1 day late", "1–3 days late",
                  "3–7 days late", ">7 days late"]
        df["delay_bucket"] = pd.cut(
            df["delivery_delay_hours"], bins=bins, labels=labels
        )
        logger.info("  Created: delay_bucket")

    logger.info("Feature engineering complete. Total columns: %d", len(df.columns))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. BUSINESS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def _on_time_pct(df: pd.DataFrame) -> float:
    """Return on-time percentage for a DataFrame slice."""
    if "on_time" not in df.columns or len(df) == 0:
        return np.nan
    return float(df["on_time"].mean() * 100)


def q1_region_performance(df: pd.DataFrame, schema: dict) -> dict[str, Any]:
    """
    Q1: Which region has the worst on-time delivery performance?
        Investigates WHY: carrier, warehouse, distance, freight cost.
    """
    logger.info("── Q1: Region Performance ────────────────────────────────────")
    region_col  = schema.get("region")
    carrier_col = schema.get("carrier")
    wh_col      = schema.get("warehouse")
    dist_col    = schema.get("distance_km")

    agg = (
        df.groupby(region_col)
        .agg(
            total_shipments   = (region_col, "count"),
            on_time_count     = ("on_time", "sum"),
            late_count        = ("is_late", "sum"),
            avg_delay_hours   = ("delivery_delay_hours", "mean"),
            median_delay_hours= ("delivery_delay_hours", "median"),
        )
        .reset_index()
    )
    agg["on_time_pct"]  = (agg["on_time_count"] / agg["total_shipments"] * 100).round(2)
    agg["late_pct"]     = (agg["late_count"]     / agg["total_shipments"] * 100).round(2)
    agg = agg.sort_values("on_time_pct")

    worst_region = agg.iloc[0][region_col]
    logger.info("  Worst region: %s  (on-time: %.1f%%)", worst_region, agg.iloc[0]["on_time_pct"])

    # ── Root-cause investigation ──────────────────────────────────────────────
    worst_df = df[df[region_col] == worst_region]
    best_df  = df[df[region_col] != worst_region]

    root_causes: dict[str, Any] = {}

    # Carrier mix in worst vs overall
    if carrier_col:
        carrier_delay = (
            worst_df.groupby(carrier_col)["delivery_delay_hours"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "avg_delay", "count": "shipments"})
            .reset_index()
            .sort_values("avg_delay", ascending=False)
        )
        root_causes["carrier_delay_in_worst_region"] = carrier_delay

    # Avg distance comparison
    if dist_col:
        root_causes["avg_distance_worst"] = float(worst_df[dist_col].mean())
        root_causes["avg_distance_others"] = float(best_df[dist_col].mean())

    # Warehouse breakdown in worst region
    if wh_col:
        wh_delay = (
            worst_df.groupby(wh_col)["delivery_delay_hours"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "avg_delay", "count": "shipments"})
            .reset_index()
            .sort_values("avg_delay", ascending=False)
        )
        root_causes["warehouse_delay_in_worst_region"] = wh_delay

    return {
        "region_summary": agg,
        "worst_region":   worst_region,
        "root_causes":    root_causes,
    }


def q2_freight_vs_distance(df: pd.DataFrame, schema: dict) -> dict[str, Any]:
    """
    Q2: Is freight cost related to distance?
        Identifies carriers that deviate from expected pricing.
    """
    logger.info("── Q2: Freight Cost vs Distance ──────────────────────────────")
    cost_col    = schema.get("freight_cost")
    dist_col    = schema.get("distance_km")
    carrier_col = schema.get("carrier")

    # Drop nulls for regression
    reg_df = df[[dist_col, cost_col, carrier_col]].dropna()

    # Global correlation
    correlation = float(reg_df[dist_col].corr(reg_df[cost_col]))
    logger.info("  Pearson correlation (distance vs cost): %.4f", correlation)

    # Linear regression
    X = reg_df[[dist_col]].values
    y = reg_df[cost_col].values
    model = LinearRegression().fit(X, y)
    slope     = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2        = float(model.score(X, y))

    reg_df = reg_df.copy()
    reg_df["expected_cost"] = model.predict(X)
    reg_df["residual"]      = reg_df[cost_col] - reg_df["expected_cost"]
    reg_df["abs_residual"]  = reg_df["residual"].abs()

    # Carrier-level deviation from expected pricing
    carrier_dev = (
        reg_df.groupby(carrier_col)
        .agg(
            mean_residual    = ("residual",      "mean"),
            mean_abs_residual= ("abs_residual",  "mean"),
            shipments        = (carrier_col,      "count"),
        )
        .reset_index()
        .sort_values("mean_residual", ascending=False)
    )
    carrier_dev["overcharge_vs_expected"] = carrier_dev["mean_residual"].round(2)

    return {
        "correlation":   correlation,
        "slope":         slope,
        "intercept":     intercept,
        "r2":            r2,
        "regression_df": reg_df,
        "carrier_deviation": carrier_dev,
    }


def q3_customer_delays(df: pd.DataFrame, schema: dict) -> dict[str, Any]:
    """
    Q3: Which customers experience the most delays?
        Investigates whether the driver is carrier, region, warehouse, or distance.
    """
    logger.info("── Q3: Customer Delays ───────────────────────────────────────")
    cust_col    = schema.get("customer")
    carrier_col = schema.get("carrier")
    region_col  = schema.get("region")
    wh_col      = schema.get("warehouse")
    dist_col    = schema.get("distance_km")

    cust_agg = (
        df.groupby(cust_col)
        .agg(
            total_shipments= (cust_col,               "count"),
            late_count     = ("is_late",               "sum"),
            avg_delay_hours= ("delivery_delay_hours",  "mean"),
        )
        .reset_index()
    )
    cust_agg["late_pct"] = (cust_agg["late_count"] / cust_agg["total_shipments"] * 100).round(2)
    cust_agg = cust_agg.sort_values("avg_delay_hours", ascending=False).reset_index(drop=True)
    top_delayed = cust_agg.head(15)

    # Root cause for top-5 delayed customers
    top5_customers = top_delayed.head(5)[cust_col].tolist()
    root_causes: dict[str, Any] = {}

    for cust in top5_customers:
        cust_df = df[df[cust_col] == cust]
        rc: dict[str, Any] = {}
        if carrier_col:
            rc["primary_carrier"] = cust_df[carrier_col].mode().iloc[0] if len(cust_df) else "N/A"
        if region_col:
            rc["primary_region"] = cust_df[region_col].mode().iloc[0] if len(cust_df) else "N/A"
        if dist_col:
            rc["avg_distance"] = float(cust_df[dist_col].mean())
        root_causes[cust] = rc

    # Overall: is customer delay driven by region or carrier?
    delay_by_carrier = (
        df[df[cust_col].isin(top5_customers)]
        .groupby(carrier_col)["delivery_delay_hours"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    ) if carrier_col else pd.DataFrame()

    delay_by_region = (
        df[df[cust_col].isin(top5_customers)]
        .groupby(region_col)["delivery_delay_hours"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    ) if region_col else pd.DataFrame()

    return {
        "customer_summary": cust_agg,
        "top_delayed":      top_delayed,
        "root_causes":      root_causes,
        "delay_by_carrier": delay_by_carrier,
        "delay_by_region":  delay_by_region,
    }


def q4_data_quality_report(cleaning_report: dict[str, Any]) -> dict[str, Any]:
    """
    Q4: Comprehensive data quality report summarising cleaning phase findings.
    """
    logger.info("── Q4: Data Quality Report ───────────────────────────────────")
    issues_df = pd.DataFrame(cleaning_report["issues"])
    total_issues = issues_df["count"].sum() if not issues_df.empty else 0

    return {
        "issues_df":       issues_df,
        "total_issues":    total_issues,
        "rows_original":   cleaning_report["rows_original"],
        "rows_cleaned":    cleaning_report.get("rows_cleaned", 0),
        "rows_removed":    cleaning_report["rows_removed"],
    }


def q5_kpi_recommendation(df: pd.DataFrame, schema: dict) -> dict[str, Any]:
    """
    Q5: Recommend the single weekly KPI that best predicts operational issues early.
        Recommended: Weekly On-Time Delivery Rate (WOTDR).
    """
    logger.info("── Q5: KPI Recommendation ────────────────────────────────────")

    # Compute the recommended KPI: Weekly On-Time Delivery Rate
    if "ship_week" not in df.columns or "on_time" not in df.columns:
        return {"kpi_series": pd.DataFrame(), "recommendation": ""}

    kpi_series = (
        df.groupby(["ship_year", "ship_week"])
        .agg(
            total_shipments = ("on_time", "count"),
            on_time_count   = ("on_time", "sum"),
            avg_delay_hours = ("delivery_delay_hours", "mean"),
        )
        .reset_index()
    )
    kpi_series["on_time_pct"] = (
        kpi_series["on_time_count"] / kpi_series["total_shipments"] * 100
    ).round(2)

    # Moving average for trend
    kpi_series["on_time_pct_ma4"] = (
        kpi_series["on_time_pct"].rolling(window=4, min_periods=1).mean().round(2)
    )

    # Week-over-week change
    kpi_series["wow_change"] = kpi_series["on_time_pct"].diff().round(2)

    recommendation = (
        "Weekly On-Time Delivery Rate (WOTDR)\n"
        "Formula: (Shipments delivered on or before expected date in a week) / "
        "(Total shipments dispatched that week) × 100\n"
        "Why it matters: WOTDR is a leading indicator of operational health. "
        "A two-week declining trend reliably precedes customer escalations, "
        "carrier SLA breaches, and revenue impact. It is simple to compute, "
        "universally understood by operations and leadership, and actionable: "
        "when WOTDR falls below 85%, the operations team should initiate a "
        "carrier review and triage regional bottlenecks immediately."
    )

    return {
        "kpi_series":       kpi_series,
        "recommendation":   recommendation,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_full_analysis(csv_path: str) -> dict[str, Any]:
    """
    Run all phases and return a single results dictionary.
    This is the primary entry point used by app.py and any CLI caller.
    """
    # Load
    df_raw = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df_raw), csv_path)

    # Phase 1 – Explore
    schema      = detect_schema(df_raw)
    exploration = explore(df_raw)

    # Phase 2 – Clean
    df_clean, cleaning_report = clean(df_raw, schema)

    # Phase 3 – Feature engineering
    df = engineer_features(df_clean, schema)

    # Phase 4 – Business analysis
    results_q1 = q1_region_performance(df, schema)
    results_q2 = q2_freight_vs_distance(df, schema)
    results_q3 = q3_customer_delays(df, schema)
    results_q4 = q4_data_quality_report(cleaning_report)
    results_q5 = q5_kpi_recommendation(df, schema)

    return {
        "df":          df,
        "df_raw":      df_raw,
        "schema":      schema,
        "exploration": exploration,
        "cleaning":    cleaning_report,
        "q1":          results_q1,
        "q2":          results_q2,
        "q3":          results_q3,
        "q4":          results_q4,
        "q5":          results_q5,
    }


if __name__ == "__main__":
    import json
    res = run_full_analysis("data/shipments.csv")
    print("\nQ1 – Worst Region:", res["q1"]["worst_region"])
    print("Q2 – Correlation:", round(res["q2"]["correlation"], 4))
    print("Q3 – Top delayed customer:", res["q3"]["top_delayed"].iloc[0]["customer"])
    print("Q4 – Total issues found:", res["q4"]["total_issues"])
    print("Q5 – KPI:", "Weekly On-Time Delivery Rate")
