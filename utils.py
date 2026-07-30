"""
utils.py
--------
Shared utility functions used across analysis.py and app.py.

Covers:
  - Logging setup
  - DataFrame display helpers
  - Colour / style constants for Plotly
  - Safe division
  - Percentage formatting
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

# ── logging ────────────────────────────────────────────────────────────────────

def get_logger(name: str = "shipment_analytics") -> logging.Logger:
    """Return a consistently configured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


# ── colour palette ─────────────────────────────────────────────────────────────

PALETTE = {
    "primary":   "#4F8EF7",
    "secondary": "#F7994F",
    "success":   "#4FC78E",
    "danger":    "#F74F4F",
    "warning":   "#F7D94F",
    "neutral":   "#A0AEC0",
    "bg":        "#0F172A",
    "card":      "#1E293B",
    "text":      "#E2E8F0",
}

CARRIER_COLORS = {
    "FastShip":      "#4F8EF7",
    "QuickMove":     "#4FC78E",
    "ReliableCo":    "#F7994F",
    "SpeedEx":       "#F74F4F",
    "GlobalFreight": "#C084FC",
}

REGION_COLORS = {
    "North":   "#4F8EF7",
    "South":   "#F74F4F",
    "East":    "#4FC78E",
    "West":    "#F7994F",
    "Central": "#C084FC",
}

PLOTLY_TEMPLATE = "plotly_dark"


# ── math helpers ───────────────────────────────────────────────────────────────

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Return numerator / denominator, or *default* when denominator is zero."""
    if denominator == 0:
        return default
    return numerator / denominator


def pct(value: float, total: float, decimals: int = 1) -> str:
    """Format value/total as a percentage string."""
    if total == 0:
        return "N/A"
    return f"{100 * value / total:.{decimals}f}%"


def fmt_currency(value: float) -> str:
    """Format a float as a dollar currency string."""
    return f"${value:,.2f}"


def fmt_number(value: float, decimals: int = 1) -> str:
    """Format a float with thousands separator."""
    return f"{value:,.{decimals}f}"


# ── dataframe helpers ──────────────────────────────────────────────────────────

def describe_df(df: pd.DataFrame) -> dict:
    """
    Return a summary dict containing:
      rows, cols, dtypes, nulls, duplicates, numeric stats, categorical stats
    """
    summary = {
        "rows":           len(df),
        "cols":           len(df.columns),
        "column_names":   list(df.columns),
        "dtypes":         df.dtypes.astype(str).to_dict(),
        "null_counts":    df.isnull().sum().to_dict(),
        "null_pct":       (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_stats":  df.select_dtypes(include="number").describe().to_dict(),
        "categorical":    {},
    }
    for col in df.select_dtypes(include=["object", "category"]).columns:
        summary["categorical"][col] = {
            "unique":    df[col].nunique(),
            "top_5":     df[col].value_counts().head(5).to_dict(),
            "null_count": int(df[col].isnull().sum()),
        }
    return summary


def column_exists(df: pd.DataFrame, col: str) -> bool:
    """Check whether *col* is present in DataFrame (case-insensitive)."""
    return col.lower() in [c.lower() for c in df.columns]


def find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """
    Return the first column name in *candidates* that exists in df.
    Returns None if none match.
    """
    col_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in col_lower:
            return col_lower[candidate.lower()]
    return None


# ── outlier helpers ────────────────────────────────────────────────────────────

def iqr_bounds(series: pd.Series, k: float = 1.5) -> tuple[float, float]:
    """Return (lower, upper) IQR fences for outlier detection."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def flag_outliers(df: pd.DataFrame, col: str, k: float = 1.5) -> pd.Series:
    """Return a boolean Series marking IQR outliers in *col*."""
    lo, hi = iqr_bounds(df[col].dropna(), k)
    return (df[col] < lo) | (df[col] > hi)
