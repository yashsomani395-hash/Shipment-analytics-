"""
generate_data.py
----------------
Generates a realistic synthetic shipments.csv dataset (~5,000 rows)
for the Shipment Analytics take-home assignment.

Intentionally introduces realistic data quality issues:
  - ~3% missing values across selected fields
  - ~1% duplicate rows
  - A handful of impossible delivery dates
  - A handful of negative / zero freight costs
  - Inconsistent region and carrier casing
  - A small number of duplicate Shipment IDs

Run once:  python generate_data.py
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── constants ──────────────────────────────────────────────────────────────────
N_ROWS = 5_000

REGIONS = ["North", "South", "East", "West", "Central"]
# Intentionally add casing variants to be cleaned later
REGION_VARIANTS = {
    "North": ["North", "north", "NORTH"],
    "South": ["South", "south"],
    "East": ["East", "EAST", "East "],
    "West": ["West", "west"],
    "Central": ["Central", "central", "CENTRAL"],
}

CARRIERS = ["FastShip", "QuickMove", "ReliableCo", "SpeedEx", "GlobalFreight"]
CARRIER_VARIANTS = {
    "FastShip": ["FastShip", "fastship", "Fast Ship"],
    "QuickMove": ["QuickMove", "Quick Move", "QUICKMOVE"],
    "ReliableCo": ["ReliableCo", "reliable co", "Reliable Co"],
    "SpeedEx": ["SpeedEx", "speedex", "Speed Ex"],
    "GlobalFreight": ["GlobalFreight", "Global Freight", "GLOBALFREIGHT"],
}

WAREHOUSES = [f"WH-{i:02d}" for i in range(1, 11)]

CUSTOMER_PREFIXES = [
    "Acme", "Global", "Prime", "Alpha", "Beta", "Delta", "Sigma", "Omega",
    "Apex", "Zenith", "Vertex", "Atlas", "Orion", "Titan", "Nova",
]
CUSTOMER_SUFFIXES = [
    "Corp", "LLC", "Inc", "Ltd", "Group", "Industries", "Solutions",
    "Enterprises", "Partners", "Co",
]
CUSTOMERS = [
    f"{p} {s}"
    for p in CUSTOMER_PREFIXES
    for s in CUSTOMER_SUFFIXES
][:80]  # 80 unique customers

# Carrier pricing characteristics: (base_cost_per_mile, noise_std)
CARRIER_PRICING = {
    "FastShip":      (0.85, 0.15),   # slightly above average
    "QuickMove":     (0.70, 0.10),   # lean / competitive
    "ReliableCo":    (0.75, 0.20),   # moderate
    "SpeedEx":       (1.10, 0.25),   # premium but overcharges on short routes
    "GlobalFreight": (0.60, 0.30),   # budget but high variance
}

# Regional delay profiles: (mean_delay_hours, std_delay_hours)
REGION_DELAY = {
    "North":   (8,  12),
    "South":   (20, 18),   # worst performer
    "East":    (5,  10),
    "West":    (10, 15),
    "Central": (3,  8),
}

START_DATE = datetime(2024, 1, 1)
END_DATE   = datetime(2025, 6, 30)


def random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def build_row(i: int, region: str, carrier: str) -> dict:
    shipment_id = f"SHP-{i:05d}"
    warehouse   = random.choice(WAREHOUSES)
    customer    = random.choice(CUSTOMERS)
    distance_km = max(50, np.random.lognormal(mean=5.5, sigma=0.6))  # ~50–2000 km

    ship_date = random_date(START_DATE, END_DATE)

    # Expected delivery: 1-7 days depending on distance
    expected_days = max(1, int(distance_km / 400) + random.randint(1, 3))
    expected_delivery = ship_date + timedelta(days=expected_days)

    # Actual delay (hours)
    mu, sigma = REGION_DELAY[region]
    delay_h = np.random.normal(mu, sigma)
    actual_delivery = expected_delivery + timedelta(hours=delay_h)

    # Freight cost
    base_rate, noise = CARRIER_PRICING[carrier]
    freight_cost = max(0.01, base_rate * distance_km + np.random.normal(0, noise * distance_km))

    return {
        "shipment_id":       shipment_id,
        "ship_date":         ship_date.strftime("%Y-%m-%d"),
        "expected_delivery": expected_delivery.strftime("%Y-%m-%d"),
        "actual_delivery":   actual_delivery.strftime("%Y-%m-%d"),
        "region":            region,
        "carrier":           carrier,
        "warehouse":         warehouse,
        "customer":          customer,
        "distance_km":       round(distance_km, 2),
        "freight_cost":      round(freight_cost, 2),
        "weight_kg":         round(max(1, np.random.lognormal(3, 0.7)), 2),
        "shipment_status":   random.choices(
            ["Delivered", "In Transit", "Delayed", "Returned"],
            weights=[75, 10, 12, 3]
        )[0],
    }


def inject_issues(df: pd.DataFrame) -> pd.DataFrame:
    """Introduce realistic data quality issues."""
    df = df.copy()
    n = len(df)

    # 1. Casing variants for region
    for _, row in df.iterrows():
        if random.random() < 0.08:
            df.at[row.name, "region"] = random.choice(REGION_VARIANTS[row["region"]])

    # 2. Casing variants for carrier
    for _, row in df.iterrows():
        if random.random() < 0.08:
            df.at[row.name, "carrier"] = random.choice(CARRIER_VARIANTS[row["carrier"]])

    # 3. Missing values (~3%)
    cols_with_missing = ["customer", "freight_cost", "weight_kg", "warehouse"]
    for col in cols_with_missing:
        mask = np.random.random(n) < 0.015
        df.loc[mask, col] = np.nan

    # 4. Negative freight cost (handful)
    neg_idx = np.random.choice(n, size=8, replace=False)
    df.loc[neg_idx, "freight_cost"] = -abs(df.loc[neg_idx, "freight_cost"])

    # 5. Zero freight cost (handful)
    zero_idx = np.random.choice(n, size=5, replace=False)
    df.loc[zero_idx, "freight_cost"] = 0.0

    # 6. Impossible delivery dates (actual before ship_date)
    bad_idx = np.random.choice(n, size=10, replace=False)
    df.loc[bad_idx, "actual_delivery"] = df.loc[bad_idx, "ship_date"]

    # 7. Duplicate rows (~1%)
    dup_idx = np.random.choice(n, size=int(n * 0.01), replace=False)
    dups = df.iloc[dup_idx].copy()
    df = pd.concat([df, dups], ignore_index=True)

    # 8. Duplicate Shipment IDs with different data (5 cases)
    dup_id_idx = np.random.choice(n, size=5, replace=False)
    extra = df.iloc[dup_id_idx].copy()
    extra["freight_cost"] = extra["freight_cost"] * 1.5
    df = pd.concat([df, extra], ignore_index=True)

    # 9. Empty customer strings (in addition to NaN)
    empty_idx = np.random.choice(len(df), size=6, replace=False)
    df.loc[empty_idx, "customer"] = ""

    # 10. Negative distance (handful)
    neg_dist_idx = np.random.choice(len(df), size=4, replace=False)
    df.loc[neg_dist_idx, "distance_km"] = -abs(df.loc[neg_dist_idx, "distance_km"])

    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def main() -> None:
    rows = []
    for i in range(1, N_ROWS + 1):
        region  = random.choices(REGIONS, weights=[20, 18, 22, 20, 20])[0]
        carrier = random.choice(CARRIERS)
        rows.append(build_row(i, region, carrier))

    df = pd.DataFrame(rows)
    df = inject_issues(df)

    output_path = "data/shipments.csv"
    import os
    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Generated {len(df):,} rows -> {output_path}")
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()
