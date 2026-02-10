"""
Data loading & preprocessing for Pipeline B.

Loads master_block_history.parquet, validates schema, derives admit_rate,
applies COVID sample weights, imputes weather (ffill→bfill→climatology).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg


def load_data() -> pd.DataFrame:
    """Load master parquet, validate, derive admit_rate, add weights, impute weather."""
    print(f"  Loading {cfg.MASTER_PARQUET} ...")
    df = pd.read_parquet(cfg.MASTER_PARQUET)

    # ── Schema validation ────────────────────────────────────────────────
    required = ["site", "date", "block", "total_enc", "admitted_enc"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["site", "block", "date"]).reset_index(drop=True)

    # ── Row-count sanity ─────────────────────────────────────────────────
    n_sites  = df["site"].nunique()
    n_dates  = df["date"].nunique()
    n_blocks = df["block"].nunique()
    expected = n_sites * n_dates * n_blocks
    assert len(df) == expected, (
        f"Row count mismatch: {len(df)} vs expected {expected} "
        f"({n_sites}×{n_dates}×{n_blocks})"
    )

    # ── Target integrity ─────────────────────────────────────────────────
    assert df["total_enc"].notna().all(), "NaN in total_enc"
    assert df["admitted_enc"].notna().all(), "NaN in admitted_enc"
    assert (df["total_enc"] >= 0).all(), "Negative total_enc"

    # ── Derive admit_rate (0/0 → 0.0, clipped to [0,1]) ─────────────────
    df["admit_rate"] = np.where(
        df["total_enc"] > 0,
        df["admitted_enc"] / df["total_enc"],
        0.0,
    )
    df["admit_rate"] = df["admit_rate"].clip(0.0, 1.0)

    # ── COVID sample weights ─────────────────────────────────────────────
    #  weight = max(total_enc, 1) × covid_factor
    covid_mask = df["is_covid_era"] if "is_covid_era" in df.columns else False
    base_weight = np.maximum(df["total_enc"], 1).astype(float)
    covid_factor = np.where(covid_mask, cfg.COVID_WEIGHT, 1.0)

    df["sample_weight"]      = base_weight * covid_factor                  # for total model
    df["sample_weight_rate"] = np.maximum(df["admitted_enc"], 1).astype(float) * covid_factor  # for rate model

    # ── Weather imputation: ffill → bfill → monthly climatology per site ─
    weather_cols = [c for c in ["temp_min", "temp_max", "precip", "snowfall"]
                    if c in df.columns]
    for col in weather_cols:
        df[col] = df.groupby("site")[col].ffill()
        df[col] = df.groupby("site")[col].bfill()
        if df[col].isna().any():
            clim = df.groupby(["site", "month"])[col].transform("mean")
            df[col] = df[col].fillna(clim)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"  {len(df):,} rows | {n_sites} sites | {n_dates:,} dates | {n_blocks} blocks")
    print(f"  Date range: {df['date'].min().date()} .. {df['date'].max().date()}")
    for t in ["total_enc", "admitted_enc"]:
        print(f"  {t}: mean={df[t].mean():.1f}  std={df[t].std():.1f}  max={df[t].max()}")
    print(f"  admit_rate: mean={df['admit_rate'].mean():.3f}")
    n_covid = covid_mask.sum() if isinstance(covid_mask, pd.Series) else 0
    print(f"  COVID-era rows: {n_covid:,} (weight={cfg.COVID_WEIGHT})")
    for col in weather_cols:
        print(f"  {col}: {df[col].isna().sum()} NaN remaining")

    return df


if __name__ == "__main__":
    load_data()
