"""
Data loading & preprocessing for Pipeline D.

Loads master_block_history.parquet, validates schema, derives admit_rate,
applies COVID sample weights (freq_weights for GLM), imputes weather.
Pipeline D uses NO lagged target features — only deterministic time-based.
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
        f"({n_sites}x{n_dates}x{n_blocks})"
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

    # ── COVID sample weights (freq_weights for statsmodels GLM) ──────────
    if "is_covid_era" not in df.columns:
        raise ValueError("Column 'is_covid_era' missing from input data. Check Data Source.")

    covid_mask = df["is_covid_era"]
    df["sample_weight"] = np.where(covid_mask, cfg.COVID_WEIGHT, 1.0)

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
    n_covid = int(covid_mask.sum())
    print(f"  COVID-era rows: {n_covid:,} (weight={cfg.COVID_WEIGHT})")
    for col in weather_cols:
        print(f"  {col}: {df[col].isna().sum()} NaN remaining")

    return df


def get_fold_data(
    df: pd.DataFrame,
    fold: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and validation for a given fold."""
    train_end = pd.Timestamp(fold["train_end"])
    val_start = pd.Timestamp(fold["val_start"])
    val_end   = pd.Timestamp(fold["val_end"])

    train = df[df["date"] <= train_end].copy()
    val   = df[(df["date"] >= val_start) & (df["date"] <= val_end)].copy()
    return train, val


def get_site_block_subset(
    df: pd.DataFrame,
    site: str,
    block: int,
) -> pd.DataFrame:
    """Extract a single (site, block) time series, sorted by date."""
    mask = (df["site"] == site) & (df["block"] == block)
    return df[mask].copy().sort_values("date").reset_index(drop=True)


if __name__ == "__main__":
    load_data()
