"""
Step 1: Load master_block_history.parquet, build daily aggregates & block shares.

Produces three derived datasets:
  1. daily_df   — aggregated across blocks per (site, date) for daily model
  2. block_df   — block-level data enriched with block_share / admit_block_share
  3. share_wide — pivoted shares (one row per site-date, 4 share columns)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg


# ── Load & validate ──────────────────────────────────────────────────────────

def _load_master() -> pd.DataFrame:
    """Load master parquet and run schema checks."""
    print(f"  Loading {cfg.MASTER_PARQUET} ...")
    df = pd.read_parquet(cfg.MASTER_PARQUET)

    required = ["site", "date", "block", "total_enc", "admitted_enc"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in master parquet: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["site", "date", "block"]).reset_index(drop=True)

    # Row-count sanity
    n_s, n_d, n_b = df["site"].nunique(), df["date"].nunique(), df["block"].nunique()
    expected = n_s * n_d * n_b
    assert len(df) == expected, f"Row mismatch: {len(df)} vs {expected}"

    # Target integrity
    assert (df["total_enc"] >= 0).all(), "Negative total_enc"
    assert (df["admitted_enc"] <= df["total_enc"]).all(), "admitted > total"

    print(f"  {len(df):,} rows | {n_s} sites | {n_d:,} dates | {n_b} blocks")
    return df


# ── Daily aggregation ────────────────────────────────────────────────────────

def _build_daily(block_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate block-level data to daily (site, date) grain."""
    # Columns that are constant across blocks for a given (site, date)
    first_cols = {
        "dow": "first", "day": "first", "week_of_year": "first",
        "month": "first", "quarter": "first", "day_of_year": "first",
        "year": "first", "is_weekend": "first", "is_covid_era": "first",
        "is_holiday": "first", "is_halloween": "first",
        "event_count": "first", "days_since_epoch": "first",
        "temp_min": "first", "temp_max": "first",
        "precip": "first", "snowfall": "first",
        "school_in_session": "first",
    }
    # Only include columns that actually exist
    agg_dict = {"total_enc": "sum", "admitted_enc": "sum"}
    for col, func in first_cols.items():
        if col in block_df.columns:
            agg_dict[col] = func

    daily = (
        block_df
        .groupby(["site", "date"], as_index=False)
        .agg(agg_dict)
        .sort_values(["site", "date"])
        .reset_index(drop=True)
    )

    # Derive admit_rate (0/0 → 0.0)
    daily["admit_rate"] = np.where(
        daily["total_enc"] > 0,
        daily["admitted_enc"] / daily["total_enc"],
        0.0,
    )
    daily["admit_rate"] = daily["admit_rate"].clip(0, 1)

    return daily


# ── Block share computation ──────────────────────────────────────────────────

def _add_block_shares(block_df: pd.DataFrame) -> pd.DataFrame:
    """Compute block_share and admit_block_share columns."""
    df = block_df.copy()

    # Total encounter shares
    daily_total = df.groupby(["site", "date"])["total_enc"].transform("sum")
    df["block_share"] = df["total_enc"] / daily_total.clip(lower=1)
    # Zero-day: distribute equally
    zero_mask = daily_total == 0
    df.loc[zero_mask, "block_share"] = 1.0 / cfg.N_BLOCKS

    # Admitted encounter shares
    daily_admitted = df.groupby(["site", "date"])["admitted_enc"].transform("sum")
    df["admit_block_share"] = df["admitted_enc"] / daily_admitted.clip(lower=1)
    zero_adm_mask = daily_admitted == 0
    df.loc[zero_adm_mask, "admit_block_share"] = 1.0 / cfg.N_BLOCKS

    return df


def _build_share_wide(block_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot block shares into wide format: one row per (site, date)."""
    wide = block_df.pivot_table(
        index=["site", "date"],
        columns="block",
        values="block_share",
        aggfunc="first",
    )
    wide.columns = [f"share_b{int(c)}" for c in wide.columns]
    wide = wide.reset_index()
    return wide


# ── Validation ───────────────────────────────────────────────────────────────

def validate(daily_df: pd.DataFrame, block_df: pd.DataFrame) -> None:
    """Run sanity checks on outputs."""
    n_sites = daily_df["site"].nunique()
    n_dates = daily_df["date"].nunique()

    # Daily row count
    assert len(daily_df) == n_sites * n_dates, (
        f"Daily rows: {len(daily_df)} vs expected {n_sites * n_dates}"
    )

    # No duplicate (site, date)
    assert daily_df.duplicated(subset=["site", "date"]).sum() == 0, "Duplicate daily rows"

    # Block share sum ≈ 1
    share_sums = block_df.groupby(["site", "date"])["block_share"].sum()
    bad = (share_sums - 1.0).abs() > 1e-6
    assert bad.sum() == 0, f"{bad.sum()} (site,date) groups with share sum != 1"

    # admitted <= total (daily)
    assert (daily_df["admitted_enc"] <= daily_df["total_enc"]).all()

    # admit_rate in [0,1]
    assert daily_df["admit_rate"].between(0, 1).all()

    # Print summary
    for site in cfg.SITES:
        s = daily_df[daily_df["site"] == site]
        print(f"    Site {site}: daily total_enc mean={s['total_enc'].mean():.1f} "
              f"std={s['total_enc'].std():.1f} max={s['total_enc'].max()}")

    mean_shares = block_df.groupby("block")["block_share"].mean()
    print(f"    Mean block shares: {dict(mean_shares.round(3))}")


# ── Main entry point ─────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load master data, return (block_df, daily_df, share_wide_df).

    block_df  — block-level with block_share + admit_block_share columns
    daily_df  — daily aggregates with admit_rate
    share_wide — pivoted shares (site, date, share_b0..share_b3)
    """
    block_df = _load_master()
    daily_df = _build_daily(block_df)
    block_df = _add_block_shares(block_df)
    share_wide = _build_share_wide(block_df)

    print(f"  Daily: {len(daily_df):,} rows | Block: {len(block_df):,} rows")
    validate(daily_df, block_df)

    # Persist intermediates
    cfg.ensure_dirs()
    daily_df.to_parquet(cfg.DATA_DIR / "daily_df.parquet", index=False)
    block_df.to_parquet(cfg.DATA_DIR / "block_shares_df.parquet", index=False)
    share_wide.to_parquet(cfg.DATA_DIR / "share_wide_df.parquet", index=False)

    return block_df, daily_df, share_wide


if __name__ == "__main__":
    load_data()
