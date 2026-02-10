"""
Step 3: Feature engineering for the block-share prediction model.

The share model predicts P(block | site, date features) — the proportion
of daily encounters that fall in each 6-hour block.

Supports three approaches (selected via config.SHARE_MODEL_TYPE):
  A. Softmax GBDT (preferred) — multiclass LightGBM
  B. Climatology fallback     — historical mean shares by (site, dow, month)

All share lags use shifts >= 63 days to prevent leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg


# ── Share lag features ───────────────────────────────────────────────────────

def _add_share_lags(block_df: pd.DataFrame) -> pd.DataFrame:
    """Per (site, block): lagged block_share and admit_block_share."""
    df = block_df.copy()

    for (_site, _blk), grp in df.groupby(["site", "block"]):
        idx = grp.index
        for lag in cfg.LAG_DAYS_SHARES:
            df.loc[idx, f"share_lag_{lag}"] = grp["block_share"].shift(lag).values
            df.loc[idx, f"admit_share_lag_{lag}"] = grp["admit_block_share"].shift(lag).values

    return df


def _add_share_rolling(block_df: pd.DataFrame) -> pd.DataFrame:
    """Per (site, block): rolling mean of block_share, shifted by 63 days."""
    df = block_df.copy()

    for (_site, _blk), grp in df.groupby(["site", "block"]):
        idx = grp.index
        shifted = grp["block_share"].shift(cfg.ROLLING_SHIFT_SHARES)
        for w in [7, 14, 28]:
            df.loc[idx, f"share_roll_mean_{w}"] = (
                shifted.rolling(w, min_periods=1).mean().values
            )
    return df


# ── Calendar features for share model ────────────────────────────────────────

def _add_share_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Add lightweight calendar features relevant to block distribution."""
    # Cyclical DOW & month
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


# ── Site encoding ────────────────────────────────────────────────────────────

def _encode_site(df: pd.DataFrame) -> pd.DataFrame:
    site_map = {s: i for i, s in enumerate(cfg.SITES)}
    df["site_enc"] = df["site"].map(site_map).astype(int)
    return df


def _cast_bools(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)
    return df


# ── Climatology computation ──────────────────────────────────────────────────

def compute_climatology(
    train_df: pd.DataFrame,
    keys: list[str] | None = None,
) -> pd.DataFrame:
    """Historical mean block shares by grouping keys — climatology baseline."""
    if keys is None:
        keys = cfg.CLIMATOLOGY_KEYS

    clim = (
        train_df
        .groupby(keys + ["block"])["block_share"]
        .mean()
        .reset_index()
        .pivot_table(index=keys, columns="block", values="block_share")
        .reset_index()
    )
    # Rename columns
    share_cols = [c for c in clim.columns if isinstance(c, (int, np.integer))]
    rename = {c: f"clim_share_b{int(c)}" for c in share_cols}
    clim = clim.rename(columns=rename)

    # Renormalize to sum to 1
    sc = [f"clim_share_b{b}" for b in cfg.BLOCKS]
    clim[sc] = clim[sc].div(clim[sc].sum(axis=1), axis=0)

    return clim


# ── Feature column resolver ─────────────────────────────────────────────────

def get_share_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return columns suitable as features for the share model."""
    candidates = [
        "site_enc",
        "dow", "month", "is_weekend", "quarter",
        "dow_sin", "dow_cos", "month_sin", "month_cos",
        "is_us_holiday", "is_halloween",
        "school_in_session",
        "event_count",
    ]
    # Add share lags & rolling
    candidates += [c for c in df.columns if c.startswith("share_lag_")]
    candidates += [c for c in df.columns if c.startswith("share_roll_mean_")]

    return sorted(c for c in candidates if c in df.columns)


# ── Main entry point ─────────────────────────────────────────────────────────

def engineer_share_features(block_df: pd.DataFrame) -> pd.DataFrame:
    """Run all share-model feature engineering. Returns enriched block_df copy."""
    df = block_df.copy()

    print("  [3a] Site encoding + bool cast ...")
    df = _encode_site(df)
    df = _cast_bools(df)

    # Carry over is_us_holiday from is_holiday if present
    if "is_holiday" in df.columns and "is_us_holiday" not in df.columns:
        df["is_us_holiday"] = df["is_holiday"].astype(int)

    print("  [3b] Share lag features ...")
    df = _add_share_lags(df)

    print("  [3c] Share rolling features ...")
    df = _add_share_rolling(df)

    print("  [3d] Calendar features for share model ...")
    df = _add_share_calendar(df)

    n_feat = len(get_share_feature_columns(df))
    print(f"  Share feature engineering complete: {n_feat} model features")

    # Persist
    cfg.ensure_dirs()
    df.to_parquet(cfg.DATA_DIR / "share_features.parquet", index=False)

    return df


if __name__ == "__main__":
    from step_01_data_loading import load_data

    block_df, _, _ = load_data()
    df = engineer_share_features(block_df)
    feats = get_share_feature_columns(df)
    print(f"\nShare feature columns ({len(feats)}):")
    for c in feats:
        print(f"  {c}")
