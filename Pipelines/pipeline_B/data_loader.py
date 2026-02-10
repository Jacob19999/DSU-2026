"""
Pipeline B — Step 1: Data Loading & Preprocessing.

Loads master_block_history, derives admit_rate, applies COVID sample weights,
imputes weather, and provides fold-aware data slicing.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from . import config as cfg

logger = logging.getLogger(__name__)


# ── 1.1  Load Master Dataset ────────────────────────────────────────────────

def load_master_data(path: Path | None = None) -> pd.DataFrame:
    """Load the unified raw dataset produced by the Data Source layer.

    Tries parquet first, falls back to CSV.  Validates required columns and
    sorts by (site, block, date) which is critical for lag computation.
    """
    path = path or cfg.MASTER_PARQUET
    if path.suffix == ".parquet" and path.exists():
        df = pd.read_parquet(path)
        logger.info("Loaded master data from parquet: %s rows", len(df))
    elif cfg.MASTER_CSV.exists():
        df = pd.read_csv(cfg.MASTER_CSV, parse_dates=["date"])
        logger.info("Loaded master data from CSV fallback: %s rows", len(df))
    else:
        raise FileNotFoundError(
            f"Master data not found at {path} or {cfg.MASTER_CSV}. "
            "Run the Data Source pipeline first."
        )

    # Normalise column names to lowercase
    df.columns = df.columns.str.strip().str.lower()

    # Ensure date is datetime
    df["date"] = pd.to_datetime(df["date"])

    _validate_schema(df)

    # Sort — critical for shift/rolling operations
    df = df.sort_values(["site", "block", "date"]).reset_index(drop=True)
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Assert the minimum required columns exist."""
    required = {"site", "date", "block", "total_enc", "admitted_enc"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Master data missing required columns: {missing}")

    # Targets should not be NaN (Data Source fills with 0)
    for col in ("total_enc", "admitted_enc"):
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            logger.warning("%s has %d NaN values — filling with 0", col, n_nan)
            df[col] = df[col].fillna(0).astype(int)


# ── 1.2  Derive Admit Rate ──────────────────────────────────────────────────

def derive_admit_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Compute admit_rate = admitted_enc / total_enc, handling zero division."""
    df = df.copy()
    df[cfg.TARGET_RATE] = np.where(
        df[cfg.TARGET_TOTAL] > 0,
        df[cfg.TARGET_ADMITTED] / df[cfg.TARGET_TOTAL],
        0.0,
    )
    # Safety clip
    df[cfg.TARGET_RATE] = df[cfg.TARGET_RATE].clip(0.0, 1.0)

    _log_rate_stats(df)
    return df


def _log_rate_stats(df: pd.DataFrame) -> None:
    rate = df[cfg.TARGET_RATE]
    logger.info(
        "admit_rate stats — mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
        rate.mean(), rate.std(), rate.min(), rate.max(),
    )
    if (rate > 1).any() or (rate < 0).any():
        logger.error("admit_rate out of [0,1] AFTER clip — this should never happen")


# ── 1.3  COVID Sample Weights ───────────────────────────────────────────────

def add_sample_weights(df: pd.DataFrame) -> pd.DataFrame:
    """Create WAPE-aligned sample weights with COVID downweighting.

    weight = max(total_enc, MIN_SAMPLE_WEIGHT) * covid_factor
    """
    df = df.copy()

    # COVID flag (may already exist from Data Source)
    if "is_covid_era" not in df.columns:
        df["is_covid_era"] = df["date"].between(
            pd.Timestamp(cfg.COVID_START),
            pd.Timestamp(cfg.COVID_END),
        )

    covid_factor = np.where(df["is_covid_era"], cfg.COVID_WEIGHT_FACTOR, 1.0)
    base_weight = np.maximum(df[cfg.TARGET_TOTAL].values, cfg.MIN_SAMPLE_WEIGHT)
    df["sample_weight"] = base_weight * covid_factor

    _log_weight_stats(df)
    return df


def _log_weight_stats(df: pd.DataFrame) -> None:
    covid_rows = df[df["is_covid_era"]]
    normal_rows = df[~df["is_covid_era"]]
    logger.info(
        "Sample weights — COVID era: n=%d, mean_w=%.2f | Normal: n=%d, mean_w=%.2f",
        len(covid_rows), covid_rows["sample_weight"].mean(),
        len(normal_rows), normal_rows["sample_weight"].mean(),
    )


# ── 1.4  Weather Imputation ─────────────────────────────────────────────────

WEATHER_COLS = ["temp_min", "temp_max", "precip", "snowfall"]


def impute_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill → backward-fill → monthly climatology per site."""
    df = df.copy()

    present_cols = [c for c in WEATHER_COLS if c in df.columns]
    if not present_cols:
        logger.warning("No weather columns found — skipping imputation")
        return df

    # Per-site forward/backward fill
    for col in present_cols:
        df[col] = df.groupby("site")[col].transform(
            lambda s: s.ffill().bfill()
        )

    # Remaining NaN: fill with site-level monthly climatology
    for col in present_cols:
        remaining_nan = df[col].isna().sum()
        if remaining_nan > 0:
            climatology = df.groupby(["site", df["date"].dt.month])[col].transform("mean")
            df[col] = df[col].fillna(climatology)
            logger.info("%s: filled %d remaining NaN with monthly climatology", col, remaining_nan)

    # Derived feature
    if "temp_min" in df.columns and "temp_max" in df.columns:
        df["temp_range"] = df["temp_max"] - df["temp_min"]

    return df


# ── 1.5  Fold Data Slicing ──────────────────────────────────────────────────

def get_fold_data(
    df: pd.DataFrame,
    train_end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train (≤ train_end) and the full df for lag context.

    Returns:
        train_df: rows with date <= train_end (for supervised examples)
        full_df:  the entire dataframe (lags may reach back before train start)
    """
    cutoff = pd.Timestamp(train_end)
    train_df = df[df["date"] <= cutoff].copy()
    logger.info(
        "Fold split: train_end=%s → train rows=%d (dates %s to %s)",
        train_end, len(train_df),
        train_df["date"].min().date(), train_df["date"].max().date(),
    )
    return train_df, df


# ── Full Preprocessing Pipeline ─────────────────────────────────────────────

def load_and_preprocess(path: Path | None = None) -> pd.DataFrame:
    """Run the complete Step 1 pipeline: load → admit_rate → weights → weather.

    Returns a preprocessed DataFrame ready for feature engineering.
    """
    df = load_master_data(path)
    df = derive_admit_rate(df)
    df = add_sample_weights(df)
    df = impute_weather(df)

    # ── Eval checks ──
    _run_step1_checks(df)
    return df


def _run_step1_checks(df: pd.DataFrame) -> None:
    """Step 1 validation checks — prints warnings/errors."""
    n_sites = df["site"].nunique()
    n_blocks = df["block"].nunique()
    n_dates = df["date"].nunique()
    expected_rows = n_sites * n_blocks * n_dates

    logger.info("Step 1 checks:")
    logger.info("  Sites=%d, Blocks=%d, Dates=%d", n_sites, n_blocks, n_dates)
    logger.info("  Actual rows=%d, Expected=%d", len(df), expected_rows)
    if len(df) != expected_rows:
        logger.warning("  ROW COUNT MISMATCH — possible missing grid entries")

    # Target NaN
    for col in (cfg.TARGET_TOTAL, cfg.TARGET_ADMITTED):
        n_nan = df[col].isna().sum()
        if n_nan > 0:
            logger.error("  %s has %d NaN values!", col, n_nan)

    # Admit rate bounds
    rate = df[cfg.TARGET_RATE]
    if rate.min() < 0 or rate.max() > 1:
        logger.error("  admit_rate out of [0,1]: min=%.4f, max=%.4f", rate.min(), rate.max())
    else:
        logger.info("  admit_rate in [0,1] ✓")

    # Weight check
    if "sample_weight" in df.columns:
        covid_w = df.loc[df["is_covid_era"], "sample_weight"].mean()
        normal_w = df.loc[~df["is_covid_era"], "sample_weight"].mean()
        ratio = normal_w / max(covid_w, 1e-6)
        logger.info("  Weight ratio (normal/covid) = %.1f (expect ~10×)", ratio)

    # Date range
    logger.info(
        "  Date range: %s to %s",
        df["date"].min().date(), df["date"].max().date(),
    )

    # Per-site target summary
    for site in cfg.SITES:
        site_df = df[df["site"] == site]
        logger.info(
            "  Site %s — total_enc: mean=%.1f, std=%.1f | admitted: mean=%.1f",
            site,
            site_df[cfg.TARGET_TOTAL].mean(),
            site_df[cfg.TARGET_TOTAL].std(),
            site_df[cfg.TARGET_ADMITTED].mean(),
        )
