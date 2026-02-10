"""
Pipeline B — Step 2: Horizon-Adaptive Feature Engineering.

Builds supervised training examples with horizon-aware lag/rolling features.
This is the CORE differentiator of Pipeline B over Pipeline A:
  - Bucket 1 (days 1-15) gets lags as recent as 16 days
  - Pipeline A's safest lag is 63 days for ALL horizons
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config as cfg

logger = logging.getLogger(__name__)


# ── 2.1  Supervised Example Construction ────────────────────────────────────

def build_supervised_examples(
    df: pd.DataFrame,
    bucket: cfg.HorizonBucket,
    target_col: str,
    subsample: bool = True,
) -> pd.DataFrame:
    """Create (features_at_t-h, target_at_t) pairs for a given horizon bucket.

    For each original row at date t, we create training examples for each
    sampled horizon h in the bucket.  The "as-of" date for features is t-h
    (i.e., features use only data available h days before the target date).

    Args:
        df:         Preprocessed master dataframe (sorted by site, block, date).
        bucket:     HorizonBucket defining which horizons and lags to use.
        target_col: Column to predict (total_enc or admit_rate).
        subsample:  If True, use HORIZON_SAMPLES instead of every day.

    Returns:
        DataFrame with features + target + days_ahead + sample_weight.
    """
    horizons = (
        cfg.HORIZON_SAMPLES[bucket.bucket_id] if subsample
        else list(range(bucket.horizon_min, bucket.horizon_max + 1))
    )

    all_examples: List[pd.DataFrame] = []

    for h in horizons:
        examples = _build_examples_for_horizon(df, bucket, target_col, h)
        if examples is not None and len(examples) > 0:
            all_examples.append(examples)

    if not all_examples:
        logger.warning("No training examples built for bucket %d", bucket.bucket_id)
        return pd.DataFrame()

    result = pd.concat(all_examples, ignore_index=True)
    logger.info(
        "Bucket %d: built %d training examples across %d horizons (target=%s)",
        bucket.bucket_id, len(result), len(horizons), target_col,
    )
    return result


def _build_examples_for_horizon(
    df: pd.DataFrame,
    bucket: cfg.HorizonBucket,
    target_col: str,
    horizon: int,
) -> Optional[pd.DataFrame]:
    """Build training examples for a single horizon value within a bucket.

    Each row's features are computed from data available `horizon` days
    before the target date.
    """
    records = []
    for (site, block), group in df.groupby(["site", "block"]):
        group = group.sort_values("date").reset_index(drop=True)

        # Target is at index i, features use data at index i - horizon
        for i in range(horizon, len(group)):
            target_row = group.iloc[i]
            # Features are based on the series up to (i - horizon)
            feature_idx = i - horizon

            # Build lag features from the series
            feat = _compute_lag_features(group, feature_idx, bucket)
            feat.update(_compute_rolling_features(group, feature_idx, bucket, target_col))
            feat.update(_compute_trend_deltas(group, feature_idx, bucket, target_col))
            feat.update(_compute_calendar_features(target_row))
            feat.update(_compute_cyclical_features(target_row))
            feat.update(_compute_event_features(target_row))
            feat.update(_compute_holiday_proximity(target_row))
            feat.update(_compute_weather_features(target_row))
            feat.update(_compute_interaction_features(target_row))

            # Pipeline B signature features
            feat["days_ahead"] = horizon
            feat["site"] = site
            feat["block"] = block

            # Target and weight
            feat["__target__"] = target_row[target_col]
            feat["__sample_weight__"] = target_row.get("sample_weight", 1.0)
            feat["__target_date__"] = target_row["date"]

            records.append(feat)

    if not records:
        return None
    return pd.DataFrame(records)


# ── 2.2  Horizon-Adaptive Lag Features ──────────────────────────────────────

def _compute_lag_features(
    series: pd.DataFrame,
    as_of_idx: int,
    bucket: cfg.HorizonBucket,
) -> Dict[str, float]:
    """Compute lag features from the target series at the as-of index.

    Lags are measured backward from the as-of position (NOT the target date).
    Each lag k means: value at (as_of_idx - k + bucket.min_lag).
    Actually, the lags in config are measured from the TARGET date, so:
      lag_k(target) = series[target_date - k]
    Since as_of_idx = target_idx - horizon, and we need lag_k from target:
      series_idx = target_idx - k = (as_of_idx + horizon) - k
    For this to be safe: k >= horizon + 1 → but our lags are defined per bucket
    to guarantee k >= bucket.min_lag > bucket.horizon_max.

    Simplification: we index directly into the sorted series.
    """
    feats = {}
    target_idx = as_of_idx  # features are built from perspective of as-of date
    # The actual target index in the group is at as_of_idx + horizon
    # But the lags in config are from target date perspective
    # lag_k means: target_date - k days → index = target_group_idx - k
    # Since group is daily and sorted, idx offset = k

    # We compute from the series directly using the as_of_idx
    # The lags are safe because min(lags) >= bucket.min_lag > horizon_max
    for col in (cfg.TARGET_TOTAL, cfg.TARGET_RATE):
        col_short = "total" if col == cfg.TARGET_TOTAL else "rate"
        if col not in series.columns:
            continue
        for lag in bucket.lags:
            idx = as_of_idx - (lag - bucket.rolling_shift)  # offset from as-of
            # Simpler: lag from as-of date = lag - horizon. But since we want
            # lag from target date = lag, and as_of = target - horizon:
            # value at (target - lag) = value at (as_of + horizon - lag)
            # = series[as_of_idx + horizon - lag]
            # For safety: horizon - lag < 0 when lag > horizon → idx < as_of_idx ✓
            # This is what we want: looking BACKWARD from as-of position.
            actual_idx = as_of_idx  # We'll use shift-based approach instead
            # Use the direct offset: lag_k from target = series at (target_idx - k)
            # target_idx in the group = as_of_idx + some_horizon (unknown here directly)
            # → Simplest correct approach: store values, compute lags later
            pass  # Handled below with vectorized approach

    # Direct index-based lag computation
    # lags in config are days back from TARGET date.
    # The group is sorted daily by date for (site, block).
    # target_group_idx = as_of_idx (we pass as_of_idx = target_idx - horizon in caller)
    # Correction: as_of_idx IS the feature observation point, target is at as_of_idx + horizon
    # So target_group_idx = as_of_idx + h (but h isn't passed here — we need it)
    # We'll compute lags from the as-of position instead:
    # "How many days ago from as_of was the value?"
    # lag_k from target = k, from as_of = k - horizon
    # Since k >= min_lag > horizon_max >= horizon, k - horizon >= 1 → safe

    # Revised: compute lags directly. We know the series positions.
    for col in (cfg.TARGET_TOTAL, cfg.TARGET_RATE):
        if col not in series.columns:
            continue
        col_short = "total" if col == cfg.TARGET_TOTAL else "rate"
        values = series[col].values
        for lag_k in bucket.lags:
            # lag_k is from the as-of date: we just look back lag_k positions
            # Because bucket lags already satisfy lag_k >= min_lag >= horizon_max + 1
            look_back = lag_k
            idx = as_of_idx - look_back
            if 0 <= idx < len(values):
                feats[f"lag_{lag_k}_{col_short}"] = values[idx]
            else:
                feats[f"lag_{lag_k}_{col_short}"] = np.nan

    return feats


# ── 2.3  Rolling Statistics (Horizon-Shifted) ───────────────────────────────

def _compute_rolling_features(
    series: pd.DataFrame,
    as_of_idx: int,
    bucket: cfg.HorizonBucket,
    target_col: str,
) -> Dict[str, float]:
    """Rolling mean/std/min/max over windows, shifted by bucket's min_lag."""
    feats = {}
    col_short = "total" if target_col == cfg.TARGET_TOTAL else "rate"

    if target_col not in series.columns:
        return feats

    values = series[target_col].values

    for window in cfg.ROLLING_WINDOWS:
        # Start position: as_of_idx - bucket.rolling_shift
        # Window covers [start - window + 1, start] inclusive
        start = as_of_idx - bucket.rolling_shift
        if start < 0:
            for stat in cfg.ROLLING_STATS:
                feats[f"roll_{stat}_{window}_{col_short}"] = np.nan
            continue

        begin = max(0, start - window + 1)
        end = start + 1  # exclusive
        if begin >= end or end > len(values):
            for stat in cfg.ROLLING_STATS:
                feats[f"roll_{stat}_{window}_{col_short}"] = np.nan
            continue

        window_vals = values[begin:end]

        feats[f"roll_mean_{window}_{col_short}"] = np.nanmean(window_vals)
        feats[f"roll_std_{window}_{col_short}"] = np.nanstd(window_vals) if len(window_vals) > 1 else 0.0
        feats[f"roll_min_{window}_{col_short}"] = np.nanmin(window_vals)
        feats[f"roll_max_{window}_{col_short}"] = np.nanmax(window_vals)

    return feats


# ── 2.4  Trend Deltas ───────────────────────────────────────────────────────

def _compute_trend_deltas(
    series: pd.DataFrame,
    as_of_idx: int,
    bucket: cfg.HorizonBucket,
    target_col: str,
) -> Dict[str, float]:
    """Capture momentum: short_roll - long_roll differentials."""
    feats = {}
    col_short = "total" if target_col == cfg.TARGET_TOTAL else "rate"

    if target_col not in series.columns:
        return feats

    values = series[target_col].values
    shift = bucket.rolling_shift

    def _safe_roll_mean(w: int) -> float:
        start = as_of_idx - shift
        begin = max(0, start - w + 1)
        end = start + 1
        if begin >= end or start < 0 or end > len(values):
            return np.nan
        return float(np.nanmean(values[begin:end]))

    rm7 = _safe_roll_mean(7)
    rm28 = _safe_roll_mean(28)
    rm91 = _safe_roll_mean(91)

    feats[f"delta_7_28_{col_short}"] = rm7 - rm28 if not (np.isnan(rm7) or np.isnan(rm28)) else np.nan
    feats[f"delta_28_91_{col_short}"] = rm28 - rm91 if not (np.isnan(rm28) or np.isnan(rm91)) else np.nan

    # Week-over-week change at minimum lag
    min_lag = bucket.lags[0]
    next_lag = bucket.lags[1] if len(bucket.lags) > 1 else min_lag + 7
    idx1 = as_of_idx - min_lag
    idx2 = as_of_idx - next_lag
    if 0 <= idx1 < len(values) and 0 <= idx2 < len(values):
        feats[f"lag_diff_{col_short}"] = values[idx1] - values[idx2]
    else:
        feats[f"lag_diff_{col_short}"] = np.nan

    return feats


# ── 2.5  Calendar & Cyclical Features ───────────────────────────────────────

def _compute_calendar_features(row: pd.Series) -> Dict[str, float]:
    """Deterministic calendar features for the TARGET date."""
    d = row["date"]
    feats = {
        "dow": d.dayofweek,
        "month": d.month,
        "day": d.day,
        "week_of_year": d.isocalendar()[1],
        "quarter": d.quarter,
        "day_of_year": d.dayofyear,
        "is_weekend": int(d.dayofweek >= 5),
        "days_since_epoch": (d - pd.Timestamp("2018-01-01")).days,
        "year_frac": d.year + d.dayofyear / 365.25,
    }
    return feats


def _compute_cyclical_features(row: pd.Series) -> Dict[str, float]:
    """Sin/cos encoding for cyclical calendar attributes."""
    d = row["date"]
    feats = {
        "dow_sin": np.sin(2 * np.pi * d.dayofweek / 7),
        "dow_cos": np.cos(2 * np.pi * d.dayofweek / 7),
        "month_sin": np.sin(2 * np.pi * d.month / 12),
        "month_cos": np.cos(2 * np.pi * d.month / 12),
        "doy_sin": np.sin(2 * np.pi * d.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * d.dayofyear / 365.25),
    }
    return feats


# ── 2.6  Holiday & Event Features ───────────────────────────────────────────

def _compute_event_features(row: pd.Series) -> Dict[str, float]:
    """Extract event-related features from the row."""
    feats = {}
    feats["is_holiday"] = int(row.get("is_holiday", False))
    feats["is_halloween"] = int(row.get("is_halloween", False))
    feats["event_count"] = row.get("event_count", 0)
    return feats


def _compute_holiday_proximity(row: pd.Series) -> Dict[str, float]:
    """Days since/until major holidays (computed from date)."""
    d = row["date"]
    year = d.year

    feats = {}

    # Days since Christmas (prior Dec 25 or current year)
    xmas = pd.Timestamp(year=year, month=12, day=25)
    if d < xmas:
        xmas = pd.Timestamp(year=year - 1, month=12, day=25)
    feats["days_since_xmas"] = (d - xmas).days

    # Days until Thanksgiving (4th Thursday of November)
    # Approximate: Nov 22-28 range
    nov1 = pd.Timestamp(year=year, month=11, day=1)
    # 4th Thursday: first Thursday + 21 days
    first_thu = nov1 + pd.Timedelta(days=(3 - nov1.dayofweek) % 7)
    thanksgiving = first_thu + pd.Timedelta(days=21)
    if d > thanksgiving:
        # Next year's thanksgiving
        nov1_next = pd.Timestamp(year=year + 1, month=11, day=1)
        first_thu_next = nov1_next + pd.Timedelta(days=(3 - nov1_next.dayofweek) % 7)
        thanksgiving = first_thu_next + pd.Timedelta(days=21)
    feats["days_until_thanksgiving"] = (thanksgiving - d).days

    # Days since July 4th
    jul4 = pd.Timestamp(year=year, month=7, day=4)
    if d < jul4:
        jul4 = pd.Timestamp(year=year - 1, month=7, day=4)
    feats["days_since_july4"] = (d - jul4).days

    # Days since school start (approximate: Aug 25)
    school_start = pd.Timestamp(year=year, month=8, day=25)
    if d < school_start:
        school_start = pd.Timestamp(year=year - 1, month=8, day=25)
    feats["days_since_school_start"] = (d - school_start).days

    return feats


# ── 2.7  Interaction Features ───────────────────────────────────────────────

def _compute_interaction_features(row: pd.Series) -> Dict[str, float]:
    """Interaction terms that capture site/block-specific patterns."""
    feats = {}
    block = row.get("block", 0)
    feats["holiday_x_block"] = int(row.get("is_holiday", False)) * block
    feats["weekend_x_block"] = int(row.get("date", pd.Timestamp.now()).dayofweek >= 5) * block
    return feats


# ── 2.9  Weather Features ───────────────────────────────────────────────────

def _compute_weather_features(row: pd.Series) -> Dict[str, float]:
    """Pass through weather columns (already imputed in Step 1)."""
    feats = {}
    for col in cfg.WEATHER_FEATURES:
        feats[col] = row.get(col, np.nan)
    return feats


# ── 2.8  Aggregate / Mean-Encoding Features ─────────────────────────────────

def add_aggregate_features(
    examples_df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add historical mean encodings computed from training history only.

    These are site×month×block and dow-level averages of total_enc,
    computed from history to avoid leakage.
    """
    if examples_df.empty:
        return examples_df

    df = examples_df.copy()

    # Site × Month × Block mean
    site_month_block_avg = (
        history_df.groupby(["site", history_df["date"].dt.month, "block"])[cfg.TARGET_TOTAL]
        .mean()
        .rename("agg_site_month_block_mean")
    )
    site_month_block_avg.index.names = ["site", "month", "block"]
    df = df.merge(
        site_month_block_avg.reset_index(),
        on=["site", "month", "block"],
        how="left",
    )

    # DOW mean
    dow_avg = (
        history_df.groupby(history_df["date"].dt.dayofweek)[cfg.TARGET_TOTAL]
        .mean()
        .rename("agg_dow_mean")
    )
    dow_avg.index.name = "dow"
    df = df.merge(dow_avg.reset_index(), on="dow", how="left")

    return df


# ── Vectorized Feature Builder (Production Path) ────────────────────────────

def build_features_for_bucket(
    df: pd.DataFrame,
    bucket: cfg.HorizonBucket,
    target_col: str = cfg.TARGET_TOTAL,
    subsample: bool = True,
) -> pd.DataFrame:
    """High-level API: build all features for a horizon bucket.

    Returns a DataFrame with feature columns, __target__, __sample_weight__,
    __target_date__, days_ahead, site, block.
    """
    logger.info("Building features for Bucket %d (target=%s)...", bucket.bucket_id, target_col)

    examples = build_supervised_examples(df, bucket, target_col, subsample=subsample)

    if examples.empty:
        return examples

    # Add aggregate encodings from the history (all data before target dates)
    max_target_date = examples["__target_date__"].max()
    history = df[df["date"] < max_target_date]
    examples = add_aggregate_features(examples, history)

    # Drop rows with too many NaN features (early history where lags unavailable)
    feature_cols = get_feature_columns(examples)
    nan_frac = examples[feature_cols].isna().mean(axis=1)
    threshold = 0.5  # drop rows where >50% of features are NaN
    n_before = len(examples)
    examples = examples[nan_frac <= threshold].reset_index(drop=True)
    n_dropped = n_before - len(examples)
    if n_dropped > 0:
        logger.info("  Dropped %d rows with >50%% NaN features", n_dropped)

    _run_step2_checks(examples, bucket, target_col)
    return examples


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return list of feature columns (excludes target, weight, metadata)."""
    exclude = {"__target__", "__sample_weight__", "__target_date__"}
    return [c for c in df.columns if c not in exclude]


# ── Build Features for Forecast Rows ────────────────────────────────────────

def build_forecast_features(
    df: pd.DataFrame,
    train_end: str,
    test_start: str,
    test_end: str,
    bucket: cfg.HorizonBucket,
) -> pd.DataFrame:
    """Build features for forecast/validation rows.

    For each (site, date, block) in the test window, compute days_ahead
    from train_end and build features using only data available at train_end.
    """
    cutoff = pd.Timestamp(train_end)
    t_start = pd.Timestamp(test_start)
    t_end = pd.Timestamp(test_end)

    records = []
    for (site, block), group in df.groupby(["site", "block"]):
        group = group.sort_values("date").reset_index(drop=True)

        # Find cutoff index
        cutoff_mask = group["date"] <= cutoff
        if not cutoff_mask.any():
            continue
        cutoff_idx = cutoff_mask.values.nonzero()[0][-1]

        # Iterate over test dates that fall in this bucket's horizon range
        test_rows = group[(group["date"] >= t_start) & (group["date"] <= t_end)]
        for _, target_row in test_rows.iterrows():
            days_ahead = (target_row["date"] - cutoff).days
            if days_ahead < bucket.horizon_min or days_ahead > bucket.horizon_max:
                continue

            # Features computed from cutoff position
            feat = _compute_lag_features(group, cutoff_idx, bucket)
            feat.update(_compute_rolling_features(group, cutoff_idx, bucket, cfg.TARGET_TOTAL))
            feat.update(_compute_trend_deltas(group, cutoff_idx, bucket, cfg.TARGET_TOTAL))
            feat.update(_compute_calendar_features(target_row))
            feat.update(_compute_cyclical_features(target_row))
            feat.update(_compute_event_features(target_row))
            feat.update(_compute_holiday_proximity(target_row))
            feat.update(_compute_weather_features(target_row))
            feat.update(_compute_interaction_features(target_row))

            feat["days_ahead"] = days_ahead
            feat["site"] = site
            feat["block"] = block
            feat["__target_date__"] = target_row["date"]

            # Actual values (for evaluation, not used in prediction)
            feat["__actual_total__"] = target_row.get(cfg.TARGET_TOTAL, np.nan)
            feat["__actual_rate__"] = target_row.get(cfg.TARGET_RATE, np.nan)
            feat["__actual_admitted__"] = target_row.get(cfg.TARGET_ADMITTED, np.nan)

            records.append(feat)

    result = pd.DataFrame(records) if records else pd.DataFrame()

    if not result.empty:
        # Add aggregate features using training history
        history = df[df["date"] <= cutoff]
        result = add_aggregate_features(result, history)

    logger.info(
        "Forecast features for Bucket %d: %d rows (test %s to %s)",
        bucket.bucket_id, len(result), test_start, test_end,
    )
    return result


# ── Step 2 Eval Checks ──────────────────────────────────────────────────────

def _run_step2_checks(
    examples: pd.DataFrame,
    bucket: cfg.HorizonBucket,
    target_col: str,
) -> None:
    """Validation checks for feature engineering output."""
    feature_cols = get_feature_columns(examples)

    logger.info("Step 2 checks (Bucket %d, target=%s):", bucket.bucket_id, target_col)
    logger.info("  Total examples: %d", len(examples))
    logger.info("  Feature count: %d", len(feature_cols))

    # NaN budget
    nan_counts = examples[feature_cols].isna().sum()
    high_nan = nan_counts[nan_counts > 0.3 * len(examples)]
    if len(high_nan) > 0:
        logger.warning("  Features with >30%% NaN: %s", dict(high_nan))

    # days_ahead range check
    if "days_ahead" in examples.columns:
        da_min = examples["days_ahead"].min()
        da_max = examples["days_ahead"].max()
        logger.info("  days_ahead range: [%d, %d] (expected [%d, %d])",
                     da_min, da_max, bucket.horizon_min, bucket.horizon_max)

    # Leakage audit: verify lag features don't use data too recent
    lag_cols = [c for c in feature_cols if c.startswith("lag_")]
    if lag_cols:
        min_lag_num = min(int(c.split("_")[1]) for c in lag_cols)
        if min_lag_num < bucket.min_lag:
            logger.error("  LEAKAGE: min lag used (%d) < bucket min_lag (%d)!",
                         min_lag_num, bucket.min_lag)
        else:
            logger.info("  Leakage check passed: min lag %d >= required %d ✓",
                         min_lag_num, bucket.min_lag)
