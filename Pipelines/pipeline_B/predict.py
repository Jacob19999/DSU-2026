"""
Pipeline B — Step 4: Prediction & Post-Processing.

Generates forecasts for a date range by:
  1. Assigning rows to horizon buckets
  2. Building features per bucket
  3. Predicting total_enc and admit_rate per bucket
  4. Deriving admitted = total × rate
  5. Enforcing all hard constraints (non-neg, admitted ≤ total, integers)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import config as cfg
from .features import build_forecast_features, get_feature_columns

logger = logging.getLogger(__name__)


# ── 4.1–4.3  Generate Predictions per Bucket ────────────────────────────────

def predict_fold(
    df: pd.DataFrame,
    train_end: str,
    test_start: str,
    test_end: str,
    bucket_models: Dict[int, Dict[str, Any]],
) -> pd.DataFrame:
    """Generate submission-shaped predictions for a test window.

    Args:
        df:             Full preprocessed master dataframe.
        train_end:      Last training date (inclusive).
        test_start:     First test date.
        test_end:       Last test date.
        bucket_models:  Dict mapping bucket_id -> {model_total, model_rate}.

    Returns:
        DataFrame with columns: Site, Date, Block, ED Enc, ED Enc Admitted
    """
    all_preds: List[pd.DataFrame] = []

    for bucket in cfg.HORIZON_BUCKETS:
        if bucket.bucket_id not in bucket_models:
            logger.warning("No models for Bucket %d — skipping", bucket.bucket_id)
            continue

        models = bucket_models[bucket.bucket_id]
        model_total = models["model_total"]
        model_rate = models["model_rate"]

        # Build features for this bucket's forecast rows
        feat_df = build_forecast_features(df, train_end, test_start, test_end, bucket)

        if feat_df.empty:
            logger.info("Bucket %d: no rows in horizon range — skipping", bucket.bucket_id)
            continue

        # Get feature columns (exclude metadata)
        feature_cols = [c for c in get_feature_columns(feat_df)
                        if not c.startswith("__")]

        # Align feature columns with what the model was trained on
        train_features_total = model_total.feature_name_ if hasattr(model_total, "feature_name_") else feature_cols
        train_features_rate = model_rate.feature_name_ if hasattr(model_rate, "feature_name_") else feature_cols

        X_total = _align_features(feat_df, train_features_total)
        X_rate = _align_features(feat_df, train_features_rate)

        # Predict
        pred_total = model_total.predict(X_total)
        pred_rate = model_rate.predict(X_rate)

        # Clip rate to [0, 1]
        pred_rate = np.clip(pred_rate, 0.0, 1.0)

        # Clip total to >= 0
        pred_total = np.clip(pred_total, 0.0, None)

        # Derive admitted
        pred_admitted = pred_total * pred_rate

        # Build result
        bucket_preds = pd.DataFrame({
            "Site": feat_df["site"].values,
            "Date": feat_df["__target_date__"].values,
            "Block": feat_df["block"].values,
            "pred_total": pred_total,
            "pred_admitted": pred_admitted,
            "days_ahead": feat_df["days_ahead"].values,
        })

        all_preds.append(bucket_preds)
        logger.info(
            "Bucket %d: predicted %d rows (days_ahead %d-%d)",
            bucket.bucket_id, len(bucket_preds),
            bucket_preds["days_ahead"].min(), bucket_preds["days_ahead"].max(),
        )

    if not all_preds:
        raise ValueError("No predictions generated — all buckets empty or missing models")

    combined = pd.concat(all_preds, ignore_index=True)

    # Post-process
    submission = post_process(combined)

    _run_step4_checks(submission, test_start, test_end)
    return submission


def _align_features(df: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
    """Ensure feature DataFrame has exactly the expected columns, in order."""
    result = pd.DataFrame(index=df.index)
    for col in expected_cols:
        if col in df.columns:
            result[col] = df[col].values
        else:
            result[col] = np.nan  # Missing feature → NaN (LightGBM handles natively)
            logger.debug("Feature '%s' missing from forecast data — filled with NaN", col)
    return result


# ── 4.4  Constraint Enforcement & Post-Processing ───────────────────────────

def post_process(preds: pd.DataFrame) -> pd.DataFrame:
    """Apply all hard constraints and convert to submission format.

    Steps:
      1. Non-negativity
      2. admitted ≤ total (already by construction, safety clip)
      3. Largest-remainder integer rounding per (Site, Date)
      4. Final constraint re-check
    """
    df = preds.copy()

    # 1. Non-negativity
    df["pred_total"] = df["pred_total"].clip(lower=0)
    df["pred_admitted"] = df["pred_admitted"].clip(lower=0)

    # 2. admitted ≤ total safety
    df["pred_admitted"] = np.minimum(df["pred_admitted"], df["pred_total"])

    # 3. Largest-remainder rounding per (Site, Date)
    df = _largest_remainder_round(df, "pred_total", "ED Enc")
    df = _largest_remainder_round(df, "pred_admitted", "ED Enc Admitted")

    # 4. Final admitted ≤ total after rounding
    df["ED Enc Admitted"] = np.minimum(df["ED Enc Admitted"], df["ED Enc"])

    # Format submission
    df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    submission = df[["Site", "Date", "Block", "ED Enc", "ED Enc Admitted"]].copy()
    submission["Block"] = submission["Block"].astype(int)
    submission["ED Enc"] = submission["ED Enc"].astype(int)
    submission["ED Enc Admitted"] = submission["ED Enc Admitted"].astype(int)

    return submission


def _largest_remainder_round(
    df: pd.DataFrame,
    float_col: str,
    int_col: str,
) -> pd.DataFrame:
    """Apply largest-remainder rounding to preserve daily totals per (Site, Date).

    For each (Site, Date) group:
      - Floor all block values
      - Compute remainder = daily_sum_float - daily_sum_floor
      - Distribute the remainder to blocks with largest fractional parts
    """
    df = df.copy()
    df["_floor"] = np.floor(df[float_col]).astype(int)
    df["_frac"] = df[float_col] - df["_floor"]

    result_vals = df["_floor"].copy()

    for (site, date), group in df.groupby(["Site", "Date"]):
        idx = group.index
        floor_sum = group["_floor"].sum()
        float_sum = group[float_col].sum()
        remainder = int(round(float_sum)) - floor_sum

        if remainder > 0:
            # Distribute to blocks with largest fractional parts
            sorted_idx = group["_frac"].sort_values(ascending=False).index
            for i, ix in enumerate(sorted_idx):
                if i >= remainder:
                    break
                result_vals.loc[ix] += 1

    df[int_col] = result_vals.clip(lower=0)
    df.drop(columns=["_floor", "_frac"], inplace=True)
    return df


# ── Step 4 Eval Checks ──────────────────────────────────────────────────────

def _run_step4_checks(submission: pd.DataFrame, test_start: str, test_end: str) -> None:
    """Validate the submission output."""
    logger.info("Step 4 checks:")

    # Row count
    n_days = (pd.Timestamp(test_end) - pd.Timestamp(test_start)).days + 1
    expected_rows = len(cfg.SITES) * n_days * len(cfg.BLOCKS)
    actual_rows = len(submission)
    logger.info("  Rows: %d (expected %d)", actual_rows, expected_rows)
    if actual_rows != expected_rows:
        logger.error("  ROW COUNT MISMATCH!")

    # Schema
    required_cols = {"Site", "Date", "Block", "ED Enc", "ED Enc Admitted"}
    missing = required_cols - set(submission.columns)
    if missing:
        logger.error("  Missing columns: %s", missing)
    else:
        logger.info("  Schema check ✓")

    # Non-negativity
    neg_total = (submission["ED Enc"] < 0).sum()
    neg_admitted = (submission["ED Enc Admitted"] < 0).sum()
    if neg_total > 0 or neg_admitted > 0:
        logger.error("  Negative values: total=%d, admitted=%d", neg_total, neg_admitted)
    else:
        logger.info("  Non-negativity ✓")

    # Admitted ≤ Total
    violations = (submission["ED Enc Admitted"] > submission["ED Enc"]).sum()
    if violations > 0:
        logger.error("  Admitted > Total violations: %d rows", violations)
    else:
        logger.info("  Admitted ≤ Total ✓")

    # Integer check
    for col in ("ED Enc", "ED Enc Admitted"):
        if not all(isinstance(v, (int, np.integer)) for v in submission[col]):
            logger.error("  %s contains non-integer values!", col)
        else:
            logger.info("  %s integer check ✓", col)

    # Distribution summary
    for col in ("ED Enc", "ED Enc Admitted"):
        vals = submission[col]
        logger.info(
            "  %s: mean=%.1f, std=%.1f, min=%d, max=%d",
            col, vals.mean(), vals.std(), vals.min(), vals.max(),
        )

    # Per-site distribution
    for site in cfg.SITES:
        site_df = submission[submission["Site"] == site]
        logger.info(
            "  Site %s: total_mean=%.1f, admitted_mean=%.1f",
            site, site_df["ED Enc"].mean(), site_df["ED Enc Admitted"].mean(),
        )
