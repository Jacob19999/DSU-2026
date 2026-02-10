"""
Pipeline B — Step 5: Evaluation.

Runs 4-fold forward validation using the eval.md contract.
Collects per-fold WAPE metrics and produces diagnostic breakdowns.
Saves OOF predictions for downstream ensemble stacking.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from . import config as cfg

logger = logging.getLogger(__name__)


# ── Metric Functions (self-contained, matching eval.md) ──────────────────────

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error — primary metric."""
    denom = float(np.sum(np.abs(y_true)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y = y_true.astype(float)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - (ss_res / ss_tot)


# ── Ground Truth Builder ─────────────────────────────────────────────────────

def build_ground_truth(raw_csv_path: Path) -> pd.DataFrame:
    """Build block-level ground truth from the raw DSU dataset.

    This matches eval.md's hourly_to_blocks_truth() logic.
    """
    df = pd.read_csv(raw_csv_path)
    df["Date"] = pd.to_datetime(df["Date"])
    df["Block"] = (df["Hour"] // 6).astype(int)

    truth = (
        df.groupby(["Site", "Date", "Block"], as_index=False)
        .agg({"ED Enc": "sum", "ED Enc Admitted": "sum"})
    )
    truth["Date"] = truth["Date"].dt.strftime("%Y-%m-%d")
    return truth


# ── Single Fold Scoring ──────────────────────────────────────────────────────

def score_fold(
    truth: pd.DataFrame,
    predictions: pd.DataFrame,
    fold: cfg.Fold,
) -> Dict[str, Any]:
    """Score a single fold's predictions against ground truth.

    Returns a dict with overall metrics + by-site + by-block breakdowns.
    """
    # Normalize prediction dates
    pred = predictions.copy()
    pred["Date"] = pd.to_datetime(pred["Date"]).dt.strftime("%Y-%m-%d")

    # Slice truth to fold window
    t = truth[(truth["Date"] >= fold.test_start) & (truth["Date"] <= fold.test_end)].copy()

    # Join
    key = ["Site", "Date", "Block"]
    joined = pred.merge(t, on=key, how="left", suffixes=("_pred", "_true"))

    missing_truth = joined["ED Enc_true"].isna().sum()
    if missing_truth > 0:
        logger.warning("Fold %d: %d rows missing ground truth!", fold.period_id, missing_truth)
        joined = joined.dropna(subset=["ED Enc_true"])

    yt_total = joined["ED Enc_true"].values
    yp_total = joined["ED Enc_pred"].values
    yt_adm = joined["ED Enc Admitted_true"].values
    yp_adm = joined["ED Enc Admitted_pred"].values

    # Overall
    result = {
        "fold_id": fold.period_id,
        "window": f"{fold.test_start}..{fold.test_end}",
        "primary_admitted_wape": wape(yt_adm, yp_adm),
        "total_wape": wape(yt_total, yp_total),
        "admitted_wape": wape(yt_adm, yp_adm),
        "total_rmse": rmse(yt_total, yp_total),
        "admitted_rmse": rmse(yt_adm, yp_adm),
        "total_r2": r2(yt_total, yp_total),
        "admitted_r2": r2(yt_adm, yp_adm),
        "total_mae": mae(yt_total, yp_total),
        "admitted_mae": mae(yt_adm, yp_adm),
    }

    # By-site
    by_site = {}
    for site in cfg.SITES:
        sub = joined[joined["Site"] == site]
        if len(sub) == 0:
            continue
        by_site[site] = {
            "total_wape": wape(sub["ED Enc_true"].values, sub["ED Enc_pred"].values),
            "admitted_wape": wape(sub["ED Enc Admitted_true"].values, sub["ED Enc Admitted_pred"].values),
            "total_rmse": rmse(sub["ED Enc_true"].values, sub["ED Enc_pred"].values),
            "admitted_rmse": rmse(sub["ED Enc Admitted_true"].values, sub["ED Enc Admitted_pred"].values),
        }
    result["by_site"] = by_site

    # By-block
    by_block = {}
    for block in cfg.BLOCKS:
        sub = joined[joined["Block"] == block]
        if len(sub) == 0:
            continue
        by_block[block] = {
            "total_wape": wape(sub["ED Enc_true"].values, sub["ED Enc_pred"].values),
            "admitted_wape": wape(sub["ED Enc Admitted_true"].values, sub["ED Enc Admitted_pred"].values),
            "total_rmse": rmse(sub["ED Enc_true"].values, sub["ED Enc_pred"].values),
            "admitted_rmse": rmse(sub["ED Enc Admitted_true"].values, sub["ED Enc Admitted_pred"].values),
        }
    result["by_block"] = by_block

    return result


# ── Full CV Evaluation ───────────────────────────────────────────────────────

def evaluate_cv(
    fold_predictions: Dict[int, pd.DataFrame],
    truth: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Run 4-fold evaluation and produce summary statistics.

    Args:
        fold_predictions: Dict mapping fold_id -> submission DataFrame
        truth:            Block-level ground truth (built from raw data if None)

    Returns:
        Dict with per-fold metrics, mean summary, and diagnostics.
    """
    if truth is None:
        truth = build_ground_truth(cfg.RAW_DATASET)

    fold_results = []
    for fold in cfg.FOLDS:
        if fold.period_id not in fold_predictions:
            logger.warning("Fold %d predictions missing — skipping", fold.period_id)
            continue

        result = score_fold(truth, fold_predictions[fold.period_id], fold)
        fold_results.append(result)

        logger.info(
            "Fold %d: admitted_wape=%.4f, total_wape=%.4f",
            fold.period_id, result["primary_admitted_wape"], result["total_wape"],
        )

    # Mean across folds
    metric_keys = [
        "primary_admitted_wape", "total_wape", "admitted_wape",
        "total_rmse", "admitted_rmse", "total_r2", "admitted_r2",
        "total_mae", "admitted_mae",
    ]
    mean_metrics = {}
    for key in metric_keys:
        values = [r[key] for r in fold_results if not np.isnan(r.get(key, np.nan))]
        mean_metrics[key] = np.mean(values) if values else np.nan

    # Per-fold variance check
    wape_values = [r["primary_admitted_wape"] for r in fold_results]
    wape_std = np.std(wape_values) if len(wape_values) > 1 else 0
    wape_mean = np.mean(wape_values) if wape_values else 0
    cv_folds = wape_std / max(wape_mean, 1e-8)

    summary = {
        "mean_metrics": mean_metrics,
        "per_fold": fold_results,
        "fold_wape_std": float(wape_std),
        "fold_wape_cv": float(cv_folds),
        "pipeline": "B (Direct Multi-Step GBDT)",
    }

    _run_step5_checks(summary)
    return summary


# ── Save Artifacts ───────────────────────────────────────────────────────────

def save_evaluation_results(
    summary: Dict[str, Any],
    fold_predictions: Dict[int, pd.DataFrame],
) -> None:
    """Save all evaluation artifacts to disk."""
    eval_dir = cfg.EVALUATION_DIR
    eval_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = cfg.PREDICTIONS_DIR
    pred_dir.mkdir(parents=True, exist_ok=True)

    # CV results table
    rows = []
    for fr in summary["per_fold"]:
        rows.append({
            "fold_id": fr["fold_id"],
            "window": fr["window"],
            "primary_admitted_wape": fr["primary_admitted_wape"],
            "total_wape": fr["total_wape"],
            "total_rmse": fr["total_rmse"],
            "admitted_rmse": fr["admitted_rmse"],
            "total_r2": fr["total_r2"],
            "admitted_r2": fr["admitted_r2"],
        })
    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(eval_dir / "cv_results.csv", index=False)

    # Summary JSON
    summary_out = {
        "pipeline": summary["pipeline"],
        "mean_admitted_wape": summary["mean_metrics"]["primary_admitted_wape"],
        "mean_total_wape": summary["mean_metrics"]["total_wape"],
        "fold_wape_std": summary["fold_wape_std"],
        "fold_wape_cv": summary["fold_wape_cv"],
        "mean_metrics": {k: float(v) for k, v in summary["mean_metrics"].items()},
    }
    with open(eval_dir / "cv_summary.json", "w") as f:
        json.dump(summary_out, f, indent=2)

    # Per-fold prediction CSVs
    for fold_id, pred_df in fold_predictions.items():
        pred_df.to_csv(pred_dir / f"fold_{fold_id}_predictions.csv", index=False)

    # OOF combined predictions
    all_oof = pd.concat(fold_predictions.values(), ignore_index=True)
    all_oof.to_csv(pred_dir / "oof_predictions.csv", index=False)

    # By-site WAPE breakdown
    site_rows = []
    for fr in summary["per_fold"]:
        for site, metrics in fr.get("by_site", {}).items():
            site_rows.append({
                "fold_id": fr["fold_id"],
                "site": site,
                **metrics,
            })
    if site_rows:
        pd.DataFrame(site_rows).to_csv(eval_dir / "by_site_wape.csv", index=False)

    # By-block WAPE breakdown
    block_rows = []
    for fr in summary["per_fold"]:
        for block, metrics in fr.get("by_block", {}).items():
            block_rows.append({
                "fold_id": fr["fold_id"],
                "block": block,
                **metrics,
            })
    if block_rows:
        pd.DataFrame(block_rows).to_csv(eval_dir / "by_block_wape.csv", index=False)

    logger.info("Evaluation artifacts saved to %s", eval_dir)


# ── Step 5 Eval Checks ──────────────────────────────────────────────────────

def _run_step5_checks(summary: Dict[str, Any]) -> None:
    """Post-evaluation sanity checks from master strategy §2.4."""
    logger.info("=" * 50)
    logger.info("Step 5 — Evaluation Summary")
    logger.info("=" * 50)

    mean_wape = summary["mean_metrics"]["primary_admitted_wape"]
    logger.info("  MEAN ADMITTED WAPE: %.4f", mean_wape)
    logger.info("  Mean total WAPE:    %.4f", summary["mean_metrics"]["total_wape"])

    # Per-fold values
    for fr in summary["per_fold"]:
        logger.info(
            "  Fold %d: admitted_wape=%.4f, total_wape=%.4f",
            fr["fold_id"], fr["primary_admitted_wape"], fr["total_wape"],
        )

    # Fold variance check
    cv = summary["fold_wape_cv"]
    if cv > 0.3:
        logger.warning("  Fold WAPE CV=%.2f > 0.3 — high variance, investigate temporal drift", cv)
    else:
        logger.info("  Fold WAPE CV=%.2f (acceptable ≤ 0.3) ✓", cv)

    # Site WAPE spread check (§2.4: no single site > 2× best)
    for fr in summary["per_fold"]:
        site_wapes = {s: m["admitted_wape"] for s, m in fr.get("by_site", {}).items()}
        if site_wapes:
            best = min(site_wapes.values())
            worst = max(site_wapes.values())
            if best > 0 and worst > 2 * best:
                worst_site = max(site_wapes, key=site_wapes.get)
                logger.warning(
                    "  Fold %d: Site %s WAPE (%.4f) > 2× best (%.4f) — investigate",
                    fr["fold_id"], worst_site, worst, best,
                )

    # Block stability check
    for fr in summary["per_fold"]:
        block_wapes = {b: m["admitted_wape"] for b, m in fr.get("by_block", {}).items()}
        if block_wapes:
            b0_wape = block_wapes.get(0, 0)
            other_mean = np.mean([v for k, v in block_wapes.items() if k != 0])
            if b0_wape > 2 * other_mean and other_mean > 0:
                logger.warning(
                    "  Fold %d: Block 0 (overnight) WAPE %.4f >> other blocks mean %.4f",
                    fr["fold_id"], b0_wape, other_mean,
                )

    logger.info("=" * 50)
