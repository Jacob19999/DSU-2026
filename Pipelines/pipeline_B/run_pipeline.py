"""
Pipeline B — Step 6: Full Orchestrator.

Entry point that runs the entire Pipeline B end-to-end:
  Data Loading → Feature Engineering → Training → Prediction → Evaluation

Usage:
    # Full 4-fold cross-validation
    python -m Pipelines.pipeline_B.run_pipeline --mode cv

    # Single fold (for debugging)
    python -m Pipelines.pipeline_B.run_pipeline --mode fold --fold-id 1

    # Final submission (Sept-Oct 2025)
    python -m Pipelines.pipeline_B.run_pipeline --mode submit

    # Quick smoke test (1 fold, 5 Optuna trials, no tuning)
    python -m Pipelines.pipeline_B.run_pipeline --mode smoke
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Pipelines.pipeline_B import config as cfg
from Pipelines.pipeline_B.data_loader import load_and_preprocess, get_fold_data
from Pipelines.pipeline_B.features import build_features_for_bucket, build_forecast_features
from Pipelines.pipeline_B.training import train_bucket_models, save_bucket_artifacts
from Pipelines.pipeline_B.predict import predict_fold
from Pipelines.pipeline_B.evaluate import (
    evaluate_cv, save_evaluation_results, build_ground_truth,
)


# ── Logging Setup ────────────────────────────────────────────────────────────

def setup_logging(log_dir: Path) -> None:
    """Configure logging to both console and file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"pipeline_b_run_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    logging.getLogger(__name__).info("Logging to %s", log_file)


logger = logging.getLogger(__name__)


# ── Single Fold Runner ───────────────────────────────────────────────────────

def run_single_fold(
    df: pd.DataFrame,
    fold: cfg.Fold,
    do_tune: bool = True,
    n_trials: int = cfg.OPTUNA_N_TRIALS,
) -> Dict[str, Any]:
    """Run Pipeline B for a single fold: features → train → predict.

    Returns:
        Dict with keys: predictions (DataFrame), bucket_models, fold_metrics
    """
    fold_start = time.time()
    logger.info("=" * 60)
    logger.info("FOLD %d: train ≤ %s → test %s..%s",
                fold.period_id, fold.train_end, fold.test_start, fold.test_end)
    logger.info("=" * 60)

    # ── Step 1: Data slicing ──
    train_df, full_df = get_fold_data(df, fold.train_end)

    # ── Step 2+3: Feature Engineering + Training per bucket ──
    bucket_models: Dict[int, Dict[str, Any]] = {}

    for bucket in cfg.HORIZON_BUCKETS:
        logger.info("─── Bucket %d (days %d-%d) ───",
                     bucket.bucket_id, bucket.horizon_min, bucket.horizon_max)

        t0 = time.time()

        # Build training examples for total_enc
        examples_total = build_features_for_bucket(
            train_df, bucket, target_col=cfg.TARGET_TOTAL, subsample=True,
        )

        # Build training examples for admit_rate
        examples_rate = build_features_for_bucket(
            train_df, bucket, target_col=cfg.TARGET_RATE, subsample=True,
        )

        if examples_total.empty or examples_rate.empty:
            logger.warning("Bucket %d: insufficient training examples — skipping", bucket.bucket_id)
            continue

        feat_time = time.time() - t0
        logger.info("  Feature engineering: %.1fs", feat_time)

        # Train models
        t0 = time.time()
        results = train_bucket_models(
            bucket, examples_total, examples_rate,
            do_tune=do_tune, n_trials=n_trials,
        )
        train_time = time.time() - t0
        logger.info("  Training: %.1fs", train_time)

        # Save model artifacts
        save_bucket_artifacts(results, bucket, fold.period_id)

        bucket_models[bucket.bucket_id] = results

    # ── Step 4: Prediction ──
    t0 = time.time()
    predictions = predict_fold(
        full_df, fold.train_end, fold.test_start, fold.test_end, bucket_models,
    )
    pred_time = time.time() - t0
    logger.info("Prediction: %.1fs → %d rows", pred_time, len(predictions))

    fold_elapsed = time.time() - fold_start
    logger.info("Fold %d complete in %.1fs (%.1f min)",
                fold.period_id, fold_elapsed, fold_elapsed / 60)

    return {
        "predictions": predictions,
        "bucket_models": bucket_models,
    }


# ── Mode: Full Cross-Validation ─────────────────────────────────────────────

def run_cv(
    do_tune: bool = True,
    n_trials: int = cfg.OPTUNA_N_TRIALS,
) -> None:
    """Run full 4-fold cross-validation."""
    total_start = time.time()
    logger.info("Pipeline B — Full CV Mode (%d folds)", len(cfg.FOLDS))

    # Step 1: Load and preprocess
    df = load_and_preprocess()

    # Step 2-4: Run each fold
    fold_predictions: Dict[int, pd.DataFrame] = {}

    for fold in cfg.FOLDS:
        result = run_single_fold(df, fold, do_tune=do_tune, n_trials=n_trials)
        fold_predictions[fold.period_id] = result["predictions"]

    # Step 5: Evaluate
    logger.info("Running evaluation across all folds...")
    summary = evaluate_cv(fold_predictions)

    # Save everything
    save_evaluation_results(summary, fold_predictions)

    total_elapsed = time.time() - total_start
    logger.info("Pipeline B CV complete in %.1fs (%.1f min)",
                total_elapsed, total_elapsed / 60)
    logger.info("FINAL MEAN ADMITTED WAPE: %.4f",
                summary["mean_metrics"]["primary_admitted_wape"])


# ── Mode: Single Fold ────────────────────────────────────────────────────────

def run_fold(
    fold_id: int,
    do_tune: bool = True,
    n_trials: int = cfg.OPTUNA_N_TRIALS,
) -> None:
    """Run a single fold (for debugging)."""
    fold = next((f for f in cfg.FOLDS if f.period_id == fold_id), None)
    if fold is None:
        raise ValueError(f"Fold {fold_id} not found. Valid: {[f.period_id for f in cfg.FOLDS]}")

    df = load_and_preprocess()
    result = run_single_fold(df, fold, do_tune=do_tune, n_trials=n_trials)

    # Score this single fold
    truth = build_ground_truth(cfg.RAW_DATASET)
    from .evaluate import score_fold
    metrics = score_fold(truth, result["predictions"], fold)

    logger.info("Single fold %d result:", fold_id)
    logger.info("  Admitted WAPE: %.4f", metrics["primary_admitted_wape"])
    logger.info("  Total WAPE:    %.4f", metrics["total_wape"])

    # Save predictions
    cfg.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    result["predictions"].to_csv(
        cfg.PREDICTIONS_DIR / f"fold_{fold_id}_predictions.csv", index=False,
    )


# ── Mode: Final Submission ───────────────────────────────────────────────────

def run_submit(
    do_tune: bool = True,
    n_trials: int = cfg.OPTUNA_N_TRIALS,
) -> None:
    """Train on all data through Aug 2025, predict Sept-Oct 2025."""
    logger.info("Pipeline B — Final Submission Mode")
    logger.info("  Train ≤ %s → Predict %s..%s",
                cfg.FINAL_TRAIN_END, cfg.FINAL_TEST_START, cfg.FINAL_TEST_END)

    df = load_and_preprocess()

    # Create a pseudo-fold for final training
    final_fold = cfg.Fold(
        period_id=0,
        train_end=cfg.FINAL_TRAIN_END,
        test_start=cfg.FINAL_TEST_START,
        test_end=cfg.FINAL_TEST_END,
    )

    result = run_single_fold(df, final_fold, do_tune=do_tune, n_trials=n_trials)

    # Save final submission
    cfg.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = cfg.PREDICTIONS_DIR / "final_submission_sept_oct_2025.csv"
    result["predictions"].to_csv(output_path, index=False)

    logger.info("Final submission saved: %s (%d rows)", output_path, len(result["predictions"]))

    # Verify 976 rows
    expected = len(cfg.SITES) * 61 * len(cfg.BLOCKS)  # 4 sites × 61 days × 4 blocks
    actual = len(result["predictions"])
    if actual != expected:
        logger.warning("Row count %d ≠ expected %d!", actual, expected)
    else:
        logger.info("Row count verified: %d ✓", actual)


# ── Mode: Smoke Test ─────────────────────────────────────────────────────────

def run_smoke() -> None:
    """Quick end-to-end test: 1 fold, no tuning, minimal iterations."""
    logger.info("Pipeline B — Smoke Test Mode")
    run_fold(fold_id=1, do_tune=False, n_trials=5)
    logger.info("Smoke test passed!")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline B: Direct Multi-Step GBDT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["cv", "fold", "submit", "smoke"],
        default="cv",
        help="Execution mode (default: cv)",
    )
    parser.add_argument(
        "--fold-id",
        type=int,
        default=1,
        help="Fold ID for --mode fold (1-4)",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Skip Optuna tuning, use default hyperparameters",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=cfg.OPTUNA_N_TRIALS,
        help=f"Number of Optuna trials (default: {cfg.OPTUNA_N_TRIALS})",
    )

    args = parser.parse_args()

    setup_logging(cfg.LOGS_DIR)

    do_tune = not args.no_tune
    n_trials = args.n_trials

    if args.mode == "cv":
        run_cv(do_tune=do_tune, n_trials=n_trials)
    elif args.mode == "fold":
        run_fold(args.fold_id, do_tune=do_tune, n_trials=n_trials)
    elif args.mode == "submit":
        run_submit(do_tune=do_tune, n_trials=n_trials)
    elif args.mode == "smoke":
        run_smoke()


if __name__ == "__main__":
    main()
