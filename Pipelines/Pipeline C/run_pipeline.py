"""
Pipeline C: Hierarchical Reconciliation (Daily → Block) — End-to-end orchestrator.

Usage
-----
# Full run (default params → train → evaluate)
python "Pipelines/Pipeline C/run_pipeline.py"

# Skip Optuna tuning (fast iteration)
python "Pipelines/Pipeline C/run_pipeline.py" --skip-tune

# Tuning only
python "Pipelines/Pipeline C/run_pipeline.py" --tune-only

# Single fold (for debugging)
python "Pipelines/Pipeline C/run_pipeline.py" --fold 1

# Generate final Sept-Oct 2025 forecast
python "Pipelines/Pipeline C/run_pipeline.py" --final-forecast

# Force share model type (override config)
python "Pipelines/Pipeline C/run_pipeline.py" --share-model climatology
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure sibling modules are importable regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline C: Hierarchical Reconciliation")
    parser.add_argument("--skip-tune", action="store_true",
                        help="Skip Optuna hyperparameter tuning")
    parser.add_argument("--tune-only", action="store_true",
                        help="Run tuning only (no default-param training)")
    parser.add_argument("--final-forecast", action="store_true",
                        help="Generate Sept-Oct 2025 submission after training")
    parser.add_argument("--fold", type=int, default=None,
                        help="Run only a single fold (for debugging)")
    parser.add_argument("--share-model", type=str, default=None,
                        choices=["softmax_gbdt", "climatology"],
                        help="Override share model type from config")
    args = parser.parse_args()

    cfg.ensure_dirs()
    t0 = time.time()

    share_type = args.share_model  # None → use config default

    print("=" * 65)
    print(" PIPELINE C: HIERARCHICAL RECONCILIATION — Starting")
    print("=" * 65)

    # ── Step 1: Data Loading & Daily Aggregation ─────────────────────────
    print("\n[Step 1/8] Loading data + building daily aggregates & block shares ...")
    from step_01_data_loading import load_data  # noqa: E402
    block_df, daily_df, share_wide = load_data()

    # ── Step 2: Feature Engineering (Daily Model) ────────────────────────
    print("\n[Step 2/8] Engineering daily-model features ...")
    from step_02_feature_eng_daily import engineer_daily_features  # noqa: E402
    daily_features = engineer_daily_features(daily_df)

    # ── Step 3: Feature Engineering (Share Model) ────────────────────────
    print("\n[Step 3/8] Engineering share-model features ...")
    from step_03_feature_eng_shares import engineer_share_features  # noqa: E402
    share_features = engineer_share_features(block_df)

    best_params_total: dict | None = None
    best_params_rate: dict | None = None
    best_params_share: dict | None = None
    best_covid: str = "downweight"

    # ── Steps 4+5: Train with default params ─────────────────────────────
    if not args.tune_only:
        if args.fold is not None:
            # Single fold mode
            fold = next(f for f in cfg.FOLDS if f["id"] == args.fold)
            print(f"\n[Step 4/8] Training daily model (fold {args.fold}, default HP) ...")
            from step_04_train_daily import train_daily_fold  # noqa: E402
            train_daily_fold(daily_features, fold)

            print(f"\n[Step 5/8] Training share model (fold {args.fold}) ...")
            from step_05_train_shares import train_share_fold  # noqa: E402
            train_share_fold(share_features, fold, share_type=share_type)

            print(f"\n[Step 7/8] Allocating daily → blocks (fold {args.fold}) ...")
            from step_07_predict import predict_fold  # noqa: E402
            predict_fold(daily_features, share_features, fold, share_type=share_type)
        else:
            print("\n[Step 4+5/8] Training daily & share models (default HP, all folds) ...")
            from step_07_predict import predict_all_folds  # noqa: E402
            predict_all_folds(daily_features, share_features, share_type=share_type)

    # ── Step 6: Hyperparameter Tuning ────────────────────────────────────
    if not args.skip_tune:
        print("\n[Step 6/8] Hyperparameter tuning (Optuna) ...")
        from step_06_tune import run_tuning  # noqa: E402
        best = run_tuning(daily_features, share_features)
        best_params_total = best["daily_total"]
        best_params_rate = best["daily_rate"]
        best_params_share = best["share"]
        best_covid = best.get("covid_policy", "downweight")
        share_type = best.get("share_type", share_type)

        # Re-train all folds with tuned params
        print("\n[Step 4b+5b/8] Re-training with tuned HP ...")
        from step_07_predict import predict_all_folds as paf  # noqa: E402
        paf(daily_features, share_features,
            best_params_total, best_params_rate, best_params_share,
            covid_policy=best_covid, share_type=share_type)
    else:
        # Try loading previously saved tuned params
        p_total = cfg.MODEL_DIR / "best_params_daily_total.json"
        if p_total.exists():
            with open(p_total) as f:
                best_params_total = json.load(f)
            print(f"  Loaded tuned daily params from disk")
        p_share = cfg.MODEL_DIR / "best_params_share.json"
        if p_share.exists():
            with open(p_share) as f:
                best_params_share = json.load(f)
            print(f"  Loaded tuned share params from disk")
        pol_path = cfg.MODEL_DIR / "best_covid_policy.txt"
        if pol_path.exists():
            best_covid = pol_path.read_text().strip()
        st_path = cfg.MODEL_DIR / "best_share_type.txt"
        if st_path.exists():
            share_type = st_path.read_text().strip()

    # ── Final forecast (optional) ────────────────────────────────────────
    if args.final_forecast:
        print("\n[FINAL] Generating Sept-Oct 2025 forecast ...")
        from step_07_predict import generate_final_forecast  # noqa: E402
        generate_final_forecast(
            daily_features, share_features,
            best_params_total, best_params_rate, best_params_share,
            covid_policy=best_covid, share_type=share_type,
        )

    # ── Step 8: Evaluation ───────────────────────────────────────────────
    if not args.tune_only:
        print("\n[Step 8/8] Evaluating against eval.md contract ...")
        from step_08_evaluate import evaluate  # noqa: E402
        evaluate()

    elapsed = time.time() - t0
    print(f"\n{'=' * 65}")
    print(f" PIPELINE C: COMPLETE ({elapsed / 60:.1f} min)")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
