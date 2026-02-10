"""
Pipeline D: GLM/GAM with Fourier Seasonality — End-to-end orchestrator.

Usage
-----
# Full cross-validation (default, with tuning)
python "Pipelines/Pipeline D/run_pipeline.py" --mode cv

# Skip tuning (fast iteration with default params)
python "Pipelines/Pipeline D/run_pipeline.py" --mode cv --skip-tune

# Tuning only (no fold training)
python "Pipelines/Pipeline D/run_pipeline.py" --mode tune

# Generate final Sept-Oct 2025 forecast
python "Pipelines/Pipeline D/run_pipeline.py" --mode submit

# Single fold for debugging
python "Pipelines/Pipeline D/run_pipeline.py" --mode fold --fold-id 1

# Submit after CV (both)
python "Pipelines/Pipeline D/run_pipeline.py" --mode cv --skip-tune --final-forecast
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure sibling modules are importable regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import config as cfg  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline D: GLM/GAM with Fourier")
    parser.add_argument(
        "--mode", choices=["cv", "submit", "fold", "tune"],
        default="cv",
        help="cv=full 4-fold CV, submit=final forecast, fold=single fold, tune=tune only",
    )
    parser.add_argument("--fold-id", type=int, default=1,
                        help="Which fold to run (only used with --mode fold)")
    parser.add_argument("--skip-tune", action="store_true",
                        help="Skip hyperparameter tuning")
    parser.add_argument("--final-forecast", action="store_true",
                        help="Also generate Sept-Oct 2025 submission after CV")
    args = parser.parse_args()

    cfg.ensure_dirs()
    t0 = time.time()

    print("=" * 68)
    print(" PIPELINE D: GLM/GAM WITH FOURIER - Starting")
    print("=" * 68)
    cfg.print_config_summary()

    # ── Step 1: Load data ────────────────────────────────────────────────
    print("\n[Step 1] Loading master data ...")
    from data_loader import load_data  # noqa: E402
    master_df = load_data()

    # ── Step 2: Feature check ────────────────────────────────────────────
    print("\n[Step 2] Design matrix info ...")
    from features import build_design_matrix, get_feature_names  # noqa: E402
    print(f"  Feature groups: {get_feature_names()}")

    # Quick sanity on one (site, block)
    from data_loader import get_site_block_subset  # noqa: E402
    sample = get_site_block_subset(master_df, "A", 0)
    X_sample = build_design_matrix(sample)
    print(f"  Sample design matrix: {X_sample.shape[0]} rows x {X_sample.shape[1]} features")
    print(f"  NaN cells: {X_sample.isna().sum().sum()}")

    # ── Step 3: Hyperparameter tuning (optional) ─────────────────────────
    from tuning import tune_pipeline_d, load_best_config, get_default_config  # noqa: E402

    best_config: dict | None = None

    if args.mode == "tune" or (args.mode in ["cv", "submit"] and not args.skip_tune):
        print("\n[Step 3] Hyperparameter tuning ...")
        best_config = tune_pipeline_d(master_df)
        if args.mode == "tune":
            elapsed = time.time() - t0
            print(f"\n{'=' * 68}")
            print(f" PIPELINE D: TUNING COMPLETE ({elapsed / 60:.1f} min)")
            print(f"{'=' * 68}")
            return
    else:
        # Try loading previously saved tuned config
        best_config = load_best_config()
        if best_config:
            print(f"\n[Step 3] Loaded tuned config from disk (skipping tuning)")
        else:
            best_config = get_default_config()
            print(f"\n[Step 3] Using default config (no tuning)")

    fourier_config = best_config.get("fourier_config", cfg.FOURIER_TERMS)
    alpha = best_config.get("alpha", cfg.GLM_ALPHA)

    # ── Step 4: Train + predict per fold ─────────────────────────────────
    from predict import train_and_predict_fold, generate_final_forecast  # noqa: E402

    if args.mode == "fold":
        fold = next(f for f in cfg.FOLDS if f["id"] == args.fold_id)
        print(f"\n[Step 4] Training single fold {args.fold_id} ...")
        result = train_and_predict_fold(
            master_df, fold, fourier_config, alpha, save=True
        )

    elif args.mode in ["cv", "submit"]:
        if args.mode == "cv":
            print(f"\n[Step 4] Training all 4 folds ...")
            all_results = []
            oof_frames = []

            for fold in cfg.FOLDS:
                print(f"\n  --- Fold {fold['id']} ---")
                result = train_and_predict_fold(
                    master_df, fold, fourier_config, alpha, save=True
                )
                all_results.append(result)
                oof_frames.append(result["submission"])

            # Summary
            valid = [r for r in all_results if "admitted_wape" in r]
            if valid:
                mean_t = np.mean([r["total_wape"] for r in valid])
                mean_a = np.mean([r["admitted_wape"] for r in valid])
                print(f"\n  4-fold mean: total_wape={mean_t:.4f}  admitted_wape={mean_a:.4f}")

            # Save OOF predictions (for ensemble stacking)
            if oof_frames:
                oof = pd.concat(oof_frames, ignore_index=True)
                oof.to_csv(cfg.PRED_DIR / "oof_predictions.csv", index=False)
                print(f"  OOF predictions: {len(oof)} rows -> {cfg.PRED_DIR / 'oof_predictions.csv'}")

    # ── Step 5: Evaluation ───────────────────────────────────────────────
    if args.mode in ["cv", "fold"]:
        print(f"\n[Step 5] Evaluating against eval.md contract ...")
        from evaluate import evaluate  # noqa: E402
        evaluate()

    # ── Optional: Final forecast ─────────────────────────────────────────
    if args.mode == "submit" or args.final_forecast:
        print(f"\n[FINAL] Generating Sept-Oct 2025 forecast ...")
        generate_final_forecast(master_df, fourier_config, alpha)

    elapsed = time.time() - t0
    print(f"\n{'=' * 68}")
    print(f" PIPELINE D: COMPLETE ({elapsed / 60:.1f} min)")
    print(f"{'=' * 68}")


if __name__ == "__main__":
    main()
