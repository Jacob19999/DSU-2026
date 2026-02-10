"""
Pipeline E: Reason-Mix Latent Factor Model — End-to-end orchestrator.

Usage
-----
# Full cross-validation (default)
python "Pipelines/Pipeline E/run_pipeline.py" --mode cv

# Skip Optuna tuning (fast iteration with default params)
python "Pipelines/Pipeline E/run_pipeline.py" --mode cv --skip-tune

# Tuning only (no default-param training)
python "Pipelines/Pipeline E/run_pipeline.py" --mode cv --tune-only

# Generate final Sept-Oct 2025 forecast
python "Pipelines/Pipeline E/run_pipeline.py" --mode submit

# Single fold for debugging
python "Pipelines/Pipeline E/run_pipeline.py" --mode fold --fold-id 1

# Reduced Optuna trials (smoke test)
python "Pipelines/Pipeline E/run_pipeline.py" --mode cv --n-trials 5

# Ablation: run without factor features
python "Pipelines/Pipeline E/run_pipeline.py" --mode cv --skip-tune --ablation-no-factors
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure sibling modules are importable regardless of CWD
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as cfg  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline E: Reason-Mix Latent Factor Model",
    )
    parser.add_argument(
        "--mode", choices=["cv", "submit", "fold"], default="cv",
        help="cv=full 4-fold CV, submit=final forecast, fold=single fold",
    )
    parser.add_argument(
        "--fold-id", type=int, default=1,
        help="Which fold to run (only used with --mode fold)",
    )
    parser.add_argument("--skip-tune", action="store_true",
                        help="Skip Optuna hyperparameter tuning")
    parser.add_argument("--tune-only", action="store_true",
                        help="Run tuning only (no default-param training)")
    parser.add_argument("--final-forecast", action="store_true",
                        help="Also generate Sept-Oct 2025 submission after CV")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Override Optuna trial count")
    parser.add_argument("--ablation-no-factors", action="store_true",
                        help="Run without factor features (ablation test)")
    args = parser.parse_args()

    cfg.ensure_dirs()
    t0 = time.time()

    print("=" * 68)
    print(" PIPELINE E: REASON-MIX LATENT FACTOR MODEL — Starting")
    print("=" * 68)
    cfg.print_config_summary()

    # ── Step 1: Load data ────────────────────────────────────────────────
    print("\n[Step 1/9] Loading master dataset ...")
    from data_loader import load_data  # noqa: E402
    base_df = load_data()

    # ── Step 2: Build share matrix ───────────────────────────────────────
    print("\n[Step 2/9] Building reason-category share matrix ...")
    from share_matrix import build_share_matrix  # noqa: E402
    base_df, share_cols = build_share_matrix(base_df)

    if args.ablation_no_factors:
        print("\n  [ABLATION] Clearing share columns — no factor features")
        share_cols = []

    # ── Step 3-5: Static + target lag features (computed once) ───────────
    print("\n[Step 3-5/9] Engineering base features (static + target lags) ...")
    from features import add_all_base_features  # noqa: E402
    base_df = add_all_base_features(base_df)

    # ── Import training modules ──────────────────────────────────────────
    from training import (  # noqa: E402
        train_fold, train_all_folds,
        run_optuna_tuning, load_tuned_params,
    )

    all_params: dict | None = None

    # ── Step 7: Optuna tuning (optional) ─────────────────────────────────
    if not args.skip_tune:
        print("\n[Step 7/9] Hyperparameter tuning (Optuna) ...")
        nt = args.n_trials
        tuned = run_optuna_tuning(
            base_df, share_cols,
            n_trials_total=nt, n_trials_rate=nt,
        )
        all_params = tuned
    else:
        # Try loading previously saved params
        all_params = load_tuned_params()
        if all_params:
            print("\n[Step 7/9] Loaded tuned params from disk (skipping Optuna)")
        else:
            print("\n[Step 7/9] No tuned params found — using defaults")

    if args.tune_only:
        elapsed = time.time() - t0
        print(f"\n{'=' * 68}")
        print(f" PIPELINE E: TUNING COMPLETE ({elapsed / 60:.1f} min)")
        print(f"{'=' * 68}")
        return

    p_total = (all_params or {}).get("total")
    p_rate  = (all_params or {}).get("rate")

    # ── Step 6: Train + predict per fold ─────────────────────────────────
    if args.mode == "fold":
        fold = next(f for f in cfg.FOLDS if f["id"] == args.fold_id)
        print(f"\n[Step 6/9] Training single fold {args.fold_id} ...")
        train_fold(
            base_df, share_cols, fold,
            params_total=p_total, params_rate=p_rate,
            save=True,
        )
    elif args.mode in ("cv", "submit"):
        print("\n[Step 6/9] Training all 4 folds ...")
        train_all_folds(
            base_df, share_cols,
            params_total=p_total, params_rate=p_rate,
            save=True,
        )

    # ── Step 9: Evaluation ───────────────────────────────────────────────
    if args.mode in ("cv", "fold"):
        print("\n[Step 9/9] Evaluating against eval.md contract ...")
        from evaluate import evaluate  # noqa: E402
        evaluate()

    # ── Optional: Final forecast ─────────────────────────────────────────
    if args.mode == "submit" or args.final_forecast:
        print("\n[FINAL] Generating Sept-Oct 2025 forecast ...")
        from predict import generate_final_forecast  # noqa: E402
        generate_final_forecast(
            base_df, share_cols,
            params_total=p_total, params_rate=p_rate,
        )

    elapsed = time.time() - t0
    print(f"\n{'=' * 68}")
    print(f" PIPELINE E: COMPLETE ({elapsed / 60:.1f} min)")
    print(f"{'=' * 68}")


if __name__ == "__main__":
    main()
