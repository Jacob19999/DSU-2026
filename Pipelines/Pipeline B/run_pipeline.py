"""
Pipeline B: Direct Multi-Step GBDT — End-to-end orchestrator.

Usage
-----
# Full cross-validation (default)
python "Pipelines/Pipeline B/run_pipeline.py" --mode cv

# Skip Optuna tuning (fast iteration with default params)
python "Pipelines/Pipeline B/run_pipeline.py" --mode cv --skip-tune

# Tuning only (no default-param training)
python "Pipelines/Pipeline B/run_pipeline.py" --mode cv --tune-only

# Generate final Sept-Oct 2025 forecast
python "Pipelines/Pipeline B/run_pipeline.py" --mode submit

# Single fold for debugging
python "Pipelines/Pipeline B/run_pipeline.py" --mode fold --fold-id 1

# Reduced Optuna trials (smoke test)
python "Pipelines/Pipeline B/run_pipeline.py" --mode cv --n-trials 5
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
    parser = argparse.ArgumentParser(description="Pipeline B: Direct Multi-Step GBDT")
    parser.add_argument("--mode", choices=["cv", "submit", "fold"], default="cv",
                        help="cv=full 4-fold CV, submit=final forecast, fold=single fold")
    parser.add_argument("--fold-id", type=int, default=1,
                        help="Which fold to run (only used with --mode fold)")
    parser.add_argument("--skip-tune", action="store_true",
                        help="Skip Optuna hyperparameter tuning")
    parser.add_argument("--tune-only", action="store_true",
                        help="Run tuning only (no default-param training)")
    parser.add_argument("--final-forecast", action="store_true",
                        help="Also generate Sept-Oct 2025 submission after CV")
    parser.add_argument("--n-trials", type=int, default=None,
                        help="Override Optuna trial count (for both total and rate)")
    args = parser.parse_args()

    cfg.ensure_dirs()
    t0 = time.time()

    print("=" * 68)
    print(" PIPELINE B: DIRECT MULTI-STEP GBDT — Starting")
    print("=" * 68)
    cfg.print_config_summary()

    # ── Step 1: Load data ────────────────────────────────────────────────
    print("\n[Step 1] Loading master data ...")
    from data_loader import load_data  # noqa: E402
    base_df = load_data()

    # ── Step 2: Static features ──────────────────────────────────────────
    print("\n[Step 2] Engineering static features ...")
    from features import add_static_features, build_bucket_data  # noqa: E402
    base_df = add_static_features(base_df)

    # ── Step 3: Build expanded bucket data (one-time cost) ───────────────
    print("\n[Step 3] Building horizon-expanded data per bucket ...")
    bucket_data_map: dict[int, object] = {}
    for bid in [1, 2, 3]:
        t_bucket = time.time()
        bucket_data_map[bid] = build_bucket_data(base_df, bid)
        elapsed_b = time.time() - t_bucket
        n_rows = len(bucket_data_map[bid])
        print(f"  Bucket {bid}: {n_rows:,} rows ({elapsed_b:.1f}s)")

    all_params: dict | None = None

    # ── Step 4: Optuna tuning (optional) ─────────────────────────────────
    if not args.skip_tune:
        print("\n[Step 4] Hyperparameter tuning (Optuna) ...")
        from training import run_optuna_tuning, load_tuned_params  # noqa: E402

        nt = args.n_trials
        all_params = run_optuna_tuning(
            base_df, bucket_data_map,
            n_trials_total=nt, n_trials_rate=nt,
        )
    else:
        # Try loading previously saved params
        from training import load_tuned_params  # noqa: E402
        all_params = load_tuned_params()
        if all_params:
            print("\n[Step 4] Loaded tuned params from disk (skipping Optuna)")
        else:
            print("\n[Step 4] No tuned params found — using defaults")

    if args.tune_only:
        elapsed = time.time() - t0
        print(f"\n{'=' * 68}")
        print(f" PIPELINE B: TUNING COMPLETE ({elapsed / 60:.1f} min)")
        print(f"{'=' * 68}")
        return

    # ── Step 5: Train + predict per fold ─────────────────────────────────
    from training import train_fold, train_all_folds  # noqa: E402

    if args.mode == "fold":
        fold = next(f for f in cfg.FOLDS if f["id"] == args.fold_id)
        print(f"\n[Step 5] Training single fold {args.fold_id} ...")
        train_fold(base_df, bucket_data_map, fold, all_params, save=True)
    else:
        print("\n[Step 5] Training all 4 folds ...")
        train_all_folds(base_df, bucket_data_map, all_params, save=True)

    # ── Step 6: Evaluation ───────────────────────────────────────────────
    if args.mode in ("cv", "fold"):
        print("\n[Step 6] Evaluating against eval.md contract ...")
        from evaluate import evaluate  # noqa: E402
        evaluate()

    # ── Optional: Final forecast ─────────────────────────────────────────
    if args.mode == "submit" or args.final_forecast:
        print("\n[FINAL] Generating Sept-Oct 2025 forecast ...")
        from predict import generate_final_forecast  # noqa: E402
        generate_final_forecast(base_df, bucket_data_map, all_params)

    elapsed = time.time() - t0
    print(f"\n{'=' * 68}")
    print(f" PIPELINE B: COMPLETE ({elapsed / 60:.1f} min)")
    print(f"{'=' * 68}")


if __name__ == "__main__":
    main()
