"""
Pipeline A: Global GBDT — End-to-end orchestrator.

Usage
-----
# Full run (default params → train → evaluate)
python "Pipelines/Pipeline A/run_pipeline.py"

# Skip Optuna tuning (fast iteration)
python "Pipelines/Pipeline A/run_pipeline.py" --skip-tune

# Tuning only
python "Pipelines/Pipeline A/run_pipeline.py" --tune-only

# Generate final Sept-Oct 2025 forecast
python "Pipelines/Pipeline A/run_pipeline.py" --final-forecast
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
    parser = argparse.ArgumentParser(description="Pipeline A: Global GBDT")
    parser.add_argument("--skip-tune", action="store_true",
                        help="Skip Optuna hyperparameter tuning")
    parser.add_argument("--tune-only", action="store_true",
                        help="Run tuning only (no default-param training)")
    parser.add_argument("--final-forecast", action="store_true",
                        help="Generate Sept-Oct 2025 submission after training")
    args = parser.parse_args()

    cfg.ensure_dirs()
    t0 = time.time()

    print("=" * 62)
    print(" PIPELINE A: GLOBAL GBDT -- Starting")
    print("=" * 62)

    # ── Step 1: Load data ────────────────────────────────────────────────
    print("\n[Step 1/6] Loading master data ...")
    from step_01_data_loading import load_data   # noqa: E402
    df = load_data()

    # ── Step 2: Feature engineering ──────────────────────────────────────
    print("\n[Step 2/6] Engineering features ...")
    from step_02_feature_eng import engineer_features  # noqa: E402
    df = engineer_features(df)

    best_params_a1: dict | None = None
    best_params_a2: dict | None = None
    best_covid: str = "downweight"

    # ── Step 3: Train with default params ────────────────────────────────
    if not args.tune_only:
        print("\n[Step 3/6] Training models (default hyperparameters) ...")
        from step_03_train import train_all_folds  # noqa: E402
        train_all_folds(df)

    # ── Step 4: Hyperparameter tuning ────────────────────────────────────
    if not args.skip_tune:
        print("\n[Step 4/6] Hyperparameter tuning (Optuna) ...")
        from step_04_tune import run_tuning  # noqa: E402
        best_params_a1, best_params_a2 = run_tuning(df)

        # Load best COVID policy
        pol_path = cfg.MODEL_DIR / "best_covid_policy.txt"
        if pol_path.exists():
            best_covid = pol_path.read_text().strip()

        # Re-train all folds with tuned params
        print("\n[Step 3b] Re-training with tuned hyperparameters ...")
        from step_03_train import train_all_folds as taf  # noqa: E402
        taf(df, best_params_a1, best_params_a2, covid_policy=best_covid)
    else:
        # Try loading previously saved tuned params
        p1 = cfg.MODEL_DIR / "best_params_a1.json"
        p2 = cfg.MODEL_DIR / "best_params_a2.json"
        if p1.exists() and p2.exists():
            with open(p1) as f:
                best_params_a1 = json.load(f)
            with open(p2) as f:
                best_params_a2 = json.load(f)
            pol_path = cfg.MODEL_DIR / "best_covid_policy.txt"
            if pol_path.exists():
                best_covid = pol_path.read_text().strip()
            print(f"  Loaded tuned params from disk (covid_policy={best_covid})")

    # ── Step 5: Predictions already saved by train_all_folds ─────────────
    print("\n[Step 5/6] Fold predictions saved -> output/")

    # ── Final forecast (optional) ────────────────────────────────────────
    if args.final_forecast:
        print("\n[FINAL] Generating Sept-Oct 2025 forecast ...")
        from step_05_predict import generate_final_forecast  # noqa: E402
        generate_final_forecast(df, best_params_a1, best_params_a2,
                                covid_policy=best_covid)

    # ── Step 6: Evaluation ───────────────────────────────────────────────
    print("\n[Step 6/6] Evaluating against eval.md contract ...")
    from step_06_evaluate import evaluate  # noqa: E402
    evaluate()

    elapsed = time.time() - t0
    print(f"\n{'=' * 62}")
    print(f" PIPELINE A: COMPLETE ({elapsed / 60:.1f} min)")
    print(f"{'=' * 62}")


if __name__ == "__main__":
    main()
