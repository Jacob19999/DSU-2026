"""
Step 4: Optuna hyperparameter search for Models A1 and A2.

Strategy:
  1. Tune A1 first (total_enc, Tweedie/Poisson + COVID policy)
     using default A2 — minimize mean admitted WAPE across 4 folds.
  2. Tune A2 second (admit_rate, regression) with A1 params frozen.
  3. Save best params to MODEL_DIR as JSON.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import optuna

import config as cfg
from step_03_train import train_fold


def _mean_admitted_wape(
    df: pd.DataFrame,
    params_a1: dict,
    params_a2: dict,
    covid_policy: str = "downweight",
) -> float:
    """Train all folds silently, return mean admitted WAPE."""
    wapes = []
    for fold in cfg.FOLDS:
        result = train_fold(
            df, fold, params_a1, params_a2,
            save=False, covid_policy=covid_policy,
        )
        wapes.append(result["admitted_wape"])
    return float(np.mean(wapes))


def run_tuning(df: pd.DataFrame) -> tuple[dict, dict]:
    """Run Optuna for A1 then A2. Returns (best_params_a1, best_params_a2)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── Phase 1: Tune A1 ────────────────────────────────────────────────
    print(f"  Phase 1 -- Tuning Model A1 ({cfg.OPTUNA_N_TRIALS_A1} trials) ...")

    def objective_a1(trial: optuna.Trial) -> float:
        obj = trial.suggest_categorical("objective", ["tweedie", "poisson"])
        params = {
            "objective": obj,
            "n_estimators": trial.suggest_int("n_estimators", 800, 3000),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "verbosity": -1,
        }
        if obj == "tweedie":
            params["tweedie_variance_power"] = trial.suggest_float(
                "tweedie_variance_power", 1.1, 1.9
            )
        covid_policy = trial.suggest_categorical("covid_policy", ["downweight", "exclude"])
        return _mean_admitted_wape(df, params, cfg.LGBM_DEFAULT_A2.copy(), covid_policy)

    study_a1 = optuna.create_study(direction="minimize", study_name="pipeline_a_a1")
    study_a1.optimize(objective_a1, n_trials=cfg.OPTUNA_N_TRIALS_A1, show_progress_bar=True)

    # Extract best A1 params
    best_a1 = {k: v for k, v in study_a1.best_params.items()
                if k not in ("covid_policy",)}
    best_a1["verbosity"] = -1
    best_covid = study_a1.best_params["covid_policy"]

    print(f"  Best A1 mean admitted WAPE: {study_a1.best_value:.4f}")
    print(f"  Best COVID policy: {best_covid}")
    print(f"  Best A1 params: {json.dumps(best_a1, indent=2)}")

    # ── Phase 2: Tune A2 ────────────────────────────────────────────────
    print(f"\n  Phase 2 -- Tuning Model A2 ({cfg.OPTUNA_N_TRIALS_A2} trials) ...")

    def objective_a2(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "n_estimators": trial.suggest_int("n_estimators", 500, 2000),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "subsample": trial.suggest_float("subsample", 0.7, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "verbosity": -1,
        }
        return _mean_admitted_wape(df, best_a1, params, best_covid)

    study_a2 = optuna.create_study(direction="minimize", study_name="pipeline_a_a2")
    study_a2.optimize(objective_a2, n_trials=cfg.OPTUNA_N_TRIALS_A2, show_progress_bar=True)

    best_a2 = study_a2.best_params.copy()
    best_a2["verbosity"] = -1

    print(f"  Best A2 mean admitted WAPE: {study_a2.best_value:.4f}")
    print(f"  Best A2 params: {json.dumps(best_a2, indent=2)}")

    # ── Persist ──────────────────────────────────────────────────────────
    cfg.ensure_dirs()
    with open(cfg.MODEL_DIR / "best_params_a1.json", "w") as f:
        json.dump(best_a1, f, indent=2)
    with open(cfg.MODEL_DIR / "best_params_a2.json", "w") as f:
        json.dump(best_a2, f, indent=2)
    with open(cfg.MODEL_DIR / "best_covid_policy.txt", "w") as f:
        f.write(best_covid)

    return best_a1, best_a2


if __name__ == "__main__":
    from step_01_data_loading import load_data
    from step_02_feature_eng import engineer_features

    df = load_data()
    df = engineer_features(df)
    run_tuning(df)
