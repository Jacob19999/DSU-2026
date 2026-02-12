"""
Step 6: Hyperparameter tuning with Optuna.

Tunes daily models and share models, selecting by mean admitted WAPE
across all 4 validation folds (full pipeline: daily → shares → block-level).
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None  # type: ignore[assignment]

import config as cfg
from step_04_train_daily import train_daily_fold, wape
from step_05_train_shares import train_share_fold
from step_07_predict import allocate_daily_to_blocks


# ── Phase 1: Tune Daily Models ──────────────────────────────────────────────

def _objective_daily(
    trial,
    daily_df: pd.DataFrame,
    share_df: pd.DataFrame,
    precomputed_shares: dict[int, pd.DataFrame] | None = None,
) -> float:
    """Optuna objective: tune daily model HP, evaluate at block level."""
    params_total = {
        "objective": trial.suggest_categorical("obj", ["tweedie", "poisson"]),
        "n_estimators": trial.suggest_int("n_est", 800, 3000),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("lr", 0.01, 0.05, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 0.95),
        "colsample_bytree": trial.suggest_float("colsample", 0.6, 0.9),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 20),
        "verbosity": -1,
    }
    if params_total["objective"] == "tweedie":
        params_total["tweedie_variance_power"] = trial.suggest_float("tvp", 1.1, 1.9)

    covid_policy = trial.suggest_categorical("covid_policy", ["downweight", "exclude"])

    wapes = []
    for fold in cfg.FOLDS:
        # Train daily model with trial params
        daily_res = train_daily_fold(
            daily_df, fold, params_total, None,
            save=False, covid_policy=covid_policy,
        )
        daily_preds = daily_res["daily_preds"]

        # Use precomputed share predictions if available
        if precomputed_shares and fold["id"] in precomputed_shares:
            share_preds = precomputed_shares[fold["id"]]
        else:
            share_res = train_share_fold(share_df, fold, save=False)
            share_preds = share_res["share_preds"]

        # Allocate and score at block level
        block_preds = allocate_daily_to_blocks(daily_preds, share_preds)

        # Get actuals
        val_start = pd.Timestamp(fold["val_start"])
        val_end = pd.Timestamp(fold["val_end"])
        truth = pd.read_parquet(cfg.MASTER_PARQUET)
        truth["date"] = pd.to_datetime(truth["date"])
        truth_w = truth[(truth["date"] >= val_start) & (truth["date"] <= val_end)]

        merged = block_preds.merge(
            truth_w[["site", "date", "block", "admitted_enc"]],
            on=["site", "date", "block"], how="inner"
        )
        if len(merged) > 0:
            fold_wape = wape(merged["admitted_enc"].values, merged["pred_admitted"].values)
            wapes.append(fold_wape)

    return float(np.mean(wapes)) if wapes else float("inf")


# ── Phase 2: Tune Share Models ──────────────────────────────────────────────

def _objective_share(
    trial,
    daily_df: pd.DataFrame,
    share_df: pd.DataFrame,
    precomputed_daily: dict[int, pd.DataFrame] | None = None,
) -> float:
    """Optuna objective: tune share model HP, evaluate at block level."""
    share_type = trial.suggest_categorical("share_type", ["softmax_gbdt", "climatology"])

    params_share = None
    if share_type == "softmax_gbdt":
        params_share = {
            "objective": "multiclass",
            "num_class": cfg.N_BLOCKS,
            "n_estimators": trial.suggest_int("s_n_est", 400, 1500),
            "max_depth": trial.suggest_int("s_max_depth", 3, 7),
            "learning_rate": trial.suggest_float("s_lr", 0.01, 0.05, log=True),
            "reg_lambda": trial.suggest_float("s_reg_lambda", 0.5, 10.0),
            "min_child_weight": trial.suggest_int("s_min_child_weight", 1, 15),
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbosity": -1,
        }

    wapes = []
    for fold in cfg.FOLDS:
        # Use precomputed daily predictions if available
        if precomputed_daily and fold["id"] in precomputed_daily:
            daily_preds = precomputed_daily[fold["id"]]
        else:
            daily_res = train_daily_fold(daily_df, fold, save=False)
            daily_preds = daily_res["daily_preds"]

        share_res = train_share_fold(
            share_df, fold, params_share, share_type=share_type, save=False
        )
        share_preds = share_res["share_preds"]

        block_preds = allocate_daily_to_blocks(daily_preds, share_preds)

        val_start = pd.Timestamp(fold["val_start"])
        val_end = pd.Timestamp(fold["val_end"])
        truth = pd.read_parquet(cfg.MASTER_PARQUET)
        truth["date"] = pd.to_datetime(truth["date"])
        truth_w = truth[(truth["date"] >= val_start) & (truth["date"] <= val_end)]

        merged = block_preds.merge(
            truth_w[["site", "date", "block", "admitted_enc"]],
            on=["site", "date", "block"], how="inner"
        )
        if len(merged) > 0:
            fold_wape = wape(merged["admitted_enc"].values, merged["pred_admitted"].values)
            wapes.append(fold_wape)

    return float(np.mean(wapes)) if wapes else float("inf")


# ── Helpers ─────────────────────────────────────────────────────────────────

def _reconstruct_daily_params(best_daily: dict) -> dict:
    """Convert Optuna trial params to train_daily_fold params_total format."""
    out = {
        "objective": best_daily.get("obj", "tweedie"),
        "n_estimators": best_daily.get("n_est", 1500),
        "max_depth": best_daily.get("max_depth", 6),
        "learning_rate": best_daily.get("lr", 0.03),
        "subsample": best_daily.get("subsample", 0.8),
        "colsample_bytree": best_daily.get("colsample", 0.8),
        "reg_lambda": best_daily.get("reg_lambda", 5.0),
        "min_child_weight": best_daily.get("min_child_weight", 10),
        "verbosity": -1,
    }
    if out["objective"] == "tweedie":
        out["tweedie_variance_power"] = best_daily.get("tvp", 1.5)
    return out


# ── Main entry point ─────────────────────────────────────────────────────────

def run_tuning(
    daily_df: pd.DataFrame,
    share_df: pd.DataFrame,
) -> dict:
    """Run Optuna tuning for daily + share models. Returns best params dict."""
    if optuna is None:
        print("  WARNING: optuna not installed — skipping tuning")
        return {"daily_total": None, "daily_rate": None, "share": None,
                "covid_policy": "downweight", "share_type": cfg.SHARE_MODEL_TYPE}

    cfg.ensure_dirs()

    # ── Phase 1: Tune daily models ───────────────────────────────────────
    print(f"  Phase 1: Tuning daily models ({cfg.OPTUNA_N_TRIALS_DAILY} trials) ...")
    study_daily = optuna.create_study(direction="minimize",
                                      study_name="pipeline_c_daily")
    study_daily.optimize(
        lambda trial: _objective_daily(trial, daily_df, share_df),
        n_trials=cfg.OPTUNA_N_TRIALS_DAILY,
        show_progress_bar=True,
    )

    best_daily = study_daily.best_params
    print(f"  Phase 1 best WAPE: {study_daily.best_value:.4f}")
    print(f"  Best daily params: {best_daily}")

    # Build Phase 1 best params for Phase 2 (share tuning needs daily preds from best daily params)
    best_params_total = _reconstruct_daily_params(best_daily)
    best_covid = best_daily.get("covid_policy", "downweight")
    precomputed_daily = {}
    for fold in cfg.FOLDS:
        daily_res = train_daily_fold(
            daily_df, fold, best_params_total, None,
            save=False, covid_policy=best_covid,
        )
        precomputed_daily[fold["id"]] = daily_res["daily_preds"]

    # ── Phase 2: Tune share models ───────────────────────────────────────
    print(f"\n  Phase 2: Tuning share models ({cfg.OPTUNA_N_TRIALS_SHARE} trials) ...")
    study_share = optuna.create_study(direction="minimize",
                                      study_name="pipeline_c_share")
    study_share.optimize(
        lambda trial: _objective_share(trial, daily_df, share_df, precomputed_daily),
        n_trials=cfg.OPTUNA_N_TRIALS_SHARE,
        show_progress_bar=True,
    )

    best_share = study_share.best_params
    print(f"  Phase 2 best WAPE: {study_share.best_value:.4f}")
    print(f"  Best share params: {best_share}")

    # ── Reconstruct best param dicts ─────────────────────────────────────
    best_params_total = _reconstruct_daily_params(best_daily)
    best_covid = best_daily.get("covid_policy", "downweight")
    best_share_type = best_share.get("share_type", cfg.SHARE_MODEL_TYPE)

    best_params_share = None
    if best_share_type == "softmax_gbdt":
        best_params_share = {
            "objective": "multiclass",
            "num_class": cfg.N_BLOCKS,
            "n_estimators": best_share.get("s_n_est", 800),
            "max_depth": best_share.get("s_max_depth", 5),
            "learning_rate": best_share.get("s_lr", 0.03),
            "reg_lambda": best_share.get("s_reg_lambda", 3.0),
            "min_child_weight": best_share.get("s_min_child_weight", 5),
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbosity": -1,
        }

    # ── Save artifacts ───────────────────────────────────────────────────
    with open(cfg.MODEL_DIR / "best_params_daily_total.json", "w") as f:
        json.dump(best_params_total, f, indent=2)
    if best_params_share:
        with open(cfg.MODEL_DIR / "best_params_share.json", "w") as f:
            json.dump(best_params_share, f, indent=2)
    (cfg.MODEL_DIR / "best_covid_policy.txt").write_text(best_covid)
    (cfg.MODEL_DIR / "best_share_type.txt").write_text(best_share_type)

    return {
        "daily_total": best_params_total,
        "daily_rate": None,   # Uses default LGBM_DAILY_RATE (rate model is less sensitive)
        "share": best_params_share,
        "covid_policy": best_covid,
        "share_type": best_share_type,
    }


if __name__ == "__main__":
    from step_01_data_loading import load_data
    from step_02_feature_eng_daily import engineer_daily_features
    from step_03_feature_eng_shares import engineer_share_features

    block_df, daily_df, _ = load_data()
    daily_df = engineer_daily_features(daily_df)
    share_df = engineer_share_features(block_df)
    run_tuning(daily_df, share_df)
