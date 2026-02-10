"""
Pipeline B — Step 3: Model Training with Optuna Hyperparameter Tuning.

Trains 6 LightGBM models: 2 targets (total_enc, admit_rate) × 3 horizon buckets.
Uses Optuna for Bayesian hyperparameter optimization with WAPE as objective.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # Deferred — will fail at runtime with clear message

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except ImportError:
    optuna = None

from . import config as cfg
from .features import get_feature_columns

logger = logging.getLogger(__name__)


# ── 3.1  Internal Train/Val Split (for early stopping) ──────────────────────

def split_train_val(
    examples: pd.DataFrame,
    val_days: int = cfg.EARLY_STOP_VAL_DAYS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split supervised examples into train and early-stopping validation.

    Uses the last `val_days` worth of TARGET dates as validation.
    This is NOT the fold validation — just for LightGBM early stopping.
    """
    max_date = examples["__target_date__"].max()
    val_cutoff = max_date - pd.Timedelta(days=val_days)

    train = examples[examples["__target_date__"] <= val_cutoff]
    val = examples[examples["__target_date__"] > val_cutoff]

    logger.info(
        "Internal split: train=%d rows (≤%s), val=%d rows (>%s)",
        len(train), val_cutoff.date(), len(val), val_cutoff.date(),
    )
    return train, val


# ── 3.2  LightGBM Training ──────────────────────────────────────────────────

def train_lgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_type: str,  # "total" or "rate"
    params: Optional[Dict[str, Any]] = None,
    feature_cols: Optional[List[str]] = None,
) -> lgb.LGBMRegressor:
    """Train a single LightGBM model with early stopping.

    Args:
        train_df:     Training examples with features + __target__ + __sample_weight__
        val_df:       Validation examples (for early stopping)
        target_type:  "total" (tweedie) or "rate" (regression)
        params:       LightGBM hyperparams (if None, uses defaults)
        feature_cols: Feature column names (if None, auto-detected)
    """
    if lgb is None:
        raise ImportError("lightgbm is required. Install: pip install lightgbm")

    if feature_cols is None:
        feature_cols = get_feature_columns(train_df)

    X_train = train_df[feature_cols]
    y_train = train_df["__target__"]
    w_train = train_df["__sample_weight__"]

    X_val = val_df[feature_cols]
    y_val = val_df["__target__"]
    w_val = val_df["__sample_weight__"]

    # Build params
    model_params = dict(cfg.LGBM_FIXED_PARAMS)
    if target_type == "total":
        model_params["objective"] = cfg.TOTAL_OBJECTIVE
        if cfg.TOTAL_OBJECTIVE == "tweedie":
            model_params["tweedie_variance_power"] = cfg.TOTAL_TWEEDIE_POWER
        model_params["metric"] = "mae"
    elif target_type == "rate":
        model_params["objective"] = cfg.RATE_OBJECTIVE
        model_params["metric"] = "mae"
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    if params:
        model_params.update(params)

    # Extract fit-time params
    n_estimators = model_params.pop("n_estimators", 1000)
    early_stopping_round = model_params.pop("early_stopping_round", 50)

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        **model_params,
    )

    # Identify categorical columns for LightGBM native handling
    cat_cols = [c for c in cfg.CATEGORICAL_FEATURES if c in feature_cols]

    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_round, verbose=False),
        lgb.log_evaluation(period=0),  # suppress per-iteration logs
    ]

    model.fit(
        X_train, y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        eval_sample_weight=[w_val],
        callbacks=callbacks,
        categorical_feature=cat_cols if cat_cols else "auto",
    )

    # Log training result
    best_iter = model.best_iteration_ if hasattr(model, "best_iteration_") else n_estimators
    logger.info(
        "  Trained %s model: best_iteration=%d/%d",
        target_type, best_iter, n_estimators,
    )

    return model


# ── 3.3  Optuna Hyperparameter Tuning ───────────────────────────────────────

def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute WAPE metric."""
    denom = float(np.sum(np.abs(y_true)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom)


def tune_hyperparams(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    target_type: str,
    n_trials: int = cfg.OPTUNA_N_TRIALS,
    feature_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run Optuna Bayesian optimization to find best LightGBM hyperparams.

    Returns best_params dict ready to pass to train_lgbm().
    """
    if optuna is None:
        raise ImportError("optuna is required. Install: pip install optuna")
    if lgb is None:
        raise ImportError("lightgbm is required. Install: pip install lightgbm")

    if feature_cols is None:
        feature_cols = get_feature_columns(train_df)

    X_train = train_df[feature_cols]
    y_train = train_df["__target__"].values
    w_train = train_df["__sample_weight__"].values
    X_val = val_df[feature_cols]
    y_val = val_df["__target__"].values
    w_val = val_df["__sample_weight__"].values

    cat_cols = [c for c in cfg.CATEGORICAL_FEATURES if c in feature_cols]

    space = cfg.OPTUNA_SEARCH_SPACE

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"]),
            "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
            "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
            "subsample": trial.suggest_float("subsample", *space["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *space["colsample_bytree"]),
            "reg_lambda": trial.suggest_float("reg_lambda", *space["reg_lambda"], log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", *space["min_child_weight"]),
            "num_leaves": trial.suggest_int("num_leaves", *space["num_leaves"]),
        }

        model_params = dict(cfg.LGBM_FIXED_PARAMS)
        if target_type == "total":
            model_params["objective"] = cfg.TOTAL_OBJECTIVE
            if cfg.TOTAL_OBJECTIVE == "tweedie":
                model_params["tweedie_variance_power"] = cfg.TOTAL_TWEEDIE_POWER
            model_params["metric"] = "mae"
        else:
            model_params["objective"] = cfg.RATE_OBJECTIVE
            model_params["metric"] = "mae"

        model_params.update(params)
        n_est = model_params.pop("n_estimators")
        early_stop = model_params.pop("early_stopping_round", 50)

        model = lgb.LGBMRegressor(n_estimators=n_est, **model_params)

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stop, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        model.fit(
            X_train, y_train,
            sample_weight=w_train,
            eval_set=[(X_val, y_val)],
            eval_sample_weight=[w_val],
            callbacks=callbacks,
            categorical_feature=cat_cols if cat_cols else "auto",
        )

        preds = model.predict(X_val)
        if target_type == "rate":
            preds = np.clip(preds, 0.0, 1.0)
        else:
            preds = np.clip(preds, 0.0, None)

        return _wape(y_val, preds)

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=cfg.SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(
        "Optuna done (%s): best WAPE=%.4f after %d trials",
        target_type, study.best_value, len(study.trials),
    )
    logger.info("  Best params: %s", study.best_params)

    # Check if best trial was in the last 10 — may need more trials
    best_trial_num = study.best_trial.number
    if best_trial_num >= n_trials - 10:
        logger.warning(
            "  Best trial (%d) was in the last 10 — consider increasing n_trials",
            best_trial_num,
        )

    return study.best_params


# ── 3.4  Train All Bucket Models ────────────────────────────────────────────

def train_bucket_models(
    bucket: cfg.HorizonBucket,
    train_examples_total: pd.DataFrame,
    train_examples_rate: pd.DataFrame,
    do_tune: bool = True,
    n_trials: int = cfg.OPTUNA_N_TRIALS,
) -> Dict[str, Any]:
    """Train both total and rate models for a single horizon bucket.

    Returns dict with keys:
        model_total, model_rate, params_total, params_rate,
        importance_total, importance_rate, metrics
    """
    results = {}
    feature_cols_total = get_feature_columns(train_examples_total)
    feature_cols_rate = get_feature_columns(train_examples_rate)

    for target_type, examples, feat_cols in [
        ("total", train_examples_total, feature_cols_total),
        ("rate", train_examples_rate, feature_cols_rate),
    ]:
        logger.info("── Bucket %d / %s ──", bucket.bucket_id, target_type)

        train, val = split_train_val(examples)

        if len(train) == 0 or len(val) == 0:
            logger.error("  Empty train or val set — skipping")
            continue

        # Tune or use defaults
        if do_tune:
            best_params = tune_hyperparams(
                train, val, target_type, n_trials=n_trials, feature_cols=feat_cols,
            )
        else:
            best_params = {
                "n_estimators": 1500,
                "max_depth": 6,
                "learning_rate": 0.03,
                "subsample": 0.85,
                "colsample_bytree": 0.8,
                "reg_lambda": 5.0,
                "min_child_weight": 5,
                "num_leaves": 63,
            }

        # Train final model with best params
        model = train_lgbm(train, val, target_type, params=best_params, feature_cols=feat_cols)

        # Feature importance
        importance = pd.DataFrame({
            "feature": feat_cols,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)

        # Validation metrics
        preds = model.predict(val[feat_cols])
        if target_type == "rate":
            preds = np.clip(preds, 0.0, 1.0)
        else:
            preds = np.clip(preds, 0.0, None)
        val_wape = _wape(val["__target__"].values, preds)
        val_mae = float(np.mean(np.abs(val["__target__"].values - preds)))

        results[f"model_{target_type}"] = model
        results[f"params_{target_type}"] = best_params
        results[f"importance_{target_type}"] = importance
        results[f"val_wape_{target_type}"] = val_wape
        results[f"val_mae_{target_type}"] = val_mae

        logger.info("  Val WAPE=%.4f, MAE=%.4f", val_wape, val_mae)

        # Eval check: top features
        top5 = importance.head(5)["feature"].tolist()
        logger.info("  Top 5 features: %s", top5)
        lag_in_top5 = any("lag" in f or "roll" in f for f in top5)
        if not lag_in_top5:
            logger.warning("  No lag/rolling features in top 5 — model may lack temporal signal")

    _run_step3_checks(results, bucket)
    return results


# ── 3.4  Model Serialization ────────────────────────────────────────────────

def save_bucket_artifacts(
    results: Dict[str, Any],
    bucket: cfg.HorizonBucket,
    fold_id: int,
) -> None:
    """Save trained models, params, and feature importance to disk."""
    fold_dir = cfg.MODELS_DIR / f"fold_{fold_id}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    imp_dir = cfg.IMPORTANCE_DIR
    imp_dir.mkdir(parents=True, exist_ok=True)

    for target_type in ("total", "rate"):
        model_key = f"model_{target_type}"
        params_key = f"params_{target_type}"
        imp_key = f"importance_{target_type}"

        if model_key in results:
            model_path = fold_dir / f"model_bucket{bucket.bucket_id}_{target_type}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(results[model_key], f)
            logger.info("  Saved model: %s", model_path)

        if params_key in results:
            params_path = fold_dir / f"best_params_bucket{bucket.bucket_id}_{target_type}.json"
            with open(params_path, "w") as f:
                json.dump(results[params_key], f, indent=2)

        if imp_key in results:
            imp_path = imp_dir / f"importance_bucket{bucket.bucket_id}_{target_type}_fold{fold_id}.csv"
            results[imp_key].to_csv(imp_path, index=False)


def load_bucket_model(
    bucket_id: int,
    target_type: str,
    fold_id: int,
) -> Any:
    """Load a previously saved model."""
    model_path = cfg.MODELS_DIR / f"fold_{fold_id}" / f"model_bucket{bucket_id}_{target_type}.pkl"
    with open(model_path, "rb") as f:
        return pickle.load(f)


# ── Step 3 Eval Checks ──────────────────────────────────────────────────────

def _run_step3_checks(results: Dict[str, Any], bucket: cfg.HorizonBucket) -> None:
    """Post-training validation."""
    logger.info("Step 3 checks (Bucket %d):", bucket.bucket_id)

    for tt in ("total", "rate"):
        wape_key = f"val_wape_{tt}"
        if wape_key in results:
            val_w = results[wape_key]
            logger.info("  %s val WAPE: %.4f", tt, val_w)
            if val_w > 1.0:
                logger.warning("  %s val WAPE > 1.0 — model may be very poor", tt)

        imp_key = f"importance_{tt}"
        if imp_key in results:
            imp = results[imp_key]
            if "days_ahead" in imp["feature"].values:
                da_rank = imp[imp["feature"] == "days_ahead"].index[0] + 1
                if da_rank <= 3:
                    logger.warning(
                        "  days_ahead ranked #%d for %s — model may just be learning horizon baselines",
                        da_rank, tt,
                    )

        model_key = f"model_{tt}"
        if model_key in results:
            model = results[model_key]
            best_iter = getattr(model, "best_iteration_", "N/A")
            n_est = getattr(model, "n_estimators", "N/A")
            logger.info("  %s: best_iteration=%s / n_estimators=%s", tt, best_iter, n_est)
