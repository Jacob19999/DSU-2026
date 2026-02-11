"""
LightGBM training for Pipeline B — 3 buckets × 2 targets = 6 models per fold.

Model B_total: total_enc (Tweedie/Poisson, volume-weighted)
Model B_rate:  admit_rate (MSE regression, admitted-weighted)

Includes Optuna hyperparameter tuning per bucket-target pair.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import lightgbm as lgb

import config as cfg
from features import (
    build_bucket_data,
    compute_fold_encodings,
    apply_fold_encodings,
    get_feature_columns,
)

CAT_FEATURES = ["site_enc", "block"]


# ── Metrics ──────────────────────────────────────────────────────────────────

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else float("nan")


def largest_remainder_round(values: np.ndarray) -> np.ndarray:
    """Round floats to ints preserving aggregate sum."""
    values = np.asarray(values, dtype=float).clip(0)
    floored = np.floor(values).astype(int)
    remainders = values - floored
    target_sum = int(round(values.sum()))
    deficit = target_sum - floored.sum()
    if deficit > 0:
        idx = np.argsort(-remainders)[:deficit]
        floored[idx] += 1
    elif deficit < 0:
        idx = np.argsort(remainders)[:(-deficit)]
        floored[idx] -= 1
    return np.maximum(floored, 0)


# ── Single-bucket training ───────────────────────────────────────────────────

def _prepare_bucket_fold(
    bucket_data: pd.DataFrame,
    base_df: pd.DataFrame,
    bucket_id: int,
    fold: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Split bucket data into train/early-stop, apply fold encodings, drop NaN.

    Returns (train_df, es_df, feature_cols).
    """
    train_end = pd.Timestamp(fold["train_end"])
    es_cutoff = train_end - pd.Timedelta(days=cfg.ES_HOLD_DAYS)
    lags = cfg.BUCKET_LAGS[bucket_id]
    max_lag_col = f"lag_total_{max(lags)}"

    # Fold-specific mean encodings (from non-expanded base data)
    enc_maps, fallback = compute_fold_encodings(base_df, fold["train_end"])

    # Filter to training window, apply encodings
    train_all = bucket_data[bucket_data["date"] <= train_end].copy()
    train_all = apply_fold_encodings(train_all, enc_maps, fallback)

    # Drop burn-in rows where longest lag is NaN
    train_all = train_all.dropna(subset=[max_lag_col])

    feature_cols = get_feature_columns(train_all)

    # Split: fit vs early-stopping hold-out
    train_df = train_all[train_all["date"] <= es_cutoff].copy()
    es_df    = train_all[train_all["date"] > es_cutoff].copy()

    # Fall back: if ES split is too small, use last 10% of training
    if len(es_df) < 100:
        n_es = max(int(len(train_all) * 0.1), 100)
        train_df = train_all.iloc[:-n_es].copy()
        es_df    = train_all.iloc[-n_es:].copy()

    return train_df, es_df, feature_cols


def train_bucket(
    train_df: pd.DataFrame,
    es_df: pd.DataFrame,
    feature_cols: list[str],
    params_total: dict | None = None,
    params_rate: dict | None = None,
) -> tuple[lgb.LGBMRegressor, lgb.LGBMRegressor]:
    """Train total_enc + admit_rate models for one bucket.

    Returns (model_total, model_rate).
    """
    p_total = (params_total or cfg.LGBM_DEFAULT_TOTAL).copy()
    p_rate  = (params_rate  or cfg.LGBM_DEFAULT_RATE).copy()

    X_train = train_df[feature_cols]
    X_es    = es_df[feature_cols]

    # ── Model B_total ────────────────────────────────────────────────────
    model_total = lgb.LGBMRegressor(**p_total)
    model_total.fit(
        X_train, train_df["total_enc"],
        sample_weight=train_df["sample_weight"].values,
        eval_set=[(X_es, es_df["total_enc"])],
        categorical_feature=CAT_FEATURES,
        callbacks=[
            lgb.early_stopping(cfg.ES_PATIENCE, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    # ── Model B_rate ─────────────────────────────────────────────────────
    model_rate = lgb.LGBMRegressor(**p_rate)
    model_rate.fit(
        X_train, train_df["admit_rate"],
        sample_weight=train_df["sample_weight_rate"].values,
        eval_set=[(X_es, es_df["admit_rate"])],
        categorical_feature=CAT_FEATURES,
        callbacks=[
            lgb.early_stopping(cfg.ES_PATIENCE, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    return model_total, model_rate


# ── Per-fold: train all buckets + predict on val window ──────────────────────

def train_fold(
    base_df: pd.DataFrame,
    bucket_data_map: dict[int, pd.DataFrame],
    fold: dict,
    all_params: dict | None = None,
    *,
    save: bool = True,
) -> dict:
    """Train 6 models for one fold, predict on validation window, post-process.

    Parameters
    ----------
    base_df         : Base DataFrame with static features (non-expanded).
    bucket_data_map : {bucket_id: pre-expanded DataFrame} for training.
    fold            : Fold definition dict.
    all_params      : {bucket_id: {"total": params, "rate": params}} or None.
    save            : Persist models + predictions to disk.

    Returns dict with fold metrics, models, and submission DataFrame.
    """
    fold_id   = fold["id"]
    train_end = pd.Timestamp(fold["train_end"])
    val_start = pd.Timestamp(fold["val_start"])
    val_end   = pd.Timestamp(fold["val_end"])

    enc_maps, fallback = compute_fold_encodings(base_df, fold["train_end"])

    all_val_preds: list[pd.DataFrame] = []
    fold_models: dict[int, tuple] = {}
    bucket_wapes: dict[int, dict] = {}

    for bid in [1, 2, 3]:
        bucket = cfg.BUCKETS[bid]
        params = (all_params or {}).get(bid, {})
        p_total = params.get("total")
        p_rate  = params.get("rate")

        # ── Prepare training data ────────────────────────────────────────
        train_df, es_df, feature_cols = _prepare_bucket_fold(
            bucket_data_map[bid], base_df, bid, fold,
        )

        # ── Train ────────────────────────────────────────────────────────
        model_total, model_rate = train_bucket(
            train_df, es_df, feature_cols, p_total, p_rate,
        )
        fold_models[bid] = (model_total, model_rate, feature_cols)

        # ── Build prediction features for val window ─────────────────────
        val_dates = pd.date_range(val_start, val_end)
        # Exact horizons for this bucket within the val window
        val_horizons = []
        valid_dates = set()
        for d in val_dates:
            h = (d - train_end).days
            if bucket["h_min"] <= h <= bucket["h_max"]:
                val_horizons.append(h)
                valid_dates.add(d)

        if not val_horizons:
            continue

        pred_data = build_bucket_data(
            base_df, bid,
            horizons=sorted(set(val_horizons)),
            target_dates=valid_dates,
        )
        pred_data = apply_fold_encodings(pred_data, enc_maps, fallback)

        # ── Predict ──────────────────────────────────────────────────────
        X_pred = pred_data[feature_cols]
        pred_total    = model_total.predict(X_pred).clip(0)
        pred_rate     = model_rate.predict(X_pred).clip(0, 1)
        pred_admitted = pred_total * pred_rate

        preds = pred_data[["site", "date", "block", "days_ahead"]].copy()
        preds["pred_total"]    = pred_total
        preds["pred_admitted"] = pred_admitted
        preds["bucket"]        = bid

        # Aggregate across horizons → one row per (site, date, block)
        preds = (
            preds.groupby(["site", "date", "block"], as_index=False)
            .agg(pred_total=("pred_total", "mean"),
                 pred_admitted=("pred_admitted", "mean"),
                 bucket=("bucket", "first"))
        )
        all_val_preds.append(preds)

        # Quick per-bucket metric (on overlapping actuals)
        merged = preds.merge(
            base_df[["site", "date", "block", "total_enc", "admitted_enc"]],
            on=["site", "date", "block"], how="left",
        )
        bw_total = wape(merged["total_enc"].values, merged["pred_total"].values)
        bw_admit = wape(merged["admitted_enc"].values, merged["pred_admitted"].values)
        bucket_wapes[bid] = {"total_wape": bw_total, "admitted_wape": bw_admit}

        # ── Save models ──────────────────────────────────────────────────
        if save:
            fdir = cfg.fold_model_dir(fold_id)
            model_total.booster_.save_model(str(fdir / f"model_bucket{bid}_total.txt"))
            model_rate.booster_.save_model(str(fdir / f"model_bucket{bid}_rate.txt"))

    # ── Assemble predictions across all buckets ──────────────────────────
    if not all_val_preds:
        print(f"    Fold {fold_id}: no predictions generated!")
        return {"fold_id": fold_id}

    val_out = pd.concat(all_val_preds, ignore_index=True)

    # ── Deduplicate: average across buckets for same (site, date, block) ─
    if val_out.duplicated(subset=["site", "date", "block"]).any():
        val_out = (
            val_out.groupby(["site", "date", "block"], as_index=False)
            .agg(pred_total=("pred_total", "mean"),
                 pred_admitted=("pred_admitted", "mean"))
        )

    # ── Post-process: largest-remainder rounding per (Site, Date) ────────
    for (_s, _d), grp in val_out.groupby(["site", "date"]):
        idx = grp.index
        val_out.loc[idx, "pred_total"]    = largest_remainder_round(grp["pred_total"].values)
        val_out.loc[idx, "pred_admitted"] = largest_remainder_round(grp["pred_admitted"].values)

    # Enforce admitted ≤ total
    val_out["pred_total"]    = val_out["pred_total"].astype(int)
    val_out["pred_admitted"] = np.minimum(
        val_out["pred_admitted"].astype(int), val_out["pred_total"]
    )

    # ── Format as eval.md submission ─────────────────────────────────────
    submission = pd.DataFrame({
        "Site":            val_out["site"].values,
        "Date":            val_out["date"].dt.strftime("%Y-%m-%d").values,
        "Block":           val_out["block"].values,
        "ED Enc":          val_out["pred_total"].values,
        "ED Enc Admitted": val_out["pred_admitted"].values,
    })

    # ── Persist ──────────────────────────────────────────────────────────
    if save:
        cfg.ensure_dirs()
        submission.to_csv(cfg.PRED_DIR / f"fold_{fold_id}_predictions.csv", index=False)

    # ── Overall fold metrics ─────────────────────────────────────────────
    actual = base_df[
        (base_df["date"] >= val_start) & (base_df["date"] <= val_end)
    ][["site", "date", "block", "total_enc", "admitted_enc"]]

    merged = submission.copy()
    merged["Date"] = pd.to_datetime(merged["Date"])
    merged = merged.merge(
        actual.rename(columns={"date": "Date", "site": "Site", "block": "Block"}),
        on=["Site", "Date", "Block"], how="left",
    )

    total_wape_val    = wape(merged["total_enc"].values, merged["ED Enc"].values)
    admitted_wape_val = wape(merged["admitted_enc"].values, merged["ED Enc Admitted"].values)

    print(f"    Fold {fold_id}: total_wape={total_wape_val:.4f}  "
          f"admitted_wape={admitted_wape_val:.4f}  "
          f"(bucket WAPEs: {', '.join(f'B{b}={w['admitted_wape']:.4f}' for b, w in bucket_wapes.items())})")

    return {
        "fold_id": fold_id,
        "total_wape": total_wape_val,
        "admitted_wape": admitted_wape_val,
        "bucket_wapes": bucket_wapes,
        "models": fold_models,
        "submission": submission,
    }


# ── All folds ────────────────────────────────────────────────────────────────

def train_all_folds(
    base_df: pd.DataFrame,
    bucket_data_map: dict[int, pd.DataFrame],
    all_params: dict | None = None,
    *,
    save: bool = True,
) -> list[dict]:
    """Train + predict across all 4 folds, print summary."""
    results = []
    for fold in cfg.FOLDS:
        result = train_fold(base_df, bucket_data_map, fold, all_params, save=save)
        results.append(result)

    # Summary
    valid = [r for r in results if "admitted_wape" in r]
    if valid:
        mean_t = np.mean([r["total_wape"] for r in valid])
        mean_a = np.mean([r["admitted_wape"] for r in valid])
        print(f"\n  4-fold mean: total_wape={mean_t:.4f}  admitted_wape={mean_a:.4f}")

        # Per-bucket summary
        for bid in [1, 2, 3]:
            bw = [r["bucket_wapes"].get(bid, {}).get("admitted_wape", float("nan"))
                  for r in valid if "bucket_wapes" in r]
            if bw:
                print(f"    Bucket {bid} mean admitted_wape: {np.nanmean(bw):.4f}")

    # Feature importance (last fold, bucket 1 total model)
    if save and valid:
        cfg.ensure_dirs()
        for bid in [1, 2, 3]:
            last_result = valid[-1]
            if bid in last_result.get("models", {}):
                m_total, _, fcols = last_result["models"][bid]
                fi = pd.Series(m_total.feature_importances_, index=fcols)
                fi = fi.sort_values(ascending=False)
                fi.to_csv(cfg.FI_DIR / f"importance_bucket{bid}_total.csv")

    # Save OOF predictions
    if save:
        oof_frames = [r["submission"] for r in valid if "submission" in r]
        if oof_frames:
            oof = pd.concat(oof_frames, ignore_index=True)
            oof.to_csv(cfg.PRED_DIR / "oof_predictions.csv", index=False)
            print(f"  OOF predictions: {len(oof)} rows -> {cfg.PRED_DIR / 'oof_predictions.csv'}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  OPTUNA HYPERPARAMETER TUNING
# ══════════════════════════════════════════════════════════════════════════════

def run_optuna_tuning(
    base_df: pd.DataFrame,
    bucket_data_map: dict[int, pd.DataFrame],
    n_trials_total: int | None = None,
    n_trials_rate: int | None = None,
) -> dict[int, dict]:
    """Tune all 6 bucket-target models via Optuna.

    Returns {bucket_id: {"total": best_params, "rate": best_params}}.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    nt_total = n_trials_total or cfg.OPTUNA_N_TRIALS_TOTAL
    nt_rate  = n_trials_rate  or cfg.OPTUNA_N_TRIALS_RATE
    all_best: dict[int, dict] = {}

    for bid in [1, 2, 3]:
        print(f"\n  Tuning Bucket {bid} ...")
        bucket_data = bucket_data_map[bid]

        # ── Phase 1: Tune total model ────────────────────────────────────
        print(f"    Phase 1 — total_enc ({nt_total} trials) ...")

        def _objective_total(trial: optuna.Trial) -> float:
            obj = trial.suggest_categorical("objective", ["tweedie", "poisson"])
            params = {
                "objective": obj,
                "n_estimators":    trial.suggest_int("n_estimators", 800, 3000),
                "max_depth":       trial.suggest_int("max_depth", 4, 8),
                "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
                "num_leaves":      trial.suggest_int("num_leaves", 31, 255),
                "subsample":       trial.suggest_float("subsample", 0.7, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
                "reg_lambda":      trial.suggest_float("reg_lambda", 1.0, 10.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "verbosity": -1,
            }
            if obj == "tweedie":
                params["tweedie_variance_power"] = trial.suggest_float(
                    "tweedie_variance_power", 1.1, 1.9
                )

            fold_wapes = []
            for fold in cfg.FOLDS:
                train_df, es_df, fcols = _prepare_bucket_fold(
                    bucket_data, base_df, bid, fold,
                )
                try:
                    m_total, _ = train_bucket(train_df, es_df, fcols, params, None)
                    # Quick WAPE on ES set
                    preds = m_total.predict(es_df[fcols]).clip(0)
                    fold_wapes.append(wape(es_df["total_enc"].values, preds))
                except Exception:
                    return float("inf")

            return float(np.mean(fold_wapes))

        study_t = optuna.create_study(direction="minimize",
                                       study_name=f"B_bucket{bid}_total")
        study_t.optimize(_objective_total, n_trials=nt_total, show_progress_bar=True)

        best_total = {k: v for k, v in study_t.best_params.items()}
        best_total["verbosity"] = -1
        print(f"    Best total WAPE: {study_t.best_value:.4f}")

        # ── Phase 2: Tune rate model (total frozen) ──────────────────────
        print(f"    Phase 2 — admit_rate ({nt_rate} trials) ...")

        def _objective_rate(trial: optuna.Trial) -> float:
            params = {
                "objective": "regression",
                "n_estimators":    trial.suggest_int("n_estimators", 500, 2000),
                "max_depth":       trial.suggest_int("max_depth", 3, 7),
                "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
                "num_leaves":      trial.suggest_int("num_leaves", 31, 127),
                "subsample":       trial.suggest_float("subsample", 0.7, 0.95),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
                "reg_lambda":      trial.suggest_float("reg_lambda", 1.0, 10.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "verbosity": -1,
            }

            fold_wapes = []
            for fold in cfg.FOLDS:
                train_df, es_df, fcols = _prepare_bucket_fold(
                    bucket_data, base_df, bid, fold,
                )
                try:
                    _, m_rate = train_bucket(train_df, es_df, fcols, best_total, params)
                    preds = m_rate.predict(es_df[fcols]).clip(0, 1)
                    fold_wapes.append(wape(es_df["admit_rate"].values, preds))
                except Exception:
                    return float("inf")

            return float(np.mean(fold_wapes))

        study_r = optuna.create_study(direction="minimize",
                                       study_name=f"B_bucket{bid}_rate")
        study_r.optimize(_objective_rate, n_trials=nt_rate, show_progress_bar=True)

        best_rate = study_r.best_params.copy()
        best_rate["verbosity"] = -1
        print(f"    Best rate WAPE: {study_r.best_value:.4f}")

        all_best[bid] = {"total": best_total, "rate": best_rate}

        # Persist per-bucket params
        cfg.ensure_dirs()
        for target_name, params in [("total", best_total), ("rate", best_rate)]:
            path = cfg.MODEL_DIR / f"best_params_bucket{bid}_{target_name}.json"
            with open(path, "w") as f:
                json.dump(params, f, indent=2)

    return all_best


def load_tuned_params() -> dict[int, dict] | None:
    """Load previously saved Optuna params from disk. Returns None if not found."""
    all_params: dict[int, dict] = {}
    for bid in [1, 2, 3]:
        bucket_params: dict[str, dict] = {}
        for target_name in ["total", "rate"]:
            path = cfg.MODEL_DIR / f"best_params_bucket{bid}_{target_name}.json"
            if not path.exists():
                return None
            with open(path) as f:
                bucket_params[target_name] = json.load(f)
        all_params[bid] = bucket_params

    print("  Loaded tuned params from disk for all 3 buckets")
    return all_params


if __name__ == "__main__":
    from data_loader import load_data
    from features import add_static_features

    df = load_data()
    df = add_static_features(df)

    bucket_data_map = {}
    for bid in [1, 2, 3]:
        print(f"  Building bucket {bid} data ...")
        bucket_data_map[bid] = build_bucket_data(df, bid)

    train_all_folds(df, bucket_data_map)
