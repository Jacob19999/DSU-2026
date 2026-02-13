"""
LightGBM training for Pipeline E — 2 models per fold (total_enc + admit_rate).

Per-fold pipeline:
  1. Fit PCA on training data → factor columns
  2. Build factor lag/rolling/momentum features
  3. Train factor forecast GBDTs (one per factor)
  4. Set factor_i_pred (actual for train, predicted for val)
  5. Apply fold encodings
  6. Train final total_enc + admit_rate models
  7. Predict on validation, post-process, save

Includes Optuna two-stage hyperparameter tuning.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import lightgbm as lgb

import config as cfg
from factor_extraction import fit_and_transform_factors
from factor_forecasting import (
    add_factor_lag_features,
    train_factor_forecast_models,
    set_predicted_factors,
)
from features import (
    compute_fold_encodings,
    apply_fold_encodings,
    get_feature_columns,
)
from transfer_learning import (
    train_teacher_abc,
    train_student_d,
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
    floored = np.maximum(floored, 0)
    # Redistribute if clipping broke sum invariant (deficit < 0 can push to -1)
    excess = floored.sum() - target_sum
    if excess > 0:
        candidates = np.where(floored > 0)[0]
        if len(candidates) >= excess:
            order = np.argsort(floored[candidates])
            floored[candidates[order[:excess]]] -= 1
    return floored


# ══════════════════════════════════════════════════════════════════════════════
#  PER-FOLD DATA PREPARATION  (factor pipeline + feature assembly)
# ══════════════════════════════════════════════════════════════════════════════

def prepare_fold_data(
    base_df: pd.DataFrame,
    share_cols: list[str],
    fold: dict,
    *,
    n_factors: int = cfg.N_FACTORS,
    factor_method: str = cfg.FACTOR_METHOD,
) -> tuple[pd.DataFrame, object | None, list]:
    """Run the full factor pipeline for one fold and return enriched DataFrame.

    Returns (fold_df, factor_model, factor_forecast_models).
    If share_cols is empty (ablation), factor steps are skipped entirely.
    """
    train_end = fold["train_end"]
    fold_df = base_df.copy()

    factor_model = None
    factor_models: list = []

    if share_cols:
        # Step 3: Factor extraction (fold-specific PCA/NMF)
        fold_df, factor_model, factor_cols = fit_and_transform_factors(
            fold_df, share_cols, train_end,
            n_factors=n_factors, method=factor_method,
        )

        # Step 4a: Factor lag features (from actual factor values)
        fold_df = add_factor_lag_features(fold_df, n_factors=n_factors)

        # Step 4b: Train factor forecast models
        factor_models = train_factor_forecast_models(
            fold_df, train_end, n_factors=n_factors,
        )

        # Step 4c: Set factor_i_pred (actual for train, predicted for val)
        fold_df = set_predicted_factors(
            fold_df, factor_models, train_end, n_factors=n_factors,
        )
    else:
        print("  [ABLATION] Skipping factor pipeline (no share columns)")

    # Fold-specific mean-target encodings
    enc_maps, fallback = compute_fold_encodings(fold_df, train_end)
    fold_df = apply_fold_encodings(fold_df, enc_maps, fallback)

    return fold_df, factor_model, factor_models


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE-FOLD TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def train_fold(
    base_df: pd.DataFrame,
    share_cols: list[str],
    fold: dict,
    params_total: dict | None = None,
    params_rate: dict | None = None,
    *,
    n_factors: int = cfg.N_FACTORS,
    factor_method: str = cfg.FACTOR_METHOD,
    save: bool = True,
) -> dict:
    """Train + predict for one fold.  Returns dict with metrics + submission."""
    fold_id   = fold["id"]
    train_end = pd.Timestamp(fold["train_end"])
    val_start = pd.Timestamp(fold["val_start"])
    val_end   = pd.Timestamp(fold["val_end"])

    p_total = (params_total or cfg.LGBM_DEFAULT_TOTAL).copy()
    p_rate  = (params_rate  or cfg.LGBM_DEFAULT_RATE).copy()

    # ── Prepare data (factor pipeline) ───────────────────────────────────
    fold_df, factor_model, factor_models = prepare_fold_data(
        base_df, share_cols, fold,
        n_factors=n_factors, factor_method=factor_method,
    )

    feature_cols = get_feature_columns(fold_df)

    # ── Split into train / val ───────────────────────────────────────────
    burn_in_col = f"lag_{max(cfg.LAG_DAYS)}"
    train_all = fold_df[fold_df["date"] <= train_end].dropna(subset=[burn_in_col]).copy()
    val_df    = fold_df[(fold_df["date"] >= val_start) & (fold_df["date"] <= val_end)].copy()

    print(f"    Total train: {len(train_all):,}  |  "
          f"Val: {len(val_df):,} rows  |  Features: {len(feature_cols)}")

    # ── Stage 1: ABC Teacher ────────────────────────────────────────────
    teacher_result = train_teacher_abc(
        fold_df, train_end, val_start, val_end,
        feature_cols=feature_cols,
        params_total=p_total, params_rate=p_rate,
    )
    model_total = teacher_result["models"]["t1"]  # alias for compat
    model_rate = teacher_result["models"]["t2"]

    abc_val_idx = teacher_result["abc_val_idx"]
    pred_total_abc = teacher_result["pred_total_abc"]
    pred_rate_abc = teacher_result["pred_rate_abc"]

    # ── Stage 2: Site D Student ──────────────────────────────────────────
    student_result = train_student_d(
        fold_df, train_end, val_start, val_end,
        teacher_models=teacher_result["models"],
        feature_cols_parent=feature_cols,
    )

    # ── Merge: ABC (teacher) + D (student) ───────────────────────────────
    pred_total    = np.empty(len(val_df), dtype=float)
    pred_rate     = np.empty(len(val_df), dtype=float)

    abc_val_pos = val_df.index.get_indexer(abc_val_idx)
    pred_total[abc_val_pos] = pred_total_abc
    pred_rate[abc_val_pos] = pred_rate_abc

    d_val_pos = val_df.index.get_indexer(student_result["d_val_idx"])
    pred_total[d_val_pos] = student_result["pred_total"]
    pred_rate[d_val_pos] = student_result["pred_rate"]

    pred_total    = pred_total.clip(0)
    pred_rate     = pred_rate.clip(0, 1)
    pred_admitted = pred_total * pred_rate

    # ── Post-process: largest-remainder rounding ─────────────────────────
    preds = val_df[["site", "date", "block"]].copy()
    preds["pred_total"]    = pred_total
    preds["pred_admitted"] = pred_admitted

    for (_s, _d), grp in preds.groupby(["site", "date"]):
        idx = grp.index
        preds.loc[idx, "pred_total"]    = largest_remainder_round(grp["pred_total"].values)
        preds.loc[idx, "pred_admitted"] = largest_remainder_round(grp["pred_admitted"].values)

    preds["pred_total"]    = preds["pred_total"].astype(int)
    preds["pred_admitted"] = np.minimum(
        preds["pred_admitted"].astype(int), preds["pred_total"]
    )

    # ── Format submission ────────────────────────────────────────────────
    submission = pd.DataFrame({
        "Site":            preds["site"].values,
        "Date":            preds["date"].dt.strftime("%Y-%m-%d").values,
        "Block":           preds["block"].values,
        "ED Enc":          preds["pred_total"].values,
        "ED Enc Admitted": preds["pred_admitted"].values,
    })

    # ── Metrics ──────────────────────────────────────────────────────────
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

    # Per-site WAPE
    site_wapes = {}
    for site in cfg.SITES:
        mask_s = merged["Site"] == site
        if mask_s.sum() == 0:
            continue
        site_wapes[site] = {
            "total": wape(merged.loc[mask_s, "total_enc"].values,
                          merged.loc[mask_s, "ED Enc"].values),
            "admitted": wape(merged.loc[mask_s, "admitted_enc"].values,
                             merged.loc[mask_s, "ED Enc Admitted"].values),
        }

    print(f"    Fold {fold_id}: total_wape={total_wape_val:.4f}  "
          f"admitted_wape={admitted_wape_val:.4f}")
    for site, sw in site_wapes.items():
        print(f"      Site {site}: total={sw['total']:.4f}  admitted={sw['admitted']:.4f}")

    # ── Save artifacts ───────────────────────────────────────────────────
    if save:
        cfg.ensure_dirs()
        submission.to_csv(cfg.PRED_DIR / f"fold_{fold_id}_predictions.csv", index=False)

        fdir = cfg.fold_model_dir(fold_id)
        # Teacher models (ABC)
        model_total.booster_.save_model(str(fdir / "teacher_t1_total.txt"))
        model_rate.booster_.save_model(str(fdir / "teacher_t2_rate.txt"))
        # Student models (Site D)
        student_result["model_s1"].booster_.save_model(str(fdir / "student_s1_total.txt"))
        student_result["model_s2"].booster_.save_model(str(fdir / "student_s2_rate.txt"))

        # Feature importance (teacher)
        fi = pd.Series(model_total.feature_importances_, index=feature_cols)
        fi = fi.sort_values(ascending=False)
        fi.to_csv(cfg.FI_DIR / f"importance_fold{fold_id}_total.csv")

        # Report factor features in top-20
        top20 = fi.head(20).index.tolist()
        factor_in_top20 = [f for f in top20 if "factor_" in f]
        print(f"    Factor features in top-20 importance: "
              f"{len(factor_in_top20)}/{len(top20)} — {factor_in_top20}")

    return {
        "fold_id": fold_id,
        "total_wape": total_wape_val,
        "admitted_wape": admitted_wape_val,
        "site_wapes": site_wapes,
        "submission": submission,
        "models": (model_total, model_rate),
        "student_result": student_result,
        "feature_cols": feature_cols,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ALL FOLDS
# ══════════════════════════════════════════════════════════════════════════════

def train_all_folds(
    base_df: pd.DataFrame,
    share_cols: list[str],
    params_total: dict | None = None,
    params_rate: dict | None = None,
    *,
    save: bool = True,
) -> list[dict]:
    """Train + predict across all 4 folds, print summary."""
    results = []
    for fold in cfg.FOLDS:
        print(f"\n--- Fold {fold['id']} ({fold['val_start']}..{fold['val_end']}) ---")
        result = train_fold(
            base_df, share_cols, fold,
            params_total=params_total, params_rate=params_rate,
            save=save,
        )
        results.append(result)

    # Summary
    valid = [r for r in results if "admitted_wape" in r]
    if valid:
        mean_t = np.mean([r["total_wape"] for r in valid])
        mean_a = np.mean([r["admitted_wape"] for r in valid])
        print(f"\n  4-fold mean: total_wape={mean_t:.4f}  admitted_wape={mean_a:.4f}")

    # Save OOF predictions
    if save:
        cfg.ensure_dirs()
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
    share_cols: list[str],
    n_trials_total: int | None = None,
    n_trials_rate: int | None = None,
) -> dict:
    """Two-stage Optuna tuning: final total_enc then admit_rate.

    Returns {"total": best_params, "rate": best_params}.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    nt_total = n_trials_total or cfg.OPTUNA_N_TRIALS_TOTAL
    nt_rate  = n_trials_rate  or cfg.OPTUNA_N_TRIALS_RATE

    # ── Stage 1: total_enc model ─────────────────────────────────────────
    print(f"\n  Stage 1 — total_enc ({nt_total} trials) ...")

    def _objective_total(trial: optuna.Trial) -> float:
        obj = trial.suggest_categorical("objective", ["tweedie", "poisson"])
        params = {
            "objective": obj,
            "n_estimators":     trial.suggest_int("n_estimators", 800, 3000),
            "max_depth":        trial.suggest_int("max_depth", 4, 8),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 31, 255),
            "subsample":        trial.suggest_float("subsample", 0.7, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1.0, 10.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "verbosity": -1,
        }
        if obj == "tweedie":
            params["tweedie_variance_power"] = trial.suggest_float(
                "tweedie_variance_power", 1.1, 1.9
            )

        fold_wapes = []
        for fold in cfg.FOLDS:
            try:
                result = train_fold(
                    base_df, share_cols, fold,
                    params_total=params, save=False,
                )
                fold_wapes.append(result["total_wape"])
            except Exception:
                return float("inf")

        return float(np.mean(fold_wapes))

    study_t = optuna.create_study(
        direction="minimize", study_name="E_total",
    )
    study_t.optimize(_objective_total, n_trials=nt_total, show_progress_bar=True)

    best_total = {k: v for k, v in study_t.best_params.items()}
    best_total["verbosity"] = -1
    print(f"  Best total WAPE: {study_t.best_value:.4f}")

    # ── Stage 2: admit_rate model (total frozen) ─────────────────────────
    print(f"\n  Stage 2 — admit_rate ({nt_rate} trials) ...")

    def _objective_rate(trial: optuna.Trial) -> float:
        params = {
            "objective": "regression",
            "n_estimators":     trial.suggest_int("n_estimators", 500, 2000),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 31, 127),
            "subsample":        trial.suggest_float("subsample", 0.7, 0.95),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1.0, 10.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "verbosity": -1,
        }

        fold_wapes = []
        for fold in cfg.FOLDS:
            try:
                result = train_fold(
                    base_df, share_cols, fold,
                    params_total=best_total, params_rate=params,
                    save=False,
                )
                fold_wapes.append(result["admitted_wape"])
            except Exception:
                return float("inf")

        return float(np.mean(fold_wapes))

    study_r = optuna.create_study(
        direction="minimize", study_name="E_rate",
    )
    study_r.optimize(_objective_rate, n_trials=nt_rate, show_progress_bar=True)

    best_rate = study_r.best_params.copy()
    best_rate["verbosity"] = -1
    print(f"  Best rate WAPE: {study_r.best_value:.4f}")

    # Persist best params
    cfg.ensure_dirs()
    for name, params in [("total", best_total), ("rate", best_rate)]:
        path = cfg.MODEL_DIR / f"best_params_{name}.json"
        with open(path, "w") as f:
            json.dump(params, f, indent=2)

    return {"total": best_total, "rate": best_rate}


def load_tuned_params() -> dict | None:
    """Load previously saved Optuna params. Returns None if not found."""
    result: dict[str, dict] = {}
    for name in ["total", "rate"]:
        path = cfg.MODEL_DIR / f"best_params_{name}.json"
        if not path.exists():
            return None
        with open(path) as f:
            result[name] = json.load(f)
    print("  Loaded tuned params from disk")
    return result


if __name__ == "__main__":
    from data_loader import load_data
    from share_matrix import build_share_matrix
    from features import add_all_base_features

    df = load_data()
    df, share_cols = build_share_matrix(df)
    df = add_all_base_features(df)
    train_all_folds(df, share_cols)
