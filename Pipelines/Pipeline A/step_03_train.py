"""
Step 3: Train LightGBM models per fold.

Model A1 — total_enc (Tweedie objective, volume-weighted)
Model A2 — admit_rate (MSE regression, admitted-weighted)

Produces submission-format CSVs and saved model files for each fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

import config as cfg
from sklearn.linear_model import LogisticRegression

from step_02_feature_eng import (
    compute_fold_aggregate_encodings,
    compute_fold_target_encodings,
    get_feature_columns,
)

# ── Site D Isolation: Residual model feature list ─────────────────────────────
# Excludes weather/reason-mix to prevent overfitting (Enhancement B, §11)

RESIDUAL_FEATURES = [
    "block", "dow", "month", "day_of_year",
    "te_site_block_mean_total", "te_site_block_mean_admitted",
    "te_site_admit_rate",
    "lag_63", "lag_91", "lag_182", "lag_364",
    "roll_mean_7", "roll_mean_28",
    "global_pred",
    "is_covid_era", "is_us_holiday", "is_weekend",
]

RESIDUAL_LGBM_PARAMS = {
    "objective": "regression",
    "n_estimators": 300,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 50,
    "learning_rate": 0.03,
    "reg_lambda": 5.0,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "verbosity": -1,
}

# Shrinkage weight for residual correction — tune via inner CV in [0.3, 0.8]
SHRINKAGE_WEIGHT = 0.5

# Admit-rate guardrail bounds per (site, block) — historical [5th, 95th] percentile
ADMIT_RATE_BOUNDS = {
    ("D", 0): (0.08, 0.30),
    ("D", 1): (0.10, 0.35),
    ("D", 2): (0.10, 0.30),
    ("D", 3): (0.08, 0.28),
}

# Zero-inflation shrinkage for admitted corrections on sparse blocks
ZERO_SHRINKAGE = 0.5


# ── Helpers ──────────────────────────────────────────────────────────────────

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error."""
    denom = float(np.sum(np.abs(y_true)))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else float("nan")


def largest_remainder_round(values: np.ndarray) -> np.ndarray:
    """Round float array to integers while preserving the aggregate sum.

    Handles the edge case where clipping negatives to 0 would break
    the sum-preservation guarantee by redistributing the deficit.
    """
    values = np.maximum(values, 0.0)  # clip inputs first
    floored = np.floor(values).astype(int)
    remainders = values - floored
    target_sum = int(round(values.sum()))
    deficit = target_sum - floored.sum()
    if deficit > 0:
        idx = np.argsort(-remainders)[:deficit]
        floored[idx] += 1
    elif deficit < 0:
        # Only decrement items that are > 0 to avoid negatives
        candidates = np.where(floored > 0)[0]
        order = candidates[np.argsort(remainders[candidates])]
        for i in order[:(-deficit)]:
            floored[i] -= 1
    return floored


# ── Single-fold training ────────────────────────────────────────────────────

def train_fold(
    df: pd.DataFrame,
    fold: dict,
    params_a1: dict | None = None,
    params_a2: dict | None = None,
    *,
    save: bool = True,
    covid_policy: str = "downweight",
) -> dict:
    """
    Train A1 + A2 for one fold, predict on val, post-process, optionally save.

    Parameters
    ----------
    covid_policy : "downweight" (default) or "exclude"
    save : write CSVs and model files to disk
    """
    p_a1 = (params_a1 or cfg.LGBM_DEFAULT_A1).copy()
    p_a2 = (params_a2 or cfg.LGBM_DEFAULT_A2).copy()

    fold_id = fold["id"]
    train_end = pd.Timestamp(fold["train_end"])
    val_start = pd.Timestamp(fold["val_start"])
    val_end = pd.Timestamp(fold["val_end"])

    # ── Split ────────────────────────────────────────────────────────────
    train_mask = df["date"] <= train_end
    val_mask = (df["date"] >= val_start) & (df["date"] <= val_end)

    df_fold = compute_fold_aggregate_encodings(df, train_mask)
    df_fold = compute_fold_target_encodings(df_fold, train_mask)
    feature_cols = get_feature_columns(df_fold)
    cat_features = [c for c in ["site_enc", "block", "site_x_dow", "site_x_month"] if c in feature_cols]

    train_data = df_fold.loc[train_mask].copy()
    val_data = df_fold.loc[val_mask].copy()

    # COVID policy
    if covid_policy == "exclude":
        if "is_covid_era" in train_data.columns:
            train_data = train_data[train_data["is_covid_era"] == 0].copy()
        else:
            print("  [WARN] covid_policy='exclude' but 'is_covid_era' not in columns; skipping exclusion.")

    # Drop rows where longest lag is NaN (burn-in period)
    train_data = train_data.dropna(subset=[f"lag_{cfg.LAG_DAYS[-1]}"])

    X_train = train_data[feature_cols]
    X_val = val_data[feature_cols]

    # ── Weights ──────────────────────────────────────────────────────────
    if covid_policy == "exclude":
        w_a1 = train_data["volume_weight"].values
        w_a2 = train_data["admitted_enc"].clip(lower=1).values.astype(float)
    else:
        w_a1 = train_data["sample_weight_a1"].values
        w_a2 = train_data["sample_weight_a2"].values

    # ── Model A1: total_enc ──────────────────────────────────────────────
    model_a1 = lgb.LGBMRegressor(**p_a1)
    model_a1.fit(
        X_train, train_data["total_enc"],
        sample_weight=w_a1,
        eval_set=[(X_val, val_data["total_enc"])],
        categorical_feature=cat_features,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    # ── Model A2: admit_rate ─────────────────────────────────────────────
    model_a2 = lgb.LGBMRegressor(**p_a2)
    model_a2.fit(
        X_train, train_data["admit_rate"],
        sample_weight=w_a2,
        eval_set=[(X_val, val_data["admit_rate"])],
        categorical_feature=cat_features,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    # ── Predict (global model) ────────────────────────────────────────────
    pred_total = model_a1.predict(X_val).clip(0)
    pred_rate = model_a2.predict(X_val).clip(0, 1)

    # ── Enhancement B: Hybrid-residual for Site D ────────────────────────
    residual_feats = [f for f in RESIDUAL_FEATURES if f in feature_cols or f == "global_pred"]

    # Add global_pred as feature for residual model
    train_data_res = train_data.copy()
    train_data_res["global_pred"] = model_a1.predict(X_train).clip(0)
    val_data_res = val_data.copy()
    val_data_res["global_pred"] = pred_total

    # Determine which residual features exist
    res_feats_available = [f for f in residual_feats if f in train_data_res.columns]

    mask_d_train = train_data_res["site"] == "D"
    mask_d_val = val_data_res["site"] == "D"
    residual_model = None

    if mask_d_train.sum() > 100 and len(res_feats_available) >= 5:
        # Compute training residuals for Site D
        y_train_d = train_data_res.loc[mask_d_train, "total_enc"].values
        pred_train_d = train_data_res.loc[mask_d_train, "global_pred"].values
        residuals_d = y_train_d - pred_train_d

        X_res_train = train_data_res.loc[mask_d_train, res_feats_available]
        X_res_val_d = val_data_res.loc[mask_d_val, res_feats_available]

        # Inner validation: last 20% of Site D training data
        n_inner = max(int(len(X_res_train) * 0.2), 30)
        X_res_tr = X_res_train.iloc[:-n_inner]
        X_res_iv = X_res_train.iloc[-n_inner:]
        y_res_tr = residuals_d[:-n_inner]
        y_res_iv = residuals_d[-n_inner:]

        residual_model = lgb.LGBMRegressor(**RESIDUAL_LGBM_PARAMS)
        residual_model.fit(
            X_res_tr, y_res_tr,
            eval_set=[(X_res_iv, y_res_iv)],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )

        # Correct Site D total predictions
        correction = SHRINKAGE_WEIGHT * residual_model.predict(X_res_val_d)
        pred_total[mask_d_val.values] += correction

    pred_total = pred_total.clip(0)
    pred_admitted = pred_total * pred_rate

    # ── Enhancement C: Zero-inflation classifier for Site D sparse blocks ─
    sparse_blocks = [0, 1, 3]
    zero_clf = None
    zero_feat_cols = ["block", "dow", "month"]
    # Add te_site_block_mean_admitted if available
    if "te_site_block_mean_admitted" in train_data.columns:
        zero_feat_cols.append("te_site_block_mean_admitted")

    mask_sparse_train = (
        (train_data["site"] == "D") & (train_data["block"].isin(sparse_blocks))
    )
    if mask_sparse_train.sum() > 50:
        zero_target = (train_data.loc[mask_sparse_train, "admitted_enc"] == 0).astype(int)
        zero_X = train_data.loc[mask_sparse_train, zero_feat_cols].fillna(0)

        if zero_target.mean() > 0.02:  # Only train if meaningful zero rate
            zero_clf = LogisticRegression(C=1.0, max_iter=500)
            zero_clf.fit(zero_X, zero_target)

            # Apply to Site D sparse blocks in validation
            mask_sparse_val = (
                (val_data["site"] == "D") & (val_data["block"].isin(sparse_blocks))
            )
            if mask_sparse_val.sum() > 0:
                X_zero_val = val_data.loc[mask_sparse_val, zero_feat_cols].fillna(0)
                p_zero = zero_clf.predict_proba(X_zero_val)[:, 1]
                pred_admitted[mask_sparse_val.values] *= (1 - p_zero * ZERO_SHRINKAGE)

    pred_admitted = pred_admitted.clip(0)

    # ── Enhancement C: Admit-rate guardrails per (site, block) ────────────
    val_out = val_data[["site", "date", "block"]].copy()
    val_out["pred_total"] = pred_total
    val_out["pred_admitted"] = pred_admitted

    for (site, block), (lo, hi) in ADMIT_RATE_BOUNDS.items():
        mask = (val_out["site"] == site) & (val_out["block"] == block)
        if mask.sum() == 0:
            continue
        safe_total = val_out.loc[mask, "pred_total"].clip(lower=1)
        raw_rate = val_out.loc[mask, "pred_admitted"] / safe_total
        clamped_rate = raw_rate.clip(lo, hi)
        val_out.loc[mask, "pred_admitted"] = (
            val_out.loc[mask, "pred_total"] * clamped_rate
        )

    # ── Post-process: largest-remainder rounding per (Site, Date) ────────
    for (_s, _d), grp in val_out.groupby(["site", "date"]):
        idx = grp.index
        val_out.loc[idx, "pred_total"] = largest_remainder_round(grp["pred_total"].values)
        val_out.loc[idx, "pred_admitted"] = largest_remainder_round(grp["pred_admitted"].values)

    # Enforce admitted <= total
    val_out["pred_total"] = val_out["pred_total"].astype(int)
    val_out["pred_admitted"] = np.minimum(
        val_out["pred_admitted"].astype(int), val_out["pred_total"]
    )

    # ── Format as eval.md submission ─────────────────────────────────────
    submission = pd.DataFrame({
        "Site": val_out["site"].values,
        "Date": val_out["date"].dt.strftime("%Y-%m-%d").values,
        "Block": val_out["block"].values,
        "ED Enc": val_out["pred_total"].values,
        "ED Enc Admitted": val_out["pred_admitted"].values,
    })

    # ── Persist ──────────────────────────────────────────────────────────
    if save:
        cfg.ensure_dirs()
        submission.to_csv(cfg.OUTPUT_DIR / f"fold_{fold_id}_predictions.csv", index=False)
        model_a1.booster_.save_model(str(cfg.MODEL_DIR / f"fold_{fold_id}_model_a1.txt"))
        model_a2.booster_.save_model(str(cfg.MODEL_DIR / f"fold_{fold_id}_model_a2.txt"))
        if residual_model is not None:
            residual_model.booster_.save_model(
                str(cfg.MODEL_DIR / f"fold_{fold_id}_residual_d.txt")
            )

    # ── Quick metrics ────────────────────────────────────────────────────
    total_wape = wape(val_data["total_enc"].values, val_out["pred_total"].values)
    admitted_wape = wape(val_data["admitted_enc"].values, val_out["pred_admitted"].values)

    print(
        f"    Fold {fold_id}: total_wape={total_wape:.4f}  admitted_wape={admitted_wape:.4f}"
        f"  (A1 iters={model_a1.best_iteration_}, A2 iters={model_a2.best_iteration_})"
    )

    return {
        "fold_id": fold_id,
        "total_wape": total_wape,
        "admitted_wape": admitted_wape,
        "model_a1": model_a1,
        "model_a2": model_a2,
        "residual_model": residual_model,
        "zero_clf": zero_clf,
        "submission": submission,
        "feature_cols": feature_cols,
    }


# ── All folds ────────────────────────────────────────────────────────────────

def train_all_folds(
    df: pd.DataFrame,
    params_a1: dict | None = None,
    params_a2: dict | None = None,
    *,
    save: bool = True,
    covid_policy: str = "downweight",
) -> list[dict]:
    """Train A1+A2 across all folds, print summary, save feature importance."""
    results = []
    for fold in cfg.FOLDS:
        result = train_fold(df, fold, params_a1, params_a2,
                            save=save, covid_policy=covid_policy)
        results.append(result)

    # ── Summary ──────────────────────────────────────────────────────────
    mean_t = np.mean([r["total_wape"] for r in results])
    mean_a = np.mean([r["admitted_wape"] for r in results])
    print(f"\n  4-fold mean: total_wape={mean_t:.4f}  admitted_wape={mean_a:.4f}")

    # ── Feature importance (average across folds) ────────────────────────
    if save and results:
        fi_frames = []
        for r in results:
            m = r["model_a1"]
            fi = pd.Series(m.feature_importances_, index=r["feature_cols"], name=f"fold_{r['fold_id']}")
            fi_frames.append(fi)
        fi_df = pd.concat(fi_frames, axis=1)
        fi_df["mean"] = fi_df.mean(axis=1)
        fi_df = fi_df.sort_values("mean", ascending=False)
        cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fi_df.to_csv(cfg.OUTPUT_DIR / "feature_importance.csv")
        print(f"\n  Top 15 features (by mean A1 gain):")
        for feat, row in fi_df.head(15).iterrows():
            print(f"    {feat:45s} {row['mean']:,.0f}")

    return results


if __name__ == "__main__":
    from step_01_data_loading import load_data
    from step_02_feature_eng import engineer_features

    df = load_data()
    df = engineer_features(df)
    train_all_folds(df)
