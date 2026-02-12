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
from step_02_feature_eng import (
    compute_fold_aggregate_encodings,
    get_feature_columns,
)


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

    # ── Predict ──────────────────────────────────────────────────────────
    pred_total = model_a1.predict(X_val).clip(0)
    pred_rate = model_a2.predict(X_val).clip(0, 1)
    pred_admitted = pred_total * pred_rate

    # ── Post-process: largest-remainder rounding per (Site, Date) ────────
    val_out = val_data[["site", "date", "block"]].copy()
    val_out["pred_total"] = pred_total
    val_out["pred_admitted"] = pred_admitted

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
