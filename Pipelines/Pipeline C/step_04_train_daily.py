"""
Step 4: Train daily-level LightGBM models per fold.

Model C1_total  — daily total_enc  (Tweedie objective, volume-weighted)
Model C1_rate   — daily admit_rate  (MSE regression, admitted-weighted)

Produces daily-level predictions for each validation fold + saved models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

import config as cfg
from step_02_feature_eng_daily import (
    compute_daily_aggregate_encodings,
    get_daily_feature_columns,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else float("nan")


# ── Single-fold training ────────────────────────────────────────────────────

def train_daily_fold(
    daily_df: pd.DataFrame,
    fold: dict,
    params_total: dict | None = None,
    params_rate: dict | None = None,
    *,
    save: bool = True,
    covid_policy: str = "downweight",
) -> dict:
    """Train C1_total + C1_rate for one fold, predict on val (daily level)."""
    p_total = (params_total or cfg.LGBM_DAILY_TOTAL).copy()
    p_rate = (params_rate or cfg.LGBM_DAILY_RATE).copy()

    fold_id = fold["id"]
    train_end = pd.Timestamp(fold["train_end"])
    val_start = pd.Timestamp(fold["val_start"])
    val_end = pd.Timestamp(fold["val_end"])

    # ── Split ────────────────────────────────────────────────────────────
    train_mask = daily_df["date"] <= train_end
    val_mask = (daily_df["date"] >= val_start) & (daily_df["date"] <= val_end)

    df_fold = compute_daily_aggregate_encodings(daily_df, train_mask)
    feature_cols = get_daily_feature_columns(df_fold)
    cat_features = [c for c in ["site_enc"] if c in feature_cols]

    train_data = df_fold.loc[train_mask].copy()
    val_data = df_fold.loc[val_mask].copy()

    # COVID policy: filter out COVID-era rows (ensure boolean mask)
    if covid_policy == "exclude":
        mask_non_covid = ~train_data["is_covid_era"].astype(bool)
        train_data = train_data[mask_non_covid].copy()

    # Drop burn-in rows (longest lag NaN)
    longest_lag = f"lag_{cfg.LAG_DAYS_DAILY[-1]}"
    if longest_lag in train_data.columns:
        train_data = train_data.dropna(subset=[longest_lag])

    X_train = train_data[feature_cols]
    X_val = val_data[feature_cols]

    # ── Weights ──────────────────────────────────────────────────────────
    if covid_policy == "exclude":
        w_total = train_data["volume_weight"].values
        w_rate = train_data["admitted_enc"].clip(lower=1).values.astype(float)
    else:
        w_total = train_data["sample_weight_total"].values
        w_rate = train_data["sample_weight_rate"].values

    # ── Model C1_total: daily total_enc ──────────────────────────────────
    model_total = lgb.LGBMRegressor(**p_total)
    model_total.fit(
        X_train, train_data["total_enc"],
        sample_weight=w_total,
        eval_set=[(X_val, val_data["total_enc"])],
        categorical_feature=cat_features,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    # ── Model C1_rate: daily admit_rate ──────────────────────────────────
    model_rate = lgb.LGBMRegressor(**p_rate)
    model_rate.fit(
        X_train, train_data["admit_rate"],
        sample_weight=w_rate,
        eval_set=[(X_val, val_data["admit_rate"])],
        categorical_feature=cat_features,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    # ── Predict (daily level) ────────────────────────────────────────────
    pred_daily_total = model_total.predict(X_val).clip(0)
    pred_daily_rate = model_rate.predict(X_val).clip(0, 1)
    pred_daily_admitted = (pred_daily_total * pred_daily_rate).clip(0)

    val_out = val_data[["site", "date"]].copy()
    val_out["pred_daily_total"] = pred_daily_total
    val_out["pred_daily_rate"] = pred_daily_rate
    val_out["pred_daily_admitted"] = pred_daily_admitted

    # ── Daily-level WAPE (intermediate diagnostic) ───────────────────────
    daily_total_wape = wape(val_data["total_enc"].values, pred_daily_total)
    daily_admitted_wape = wape(val_data["admitted_enc"].values,
                              np.round(pred_daily_admitted).astype(int))

    print(f"    Fold {fold_id}: daily_total_wape={daily_total_wape:.4f}  "
          f"daily_admitted_wape={daily_admitted_wape:.4f}  "
          f"(C1_total iters={model_total.best_iteration_}, "
          f"C1_rate iters={model_rate.best_iteration_})")

    # ── Persist ──────────────────────────────────────────────────────────
    if save:
        cfg.ensure_dirs()
        val_out.to_parquet(cfg.DATA_DIR / f"fold_{fold_id}_daily_predictions.parquet",
                           index=False)
        model_total.booster_.save_model(
            str(cfg.MODEL_DIR / f"fold_{fold_id}_model_c1_total.txt"))
        model_rate.booster_.save_model(
            str(cfg.MODEL_DIR / f"fold_{fold_id}_model_c1_rate.txt"))

    return {
        "fold_id": fold_id,
        "daily_total_wape": daily_total_wape,
        "daily_admitted_wape": daily_admitted_wape,
        "model_total": model_total,
        "model_rate": model_rate,
        "daily_preds": val_out,
        "feature_cols": feature_cols,
    }


# ── All folds ────────────────────────────────────────────────────────────────

def train_all_daily_folds(
    daily_df: pd.DataFrame,
    params_total: dict | None = None,
    params_rate: dict | None = None,
    *,
    save: bool = True,
    covid_policy: str = "downweight",
) -> list[dict]:
    """Train daily models across all folds, print summary."""
    results = []
    for fold in cfg.FOLDS:
        result = train_daily_fold(daily_df, fold, params_total, params_rate,
                                  save=save, covid_policy=covid_policy)
        results.append(result)

    mean_tw = np.mean([r["daily_total_wape"] for r in results])
    mean_aw = np.mean([r["daily_admitted_wape"] for r in results])
    print(f"\n  Daily 4-fold mean: total_wape={mean_tw:.4f}  admitted_wape={mean_aw:.4f}")

    # Feature importance (average across folds)
    if save and results:
        fi_frames = []
        for r in results:
            m = r["model_total"]
            fi = pd.Series(m.feature_importances_, index=r["feature_cols"],
                           name=f"fold_{r['fold_id']}")
            fi_frames.append(fi)
        fi_df = pd.concat(fi_frames, axis=1)
        fi_df["mean"] = fi_df.mean(axis=1)
        fi_df = fi_df.sort_values("mean", ascending=False)
        cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fi_df.to_csv(cfg.OUTPUT_DIR / "daily_feature_importance.csv")
        print(f"\n  Top 15 daily features (by mean C1_total gain):")
        for feat, row in fi_df.head(15).iterrows():
            print(f"    {feat:45s} {row['mean']:,.0f}")

    return results


if __name__ == "__main__":
    from step_01_data_loading import load_data
    from step_02_feature_eng_daily import engineer_daily_features

    _, daily_df, _ = load_data()
    daily_df = engineer_daily_features(daily_df)
    train_all_daily_folds(daily_df)
