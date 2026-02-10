"""
Step 5: Generate final Sept-Oct 2025 forecast (competition submission).

Trains on ALL data <= 2025-08-31 with best (or default) params,
predicts 2025-09-01 → 2025-10-31 = 976 rows.
Uses last 2 months of training as early-stopping hold-out.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import lightgbm as lgb

import config as cfg
from step_02_feature_eng import (
    compute_fold_aggregate_encodings,
    get_feature_columns,
)
from step_03_train import largest_remainder_round, wape


def generate_final_forecast(
    df: pd.DataFrame,
    params_a1: dict | None = None,
    params_a2: dict | None = None,
    covid_policy: str = "downweight",
) -> pd.DataFrame:
    """Train on data ≤ 2025-08-31, predict Sept-Oct 2025."""

    # Try loading tuned params from disk if not supplied
    if params_a1 is None:
        p = cfg.MODEL_DIR / "best_params_a1.json"
        if p.exists():
            with open(p) as f:
                params_a1 = json.load(f)
            print(f"  Loaded tuned A1 params from {p}")
    if params_a2 is None:
        p = cfg.MODEL_DIR / "best_params_a2.json"
        if p.exists():
            with open(p) as f:
                params_a2 = json.load(f)
            print(f"  Loaded tuned A2 params from {p}")

    p_a1 = (params_a1 or cfg.LGBM_DEFAULT_A1).copy()
    p_a2 = (params_a2 or cfg.LGBM_DEFAULT_A2).copy()

    train_end = pd.Timestamp("2025-08-31")
    pred_start = pd.Timestamp("2025-09-01")
    pred_end = pd.Timestamp("2025-10-31")
    es_start = pd.Timestamp("2025-07-01")     # early-stopping hold-out

    train_mask = df["date"] <= train_end
    pred_mask = (df["date"] >= pred_start) & (df["date"] <= pred_end)

    df_fold = compute_fold_aggregate_encodings(df, train_mask)
    feature_cols = get_feature_columns(df_fold)
    cat_features = ["site_enc", "block", "site_x_dow", "site_x_month"]

    # Split training into fit + early-stopping hold-out
    train_fit = df_fold.loc[train_mask & (df["date"] < es_start)].copy()
    train_es = df_fold.loc[train_mask & (df["date"] >= es_start)].copy()
    pred_data = df_fold.loc[pred_mask].copy()

    if covid_policy == "exclude":
        train_fit = train_fit[~train_fit["is_covid_era"]].copy()

    train_fit = train_fit.dropna(subset=[f"lag_{cfg.LAG_DAYS[-1]}"])

    X_fit = train_fit[feature_cols]
    X_es = train_es[feature_cols]
    X_pred = pred_data[feature_cols]

    # Weights
    if covid_policy == "exclude":
        w_a1 = train_fit["volume_weight"].values
        w_a2 = train_fit["admitted_enc"].clip(lower=1).values.astype(float)
    else:
        w_a1 = train_fit["sample_weight_a1"].values
        w_a2 = train_fit["sample_weight_a2"].values

    # ── A1 ───────────────────────────────────────────────────────────────
    model_a1 = lgb.LGBMRegressor(**p_a1)
    model_a1.fit(
        X_fit, train_fit["total_enc"],
        sample_weight=w_a1,
        eval_set=[(X_es, train_es["total_enc"])],
        categorical_feature=cat_features,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    # ── A2 ───────────────────────────────────────────────────────────────
    model_a2 = lgb.LGBMRegressor(**p_a2)
    model_a2.fit(
        X_fit, train_fit["admit_rate"],
        sample_weight=w_a2,
        eval_set=[(X_es, train_es["admit_rate"])],
        categorical_feature=cat_features,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    # ── Predict ──────────────────────────────────────────────────────────
    raw_total = model_a1.predict(X_pred).clip(0)
    raw_rate = model_a2.predict(X_pred).clip(0, 1)
    raw_admitted = raw_total * raw_rate

    result = pred_data[["site", "date", "block"]].copy()
    result["pred_total"] = raw_total
    result["pred_admitted"] = raw_admitted

    for (_s, _d), grp in result.groupby(["site", "date"]):
        idx = grp.index
        result.loc[idx, "pred_total"] = largest_remainder_round(grp["pred_total"].values)
        result.loc[idx, "pred_admitted"] = largest_remainder_round(grp["pred_admitted"].values)

    result["pred_total"] = result["pred_total"].astype(int)
    result["pred_admitted"] = np.minimum(
        result["pred_admitted"].astype(int), result["pred_total"]
    )

    # ── Submission CSV ───────────────────────────────────────────────────
    submission = pd.DataFrame({
        "Site": result["site"].values,
        "Date": result["date"].dt.strftime("%Y-%m-%d").values,
        "Block": result["block"].values,
        "ED Enc": result["pred_total"].values,
        "ED Enc Admitted": result["pred_admitted"].values,
    })

    cfg.ensure_dirs()
    out_path = cfg.OUTPUT_DIR / "final_sept_oct_2025.csv"
    submission.to_csv(out_path, index=False)
    print(f"  Saved {len(submission)} rows -> {out_path}")
    print(f"  A1 iters={model_a1.best_iteration_}, A2 iters={model_a2.best_iteration_}")

    # Save final models
    model_a1.booster_.save_model(str(cfg.MODEL_DIR / "final_model_a1.txt"))
    model_a2.booster_.save_model(str(cfg.MODEL_DIR / "final_model_a2.txt"))

    return submission


if __name__ == "__main__":
    from step_01_data_loading import load_data
    from step_02_feature_eng import engineer_features

    df = load_data()
    df = engineer_features(df)
    generate_final_forecast(df)
