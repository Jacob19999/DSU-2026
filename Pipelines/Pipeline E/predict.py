"""
Prediction & post-processing for Pipeline E.

Handles:
  - Final Sept-Oct 2025 forecast (competition submission)
  - Full retrain on all data ≤ 2025-08-31
  - Constraint enforcement (non-negativity, admitted ≤ total, integer rounding)
"""

from __future__ import annotations

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
from training import largest_remainder_round, wape, CAT_FEATURES


def generate_final_forecast(
    base_df: pd.DataFrame,
    share_cols: list[str],
    params_total: dict | None = None,
    params_rate: dict | None = None,
) -> pd.DataFrame:
    """Train on all data ≤ 2025-08-31, predict Sept-Oct 2025 (976 rows).

    Uses Jul-Aug 2025 as early-stopping hold-out.
    """
    train_end  = pd.Timestamp("2025-08-31")
    pred_start = pd.Timestamp("2025-09-01")
    pred_end   = pd.Timestamp("2025-10-31")
    es_start   = pd.Timestamp("2025-07-01")

    p_total = (params_total or cfg.LGBM_DEFAULT_TOTAL).copy()
    p_rate  = (params_rate  or cfg.LGBM_DEFAULT_RATE).copy()

    print("  Running factor pipeline on full history ...")

    # ── Factor pipeline ──────────────────────────────────────────────────
    fold_df, factor_model, factor_cols = fit_and_transform_factors(
        base_df, share_cols, train_end,
    )
    fold_df = add_factor_lag_features(fold_df)
    factor_models = train_factor_forecast_models(fold_df, train_end)
    fold_df = set_predicted_factors(fold_df, factor_models, train_end)

    # Fold encodings (trained on everything ≤ train_end)
    enc_maps, fallback = compute_fold_encodings(fold_df, train_end)
    fold_df = apply_fold_encodings(fold_df, enc_maps, fallback)

    feature_cols = get_feature_columns(fold_df)

    # ── Prepare training data ────────────────────────────────────────────
    burn_in_col = f"lag_{max(cfg.LAG_DAYS)}"
    train_all = fold_df[fold_df["date"] <= train_end].dropna(subset=[burn_in_col]).copy()

    train_fit = train_all[train_all["date"] < es_start].copy()
    train_es  = train_all[train_all["date"] >= es_start].copy()

    if len(train_es) < 50:
        n_es = max(int(len(train_all) * 0.1), 50)
        train_fit = train_all.iloc[:-n_es]
        train_es  = train_all.iloc[-n_es:]

    print(f"  Train: {len(train_fit):,} fit + {len(train_es):,} ES  |  "
          f"Features: {len(feature_cols)}")

    # ── Train total_enc ──────────────────────────────────────────────────
    model_total = lgb.LGBMRegressor(**p_total)
    model_total.fit(
        train_fit[feature_cols], train_fit["total_enc"],
        sample_weight=train_fit["sample_weight"].values,
        eval_set=[(train_es[feature_cols], train_es["total_enc"])],
        categorical_feature=CAT_FEATURES,
        callbacks=[
            lgb.early_stopping(cfg.ES_PATIENCE, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    # ── Train admit_rate ─────────────────────────────────────────────────
    model_rate = lgb.LGBMRegressor(**p_rate)
    model_rate.fit(
        train_fit[feature_cols], train_fit["admit_rate"],
        sample_weight=train_fit["sample_weight_rate"].values,
        eval_set=[(train_es[feature_cols], train_es["admit_rate"])],
        categorical_feature=CAT_FEATURES,
        callbacks=[
            lgb.early_stopping(cfg.ES_PATIENCE, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    print(f"  total iters={model_total.best_iteration_}  "
          f"rate iters={model_rate.best_iteration_}")

    # ── Predict Sept-Oct 2025 ────────────────────────────────────────────
    pred_mask = (fold_df["date"] >= pred_start) & (fold_df["date"] <= pred_end)
    pred_df   = fold_df[pred_mask].copy()

    if len(pred_df) == 0:
        raise ValueError("No prediction rows — master grid may not cover Sept-Oct 2025")

    raw_total    = model_total.predict(pred_df[feature_cols]).clip(0)
    raw_rate     = model_rate.predict(pred_df[feature_cols]).clip(0, 1)
    raw_admitted = raw_total * raw_rate

    preds = pred_df[["site", "date", "block"]].copy()
    preds["pred_total"]    = raw_total
    preds["pred_admitted"] = raw_admitted

    # ── Post-process: largest-remainder rounding ─────────────────────────
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

    # ── Save ─────────────────────────────────────────────────────────────
    cfg.ensure_dirs()
    out_path = cfg.PRED_DIR / "final_sept_oct_2025.csv"
    submission.to_csv(out_path, index=False)

    model_total.booster_.save_model(str(cfg.MODEL_DIR / "final_model_e1_total.txt"))
    model_rate.booster_.save_model(str(cfg.MODEL_DIR / "final_model_e2_rate.txt"))

    print(f"  Saved {len(submission)} rows -> {out_path}")

    # ── Sanity checks ────────────────────────────────────────────────────
    expected = len(cfg.SITES) * len(pd.date_range(pred_start, pred_end)) * len(cfg.BLOCKS)
    assert len(submission) == expected, (
        f"Row count {len(submission)} != expected {expected}"
    )
    assert (submission["ED Enc"] >= 0).all(), "Negative ED Enc"
    assert (submission["ED Enc Admitted"] <= submission["ED Enc"]).all(), "admitted > total"
    assert submission.duplicated(subset=["Site", "Date", "Block"]).sum() == 0, "Duplicates"

    # Summary
    print(f"  Mean ED Enc: {submission['ED Enc'].mean():.1f}  "
          f"Mean Admitted: {submission['ED Enc Admitted'].mean():.1f}")
    for site in cfg.SITES:
        s = submission[submission["Site"] == site]
        print(f"    Site {site}: mean total={s['ED Enc'].mean():.1f}  "
              f"mean admitted={s['ED Enc Admitted'].mean():.1f}")

    return submission


if __name__ == "__main__":
    from data_loader import load_data
    from share_matrix import build_share_matrix
    from features import add_all_base_features

    df = load_data()
    df, share_cols = build_share_matrix(df)
    df = add_all_base_features(df)
    generate_final_forecast(df, share_cols)
