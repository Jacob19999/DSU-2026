"""
Step 7: Prediction & Block Allocation.

Combines daily forecasts with predicted block shares to produce
final block-level predictions. Applies hard constraints, largest-
remainder rounding, and produces submission-format CSVs.
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import lightgbm as lgb

import config as cfg


# ── Largest-remainder rounding ───────────────────────────────────────────────

def largest_remainder_round(
    values: np.ndarray,
    target_sum: int | None = None,
) -> np.ndarray:
    """Round float array to non-negative integers preserving the sum."""
    values = np.maximum(values, 0)
    if target_sum is None:
        target_sum = int(round(np.sum(values)))

    floored = np.floor(values).astype(int)
    remainders = values - floored
    deficit = target_sum - floored.sum()

    if deficit > 0:
        indices = np.argsort(-remainders)[:deficit]
        floored[indices] += 1
    elif deficit < 0:
        indices = np.argsort(remainders)[:abs(deficit)]
        floored[indices] = np.maximum(floored[indices] - 1, 0)

    return floored


# ── Allocate daily to blocks ─────────────────────────────────────────────────

def allocate_daily_to_blocks(
    daily_preds: pd.DataFrame,
    share_preds: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine daily predictions with share predictions → block-level.

    daily_preds: (site, date, pred_daily_total, pred_daily_rate, pred_daily_admitted)
    share_preds: (site, date, block, pred_share_total_b0..b3, pred_share_admitted_b0..b3)

    Returns DataFrame with (site, date, block, pred_total, pred_admitted).
    """
    # Get unique (site, date) from share_preds
    share_cols_total = [f"pred_share_total_b{b}" for b in cfg.BLOCKS]
    share_cols_admitted = [f"pred_share_admitted_b{b}" for b in cfg.BLOCKS]

    # Merge daily preds with share preds on (site, date)
    # share_preds is block-level; pivot to get one row per (site, date) with 4 share columns
    share_daily = (
        share_preds
        .drop_duplicates(subset=["site", "date"])
        [["site", "date"] + share_cols_total + share_cols_admitted]
    )

    merged = daily_preds.merge(share_daily, on=["site", "date"], how="inner")

    results = []
    for _, row in merged.iterrows():
        site = row["site"]
        date = row["date"]
        daily_total = max(row["pred_daily_total"], 0)
        daily_admitted = max(row["pred_daily_admitted"], 0)

        shares_total = np.array([row[f"pred_share_total_b{b}"] for b in cfg.BLOCKS])
        shares_admitted = np.array([row[f"pred_share_admitted_b{b}"] for b in cfg.BLOCKS])

        # Normalize shares
        shares_total = np.maximum(shares_total, 0)
        s_sum = shares_total.sum()
        if s_sum > 0:
            shares_total = shares_total / s_sum
        else:
            shares_total = np.full(cfg.N_BLOCKS, 1.0 / cfg.N_BLOCKS)

        shares_admitted = np.maximum(shares_admitted, 0)
        s_sum_a = shares_admitted.sum()
        if s_sum_a > 0:
            shares_admitted = shares_admitted / s_sum_a
        else:
            shares_admitted = np.full(cfg.N_BLOCKS, 1.0 / cfg.N_BLOCKS)

        # Allocate
        block_total_raw = daily_total * shares_total
        block_total_int = largest_remainder_round(block_total_raw)

        block_admitted_raw = daily_admitted * shares_admitted
        block_admitted_int = largest_remainder_round(block_admitted_raw)

        # Enforce admitted <= total
        block_admitted_int = np.minimum(block_admitted_int, block_total_int)

        for b in cfg.BLOCKS:
            results.append({
                "site": site,
                "date": date,
                "block": b,
                "pred_total": int(block_total_int[b]),
                "pred_admitted": int(block_admitted_int[b]),
            })

    return pd.DataFrame(results)


# ── Per-fold prediction pipeline ─────────────────────────────────────────────

def predict_fold(
    daily_df: pd.DataFrame,
    share_df: pd.DataFrame,
    fold: dict,
    params_total: dict | None = None,
    params_rate: dict | None = None,
    params_share: dict | None = None,
    *,
    covid_policy: str = "downweight",
    share_type: str | None = None,
) -> pd.DataFrame:
    """Run full prediction pipeline for one fold: daily → share → allocate."""
    from step_04_train_daily import train_daily_fold
    from step_05_train_shares import train_share_fold

    fold_id = fold["id"]

    # Train & predict daily
    daily_res = train_daily_fold(
        daily_df, fold, params_total, params_rate,
        save=True, covid_policy=covid_policy,
    )

    # Train & predict shares
    share_res = train_share_fold(
        share_df, fold, params_share, share_type=share_type, save=True,
    )

    # Allocate
    block_preds = allocate_daily_to_blocks(daily_res["daily_preds"],
                                           share_res["share_preds"])

    # Format as submission CSV
    submission = pd.DataFrame({
        "Site": block_preds["site"].values,
        "Date": pd.to_datetime(block_preds["date"]).dt.strftime("%Y-%m-%d").values,
        "Block": block_preds["block"].values,
        "ED Enc": block_preds["pred_total"].values,
        "ED Enc Admitted": block_preds["pred_admitted"].values,
    })

    cfg.ensure_dirs()
    submission.to_csv(cfg.OUTPUT_DIR / f"fold_{fold_id}_predictions.csv", index=False)
    print(f"    Fold {fold_id}: saved {len(submission)} rows → output/fold_{fold_id}_predictions.csv")

    return submission


# ── All folds ────────────────────────────────────────────────────────────────

def predict_all_folds(
    daily_df: pd.DataFrame,
    share_df: pd.DataFrame,
    params_total: dict | None = None,
    params_rate: dict | None = None,
    params_share: dict | None = None,
    *,
    covid_policy: str = "downweight",
    share_type: str | None = None,
) -> list[pd.DataFrame]:
    """Run prediction for all folds, save OOF predictions."""
    all_subs = []
    for fold in cfg.FOLDS:
        sub = predict_fold(
            daily_df, share_df, fold,
            params_total, params_rate, params_share,
            covid_policy=covid_policy, share_type=share_type,
        )
        all_subs.append(sub)

    # Combine OOF
    oof = pd.concat(all_subs, ignore_index=True)
    oof.to_csv(cfg.OUTPUT_DIR / "oof_predictions.csv", index=False)
    print(f"\n  OOF predictions: {len(oof)} rows → output/oof_predictions.csv")

    return all_subs


# ── Final forecast (Sept-Oct 2025) ──────────────────────────────────────────

def generate_final_forecast(
    daily_df: pd.DataFrame,
    share_df: pd.DataFrame,
    params_total: dict | None = None,
    params_rate: dict | None = None,
    params_share: dict | None = None,
    *,
    covid_policy: str = "downweight",
    share_type: str | None = None,
) -> pd.DataFrame:
    """Train on data <= 2025-08-31, predict Sept-Oct 2025."""
    from step_02_feature_eng_daily import (
        compute_daily_aggregate_encodings,
        get_daily_feature_columns,
    )
    from step_03_feature_eng_shares import get_share_feature_columns

    # Try loading tuned params from disk
    if params_total is None:
        p = cfg.MODEL_DIR / "best_params_daily_total.json"
        if p.exists():
            with open(p) as f:
                params_total = json.load(f)
            print(f"  Loaded tuned daily params from {p}")

    if params_share is None:
        p = cfg.MODEL_DIR / "best_params_share.json"
        if p.exists():
            with open(p) as f:
                params_share = json.load(f)
            print(f"  Loaded tuned share params from {p}")

    p_total = (params_total or cfg.LGBM_DAILY_TOTAL).copy()
    p_rate = (params_rate or cfg.LGBM_DAILY_RATE).copy()
    p_share = (params_share or cfg.LGBM_SHARE).copy()

    train_end = pd.Timestamp("2025-08-31")
    pred_start = pd.Timestamp("2025-09-01")
    pred_end = pd.Timestamp("2025-10-31")
    es_start = pd.Timestamp("2025-07-01")

    # ── Daily model ──────────────────────────────────────────────────────
    train_mask = daily_df["date"] <= train_end
    pred_mask = (daily_df["date"] >= pred_start) & (daily_df["date"] <= pred_end)

    df_fold = compute_daily_aggregate_encodings(daily_df, train_mask)
    daily_feature_cols = get_daily_feature_columns(df_fold)
    cat_daily = [c for c in ["site_enc"] if c in daily_feature_cols]

    train_fit = df_fold.loc[train_mask & (daily_df["date"] < es_start)].copy()
    train_es = df_fold.loc[train_mask & (daily_df["date"] >= es_start)].copy()
    pred_data = df_fold.loc[pred_mask].copy()

    if covid_policy == "exclude":
        train_fit = train_fit[~train_fit["is_covid_era"]].copy()

    longest_lag = f"lag_{cfg.LAG_DAYS_DAILY[-1]}"
    if longest_lag in train_fit.columns:
        train_fit = train_fit.dropna(subset=[longest_lag])

    w_total = train_fit["sample_weight_total"].values
    w_rate = train_fit["sample_weight_rate"].values

    model_total = lgb.LGBMRegressor(**p_total)
    model_total.fit(
        train_fit[daily_feature_cols], train_fit["total_enc"],
        sample_weight=w_total,
        eval_set=[(train_es[daily_feature_cols], train_es["total_enc"])],
        categorical_feature=cat_daily,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    model_rate = lgb.LGBMRegressor(**p_rate)
    model_rate.fit(
        train_fit[daily_feature_cols], train_fit["admit_rate"],
        sample_weight=w_rate,
        eval_set=[(train_es[daily_feature_cols], train_es["admit_rate"])],
        categorical_feature=cat_daily,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    raw_total = model_total.predict(pred_data[daily_feature_cols]).clip(0)
    raw_rate = model_rate.predict(pred_data[daily_feature_cols]).clip(0, 1)

    daily_preds = pred_data[["site", "date"]].copy()
    daily_preds["pred_daily_total"] = raw_total
    daily_preds["pred_daily_rate"] = raw_rate
    daily_preds["pred_daily_admitted"] = raw_total * raw_rate

    # ── Share model ──────────────────────────────────────────────────────
    st = share_type or cfg.SHARE_MODEL_TYPE
    share_feature_cols = get_share_feature_columns(share_df)
    cat_share = [c for c in ["site_enc"] if c in share_feature_cols]

    share_train = share_df[share_df["date"] <= train_end].copy()
    share_pred = share_df[(share_df["date"] >= pred_start) & (share_df["date"] <= pred_end)].copy()

    longest_share_lag = f"share_lag_{cfg.LAG_DAYS_SHARES[-1]}"
    if longest_share_lag in share_train.columns:
        share_train = share_train.dropna(subset=[longest_share_lag])

    if st == "softmax_gbdt":
        share_model = lgb.LGBMClassifier(**p_share)
        share_model.fit(
            share_train[share_feature_cols], share_train["block"],
            sample_weight=share_train["total_enc"].clip(lower=1).values.astype(float),
            categorical_feature=cat_share,
            callbacks=[lgb.log_evaluation(0)],
        )
        pred_probs = share_model.predict_proba(share_pred[share_feature_cols])
        pred_probs = pred_probs / pred_probs.sum(axis=1, keepdims=True)
    else:
        # Climatology
        from step_05_train_shares import _predict_climatology
        pred_probs = _predict_climatology(share_train, share_pred)

    share_preds_out = share_pred[["site", "date", "block"]].copy()
    for b in cfg.BLOCKS:
        share_preds_out[f"pred_share_total_b{b}"] = pred_probs[:, b]
        share_preds_out[f"pred_share_admitted_b{b}"] = pred_probs[:, b]

    # ── Allocate ─────────────────────────────────────────────────────────
    block_preds = allocate_daily_to_blocks(daily_preds, share_preds_out)

    submission = pd.DataFrame({
        "Site": block_preds["site"].values,
        "Date": pd.to_datetime(block_preds["date"]).dt.strftime("%Y-%m-%d").values,
        "Block": block_preds["block"].values,
        "ED Enc": block_preds["pred_total"].values,
        "ED Enc Admitted": block_preds["pred_admitted"].values,
    })

    cfg.ensure_dirs()
    out_path = cfg.OUTPUT_DIR / "final_sept_oct_2025.csv"
    submission.to_csv(out_path, index=False)
    print(f"  Saved {len(submission)} rows → {out_path}")

    # Save final models
    model_total.booster_.save_model(str(cfg.MODEL_DIR / "final_model_c1_total.txt"))
    model_rate.booster_.save_model(str(cfg.MODEL_DIR / "final_model_c1_rate.txt"))

    return submission


if __name__ == "__main__":
    from step_01_data_loading import load_data
    from step_02_feature_eng_daily import engineer_daily_features
    from step_03_feature_eng_shares import engineer_share_features

    block_df, daily_df, _ = load_data()
    daily_df = engineer_daily_features(daily_df)
    share_df = engineer_share_features(block_df)
    predict_all_folds(daily_df, share_df)
