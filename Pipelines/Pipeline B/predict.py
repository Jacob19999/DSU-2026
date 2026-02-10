"""
Prediction & post-processing for Pipeline B.

Handles:
  - Final Sept-Oct 2025 forecast (competition submission)
  - Constraint enforcement (non-negativity, admitted ≤ total, integer rounding)
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
from training import train_bucket, largest_remainder_round, wape, CAT_FEATURES


def generate_final_forecast(
    base_df: pd.DataFrame,
    bucket_data_map: dict[int, pd.DataFrame],
    all_params: dict | None = None,
) -> pd.DataFrame:
    """Train on all data ≤ 2025-08-31, predict Sept-Oct 2025 (976 rows).

    Uses last 2 months as early-stopping hold-out.
    """
    train_end = pd.Timestamp("2025-08-31")
    pred_start = pd.Timestamp("2025-09-01")
    pred_end   = pd.Timestamp("2025-10-31")
    es_start   = pd.Timestamp("2025-07-01")

    enc_maps, fallback = compute_fold_encodings(base_df, train_end)

    all_preds: list[pd.DataFrame] = []

    for bid in [1, 2, 3]:
        bucket = cfg.BUCKETS[bid]
        lags = cfg.BUCKET_LAGS[bid]
        max_lag_col = f"lag_total_{max(lags)}"
        params = (all_params or {}).get(bid, {})
        p_total = params.get("total")
        p_rate  = params.get("rate")

        # ── Prepare training data ────────────────────────────────────────
        bucket_data = bucket_data_map[bid]
        train_all = bucket_data[bucket_data["date"] <= train_end].copy()
        train_all = apply_fold_encodings(train_all, enc_maps, fallback)
        train_all = train_all.dropna(subset=[max_lag_col])

        feature_cols = get_feature_columns(train_all)

        train_fit = train_all[train_all["date"] < es_start].copy()
        train_es  = train_all[train_all["date"] >= es_start].copy()

        if len(train_es) < 50:
            n_es = max(int(len(train_all) * 0.1), 50)
            train_fit = train_all.iloc[:-n_es].copy()
            train_es  = train_all.iloc[-n_es:].copy()

        # ── Train ────────────────────────────────────────────────────────
        model_total, model_rate = train_bucket(
            train_fit, train_es, feature_cols, p_total, p_rate,
        )

        # ── Build prediction features ────────────────────────────────────
        val_dates = pd.date_range(pred_start, pred_end)
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
        raw_total    = model_total.predict(X_pred).clip(0)
        raw_rate     = model_rate.predict(X_pred).clip(0, 1)
        raw_admitted = raw_total * raw_rate

        preds = pred_data[["site", "date", "block"]].copy()
        preds["pred_total"]    = raw_total
        preds["pred_admitted"] = raw_admitted
        all_preds.append(preds)

        # Save final models
        cfg.ensure_dirs()
        model_total.booster_.save_model(str(cfg.MODEL_DIR / f"final_bucket{bid}_total.txt"))
        model_rate.booster_.save_model(str(cfg.MODEL_DIR / f"final_bucket{bid}_rate.txt"))

        print(f"    Bucket {bid}: {len(preds)} prediction rows  "
              f"(total iters={model_total.best_iteration_}, "
              f"rate iters={model_rate.best_iteration_})")

    # ── Assemble all buckets ─────────────────────────────────────────────
    if not all_preds:
        print("  WARNING: No prediction data generated!")
        return pd.DataFrame()

    result = pd.concat(all_preds, ignore_index=True)

    # ── Post-process: largest-remainder rounding ─────────────────────────
    for (_s, _d), grp in result.groupby(["site", "date"]):
        idx = grp.index
        result.loc[idx, "pred_total"]    = largest_remainder_round(grp["pred_total"].values)
        result.loc[idx, "pred_admitted"] = largest_remainder_round(grp["pred_admitted"].values)

    result["pred_total"]    = result["pred_total"].astype(int)
    result["pred_admitted"] = np.minimum(
        result["pred_admitted"].astype(int), result["pred_total"]
    )

    # ── Format submission ────────────────────────────────────────────────
    submission = pd.DataFrame({
        "Site":            result["site"].values,
        "Date":            result["date"].dt.strftime("%Y-%m-%d").values,
        "Block":           result["block"].values,
        "ED Enc":          result["pred_total"].values,
        "ED Enc Admitted": result["pred_admitted"].values,
    })

    out_path = cfg.PRED_DIR / "final_sept_oct_2025.csv"
    submission.to_csv(out_path, index=False)
    print(f"  Saved {len(submission)} rows -> {out_path}")

    # ── Sanity checks ────────────────────────────────────────────────────
    assert (submission["ED Enc"] >= 0).all(), "Negative ED Enc"
    assert (submission["ED Enc Admitted"] >= submission["ED Enc"]).sum() == 0 or \
           (submission["ED Enc Admitted"] <= submission["ED Enc"]).all(), "admitted > total"
    expected_rows = len(cfg.SITES) * len(pd.date_range(pred_start, pred_end)) * len(cfg.BLOCKS)
    assert len(submission) == expected_rows, (
        f"Row count {len(submission)} != expected {expected_rows}"
    )

    return submission


if __name__ == "__main__":
    from data_loader import load_data
    from features import add_static_features

    df = load_data()
    df = add_static_features(df)

    bucket_data_map = {}
    for bid in [1, 2, 3]:
        print(f"  Building bucket {bid} data ...")
        bucket_data_map[bid] = build_bucket_data(df, bid)

    generate_final_forecast(df, bucket_data_map)
