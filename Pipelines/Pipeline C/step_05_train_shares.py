"""
Step 5: Train block-share models per fold.

Supports two approaches (selected via config.SHARE_MODEL_TYPE):
  - "softmax_gbdt" : LightGBM multiclass (num_class=4) predicting block index
  - "climatology"  : Historical mean shares by (site, dow, month)

Produces predicted block shares for each validation fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

import config as cfg
from step_03_feature_eng_shares import (
    compute_climatology,
    get_share_feature_columns,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _predict_climatology(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    keys: list[str] | None = None,
) -> np.ndarray:
    """Predict shares using climatology fallback. Returns (N, 4) array."""
    if keys is None:
        keys = cfg.CLIMATOLOGY_KEYS

    clim = compute_climatology(train_df, keys)
    share_cols = [f"clim_share_b{b}" for b in cfg.BLOCKS]

    merged = val_df.merge(clim, on=keys, how="left")

    # Fallback for unseen combos: uniform shares
    for sc in share_cols:
        merged[sc] = merged[sc].fillna(1.0 / cfg.N_BLOCKS)

    shares = merged[share_cols].values
    # Renormalize
    shares = shares / shares.sum(axis=1, keepdims=True)
    return shares


# ── Single-fold training ────────────────────────────────────────────────────

def train_share_fold(
    share_df: pd.DataFrame,
    fold: dict,
    params_share: dict | None = None,
    *,
    share_type: str | None = None,
    save: bool = True,
) -> dict:
    """Train share model for one fold. Returns dict with share predictions."""
    share_type = share_type or cfg.SHARE_MODEL_TYPE
    p_share = (params_share or cfg.LGBM_SHARE).copy()

    fold_id = fold["id"]
    train_end = pd.Timestamp(fold["train_end"])
    val_start = pd.Timestamp(fold["val_start"])
    val_end = pd.Timestamp(fold["val_end"])

    train_mask = share_df["date"] <= train_end
    val_mask = (share_df["date"] >= val_start) & (share_df["date"] <= val_end)

    train_data = share_df.loc[train_mask].copy()
    val_data = share_df.loc[val_mask].copy()

    feature_cols = get_share_feature_columns(share_df)
    cat_features = [c for c in ["site_enc"] if c in feature_cols]

    model_total = None
    model_admitted = None
    pred_shares_total: np.ndarray
    pred_shares_admitted: np.ndarray

    if share_type == "softmax_gbdt":
        # Drop burn-in rows with NaN share lags
        longest_share_lag = f"share_lag_{cfg.LAG_DAYS_SHARES[-1]}"
        if longest_share_lag in train_data.columns:
            train_data = train_data.dropna(subset=[longest_share_lag])

        X_train = train_data[feature_cols]
        X_val = val_data[feature_cols]

        # ── Total encounter share model ──────────────────────────────────
        model_total = lgb.LGBMClassifier(**p_share)
        model_total.fit(
            X_train, train_data["block"],
            sample_weight=train_data["total_enc"].clip(lower=1).values.astype(float),
            eval_set=[(X_val, val_data["block"])],
            categorical_feature=cat_features,
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )

        pred_shares_total = model_total.predict_proba(X_val)
        # Normalize safety
        pred_shares_total = pred_shares_total / pred_shares_total.sum(axis=1, keepdims=True)

        # ── Admitted encounter share model ───────────────────────────────
        # Train separate model weighted by admitted_enc
        model_admitted = lgb.LGBMClassifier(**p_share)
        model_admitted.fit(
            X_train, train_data["block"],
            sample_weight=train_data["admitted_enc"].clip(lower=1).values.astype(float),
            eval_set=[(X_val, val_data["block"])],
            categorical_feature=cat_features,
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
        )

        pred_shares_admitted = model_admitted.predict_proba(X_val)
        pred_shares_admitted = pred_shares_admitted / pred_shares_admitted.sum(axis=1, keepdims=True)

        print(f"    Fold {fold_id} [softmax_gbdt]: share_total iters={model_total.best_iteration_}, "
              f"share_admitted iters={model_admitted.best_iteration_}")

    elif share_type == "climatology":
        # Use climatology for both total and admitted shares
        pred_shares_total = _predict_climatology(train_data, val_data)
        pred_shares_admitted = _predict_climatology(
            train_data, val_data
        )  # Same keys; admitted share is close enough
        print(f"    Fold {fold_id} [climatology]: using historical mean shares")

    else:
        raise ValueError(f"Unknown share_type: {share_type}")

    # ── Build output DataFrame ───────────────────────────────────────────
    # val_data has block-level rows; we need to assign predicted shares per (site, date, block)
    # pred_shares is ordered by val_data index, shape (N_val_rows, 4)
    # Each row in val_data is (site, date, block) — the shares need to map block column → share column
    val_out = val_data[["site", "date", "block"]].copy()

    # Map: for each row, pick the share corresponding to that row's block
    for b in cfg.BLOCKS:
        val_out[f"pred_share_total_b{b}"] = pred_shares_total[:, b]
        val_out[f"pred_share_admitted_b{b}"] = pred_shares_admitted[:, b]

    # ── Persist ──────────────────────────────────────────────────────────
    if save:
        cfg.ensure_dirs()
        val_out.to_parquet(cfg.DATA_DIR / f"fold_{fold_id}_share_predictions.parquet",
                           index=False)
        if model_total is not None:
            model_total.booster_.save_model(
                str(cfg.MODEL_DIR / f"fold_{fold_id}_model_share_total.txt"))
        if model_admitted is not None:
            model_admitted.booster_.save_model(
                str(cfg.MODEL_DIR / f"fold_{fold_id}_model_share_admitted.txt"))

    return {
        "fold_id": fold_id,
        "share_type": share_type,
        "model_total": model_total,
        "model_admitted": model_admitted,
        "share_preds": val_out,
        "feature_cols": feature_cols,
    }


# ── All folds ────────────────────────────────────────────────────────────────

def train_all_share_folds(
    share_df: pd.DataFrame,
    params_share: dict | None = None,
    *,
    share_type: str | None = None,
    save: bool = True,
) -> list[dict]:
    """Train share models across all folds."""
    results = []
    for fold in cfg.FOLDS:
        result = train_share_fold(share_df, fold, params_share,
                                  share_type=share_type, save=save)
        results.append(result)

    # Feature importance for softmax models
    st = share_type or cfg.SHARE_MODEL_TYPE
    if save and st == "softmax_gbdt" and results:
        fi_frames = []
        for r in results:
            m = r["model_total"]
            if m is not None:
                fi = pd.Series(m.feature_importances_, index=r["feature_cols"],
                               name=f"fold_{r['fold_id']}")
                fi_frames.append(fi)
        if fi_frames:
            fi_df = pd.concat(fi_frames, axis=1)
            fi_df["mean"] = fi_df.mean(axis=1)
            fi_df = fi_df.sort_values("mean", ascending=False)
            cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            fi_df.to_csv(cfg.OUTPUT_DIR / "share_feature_importance.csv")
            print(f"\n  Top 10 share features (by mean gain):")
            for feat, row in fi_df.head(10).iterrows():
                print(f"    {feat:45s} {row['mean']:,.0f}")

    return results


if __name__ == "__main__":
    from step_01_data_loading import load_data
    from step_03_feature_eng_shares import engineer_share_features

    block_df, _, _ = load_data()
    share_df = engineer_share_features(block_df)
    train_all_share_folds(share_df)
