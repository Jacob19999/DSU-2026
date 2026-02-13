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
from transfer_learning import train_teacher_abc, train_student_d

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

    val_data = df_fold.loc[val_mask].copy()

    # train_data needed for Enhancement C (zero-inflation) — Site D sparse blocks
    train_data = df_fold.loc[train_mask].copy()
    if covid_policy == "exclude":
        if "is_covid_era" in train_data.columns:
            train_data = train_data[train_data["is_covid_era"] == 0].copy()
    train_data = train_data.dropna(subset=[f"lag_{cfg.LAG_DAYS[-1]}"])

    # ── Stage 1: ABC Teacher ────────────────────────────────────────────
    teacher_result = train_teacher_abc(
        df_fold, train_mask, val_mask,
        feature_cols=feature_cols,
        cat_features=cat_features,
        params_a1=p_a1, params_a2=p_a2,
        covid_policy=covid_policy,
    )
    model_t1 = teacher_result["models"]["t1"]
    model_t2 = teacher_result["models"]["t2"]

    # ABC val predictions
    abc_val_idx = teacher_result["abc_val_idx"]
    pred_total_abc = teacher_result["pred_total_abc"]
    pred_rate_abc = teacher_result["pred_rate_abc"]

    # ── Stage 2: Site D Student ──────────────────────────────────────────
    student_result = train_student_d(
        df_fold, train_mask, val_mask,
        teacher_models=teacher_result["models"],
        feature_cols_parent=feature_cols,
    )

    # ── Merge: ABC (teacher) + D (student) ───────────────────────────────
    mask_d_val = val_mask & (df_fold["site"] == "D")
    mask_abc_val = val_mask & df_fold["site"].isin(["A", "B", "C"])

    # Allocate full-length prediction arrays aligned to val_data index
    pred_total = np.empty(len(val_data), dtype=float)
    pred_rate = np.empty(len(val_data), dtype=float)

    # Map ABC predictions
    abc_val_pos = val_data.index.get_indexer(abc_val_idx)
    pred_total[abc_val_pos] = pred_total_abc
    pred_rate[abc_val_pos] = pred_rate_abc

    # Map D predictions
    d_val_pos = val_data.index.get_indexer(student_result["d_val_idx"])
    pred_total[d_val_pos] = student_result["pred_total"]
    pred_rate[d_val_pos] = student_result["pred_rate"]

    pred_total = pred_total.clip(0)
    pred_rate = pred_rate.clip(0, 1)
    pred_admitted = pred_total * pred_rate

    # Aliases for backward compat (model saving, return dict)
    model_a1 = model_t1
    model_a2 = model_t2
    residual_model = None  # No longer used

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
        # Teacher models (ABC)
        model_t1.booster_.save_model(str(cfg.MODEL_DIR / f"fold_{fold_id}_teacher_t1.txt"))
        model_t2.booster_.save_model(str(cfg.MODEL_DIR / f"fold_{fold_id}_teacher_t2.txt"))
        # Student models (Site D)
        student_result["model_s1"].booster_.save_model(
            str(cfg.MODEL_DIR / f"fold_{fold_id}_student_s1.txt")
        )
        student_result["model_s2"].booster_.save_model(
            str(cfg.MODEL_DIR / f"fold_{fold_id}_student_s2.txt")
        )

    # ── Quick metrics ────────────────────────────────────────────────────
    total_wape_all = wape(val_data["total_enc"].values, val_out["pred_total"].values)
    admitted_wape_all = wape(val_data["admitted_enc"].values, val_out["pred_admitted"].values)

    # Per-site WAPE (diagnostic)
    site_wapes = {}
    for site in cfg.SITES:
        mask_s = val_out["site"] == site
        if mask_s.sum() == 0:
            continue
        s_actual_t = val_data.loc[val_out.index[mask_s], "total_enc"].values
        s_pred_t = val_out.loc[mask_s, "pred_total"].values
        s_actual_a = val_data.loc[val_out.index[mask_s], "admitted_enc"].values
        s_pred_a = val_out.loc[mask_s, "pred_admitted"].values
        site_wapes[site] = {
            "total": wape(s_actual_t, s_pred_t),
            "admitted": wape(s_actual_a, s_pred_a),
        }

    print(
        f"    Fold {fold_id}: total_wape={total_wape_all:.4f}  "
        f"admitted_wape={admitted_wape_all:.4f}"
    )
    for site, sw in site_wapes.items():
        print(f"      Site {site}: total={sw['total']:.4f}  admitted={sw['admitted']:.4f}")

    return {
        "fold_id": fold_id,
        "total_wape": total_wape_all,
        "admitted_wape": admitted_wape_all,
        "site_wapes": site_wapes,
        "model_a1": model_a1,
        "model_a2": model_a2,
        "student_result": student_result,
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
