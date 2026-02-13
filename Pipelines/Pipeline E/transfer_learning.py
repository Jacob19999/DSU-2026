"""
Transfer Learning for Pipeline E: Two-stage ABC Teacher → Site D Student.

Same architecture as Pipeline A's transfer_learning.py, but adapted for
Pipeline E's factor features:
  - Teacher uses full factor set (all N_FACTORS)
  - Student uses REDUCED factor set (top 2 factors only — factors are
    noisier for Site D due to 52.5% "other" reason-category share)

Pipeline E differences vs Pipeline A:
  - Weight columns: sample_weight / sample_weight_rate (not sample_weight_a1/a2)
  - No categorical features passed to teacher (Pipeline E uses site_enc, block)
  - Early stopping uses ES_HOLD_DAYS holdout instead of val set
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

import config as cfg

CAT_FEATURES = ["site_enc", "block"]

# ── Student feature set (Pipeline E variant) ─────────────────────────────────

D_OWN_FEATURES = [
    # Block identifier
    "block",
    # Target lags (D-specific history)
    "lag_63", "lag_70", "lag_77", "lag_91", "lag_182", "lag_364",
    # Admit-rate lags
    "lag_rate_63", "lag_rate_70", "lag_rate_77", "lag_rate_91",
    "lag_rate_182", "lag_rate_364",
    # Rolling stats (D-specific)
    "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_mean_56", "roll_mean_91",
    "roll_std_7", "roll_std_14", "roll_std_28",
    # Rolling min/max (Pipeline E has these)
    "roll_min_7", "roll_min_28",
    "roll_max_7", "roll_max_28",
    # Trend deltas
    "delta_7_28", "delta_28_91", "lag_diff",
    # Calendar (deterministic)
    "dow", "day", "week_of_year", "month", "quarter", "day_of_year",
    "is_weekend", "is_halloween",
    "dow_sin", "dow_cos", "doy_sin", "doy_cos", "month_sin", "month_cos",
    "days_since_epoch", "year_frac",
    # Holidays
    "is_us_holiday", "days_to_nearest_holiday",
    "days_since_xmas", "days_until_thanksgiving", "days_since_july4",
    # School
    "school_in_session", "days_since_school_start", "days_until_school_start",
    # Events
    "event_count",
    # COVID
    "is_covid_era",
    # Target encodings (D-specific baselines)
    "te_site_block_mean_total", "te_site_block_mean_admitted",
    "te_site_admit_rate", "te_site_dow_mean",
    "te_site_mean_total_enc", "te_site_mean_admitted_enc",
    # Fold encodings
    "site_month_block_mean", "site_dow_mean",
    # Cross-block lags (block 0 only)
    "xblock_b3_total_63", "xblock_b3_total_91",
    "xblock_b3_roll_mean_7", "xblock_b3_roll_mean_28",
    # Interactions
    "holiday_x_block", "weekend_x_block",
]

TRANSFER_FEATURES = [
    "teacher_pred_total",
    "teacher_pred_rate",
    "teacher_pred_admitted",
    "teacher_residual_roll_7",
    "teacher_residual_roll_14",
    "teacher_residual_roll_28",
    "abc_month_block_mean_total",
    "abc_dow_mean_total",
    "abc_month_block_admit_rate",
]

# Reduced factor features for student (top 2 factors only — noisier for Site D)
FACTOR_TRANSFER_FEATURES = [
    "factor_0_pred", "factor_1_pred",
    "factor_0_momentum", "factor_1_momentum",
]

STUDENT_FEATURE_COLS = D_OWN_FEATURES + TRANSFER_FEATURES + FACTOR_TRANSFER_FEATURES


# ── Transfer feature generation ──────────────────────────────────────────────

def generate_transfer_features(
    df_fold: pd.DataFrame,
    train_mask: pd.Series,
    teacher_models: dict,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Compute transfer artifacts from teacher models for Site D rows."""
    df = df_fold.copy()
    model_t1 = teacher_models["t1"]
    model_t2 = teacher_models["t2"]

    mask_d = df["site"] == "D"

    # 1. Teacher predictions on ALL Site D rows
    X_all_d = df.loc[mask_d, feature_cols]
    teacher_pred_total = model_t1.predict(X_all_d).clip(0)
    teacher_pred_rate = model_t2.predict(X_all_d).clip(0, 1)

    df.loc[mask_d, "teacher_pred_total"] = teacher_pred_total
    df.loc[mask_d, "teacher_pred_rate"] = teacher_pred_rate
    df.loc[mask_d, "teacher_pred_admitted"] = teacher_pred_total * teacher_pred_rate

    # 2. Teacher residual on D's training history
    mask_d_train = mask_d & train_mask
    if mask_d_train.sum() > 0:
        actual_total_d = df.loc[mask_d_train, "total_enc"].values
        teacher_total_d = df.loc[mask_d_train, "teacher_pred_total"].values
        df.loc[mask_d_train, "teacher_residual_total"] = actual_total_d - teacher_total_d

        d_train_sorted = df.loc[mask_d_train].sort_values(["block", "date"])
        for w in [7, 14, 28]:
            col = f"teacher_residual_roll_{w}"
            df.loc[d_train_sorted.index, col] = (
                d_train_sorted.groupby("block")["teacher_residual_total"]
                .transform(lambda s: s.shift(cfg.MAX_HORIZON).rolling(w, min_periods=1).mean())
            )

    # 3. ABC cross-site aggregate priors
    abc_train = df.loc[train_mask & df["site"].isin(["A", "B", "C"])]

    mb_mean = abc_train.groupby(["month", "block"])["total_enc"].mean()
    keys_mb = list(zip(df.loc[mask_d, "month"], df.loc[mask_d, "block"]))
    fallback_total = float(abc_train["total_enc"].mean()) if len(abc_train) > 0 else 0.0
    df.loc[mask_d, "abc_month_block_mean_total"] = [
        mb_mean.get(k, fallback_total) for k in keys_mb
    ]

    dow_mean = abc_train.groupby("dow")["total_enc"].mean()
    df.loc[mask_d, "abc_dow_mean_total"] = (
        df.loc[mask_d, "dow"].map(dow_mean).fillna(fallback_total).values
    )

    if len(abc_train) > 0:
        mb_rate = abc_train.groupby(["month", "block"]).apply(
            lambda g: g["admitted_enc"].sum() / g["total_enc"].sum()
            if g["total_enc"].sum() > 0 else 0.0
        )
        fallback_rate = float(
            abc_train["admitted_enc"].sum() / abc_train["total_enc"].sum()
        ) if abc_train["total_enc"].sum() > 0 else 0.0
        df.loc[mask_d, "abc_month_block_admit_rate"] = [
            mb_rate.get(k, fallback_rate) for k in keys_mb
        ]

    return df


# ── Stage 1: ABC Teacher ─────────────────────────────────────────────────────

def train_teacher_abc(
    fold_df: pd.DataFrame,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    feature_cols: list[str],
    params_total: dict,
    params_rate: dict,
) -> dict:
    """Train teacher models on Sites A, B, C only.

    Uses Pipeline E's ES_HOLD_DAYS early-stopping strategy.
    """
    burn_in_col = f"lag_{max(cfg.LAG_DAYS)}"

    # ABC training data
    abc_mask = fold_df["site"].isin(["A", "B", "C"])
    train_all_abc = (
        fold_df[abc_mask & (fold_df["date"] <= train_end)]
        .dropna(subset=[burn_in_col])
        .copy()
    )
    val_abc = fold_df[
        abc_mask
        & (fold_df["date"] >= val_start)
        & (fold_df["date"] <= val_end)
    ].copy()

    # Early-stopping split (Pipeline E style: last ES_HOLD_DAYS of training)
    es_cutoff = train_end - pd.Timedelta(days=cfg.ES_HOLD_DAYS)
    train_fit = train_all_abc[train_all_abc["date"] <= es_cutoff]
    train_es = train_all_abc[train_all_abc["date"] > es_cutoff]

    if len(train_es) < 100:
        n_es = max(int(len(train_all_abc) * 0.1), 100)
        train_fit = train_all_abc.iloc[:-n_es]
        train_es = train_all_abc.iloc[-n_es:]

    print(f"    [Teacher] ABC train: {len(train_fit):,} fit + {len(train_es):,} ES  |  "
          f"ABC val: {len(val_abc):,}")

    # T1: total_enc
    model_t1 = lgb.LGBMRegressor(**params_total)
    model_t1.fit(
        train_fit[feature_cols], train_fit["total_enc"],
        sample_weight=train_fit["sample_weight"].values,
        eval_set=[(train_es[feature_cols], train_es["total_enc"])],
        categorical_feature=CAT_FEATURES,
        callbacks=[
            lgb.early_stopping(cfg.ES_PATIENCE, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    # T2: admit_rate
    model_t2 = lgb.LGBMRegressor(**params_rate)
    model_t2.fit(
        train_fit[feature_cols], train_fit["admit_rate"],
        sample_weight=train_fit["sample_weight_rate"].values,
        eval_set=[(train_es[feature_cols], train_es["admit_rate"])],
        categorical_feature=CAT_FEATURES,
        callbacks=[
            lgb.early_stopping(cfg.ES_PATIENCE, verbose=False),
            lgb.log_evaluation(0),
        ],
    )

    # ABC val predictions
    pred_total_abc = model_t1.predict(val_abc[feature_cols]).clip(0)
    pred_rate_abc = model_t2.predict(val_abc[feature_cols]).clip(0, 1)

    # Diagnostics — teacher prediction range on D
    mask_d = fold_df["site"] == "D"
    X_d_all = fold_df.loc[mask_d, feature_cols]
    if len(X_d_all) > 0:
        teacher_pred_d = model_t1.predict(X_d_all).clip(0)
        print(f"    [Teacher] Pred on D: mean={teacher_pred_d.mean():.1f}, "
              f"min={teacher_pred_d.min():.1f}, max={teacher_pred_d.max():.1f}")

        # Residual on D train
        mask_d_train = mask_d & (fold_df["date"] <= train_end)
        if mask_d_train.sum() > 0:
            actual_d = fold_df.loc[mask_d_train, "total_enc"].values
            pred_d_tr = model_t1.predict(
                fold_df.loc[mask_d_train, feature_cols]
            ).clip(0)
            residuals = actual_d - pred_d_tr
            print(f"    [Teacher] D residuals: mean={residuals.mean():.2f}, "
                  f"std={residuals.std():.2f}, "
                  f"skew={pd.Series(residuals).skew():.2f}")

    print(f"    [Teacher] T1 iters={model_t1.best_iteration_}, "
          f"T2 iters={model_t2.best_iteration_}")

    return {
        "models": {"t1": model_t1, "t2": model_t2},
        "abc_val_idx": val_abc.index,
        "pred_total_abc": pred_total_abc,
        "pred_rate_abc": pred_rate_abc,
    }


# ── Stage 2: Site D Student ──────────────────────────────────────────────────

def train_student_d(
    fold_df: pd.DataFrame,
    train_end: pd.Timestamp,
    val_start: pd.Timestamp,
    val_end: pd.Timestamp,
    teacher_models: dict,
    feature_cols_parent: list[str],
    params_s1: dict | None = None,
    params_s2: dict | None = None,
    blend_alpha: float | None = None,
) -> dict:
    """Train Site D student with transfer features (Pipeline E variant)."""
    p_s1 = (params_s1 or cfg.STUDENT_LGBM_PARAMS_S1).copy()
    p_s2 = (params_s2 or cfg.STUDENT_LGBM_PARAMS_S2).copy()
    alpha = blend_alpha if blend_alpha is not None else cfg.STUDENT_BLEND_ALPHA

    burn_in_col = f"lag_{max(cfg.LAG_DAYS)}"

    # Build train/val masks for transfer feature generation
    train_mask = fold_df["date"] <= train_end
    val_mask = (fold_df["date"] >= val_start) & (fold_df["date"] <= val_end)

    # Generate transfer features
    fold_df = generate_transfer_features(
        fold_df, train_mask, teacher_models, feature_cols_parent,
    )

    # Site D split
    mask_d = fold_df["site"] == "D"
    train_d = (
        fold_df[mask_d & train_mask]
        .dropna(subset=[burn_in_col])
        .copy()
    )
    val_d = fold_df[mask_d & val_mask].copy()

    # Resolve available student features
    student_features = [f for f in STUDENT_FEATURE_COLS if f in train_d.columns]

    X_train = train_d[student_features]
    X_val = val_d[student_features]

    # Inner validation split (last 20%)
    n_inner = max(int(len(X_train) * 0.2), 50)
    X_tr = X_train.iloc[:-n_inner]
    X_iv = X_train.iloc[-n_inner:]
    y_tr_total = train_d["total_enc"].iloc[:-n_inner]
    y_iv_total = train_d["total_enc"].iloc[-n_inner:]
    y_tr_rate = train_d["admit_rate"].iloc[:-n_inner]
    y_iv_rate = train_d["admit_rate"].iloc[-n_inner:]

    # Weights
    w_s1 = train_d["sample_weight"].iloc[:-n_inner].values
    w_s2 = train_d["sample_weight_rate"].iloc[:-n_inner].values

    print(f"    [Student] D train: {len(X_tr):,} fit + {len(X_iv):,} inner-val  |  "
          f"D val: {len(X_val):,}  |  Features: {len(student_features)}")

    # S1: total_enc
    model_s1 = lgb.LGBMRegressor(**p_s1)
    model_s1.fit(
        X_tr, y_tr_total,
        sample_weight=w_s1,
        eval_set=[(X_iv, y_iv_total)],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )

    # S2: admit_rate
    model_s2 = lgb.LGBMRegressor(**p_s2)
    model_s2.fit(
        X_tr, y_tr_rate,
        sample_weight=w_s2,
        eval_set=[(X_iv, y_iv_rate)],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )

    # Predict
    student_pred_total = model_s1.predict(X_val).clip(0)
    student_pred_rate = model_s2.predict(X_val).clip(0, 1)

    # Teacher predictions on D val (for blending)
    teacher_val_pred_total = teacher_models["t1"].predict(
        val_d[feature_cols_parent]
    ).clip(0)
    teacher_val_pred_rate = teacher_models["t2"].predict(
        val_d[feature_cols_parent]
    ).clip(0, 1)

    # Blend
    pred_total_d = alpha * student_pred_total + (1 - alpha) * teacher_val_pred_total
    pred_rate_d = alpha * student_pred_rate + (1 - alpha) * teacher_val_pred_rate

    # Diagnostics
    print(f"    [Student] S1 iters={model_s1.best_iteration_}, "
          f"S2 iters={model_s2.best_iteration_}  |  blend α={alpha:.2f}")

    fi_s1 = pd.Series(
        model_s1.feature_importances_, index=student_features,
    ).sort_values(ascending=False)
    print(f"    [Student] Top 10 S1 features (by gain):")
    for feat, imp in fi_s1.head(10).items():
        marker = " ← TRANSFER" if feat in TRANSFER_FEATURES else ""
        if feat in FACTOR_TRANSFER_FEATURES:
            marker = " ← FACTOR"
        print(f"      {feat:45s} {imp:>8,.0f}{marker}")

    transfer_in_top10 = [
        f for f in fi_s1.head(10).index
        if f in TRANSFER_FEATURES or f in FACTOR_TRANSFER_FEATURES
    ]
    if not transfer_in_top10:
        print("    [WARN] No transfer/factor features in top 10 — signal may be weak")

    return {
        "model_s1": model_s1,
        "model_s2": model_s2,
        "d_val_idx": val_d.index,
        "pred_total": pred_total_d,
        "pred_rate": pred_rate_d,
        "student_features": student_features,
        "fi_s1": fi_s1,
    }
