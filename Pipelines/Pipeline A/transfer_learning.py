"""
Transfer Learning: Two-stage ABC Teacher → Site D Student.

Stage 1 — Train ABC teacher (T1: total_enc, T2: admit_rate) on Sites A/B/C only.
Stage 2 — Train Site D student (S1, S2) on D only, augmented with transfer features
           derived from the teacher's predictions on D's history.

The teacher's knowledge flows through *features* (predictions, residuals, priors),
not weight sharing. Each model is independently trained.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

import config as cfg

# ── Student feature set ──────────────────────────────────────────────────────

D_OWN_FEATURES = [
    # Block identifier
    "block",
    # Target lags (D-specific history)
    "lag_63", "lag_70", "lag_77", "lag_91", "lag_182", "lag_364",
    # Admit-rate lags
    "lag_admit_63", "lag_admit_70", "lag_admit_77", "lag_admit_91",
    "lag_admit_182", "lag_admit_364",
    # Rolling stats (D-specific)
    "roll_mean_7", "roll_mean_14", "roll_mean_28", "roll_mean_56", "roll_mean_91",
    "roll_std_7", "roll_std_14", "roll_std_28",
    # Trend deltas
    "delta_7_28", "delta_28_91", "delta_lag_63_70",
    # Calendar (deterministic, always safe)
    "dow", "day", "week_of_year", "month", "quarter", "day_of_year",
    "is_weekend", "is_halloween",
    "dow_sin", "dow_cos", "doy_sin", "doy_cos", "month_sin", "month_cos",
    "days_since_epoch", "year_frac",
    # Holidays
    "is_us_holiday", "days_to_nearest_holiday",
    # School
    "school_in_session",
    # Events
    "event_count",
    # COVID
    "is_covid_era",
    # Target encodings (D-specific baselines)
    "te_site_block_mean_total", "te_site_block_mean_admitted",
    "te_site_admit_rate", "te_site_dow_mean",
    # Cross-block lags (block 0 only — will be NaN for other blocks, handled by LGB)
    "xblock_b3_total_63", "xblock_b3_total_91",
    "xblock_b3_roll_mean_7", "xblock_b3_roll_mean_28",
]

TRANSFER_FEATURES = [
    # Teacher predictions (informed prior — "what ABC patterns predict for D")
    "teacher_pred_total",
    "teacher_pred_rate",
    "teacher_pred_admitted",
    # Teacher rolling error (how biased is ABC model on D recently?)
    "teacher_residual_roll_7",
    "teacher_residual_roll_14",
    "teacher_residual_roll_28",
    # ABC population priors (what is "normal" for comparable days across ABC?)
    "abc_month_block_mean_total",
    "abc_dow_mean_total",
    "abc_month_block_admit_rate",
]

STUDENT_FEATURE_COLS = D_OWN_FEATURES + TRANSFER_FEATURES


# ── Transfer feature generation ──────────────────────────────────────────────

def generate_transfer_features(
    df_fold: pd.DataFrame,
    train_mask: pd.Series,
    teacher_models: dict,
    feature_cols: list[str],
) -> pd.DataFrame:
    """Compute all transfer artifacts from teacher models for Site D rows.

    Parameters
    ----------
    df_fold : full fold DataFrame (all sites)
    train_mask : boolean mask for training rows
    teacher_models : {"t1": model_t1, "t2": model_t2}
    feature_cols : feature columns used by the teacher models
    """
    df = df_fold.copy()
    model_t1 = teacher_models["t1"]
    model_t2 = teacher_models["t2"]

    mask_d = df["site"] == "D"

    # 1. Teacher predictions on ALL Site D rows (train + val)
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

        # Rolling teacher error per block
        d_train_sorted = df.loc[mask_d_train].sort_values(["block", "date"])
        for w in [7, 14, 28]:
            col = f"teacher_residual_roll_{w}"
            df.loc[d_train_sorted.index, col] = (
                d_train_sorted.groupby("block")["teacher_residual_total"]
                .transform(lambda s: s.shift(cfg.MAX_HORIZON).rolling(w, min_periods=1).mean())
            )

    # 3. ABC cross-site aggregate priors
    abc_train = df.loc[train_mask & df["site"].isin(["A", "B", "C"])]

    # Month × Block mean total
    mb_mean = abc_train.groupby(["month", "block"])["total_enc"].mean()
    keys_mb = list(zip(df.loc[mask_d, "month"], df.loc[mask_d, "block"]))
    fallback_total = float(abc_train["total_enc"].mean()) if len(abc_train) > 0 else 0.0
    df.loc[mask_d, "abc_month_block_mean_total"] = [
        mb_mean.get(k, fallback_total) for k in keys_mb
    ]

    # DOW mean total
    dow_mean = abc_train.groupby("dow")["total_enc"].mean()
    df.loc[mask_d, "abc_dow_mean_total"] = (
        df.loc[mask_d, "dow"].map(dow_mean).fillna(fallback_total).values
    )

    # Month × Block admit rate
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
    df_fold: pd.DataFrame,
    train_mask: pd.Series,
    val_mask: pd.Series,
    feature_cols: list[str],
    cat_features: list[str],
    params_a1: dict,
    params_a2: dict,
    *,
    covid_policy: str = "downweight",
) -> dict:
    """Train teacher models on Sites A, B, C only.

    Returns dict with models, ABC predictions, and diagnostics.
    """
    # Filter to ABC
    abc_train_mask = train_mask & df_fold["site"].isin(["A", "B", "C"])
    abc_val_mask = val_mask & df_fold["site"].isin(["A", "B", "C"])

    train_abc = df_fold.loc[abc_train_mask].copy()
    val_abc = df_fold.loc[abc_val_mask].copy()

    # COVID policy
    if covid_policy == "exclude":
        if "is_covid_era" in train_abc.columns:
            train_abc = train_abc[train_abc["is_covid_era"] == 0].copy()

    # Drop burn-in
    train_abc = train_abc.dropna(subset=[f"lag_{cfg.LAG_DAYS[-1]}"])

    X_train = train_abc[feature_cols]
    X_val = val_abc[feature_cols]

    # Weights
    if covid_policy == "exclude":
        w_a1 = train_abc["volume_weight"].values
        w_a2 = train_abc["admitted_enc"].clip(lower=1).values.astype(float)
    else:
        w_a1 = train_abc["sample_weight_a1"].values
        w_a2 = train_abc["sample_weight_a2"].values

    cat_feats = [c for c in cat_features if c in feature_cols]

    # Model T1: total_enc — ABC only
    model_t1 = lgb.LGBMRegressor(**params_a1)
    model_t1.fit(
        X_train, train_abc["total_enc"],
        sample_weight=w_a1,
        eval_set=[(X_val, val_abc["total_enc"])],
        categorical_feature=cat_feats,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    # Model T2: admit_rate — ABC only
    model_t2 = lgb.LGBMRegressor(**params_a2)
    model_t2.fit(
        X_train, train_abc["admit_rate"],
        sample_weight=w_a2,
        eval_set=[(X_val, val_abc["admit_rate"])],
        categorical_feature=cat_feats,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )

    # ABC val predictions (for Sites A/B/C output)
    pred_total_abc = model_t1.predict(
        df_fold.loc[val_mask & df_fold["site"].isin(["A", "B", "C"]), feature_cols]
    ).clip(0)
    pred_rate_abc = model_t2.predict(
        df_fold.loc[val_mask & df_fold["site"].isin(["A", "B", "C"]), feature_cols]
    ).clip(0, 1)

    # ── Diagnostics ──────────────────────────────────────────────────────
    # Teacher prediction range on D (full history)
    mask_d = df_fold["site"] == "D"
    X_d_all = df_fold.loc[mask_d, feature_cols]
    teacher_pred_d = model_t1.predict(X_d_all).clip(0)

    print(f"    [Teacher] ABC train: {len(train_abc):,} rows  |  "
          f"T1 iters={model_t1.best_iteration_}, T2 iters={model_t2.best_iteration_}")
    print(f"    [Teacher] Pred on D: mean={teacher_pred_d.mean():.1f}, "
          f"min={teacher_pred_d.min():.1f}, max={teacher_pred_d.max():.1f}")

    # Teacher residual on D train set
    mask_d_train = mask_d & train_mask
    if mask_d_train.sum() > 0:
        actual_d = df_fold.loc[mask_d_train, "total_enc"].values
        pred_d_train = model_t1.predict(
            df_fold.loc[mask_d_train, feature_cols]
        ).clip(0)
        residuals = actual_d - pred_d_train
        print(f"    [Teacher] D residuals: mean={residuals.mean():.2f}, "
              f"std={residuals.std():.2f}, "
              f"skew={pd.Series(residuals).skew():.2f}")

    return {
        "models": {"t1": model_t1, "t2": model_t2},
        "abc_val_idx": df_fold.index[val_mask & df_fold["site"].isin(["A", "B", "C"])],
        "pred_total_abc": pred_total_abc,
        "pred_rate_abc": pred_rate_abc,
    }


# ── Stage 2: Site D Student ──────────────────────────────────────────────────

def train_student_d(
    df_fold: pd.DataFrame,
    train_mask: pd.Series,
    val_mask: pd.Series,
    teacher_models: dict,
    feature_cols_parent: list[str],
    params_s1: dict | None = None,
    params_s2: dict | None = None,
    blend_alpha: float | None = None,
    *,
    teacher_val_pred_total: np.ndarray | None = None,
) -> dict:
    """Train Site D student models with transfer features.

    Parameters
    ----------
    teacher_val_pred_total : teacher's prediction of total_enc on D val set.
        Used for blending. If None, computed internally.
    """
    p_s1 = (params_s1 or cfg.STUDENT_LGBM_PARAMS_S1).copy()
    p_s2 = (params_s2 or cfg.STUDENT_LGBM_PARAMS_S2).copy()
    alpha = blend_alpha if blend_alpha is not None else cfg.STUDENT_BLEND_ALPHA

    # Generate transfer features (modifies df_fold copy)
    df_fold = generate_transfer_features(
        df_fold, train_mask, teacher_models, feature_cols_parent,
    )

    # Split — Site D only
    mask_d_train = train_mask & (df_fold["site"] == "D")
    mask_d_val = val_mask & (df_fold["site"] == "D")

    train_d = df_fold.loc[mask_d_train].dropna(subset=[f"lag_{cfg.LAG_DAYS[-1]}"]).copy()
    val_d = df_fold.loc[mask_d_val].copy()

    # Resolve available student features
    student_features = [f for f in STUDENT_FEATURE_COLS if f in train_d.columns]

    X_train = train_d[student_features]
    X_val = val_d[student_features]

    # Inner validation split for early stopping (last 20% of D training)
    n_inner = max(int(len(X_train) * 0.2), 50)
    X_tr = X_train.iloc[:-n_inner]
    X_iv = X_train.iloc[-n_inner:]
    y_tr_total = train_d["total_enc"].iloc[:-n_inner]
    y_iv_total = train_d["total_enc"].iloc[-n_inner:]
    y_tr_rate = train_d["admit_rate"].iloc[:-n_inner]
    y_iv_rate = train_d["admit_rate"].iloc[-n_inner:]

    # Weights (COVID-aware)
    w_s1 = train_d["sample_weight_a1"].iloc[:-n_inner].values
    w_s2 = train_d["sample_weight_a2"].iloc[:-n_inner].values

    print(f"    [Student] D train: {len(X_tr):,} fit + {len(X_iv):,} inner-val  |  "
          f"D val: {len(X_val):,}  |  Features: {len(student_features)}")

    # Model S1: total_enc for Site D
    model_s1 = lgb.LGBMRegressor(**p_s1)
    model_s1.fit(
        X_tr, y_tr_total,
        sample_weight=w_s1,
        eval_set=[(X_iv, y_iv_total)],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )

    # Model S2: admit_rate for Site D
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
    if teacher_val_pred_total is None:
        teacher_val_pred_total = teacher_models["t1"].predict(
            val_d[feature_cols_parent]
        ).clip(0)
    teacher_val_pred_rate = teacher_models["t2"].predict(
        val_d[feature_cols_parent]
    ).clip(0, 1)

    # Blend student + teacher
    pred_total_d = (
        alpha * student_pred_total
        + (1 - alpha) * teacher_val_pred_total
    )
    pred_rate_d = (
        alpha * student_pred_rate
        + (1 - alpha) * teacher_val_pred_rate
    )

    # ── Diagnostics ──────────────────────────────────────────────────────
    print(f"    [Student] S1 iters={model_s1.best_iteration_}, "
          f"S2 iters={model_s2.best_iteration_}  |  "
          f"blend α={alpha:.2f}")

    # Feature importance top 15
    fi_s1 = pd.Series(
        model_s1.feature_importances_, index=student_features,
    ).sort_values(ascending=False)
    print(f"    [Student] Top 10 S1 features (by gain):")
    for feat, imp in fi_s1.head(10).items():
        marker = " ← TRANSFER" if feat in TRANSFER_FEATURES else ""
        print(f"      {feat:45s} {imp:>8,.0f}{marker}")

    # Check transfer signal
    transfer_in_top10 = [f for f in fi_s1.head(10).index if f in TRANSFER_FEATURES]
    if not transfer_in_top10:
        print("    [WARN] No transfer features in top 10 — teacher signal may be weak")

    return {
        "model_s1": model_s1,
        "model_s2": model_s2,
        "d_val_idx": val_d.index,
        "pred_total": pred_total_d,
        "pred_rate": pred_rate_d,
        "student_features": student_features,
        "fi_s1": fi_s1,
    }
