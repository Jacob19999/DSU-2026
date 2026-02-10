"""
Factor forecasting for Pipeline E.

Trains one LightGBM model per latent factor to predict current factor values
from lagged actual factors + calendar features.  Uses safe-lag approach
(all lags >= 63) for v1 to avoid recursive prediction error accumulation.

For validation/forecast rows where actual factors are unknown, the trained
models provide factor_i_pred estimates that feed into the final GBDT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

import config as cfg


# ══════════════════════════════════════════════════════════════════════════════
#  FACTOR FEATURE ENGINEERING  (lags, rolling, momentum from actual factors)
# ══════════════════════════════════════════════════════════════════════════════

def add_factor_lag_features(
    df: pd.DataFrame,
    n_factors: int = cfg.N_FACTORS,
) -> pd.DataFrame:
    """Add safe-lagged factor features per (site, block) group.

    All shifts >= FACTOR_ROLLING_SHIFT (63) to guarantee no leakage.
    """
    df = df.copy()

    for (_site, _blk), grp in df.groupby(["site", "block"]):
        idx = grp.index

        for i in range(n_factors):
            series = grp[f"factor_{i}"]

            # Lag features
            for lag in cfg.FACTOR_LAG_DAYS:
                df.loc[idx, f"factor_{i}_lag_{lag}"] = series.shift(lag).values

            # Rolling means (shifted by 63)
            shifted = series.shift(cfg.FACTOR_ROLLING_SHIFT)
            for w in cfg.FACTOR_ROLLING_WINDOWS:
                df.loc[idx, f"factor_{i}_roll_mean_{w}"] = (
                    shifted.rolling(w, min_periods=1).mean().values
                )

            # Year-over-year lag (63+364 = 427 days back)
            df.loc[idx, f"factor_{i}_lag_427"] = series.shift(427).values

    # ── Derived: momentum & yearly deviation ─────────────────────────────
    for i in range(n_factors):
        # Momentum: week-over-week composition shift (safe: lag63 − lag70)
        df[f"factor_{i}_momentum"] = (
            df[f"factor_{i}_lag_63"] - df[f"factor_{i}_lag_70"]
        )
        # Yearly deviation: same period last year vs 63-day-ago
        df[f"factor_{i}_deviation_yearly"] = (
            df[f"factor_{i}_lag_63"] - df[f"factor_{i}_lag_427"]
        )

    return df


def _get_ff_feature_cols(factor_idx: int, df: pd.DataFrame) -> list[str]:
    """Return feature columns for forecasting a single factor."""
    cols = []

    # Factor-specific lags + rolling + momentum
    for lag in cfg.FACTOR_LAG_DAYS:
        c = f"factor_{factor_idx}_lag_{lag}"
        if c in df.columns:
            cols.append(c)
    for w in cfg.FACTOR_ROLLING_WINDOWS:
        c = f"factor_{factor_idx}_roll_mean_{w}"
        if c in df.columns:
            cols.append(c)
    for c in [f"factor_{factor_idx}_momentum",
              f"factor_{factor_idx}_deviation_yearly"]:
        if c in df.columns:
            cols.append(c)

    # Calendar (deterministic, always available)
    for c in ["dow", "month", "day_of_year", "is_weekend", "week_of_year",
              "site_enc", "block"]:
        if c in df.columns:
            cols.append(c)

    return cols


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN FACTOR FORECAST MODELS
# ══════════════════════════════════════════════════════════════════════════════

def train_factor_forecast_models(
    df: pd.DataFrame,
    train_end: str | pd.Timestamp,
    n_factors: int = cfg.N_FACTORS,
    params: dict | None = None,
) -> list[lgb.LGBMRegressor]:
    """Train one GBDT per factor on training data.

    Returns list of N_FACTORS fitted LGBMRegressor models.
    """
    train_mask = df["date"] <= pd.Timestamp(train_end)
    p = (params or cfg.LGBM_FACTOR_FORECAST).copy()

    models: list[lgb.LGBMRegressor] = []
    print("  Training factor forecast models ...")

    for i in range(n_factors):
        target_col = f"factor_{i}"
        feat_cols  = _get_ff_feature_cols(i, df)

        # Filter to training, drop NaN burn-in rows
        burn_in_col = f"factor_{i}_lag_{max(cfg.FACTOR_LAG_DAYS)}"
        train_data = df[train_mask].dropna(subset=[burn_in_col])

        # Early-stopping split
        es_cutoff = pd.Timestamp(train_end) - pd.Timedelta(days=cfg.ES_HOLD_DAYS)
        fit_data = train_data[train_data["date"] <= es_cutoff]
        es_data  = train_data[train_data["date"] > es_cutoff]

        if len(es_data) < 50:
            n_es = max(int(len(train_data) * 0.1), 50)
            fit_data = train_data.iloc[:-n_es]
            es_data  = train_data.iloc[-n_es:]

        model = lgb.LGBMRegressor(**p)
        model.fit(
            fit_data[feat_cols], fit_data[target_col],
            eval_set=[(es_data[feat_cols], es_data[target_col])],
            callbacks=[
                lgb.early_stopping(cfg.ES_PATIENCE, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        models.append(model)

        # Diagnostics
        pred = model.predict(es_data[feat_cols])
        actual = es_data[target_col].values
        mae = float(np.mean(np.abs(actual - pred)))
        ss_res = float(np.sum((actual - pred) ** 2))
        ss_tot = float(np.sum((actual - actual.mean()) ** 2))
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        print(f"    Factor {i}: MAE={mae:.4f}  R²={r2:.3f}  "
              f"(iters={model.best_iteration_})")

    return models


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICT FACTORS  (actual for train, predicted for val/forecast)
# ══════════════════════════════════════════════════════════════════════════════

def set_predicted_factors(
    df: pd.DataFrame,
    factor_models: list[lgb.LGBMRegressor],
    train_end: str | pd.Timestamp,
    n_factors: int = cfg.N_FACTORS,
) -> pd.DataFrame:
    """Set factor_i_pred column: actual factors for train, predicted for val.

    For training rows the actual (PCA-derived) factor is used.
    For validation/forecast rows the factor forecast model provides estimates.
    """
    df = df.copy()
    train_mask = df["date"] <= pd.Timestamp(train_end)

    for i in range(n_factors):
        # Training: use actual PCA factors
        df.loc[train_mask, f"factor_{i}_pred"] = df.loc[train_mask, f"factor_{i}"]

        # Validation: use predicted
        val_mask = ~train_mask
        if val_mask.any():
            feat_cols = _get_ff_feature_cols(i, df)
            df.loc[val_mask, f"factor_{i}_pred"] = (
                factor_models[i].predict(df.loc[val_mask, feat_cols])
            )

    return df


if __name__ == "__main__":
    from data_loader import load_data
    from share_matrix import build_share_matrix
    from factor_extraction import fit_and_transform_factors
    from features import add_static_features

    df = load_data()
    df, share_cols = build_share_matrix(df)
    df = add_static_features(df)
    df, _, _ = fit_and_transform_factors(df, share_cols, "2024-12-31")
    df = add_factor_lag_features(df)
    models = train_factor_forecast_models(df, "2024-12-31")
    df = set_predicted_factors(df, models, "2024-12-31")
    print(f"\nPredicted factor columns present: "
          f"{[c for c in df.columns if c.endswith('_pred')]}")
