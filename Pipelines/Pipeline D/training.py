"""
GLM training for Pipeline D — 16 Poisson GLMs (total_enc) + 16 Binomial GLMs (admit_rate).

One model per (Site, Block) pair.  Uses statsmodels with L2 regularization.
COVID-era rows are downweighted via freq_weights.
"""

from __future__ import annotations

import pickle
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm

import config as cfg
from data_loader import get_site_block_subset
from features import build_design_matrix


# ── Metrics ──────────────────────────────────────────────────────────────────

def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum(np.abs(y_true)))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else float("nan")


def largest_remainder_round(values: np.ndarray) -> np.ndarray:
    """Round floats to ints preserving aggregate sum."""
    values = np.asarray(values, dtype=float)
    # Replace inf/nan with 0 before rounding
    values = np.where(np.isfinite(values), values, 0.0)
    values = values.clip(0)
    floored = np.floor(values).astype(int)
    remainders = values - floored
    target_sum = int(round(values.sum()))
    deficit = target_sum - floored.sum()
    if deficit > 0:
        idx = np.argsort(-remainders)[:deficit]
        floored[idx] += 1
    elif deficit < 0:
        idx = np.argsort(remainders)[:(-deficit)]
        floored[idx] -= 1
    return np.maximum(floored, 0)


# ══════════════════════════════════════════════════════════════════════════════
#  SINGLE-MODEL FITTING
# ══════════════════════════════════════════════════════════════════════════════

def train_total_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    weights: pd.Series,
    alpha: float = cfg.GLM_ALPHA,
):
    """Fit a regularised Poisson GLM for total_enc.

    Returns fitted GLMResultsRegularized object.
    """
    family = sm.families.Poisson(link=sm.families.links.Log())
    model = sm.GLM(
        endog=y_train.values,
        exog=X_train.values,
        family=family,
        freq_weights=weights.values,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit_regularized(
            alpha=alpha,
            L1_wt=cfg.GLM_L1_WT,
            maxiter=cfg.GLM_MAXITER,
        )
    # Attach column names for later prediction
    result._feature_names = list(X_train.columns)
    return result


def train_rate_model(
    X_train: pd.DataFrame,
    y_rate: pd.Series,
    weights: pd.Series,
    total_enc: pd.Series,
    alpha: float = cfg.GLM_ALPHA,
):
    """Fit a quasi-Binomial GLM for admit_rate.

    Uses total_enc as var_weights (number of trials) and sample_weight as
    freq_weights (COVID downweighting).
    """
    family = sm.families.Binomial(link=sm.families.links.Logit())

    # Clamp rate away from exact 0/1 to avoid log(0) in logit link
    y_clamped = y_rate.clip(1e-6, 1 - 1e-6).values

    # var_weights = number of Bernoulli trials; freq_weights = importance weight
    var_w = np.maximum(total_enc.values.astype(float), 1.0)

    model = sm.GLM(
        endog=y_clamped,
        exog=X_train.values,
        family=family,
        var_weights=var_w,
        freq_weights=weights.values,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit_regularized(
            alpha=alpha,
            L1_wt=cfg.GLM_L1_WT,
            maxiter=cfg.GLM_MAXITER,
        )
    result._feature_names = list(X_train.columns)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN ALL 32 MODELS
# ══════════════════════════════════════════════════════════════════════════════

def train_all_models(
    train_df: pd.DataFrame,
    fourier_config: list[dict] | None = None,
    alpha: float = cfg.GLM_ALPHA,
    *,
    verbose: bool = True,
) -> dict:
    """Train 16 total_enc + 16 admit_rate models (one per site × block).

    Returns dict keyed by (site, block) → {"total_model": ..., "rate_model": ...}
    """
    models: dict[tuple, dict] = {}

    for site in cfg.SITES:
        for block in cfg.BLOCKS:
            series = get_site_block_subset(train_df, site, block)
            if len(series) == 0:
                if verbose:
                    print(f"  [{site}, Block {block}] SKIPPED — no training data")
                continue

            # Build design matrix
            X = build_design_matrix(series, fourier_config)
            y_total = series["total_enc"]
            y_rate  = series["admit_rate"]
            weights = series["sample_weight"]

            # Drop any rows with NaN in design matrix
            valid_mask = X.notna().all(axis=1) & y_total.notna()
            X_c      = X[valid_mask]
            y_total_c = y_total[valid_mask]
            y_rate_c  = y_rate[valid_mask]
            w_c       = weights[valid_mask]

            if len(X_c) < 50:
                if verbose:
                    print(f"  [{site}, Block {block}] SKIPPED — only {len(X_c)} valid rows")
                continue

            # Train total_enc (Poisson)
            total_model = train_total_model(X_c, y_total_c, w_c, alpha)

            # Train admit_rate (Binomial)
            rate_model = train_rate_model(X_c, y_rate_c, w_c, y_total_c, alpha)

            models[(site, block)] = {
                "total_model": total_model,
                "rate_model":  rate_model,
            }

            if verbose:
                # In-sample fit check
                eta = np.clip(X_c.values @ total_model.params, -20, 20)
                pred_total = np.exp(eta)
                mean_actual = y_total_c.mean()
                mean_pred   = pred_total.mean()
                print(f"  [{site}, Block {block}] n={len(X_c):,}  "
                      f"mean(y)={mean_actual:.1f}  mean(ŷ)={mean_pred:.1f}  "
                      f"n_params={len(total_model.params)}")

    return models


# ══════════════════════════════════════════════════════════════════════════════
#  SERIALISATION
# ══════════════════════════════════════════════════════════════════════════════

def save_models(
    models: dict,
    fold_id: int,
) -> None:
    """Persist fitted models + coefficient tables to disk."""
    fdir = cfg.fold_model_dir(fold_id)

    coeff_rows = []
    for (site, block), m in models.items():
        # Total model
        total_path = fdir / f"total_model_{site}_{block}.pkl"
        with open(total_path, "wb") as f:
            pickle.dump(m["total_model"], f)

        # Rate model
        rate_path = fdir / f"rate_model_{site}_{block}.pkl"
        with open(rate_path, "wb") as f:
            pickle.dump(m["rate_model"], f)

        # Coefficient table
        names = m["total_model"]._feature_names
        for i, name in enumerate(names):
            coeff_rows.append({
                "site": site,
                "block": block,
                "feature": name,
                "total_coef": m["total_model"].params[i],
                "rate_coef": m["rate_model"].params[i],
            })

    if coeff_rows:
        pd.DataFrame(coeff_rows).to_csv(fdir / "coefficients.csv", index=False)


def load_models(fold_id: int) -> dict | None:
    """Load previously saved models from disk."""
    fdir = cfg.fold_model_dir(fold_id)
    models = {}
    for site in cfg.SITES:
        for block in cfg.BLOCKS:
            total_path = fdir / f"total_model_{site}_{block}.pkl"
            rate_path  = fdir / f"rate_model_{site}_{block}.pkl"
            if not total_path.exists() or not rate_path.exists():
                return None
            with open(total_path, "rb") as f:
                total_model = pickle.load(f)
            with open(rate_path, "rb") as f:
                rate_model = pickle.load(f)
            models[(site, block)] = {
                "total_model": total_model,
                "rate_model":  rate_model,
            }
    return models


if __name__ == "__main__":
    from data_loader import load_data, get_fold_data

    df = load_data()
    train_df, val_df = get_fold_data(df, cfg.FOLDS[0])
    print(f"\n  Training on fold 1: {len(train_df):,} train rows ...")
    models = train_all_models(train_df)
    print(f"\n  Trained {len(models)} (site, block) model pairs")
