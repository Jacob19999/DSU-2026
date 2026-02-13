"""
GLM training for Pipeline D.

Supports two modes (config.MODEL_TYPE):
  "mixed_effects" (DEFAULT):
    Single statsmodels.mixedlm with (1 | site) + (1 | site:block) random
    intercepts replaces 16 per-series GLMs.  Principled partial pooling
    shrinks Site D toward the population mean.
  "per_series" (FALLBACK):
    16 Poisson GLMs (total_enc) + 16 Binomial GLMs (admit_rate).
    One model per (Site, Block) pair.

COVID-era rows are downweighted via freq_weights / sample_weight.
"""

from __future__ import annotations

import pickle
import warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

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
    alpha_vec = np.array([0.0] + [alpha] * (X_train.shape[1] - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit_regularized(
            alpha=alpha_vec,
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
    alpha_vec = np.array([0.0] + [alpha] * (X_train.shape[1] - 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = model.fit_regularized(
            alpha=alpha_vec,
            L1_wt=cfg.GLM_L1_WT,
            maxiter=cfg.GLM_MAXITER,
        )
    result._feature_names = list(X_train.columns)
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  MIXED-EFFECTS MODEL (replaces 16 per-series GLMs)
# ══════════════════════════════════════════════════════════════════════════════

def train_mixed_effects_total(
    train_df: pd.DataFrame,
    fourier_config: list[dict] | None = None,
    *,
    verbose: bool = True,
):
    """Fit a single log-Gaussian mixed-effects model for total_enc.

    Fixed effects: Fourier + DOW + holidays + trend + weather (shared).
    Random effects: (1 | site) + (1 | site_block) intercepts.

    Returns the fitted MixedLMResults object.
    """
    df = train_df.copy()

    # Build design matrix features (without intercept for formula-based API)
    X = build_design_matrix(df, fourier_config)
    feature_cols = [c for c in X.columns if c != "const"]

    # Merge features into df
    for col in feature_cols:
        df[col] = X[col].values

    # Create grouping variables
    df["site_block"] = df["site"].astype(str) + "_" + df["block"].astype(str)

    # Log-transform target (log1p for zero-safety)
    df["log_total"] = np.log1p(df["total_enc"])

    # Build formula string
    formula_rhs = " + ".join(feature_cols)
    formula = f"log_total ~ {formula_rhs}"

    # Fit mixed-effects model
    try:
        model = smf.mixedlm(
            formula=formula,
            data=df,
            groups=df["site"],
            re_formula="1",
            vc_formula={"site_block": "0 + C(site_block)"},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(method="lbfgs", maxiter=500, reml=True)

        if verbose:
            print(f"  [Mixed-effects total_enc] converged={result.converged}  "
                  f"n={len(df):,}  n_groups(site)={df['site'].nunique()}  "
                  f"n_groups(site_block)={df['site_block'].nunique()}")
            # Print random effects summary
            re = result.random_effects
            for site, effects in re.items():
                intercept = effects.get("Group", effects.get("Intercept", 0.0))
                print(f"    Site {site}: random intercept = {intercept:+.4f}")

        result._feature_cols = feature_cols
        result._fourier_config = fourier_config
        return result

    except Exception as e:
        if verbose:
            print(f"  [Mixed-effects total_enc] FAILED: {e}")
            print("  Falling back to per-series GLMs...")
        return None


def train_mixed_effects_rate(
    train_df: pd.DataFrame,
    fourier_config: list[dict] | None = None,
    *,
    verbose: bool = True,
):
    """Fit a logit-transformed mixed-effects model for admit_rate.

    Encodes the 17.4% vs 31.6% structural gap via site random intercept.
    """
    df = train_df.copy()

    X = build_design_matrix(df, fourier_config)
    feature_cols = [c for c in X.columns if c != "const"]

    for col in feature_cols:
        df[col] = X[col].values

    df["site_block"] = df["site"].astype(str) + "_" + df["block"].astype(str)

    # Logit-transform admit_rate (clamp away from 0/1)
    rate_clamped = df["admit_rate"].clip(0.01, 0.99)
    df["logit_rate"] = np.log(rate_clamped / (1 - rate_clamped))

    formula_rhs = " + ".join(feature_cols)
    formula = f"logit_rate ~ {formula_rhs}"

    try:
        model = smf.mixedlm(
            formula=formula,
            data=df,
            groups=df["site"],
            re_formula="1",
            vc_formula={"site_block": "0 + C(site_block)"},
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(method="lbfgs", maxiter=500, reml=True)

        if verbose:
            print(f"  [Mixed-effects admit_rate] converged={result.converged}  "
                  f"n={len(df):,}")
            re = result.random_effects
            for site, effects in re.items():
                intercept = effects.get("Group", effects.get("Intercept", 0.0))
                print(f"    Site {site}: random intercept = {intercept:+.4f}")

        result._feature_cols = feature_cols
        result._fourier_config = fourier_config
        return result

    except Exception as e:
        if verbose:
            print(f"  [Mixed-effects admit_rate] FAILED: {e}")
            print("  Falling back to per-series GLMs...")
        return None


def predict_mixed_effects(
    model_result,
    pred_df: pd.DataFrame,
    target: str = "total",
) -> np.ndarray:
    """Generate predictions from a mixed-effects model.

    For total: predictions are on log1p scale → expm1 back to counts.
    For rate:  predictions are on logit scale → expit back to [0,1].
    """
    df = pred_df.copy()
    feature_cols = model_result._feature_cols
    fourier_config = model_result._fourier_config

    X = build_design_matrix(df, fourier_config)
    for col in feature_cols:
        if col in X.columns:
            df[col] = X[col].values
        else:
            df[col] = 0.0

    df["site_block"] = df["site"].astype(str) + "_" + df["block"].astype(str)

    pred = model_result.predict(df)

    if target == "total":
        return np.expm1(pred.values).clip(0)
    else:
        # Inverse logit (expit)
        return (1 / (1 + np.exp(-pred.values))).clip(0, 1)


def train_mixed_effects_all(
    train_df: pd.DataFrame,
    fourier_config: list[dict] | None = None,
    *,
    verbose: bool = True,
) -> dict | None:
    """Train mixed-effects models for both total_enc and admit_rate.

    Returns dict {"total_model": ..., "rate_model": ..., "model_type": "mixed_effects"}
    or None if training fails (caller should fall back to per-series).
    """
    total_model = train_mixed_effects_total(train_df, fourier_config, verbose=verbose)
    if total_model is None:
        return None

    rate_model = train_mixed_effects_rate(train_df, fourier_config, verbose=verbose)
    if rate_model is None:
        return None

    return {
        "total_model": total_model,
        "rate_model": rate_model,
        "model_type": "mixed_effects",
    }


# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN ALL 32 MODELS (per-series fallback)
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
                    print(f"  [{site}, Block {block}] SKIPPED - no training data")
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
                    print(f"  [{site}, Block {block}] SKIPPED - only {len(X_c)} valid rows")
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
                print(
                    f"  [{site}, Block {block}] n={len(X_c):,}  "
                    f"mean(y)={mean_actual:.1f}  mean(y_hat)={mean_pred:.1f}  "
                    f"n_params={len(total_model.params)}"
                )

    return models


def train_models(
    train_df: pd.DataFrame,
    fourier_config: list[dict] | None = None,
    alpha: float = cfg.GLM_ALPHA,
    *,
    verbose: bool = True,
) -> dict:
    """Unified entry point: tries mixed-effects first, falls back to per-series.

    Returns either:
      - {"model_type": "mixed_effects", "total_model": MixedLMResult, "rate_model": MixedLMResult}
      - Per-series dict keyed by (site, block) → {"total_model": ..., "rate_model": ...}
        with added key "model_type": "per_series"
    """
    model_type = getattr(cfg, "MODEL_TYPE", "per_series")

    if model_type == "mixed_effects":
        if verbose:
            print("  Attempting mixed-effects model (single pooled model)...")
        result = train_mixed_effects_all(train_df, fourier_config, verbose=verbose)
        if result is not None:
            return result
        if verbose:
            print("  Mixed-effects failed → falling back to per-series GLMs")

    # Per-series fallback
    if verbose:
        print("  Training 32 per-series GLMs...")
    models = train_all_models(train_df, fourier_config, alpha, verbose=verbose)
    models["model_type"] = "per_series"
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

    model_type = models.get("model_type", "per_series")

    if model_type == "mixed_effects":
        # Save the two mixed-effects models
        with open(fdir / "mixed_total_model.pkl", "wb") as f:
            pickle.dump(models["total_model"], f)
        with open(fdir / "mixed_rate_model.pkl", "wb") as f:
            pickle.dump(models["rate_model"], f)
        # Save type marker
        (fdir / "model_type.txt").write_text("mixed_effects")
        return

    # Per-series models
    (fdir / "model_type.txt").write_text("per_series")
    coeff_rows = []
    for key, m in models.items():
        if key == "model_type" or not isinstance(key, tuple):
            continue
        site, block = key

        total_path = fdir / f"total_model_{site}_{block}.pkl"
        with open(total_path, "wb") as f:
            pickle.dump(m["total_model"], f)

        rate_path = fdir / f"rate_model_{site}_{block}.pkl"
        with open(rate_path, "wb") as f:
            pickle.dump(m["rate_model"], f)

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
    """Load previously saved models from disk (handles both model types)."""
    fdir = cfg.fold_model_dir(fold_id)

    # Check model type
    type_path = fdir / "model_type.txt"
    if type_path.exists():
        model_type = type_path.read_text().strip()
    else:
        model_type = "per_series"

    if model_type == "mixed_effects":
        total_path = fdir / "mixed_total_model.pkl"
        rate_path = fdir / "mixed_rate_model.pkl"
        if not total_path.exists() or not rate_path.exists():
            return None
        with open(total_path, "rb") as f:
            total_model = pickle.load(f)
        with open(rate_path, "rb") as f:
            rate_model = pickle.load(f)
        return {
            "total_model": total_model,
            "rate_model": rate_model,
            "model_type": "mixed_effects",
        }

    # Per-series fallback
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
    models["model_type"] = "per_series"
    return models


if __name__ == "__main__":
    from data_loader import load_data, get_fold_data

    df = load_data()
    train_df, val_df = get_fold_data(df, cfg.FOLDS[0])
    print(f"\n  Training on fold 1: {len(train_df):,} train rows ...")
    models = train_all_models(train_df)
    print(f"\n  Trained {len(models)} (site, block) model pairs")
