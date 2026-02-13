"""
Prediction & post-processing for Pipeline D.

Generates forecasts from fitted GLM models, enforces hard constraints
(non-negativity, admitted ≤ total, integer rounding via largest-remainder),
and produces eval.md-compatible submission CSVs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg
from data_loader import get_site_block_subset, get_fold_data
from features import build_design_matrix
from training import (
    train_models,
    train_all_models,
    predict_mixed_effects,
    save_models,
    largest_remainder_round,
    wape,
)


# ══════════════════════════════════════════════════════════════════════════════
#  RAW PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def predict_window(
    master_df: pd.DataFrame,
    models: dict,
    forecast_start: str,
    forecast_end: str,
    fourier_config: list[dict] | None = None,
) -> pd.DataFrame:
    """Generate raw predictions for all (site, block) on a date window."""
    model_type = models.get("model_type", "per_series")

    if model_type == "mixed_effects":
        return _predict_window_mixed(master_df, models, forecast_start, forecast_end, fourier_config)
    return _predict_window_per_series(master_df, models, forecast_start, forecast_end, fourier_config)


def _predict_window_mixed(
    master_df: pd.DataFrame,
    models: dict,
    forecast_start: str,
    forecast_end: str,
    fourier_config: list[dict] | None = None,
) -> pd.DataFrame:
    """Predict using unified mixed-effects models."""
    start = pd.Timestamp(forecast_start)
    end = pd.Timestamp(forecast_end)

    forecast_df = master_df[
        (master_df["date"] >= start) & (master_df["date"] <= end)
    ].copy()

    if forecast_df.empty:
        return pd.DataFrame()

    pred_total = predict_mixed_effects(models["total_model"], forecast_df, target="total")
    pred_rate = predict_mixed_effects(models["rate_model"], forecast_df, target="rate")
    pred_admitted = pred_total * pred_rate

    rows = []
    for i in range(len(forecast_df)):
        rows.append({
            "site": forecast_df.iloc[i]["site"],
            "date": forecast_df.iloc[i]["date"],
            "block": forecast_df.iloc[i]["block"],
            "pred_total": pred_total[i],
            "pred_rate": pred_rate[i],
            "pred_admitted": pred_admitted[i],
        })
    return pd.DataFrame(rows)


def _predict_window_per_series(
    master_df: pd.DataFrame,
    models: dict,
    forecast_start: str,
    forecast_end: str,
    fourier_config: list[dict] | None = None,
) -> pd.DataFrame:
    """Predict using per-series GLMs (original logic)."""
    fc = fourier_config or cfg.FOURIER_TERMS
    start = pd.Timestamp(forecast_start)
    end   = pd.Timestamp(forecast_end)

    rows: list[dict] = []

    for site in cfg.SITES:
        for block in cfg.BLOCKS:
            key = (site, block)
            if key not in models:
                continue

            total_model = models[key]["total_model"]
            rate_model  = models[key]["rate_model"]

            # Extract forecast-window subset from master grid
            series = get_site_block_subset(master_df, site, block)
            forecast_df = series[
                (series["date"] >= start) & (series["date"] <= end)
            ].copy()

            if forecast_df.empty:
                print(f"  WARNING: No data for {site} Block {block} in {start}..{end}")
                continue

            # Build design matrix (fully deterministic)
            X = build_design_matrix(forecast_df, fc)

            # Handle NaN in weather cols for future dates → climatology fill
            if X.isna().any().any():
                X = X.fillna(0.0)

            # ── Predict total_enc (Poisson log-link -> exp) ───────────────
            eta_total = X.values @ total_model.params
            eta_total = np.clip(eta_total, -20, 20)  # prevent exp overflow
            pred_total = np.exp(eta_total)

            # ── Predict admit_rate (Binomial logit-link -> sigmoid) ──────
            eta_rate = X.values @ rate_model.params
            eta_rate = np.clip(eta_rate, -20, 20)
            pred_rate = 1.0 / (1.0 + np.exp(-eta_rate))
            pred_rate = np.clip(pred_rate, 0, 1)

            # ── Derive admitted ──────────────────────────────────────────
            pred_admitted = pred_total * pred_rate

            for i in range(len(forecast_df)):
                date = forecast_df.iloc[i]["date"]
                rows.append({
                    "site":  site,
                    "date":  date,
                    "block": block,
                    "pred_total":    pred_total[i],
                    "pred_rate":     pred_rate[i],
                    "pred_admitted": pred_admitted[i],
                })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
#  POST-PROCESSING (CONSTRAINT ENFORCEMENT)
# ══════════════════════════════════════════════════════════════════════════════

def post_process(pred_df: pd.DataFrame) -> pd.DataFrame:
    """Apply all hard constraints from eval.md.

    1. Non-negativity
    2. Largest-remainder rounding per (site, date)
    3. Admitted ≤ total
    4. Integer types
    5. Strict date format
    """
    pred_df = pred_df.copy()

    # Clip negatives (Poisson log-link shouldn't produce any, but safety net)
    pred_df["pred_total"]    = pred_df["pred_total"].clip(lower=0)
    pred_df["pred_admitted"] = pred_df["pred_admitted"].clip(lower=0)

    # Largest-remainder rounding per (site, date) → preserves daily sum
    pred_df["ED Enc"]          = 0
    pred_df["ED Enc Admitted"] = 0

    for (site, date), grp in pred_df.groupby(["site", "date"]):
        idx = grp.index
        pred_df.loc[idx, "ED Enc"] = largest_remainder_round(
            grp["pred_total"].values
        )
        pred_df.loc[idx, "ED Enc Admitted"] = largest_remainder_round(
            grp["pred_admitted"].values
        )

    # Enforce admitted ≤ total (rounding can violate)
    pred_df["ED Enc"]          = pred_df["ED Enc"].astype(int)
    pred_df["ED Enc Admitted"] = np.minimum(
        pred_df["ED Enc Admitted"].astype(int),
        pred_df["ED Enc"],
    )

    # Format as eval.md submission
    submission = pd.DataFrame({
        "Site":            pred_df["site"].values,
        "Date":            pd.to_datetime(pred_df["date"]).dt.strftime("%Y-%m-%d").values,
        "Block":           pred_df["block"].values,
        "ED Enc":          pred_df["ED Enc"].values,
        "ED Enc Admitted": pred_df["ED Enc Admitted"].values,
    })

    return submission


# ══════════════════════════════════════════════════════════════════════════════
#  FOLD-LEVEL TRAINING + PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

def train_and_predict_fold(
    master_df: pd.DataFrame,
    fold: dict,
    fourier_config: list[dict] | None = None,
    alpha: float = cfg.GLM_ALPHA,
    *,
    save: bool = True,
) -> dict:
    """Train all models on fold's training data, predict on val window, score.

    Returns dict with fold metrics and submission DataFrame.
    """
    fold_id = fold["id"]
    train_df, val_df = get_fold_data(master_df, fold)

    # ── Train models (mixed-effects attempt → per-series fallback) ─────
    models = train_models(train_df, fourier_config, alpha, verbose=True)

    # ── Predict on validation window ─────────────────────────────────────
    raw_pred = predict_window(
        master_df, models, fold["val_start"], fold["val_end"], fourier_config
    )
    submission = post_process(raw_pred)

    # ── Score ────────────────────────────────────────────────────────────
    actual = val_df[["site", "date", "block", "total_enc", "admitted_enc"]].copy()
    actual["Date"] = actual["date"].dt.strftime("%Y-%m-%d")

    merged = submission.merge(
        actual.rename(columns={"site": "Site", "block": "Block"}),
        on=["Site", "Date", "Block"],
        how="left",
    )

    total_wape_val    = wape(merged["total_enc"].values, merged["ED Enc"].values)
    admitted_wape_val = wape(merged["admitted_enc"].values, merged["ED Enc Admitted"].values)

    print(f"    Fold {fold_id}: total_wape={total_wape_val:.4f}  "
          f"admitted_wape={admitted_wape_val:.4f}  "
          f"({len(submission)} rows)")

    # ── Save ─────────────────────────────────────────────────────────────
    if save:
        cfg.ensure_dirs()
        submission.to_csv(
            cfg.PRED_DIR / f"fold_{fold_id}_predictions.csv", index=False
        )
        save_models(models, fold_id)

    return {
        "fold_id":        fold_id,
        "total_wape":     total_wape_val,
        "admitted_wape":  admitted_wape_val,
        "models":         models,
        "submission":     submission,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  FINAL SEPT-OCT 2025 FORECAST
# ══════════════════════════════════════════════════════════════════════════════

def generate_final_forecast(
    master_df: pd.DataFrame,
    fourier_config: list[dict] | None = None,
    alpha: float = cfg.GLM_ALPHA,
) -> pd.DataFrame:
    """Train on all data ≤ 2025-08-31, predict Sept-Oct 2025 (976 rows)."""
    train_end  = "2025-08-31"
    pred_start = "2025-09-01"
    pred_end   = "2025-10-31"

    train_df = master_df[master_df["date"] <= pd.Timestamp(train_end)].copy()

    print(f"  Training on {len(train_df):,} rows (through {train_end}) ...")
    models = train_models(train_df, fourier_config, alpha, verbose=True)

    raw_pred = predict_window(master_df, models, pred_start, pred_end, fourier_config)
    submission = post_process(raw_pred)

    # ── Sanity checks ────────────────────────────────────────────────────
    expected_rows = (
        len(cfg.SITES) * len(pd.date_range(pred_start, pred_end)) * len(cfg.BLOCKS)
    )
    assert len(submission) == expected_rows, (
        f"Row count {len(submission)} != expected {expected_rows}"
    )
    assert (submission["ED Enc"] >= 0).all(), "Negative ED Enc"
    assert (submission["ED Enc Admitted"] <= submission["ED Enc"]).all(), "admitted > total"

    # ── Save ─────────────────────────────────────────────────────────────
    cfg.ensure_dirs()
    out_path = cfg.PRED_DIR / "final_sept_oct_2025.csv"
    submission.to_csv(out_path, index=False)
    print(f"  Saved {len(submission)} rows -> {out_path}")

    # Save final models
    save_models(models, fold_id=0)

    return submission


if __name__ == "__main__":
    from data_loader import load_data

    df = load_data()

    # Quick smoke test: single fold
    result = train_and_predict_fold(df, cfg.FOLDS[0], save=False)
    print(f"\n  Fold 1 admitted WAPE: {result['admitted_wape']:.4f}")
