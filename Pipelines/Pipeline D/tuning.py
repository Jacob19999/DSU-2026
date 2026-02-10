"""
Hyperparameter tuning for Pipeline D — two-phase grid search.

Phase 1: Coarse grid on (weekly_order × annual_order × alpha), fold 1 only.
Phase 2: Full 4-fold CV on top-K candidates from Phase 1.

Far fewer hyperparameters than GBDT pipelines → total tuning ~1-2 hours.
"""

from __future__ import annotations

import json
import time
import numpy as np
import pandas as pd

import config as cfg
from data_loader import get_fold_data, get_site_block_subset
from features import build_design_matrix
from training import train_total_model, train_rate_model, wape


def _evaluate_config(
    master_df: pd.DataFrame,
    fold: dict,
    fourier_config: list[dict],
    alpha: float,
) -> float:
    """Train all 16 total_enc models on one fold, return admitted WAPE on val set."""
    train_df, val_df = get_fold_data(master_df, fold)

    total_errors = 0.0
    total_actual = 0.0
    admitted_errors = 0.0
    admitted_actual = 0.0

    for site in cfg.SITES:
        for block in cfg.BLOCKS:
            # ── Train ────────────────────────────────────────────────────
            tr = get_site_block_subset(train_df, site, block)
            if len(tr) < 50:
                continue

            X_tr = build_design_matrix(tr, fourier_config)
            valid = X_tr.notna().all(axis=1) & tr["total_enc"].notna()
            X_tr = X_tr[valid]
            y_total = tr.loc[valid, "total_enc"]
            y_rate  = tr.loc[valid, "admit_rate"]
            w       = tr.loc[valid, "sample_weight"]

            try:
                total_model = train_total_model(X_tr, y_total, w, alpha)
                rate_model  = train_rate_model(X_tr, y_rate, w, y_total, alpha)
            except Exception:
                continue

            # ── Predict on val ───────────────────────────────────────────
            va = get_site_block_subset(val_df, site, block)
            if len(va) == 0:
                continue
            X_va = build_design_matrix(va, fourier_config)
            valid_va = X_va.notna().all(axis=1)
            X_va = X_va[valid_va]
            va = va[valid_va]

            eta_total = np.clip(X_va.values @ total_model.params, -20, 20)
            pred_total = np.exp(eta_total)
            eta_rate = np.clip(X_va.values @ rate_model.params, -20, 20)
            pred_rate = (1.0 / (1.0 + np.exp(-eta_rate))).clip(0, 1)
            pred_admitted = pred_total * pred_rate

            total_errors    += np.abs(va["total_enc"].values - pred_total).sum()
            total_actual    += va["total_enc"].values.sum()
            admitted_errors += np.abs(va["admitted_enc"].values - pred_admitted).sum()
            admitted_actual += va["admitted_enc"].values.sum()

    # Return admitted WAPE (primary metric)
    if admitted_actual > 0:
        return float(admitted_errors / admitted_actual)
    return float("inf")


def tune_pipeline_d(
    master_df: pd.DataFrame,
    *,
    top_k: int = 10,
    verbose: bool = True,
) -> dict:
    """Two-phase hyperparameter tuning.

    Phase 1: Coarse grid on fold 1 only (fast screening).
    Phase 2: Full 4-fold CV on top-K candidates.

    Returns best_config dict with keys: weekly_order, annual_order, alpha, fourier_config.
    """
    cfg.ensure_dirs()

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 1: Coarse grid (fold 1)
    # ══════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"  Phase 1: Coarse grid ({len(cfg.FOURIER_ORDER_SEARCH['weekly_order'])} x "
              f"{len(cfg.FOURIER_ORDER_SEARCH['annual_order'])} x "
              f"{len(cfg.ALPHA_SEARCH)} = "
              f"{len(cfg.FOURIER_ORDER_SEARCH['weekly_order']) * len(cfg.FOURIER_ORDER_SEARCH['annual_order']) * len(cfg.ALPHA_SEARCH)} combos, fold 1) ...")

    fold1 = cfg.FOLDS[0]
    results: list[dict] = []
    t0 = time.time()
    n_total = (len(cfg.FOURIER_ORDER_SEARCH["weekly_order"]) *
               len(cfg.FOURIER_ORDER_SEARCH["annual_order"]) *
               len(cfg.ALPHA_SEARCH))
    n_done = 0

    for w_order in cfg.FOURIER_ORDER_SEARCH["weekly_order"]:
        for a_order in cfg.FOURIER_ORDER_SEARCH["annual_order"]:
            for alpha in cfg.ALPHA_SEARCH:
                n_done += 1
                fourier_config = [
                    {"period": 7,      "order": w_order},
                    {"period": 365.25, "order": a_order},
                ]
                try:
                    fold1_wape = _evaluate_config(master_df, fold1, fourier_config, alpha)
                except Exception as e:
                    fold1_wape = float("inf")
                    if verbose:
                        print(f"    ERROR: w={w_order} a={a_order} alpha={alpha}: {e}")

                results.append({
                    "weekly_order": w_order,
                    "annual_order": a_order,
                    "alpha": alpha,
                    "fold1_wape": fold1_wape,
                })

                if verbose and n_done % 30 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / n_done * (n_total - n_done)
                    print(f"    {n_done}/{n_total} done ({elapsed:.0f}s elapsed, "
                          f"~{eta:.0f}s remaining)")

    # Sort by fold1 WAPE, take top K
    results.sort(key=lambda x: x["fold1_wape"])
    top_candidates = results[:top_k]

    if verbose:
        print(f"\n  Phase 1 top {top_k}:")
        for i, r in enumerate(top_candidates):
            print(f"    {i+1}. w={r['weekly_order']} a={r['annual_order']} "
                  f"a={r['alpha']:.3f} -> WAPE={r['fold1_wape']:.4f}")

    # ══════════════════════════════════════════════════════════════════════
    #  PHASE 2: Full 4-fold CV on top K
    # ══════════════════════════════════════════════════════════════════════
    if verbose:
        print(f"\n  Phase 2: Full 4-fold CV on top {top_k} candidates ...")

    best_config = None
    best_wape = float("inf")

    for r in top_candidates:
        fourier_config = [
            {"period": 7,      "order": r["weekly_order"]},
            {"period": 365.25, "order": r["annual_order"]},
        ]
        fold_wapes = []
        for fold in cfg.FOLDS:
            try:
                w = _evaluate_config(master_df, fold, fourier_config, r["alpha"])
                fold_wapes.append(w)
            except Exception:
                fold_wapes.append(float("inf"))

        mean_wape = float(np.mean(fold_wapes))

        if verbose:
            print(
                f"    w={r['weekly_order']} a={r['annual_order']} alpha={r['alpha']:.3f}  "
                f"-> folds=[{', '.join(f'{w:.4f}' for w in fold_wapes)}]  "
                f"mean={mean_wape:.4f}"
            )

        if mean_wape < best_wape:
            best_wape = mean_wape
            best_config = {
                "weekly_order": r["weekly_order"],
                "annual_order": r["annual_order"],
                "alpha":        r["alpha"],
                "mean_admitted_wape": mean_wape,
                "fold_wapes":   fold_wapes,
            }

    # Add fourier_config to best_config
    if best_config:
        best_config["fourier_config"] = [
            {"period": 7,      "order": best_config["weekly_order"]},
            {"period": 365.25, "order": best_config["annual_order"]},
        ]

    # ── Save ─────────────────────────────────────────────────────────────
    if best_config:
        save_path = cfg.MODEL_DIR / "best_config.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        serialisable = {k: v for k, v in best_config.items()}
        with open(save_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        if verbose:
            print(f"\n  Best config saved -> {save_path}")
            print(f"  Best: w={best_config['weekly_order']} a={best_config['annual_order']} "
                  f"a={best_config['alpha']:.3f} -> mean_admitted_wape={best_wape:.4f}")

    return best_config or {
        "weekly_order": cfg.FOURIER_TERMS[0]["order"],
        "annual_order": cfg.FOURIER_TERMS[1]["order"],
        "alpha": cfg.GLM_ALPHA,
        "fourier_config": cfg.FOURIER_TERMS,
    }


def load_best_config() -> dict | None:
    """Load previously saved best config from disk."""
    path = cfg.MODEL_DIR / "best_config.json"
    if not path.exists():
        return None
    with open(path) as f:
        config = json.load(f)
    # Reconstruct fourier_config if missing
    if "fourier_config" not in config:
        config["fourier_config"] = [
            {"period": 7,      "order": config["weekly_order"]},
            {"period": 365.25, "order": config["annual_order"]},
        ]
    print(
        f"  Loaded tuned config: w={config['weekly_order']} "
        f"a={config['annual_order']} alpha={config['alpha']}"
    )
    return config


def get_default_config() -> dict:
    """Return default config (no tuning)."""
    return {
        "weekly_order":  cfg.FOURIER_TERMS[0]["order"],
        "annual_order":  cfg.FOURIER_TERMS[1]["order"],
        "alpha":         cfg.GLM_ALPHA,
        "fourier_config": cfg.FOURIER_TERMS,
    }


if __name__ == "__main__":
    from data_loader import load_data

    df = load_data()
    best = tune_pipeline_d(df)
    print(f"\n  Final best: {best}")
