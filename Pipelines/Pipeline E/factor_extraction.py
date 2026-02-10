"""
PCA / NMF factor extraction for Pipeline E.

Reduces the ~20-dimensional share matrix to k latent factors that capture
dominant composition patterns (e.g. "respiratory season", "trauma", "GI").

CRITICAL: PCA/NMF is fit on TRAINING data only per fold to prevent leakage.
The fitted transform is then applied to all rows (train + val).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, NMF

import config as cfg


def fit_and_transform_factors(
    df: pd.DataFrame,
    share_cols: list[str],
    train_end: str | pd.Timestamp,
    n_factors: int = cfg.N_FACTORS,
    method: str = cfg.FACTOR_METHOD,
) -> tuple[pd.DataFrame, object, list[str]]:
    """Fit factor model on training data only, transform all rows.

    Returns
    -------
    df          : DataFrame with factor_0 … factor_{k-1} columns added.
    model       : Fitted PCA or NMF model object.
    factor_cols : List of factor column names.
    """
    df = df.copy()
    train_mask = df["date"] <= pd.Timestamp(train_end)

    X_train = df.loc[train_mask, share_cols].values
    X_all   = df[share_cols].values

    # ── Fit ───────────────────────────────────────────────────────────────
    if method == "pca":
        model = PCA(n_components=n_factors, random_state=cfg.SEED)
    elif method == "nmf":
        # NMF requires non-negative input; shares are [0,1] so this is fine
        model = NMF(
            n_components=n_factors, init="nndsvda",
            max_iter=500, random_state=cfg.SEED,
        )
    else:
        raise ValueError(f"Unknown factor method: {method}")

    model.fit(X_train)
    factors_all = model.transform(X_all)

    # ── Add factor columns ───────────────────────────────────────────────
    factor_cols: list[str] = []
    for i in range(n_factors):
        col = f"factor_{i}"
        df[col] = factors_all[:, i]
        factor_cols.append(col)

    # ── Diagnostics ──────────────────────────────────────────────────────
    if method == "pca":
        ev = model.explained_variance_ratio_
        cumev = np.cumsum(ev)
        print(f"  PCA explained variance: {[f'{v:.3f}' for v in ev]}")
        print(f"  Cumulative ({n_factors} factors): {cumev[-1]:.3f}")
    else:
        recon = model.inverse_transform(factors_all)
        err = np.mean((X_all - recon) ** 2)
        print(f"  NMF reconstruction MSE: {err:.6f}")

    print(f"  Factor columns: {factor_cols}")
    print(f"  Training rows for fit: {train_mask.sum():,}")

    return df, model, factor_cols


def print_factor_loadings(
    model,
    share_cols: list[str],
    method: str = cfg.FACTOR_METHOD,
) -> None:
    """Print top contributors per factor for interpretation."""
    if method == "pca":
        loadings = pd.DataFrame(
            model.components_.T,
            index=share_cols,
            columns=[f"PC{i}" for i in range(model.n_components)],
        )
    else:
        # NMF: components_ rows are factors, columns are features
        loadings = pd.DataFrame(
            model.components_.T,
            index=share_cols,
            columns=[f"NMF{i}" for i in range(model.n_components)],
        )

    print("  Factor loadings (top 3 contributors per factor):")
    for i in range(loadings.shape[1]):
        col = loadings.columns[i]
        top_pos = loadings[col].nlargest(3).index.tolist()
        top_neg = loadings[col].nsmallest(3).index.tolist()
        print(f"    {col}: + {top_pos},  − {top_neg}")


if __name__ == "__main__":
    from data_loader import load_data
    from share_matrix import build_share_matrix

    df = load_data()
    df, share_cols = build_share_matrix(df)
    df, model, factor_cols = fit_and_transform_factors(
        df, share_cols, "2024-12-31",
    )
    print_factor_loadings(model, share_cols)
