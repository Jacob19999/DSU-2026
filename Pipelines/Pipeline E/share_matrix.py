"""
Build reason-category share matrix for Pipeline E.

Transforms raw count_reason_* columns into normalised shares (sum ≈ 1.0
per row), with optional rolling smoothing to stabilise noisy block-level
estimates where small denominators dominate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg


def get_reason_columns(df: pd.DataFrame) -> list[str]:
    """Discover all count_reason_* columns in the DataFrame."""
    return sorted([c for c in df.columns if c.startswith("count_reason_")])


def build_share_matrix(
    df: pd.DataFrame,
    top_n: int = cfg.TOP_N_REASONS,
    min_volume: int = cfg.MIN_CATEGORY_VOLUME,
    smooth_window: int = cfg.SHARE_SMOOTH_WINDOW,
) -> tuple[pd.DataFrame, list[str]]:
    """Build smoothed share matrix from reason count columns.

    Returns
    -------
    df : DataFrame with share_* columns appended.
    share_cols : list of share column names (ordered).
    """
    df = df.copy()
    reason_cols = get_reason_columns(df)

    # ── Identify top categories by volume ────────────────────────────────
    category_volumes = df[reason_cols].sum().sort_values(ascending=False)
    top_categories = category_volumes[
        category_volumes >= min_volume
    ].head(top_n).index.tolist()

    # ── Aggregate "other" bucket ─────────────────────────────────────────
    # Separate count_reason_other (Data Source catch-all) from our selection
    selected = [c for c in top_categories if c != "count_reason_other"]
    remaining = [c for c in reason_cols if c not in selected]
    df["_other_combined"] = df[remaining].sum(axis=1)

    # ── Compute shares ───────────────────────────────────────────────────
    all_count_cols = selected + ["_other_combined"]
    row_totals = df[all_count_cols].sum(axis=1).clip(lower=1)  # avoid div/0

    share_cols: list[str] = []
    for col in selected:
        share_name = col.replace("count_reason_", "share_")
        df[share_name] = df[col] / row_totals
        share_cols.append(share_name)

    df["share_other"] = df["_other_combined"] / row_totals
    share_cols.append("share_other")

    # ── Rolling smoothing per (site, block) ──────────────────────────────
    if smooth_window > 1:
        for col in share_cols:
            df[col] = (
                df.groupby(["site", "block"])[col]
                .transform(lambda s: s.rolling(smooth_window, min_periods=1).mean())
            )

    # ── Cleanup ──────────────────────────────────────────────────────────
    df.drop(columns=["_other_combined"], inplace=True, errors="ignore")

    # ── Validation ───────────────────────────────────────────────────────
    share_sum = df[share_cols].sum(axis=1)
    max_dev = (share_sum - 1.0).abs().max()
    zero_rows = (row_totals <= 1).sum()

    print(f"  Share columns: {len(share_cols)}")
    print(f"  Max share-sum deviation from 1.0: {max_dev:.4f}")
    print(f"  Zero-visit rows (clipped): {zero_rows:,}")
    top5 = df[share_cols].mean().nlargest(5)
    for name, val in top5.items():
        print(f"    {name}: {val:.4f}")

    return df, share_cols


if __name__ == "__main__":
    from data_loader import load_data

    df = load_data()
    df, share_cols = build_share_matrix(df)
    print(f"\nFinal share columns ({len(share_cols)}): {share_cols}")
