"""
Step 1: Load master_block_history.parquet, validate schema, derive admit_rate.

Consumes the unified dataset produced by the Data Source pipeline.
No raw CSV processing happens here — that's the Data Source's job.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config as cfg


def load_data() -> pd.DataFrame:
    """Load master parquet, validate, derive admit_rate, return enriched df."""
    print(f"  Loading {cfg.MASTER_PARQUET} ...")
    df = pd.read_parquet(cfg.MASTER_PARQUET)

    # ── Schema validation ────────────────────────────────────────────────
    required = ["site", "date", "block", "total_enc", "admitted_enc"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in master parquet: {missing}")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["site", "date", "block"]).reset_index(drop=True)

    # ── Row-count sanity ─────────────────────────────────────────────────
    n_sites = df["site"].nunique()
    n_dates = df["date"].nunique()
    n_blocks = df["block"].nunique()
    expected = n_sites * n_dates * n_blocks
    assert len(df) == expected, (
        f"Row count mismatch: got {len(df)}, expected {expected} "
        f"({n_sites} sites x {n_dates} dates x {n_blocks} blocks)"
    )

    # ── Target integrity ─────────────────────────────────────────────────
    assert df["total_enc"].notna().all(), "NaN found in total_enc"
    assert df["admitted_enc"].notna().all(), "NaN found in admitted_enc"
    assert (df["total_enc"] >= 0).all(), "Negative total_enc found"
    assert (df["admitted_enc"] <= df["total_enc"]).all(), "admitted_enc > total_enc"

    # ── Derive admit_rate (0/0 → 0.0) ───────────────────────────────────
    df["admit_rate"] = np.where(
        df["total_enc"] > 0,
        df["admitted_enc"] / df["total_enc"],
        0.0,
    )

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"  {len(df):,} rows | {n_sites} sites | {n_dates:,} dates | {n_blocks} blocks")
    print(f"  Date range: {df['date'].min().date()} .. {df['date'].max().date()}")
    for t in ["total_enc", "admitted_enc"]:
        print(f"  {t}: mean={df[t].mean():.1f}  std={df[t].std():.1f}  max={df[t].max()}")
    print(f"  admit_rate: mean={df['admit_rate'].mean():.3f}")

    return df


if __name__ == "__main__":
    load_data()
