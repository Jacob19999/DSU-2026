from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd


def add_block(df: pd.DataFrame, *, hour_col: str = "Hour") -> pd.DataFrame:
    out = df.copy()
    out["Block"] = (pd.to_numeric(out[hour_col]) // 6).astype(int)
    return out


def aggregate_targets_to_block(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate raw rows to (Site, Date, Block) with target sums.
    """
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = add_block(out)
    agg = (
        out.groupby(["Site", "Date", "Block"], as_index=False)
        .agg(total_enc=("ED Enc", "sum"), admitted_enc=("ED Enc Admitted", "sum"))
        .sort_values(["Site", "Date", "Block"])
    )
    return agg


def make_forecast_grid(
    *,
    sites: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    days = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date), freq="D")
    blocks = [0, 1, 2, 3]
    idx = pd.MultiIndex.from_product([sites, days, blocks], names=["Site", "Date", "Block"])
    out = idx.to_frame(index=False)
    return out

