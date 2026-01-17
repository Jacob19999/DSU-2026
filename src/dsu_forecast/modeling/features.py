from __future__ import annotations

import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    sort_cols: list[str],
    target_cols: list[str],
    lags: list[int],
) -> pd.DataFrame:
    out = df.sort_values(group_cols + sort_cols).copy()
    g = out.groupby(group_cols, sort=False)
    for t in target_cols:
        for k in lags:
            out[f"{t}_lag_{k}"] = g[t].shift(k)
    return out


def add_rolling_mean_features(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    sort_cols: list[str],
    target_cols: list[str],
    windows: list[int],
) -> pd.DataFrame:
    out = df.sort_values(group_cols + sort_cols).copy()
    g = out.groupby(group_cols, sort=False)
    for t in target_cols:
        for w in windows:
            out[f"{t}_rmean_{w}"] = g[t].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
    return out

