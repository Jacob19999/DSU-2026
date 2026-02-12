"""
Step 2: Feature engineering — lags, rolling, calendar, cyclical,
holiday proximity, school proximity, weather, case-mix shares,
interaction features, and sample weights.

All per-(site, block) features use shifts >= MAX_HORIZON (63d) to
prevent future leakage.  Aggregate mean-encodings are NOT computed
here — they must be recomputed per fold in step_03 to avoid leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays as _holidays_lib
except ImportError:
    _holidays_lib = None

import config as cfg

# ── Datetime helpers ─────────────────────────────────────────────────────────

_EPOCH_D = np.datetime64("1970-01-01", "D")


def _to_days(dt_array) -> np.ndarray:
    """Convert datetime-like array to int (days since unix epoch)."""
    return (np.asarray(dt_array, dtype="datetime64[D]") - _EPOCH_D).astype(np.int64)


def _days_since_last(dates: pd.Series, ref_dates) -> np.ndarray:
    """Vectorized: days since most-recent ref_date <= each date."""
    d = _to_days(dates.values)
    r = np.sort(_to_days(np.asarray(ref_dates, dtype="datetime64[D]")))
    idx = np.searchsorted(r, d, side="right") - 1
    out = np.full(len(d), np.nan)
    valid = idx >= 0
    out[valid] = d[valid] - r[idx[valid]]
    return out


def _days_until_next(dates: pd.Series, ref_dates) -> np.ndarray:
    """Vectorized: days until next ref_date >= each date."""
    d = _to_days(dates.values)
    r = np.sort(_to_days(np.asarray(ref_dates, dtype="datetime64[D]")))
    idx = np.searchsorted(r, d, side="left")
    out = np.full(len(d), np.nan)
    valid = idx < len(r)
    out[valid] = r[idx[valid]] - d[valid]
    return out


def _days_to_nearest(dates: pd.Series, ref_dates) -> np.ndarray:
    """Vectorized: min absolute distance to nearest ref_date."""
    since = _days_since_last(dates, ref_dates)
    until = _days_until_next(dates, ref_dates)
    return np.fmin(since, until)          # np.fmin ignores NaN


# ── Feature Groups ───────────────────────────────────────────────────────────

def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per (site, block): lagged total_enc and admit_rate."""
    for (_site, _blk), grp in df.groupby(["site", "block"]):
        idx = grp.index
        for lag in cfg.LAG_DAYS:
            df.loc[idx, f"lag_{lag}"] = grp["total_enc"].shift(lag).values
            df.loc[idx, f"lag_admit_{lag}"] = grp["admit_rate"].shift(lag).values
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per (site, block): rolling stats on total_enc, shifted by ROLLING_SHIFT."""
    for (_site, _blk), grp in df.groupby(["site", "block"]):
        idx = grp.index
        shifted = grp["total_enc"].shift(cfg.ROLLING_SHIFT)
        for w in cfg.ROLLING_WINDOWS:
            roll = shifted.rolling(w, min_periods=1)
            df.loc[idx, f"roll_mean_{w}"] = roll.mean().values
            df.loc[idx, f"roll_std_{w}"] = roll.std().values
            df.loc[idx, f"roll_min_{w}"] = roll.min().values
            df.loc[idx, f"roll_max_{w}"] = roll.max().values
    return df


def _add_trend_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Short-term momentum and horizon-boundary deltas (M5 trick)."""
    df["delta_7_28"] = df["roll_mean_7"] - df["roll_mean_28"]
    df["delta_28_91"] = df["roll_mean_28"] - df["roll_mean_91"]
    df["delta_lag_63_70"] = df["lag_63"] - df["lag_70"]
    return df


def _add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sin/cos encodings for DOW, day-of-year, month."""
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["year_frac"] = df["year"] + (df["day_of_year"] - 1) / 365.25
    return df


def _add_holiday_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """Holiday proximity features via the ``holidays`` library."""
    if _holidays_lib is None:
        print("  WARNING: holidays lib missing -- skipping holiday proximity")
        df["is_us_holiday"] = df.get("is_holiday", pd.Series(False, index=df.index)).astype(int)
        return df

    us_hol = _holidays_lib.US(years=range(2017, 2027))

    xmas = pd.to_datetime([f"{y}-12-25" for y in range(2017, 2026)])
    jul4 = pd.to_datetime([f"{y}-07-04" for y in range(2017, 2026)])
    tday = pd.to_datetime([d for d, n in sorted(us_hol.items()) if "Thanksgiving" in n])
    all_hol = pd.to_datetime(sorted(us_hol.keys()))

    # Binary flag (reuse master-data column when available)
    df["is_us_holiday"] = df["is_holiday"].astype(int) if "is_holiday" in df.columns else 0

    dates = df["date"]
    df["days_since_xmas"] = _days_since_last(dates, xmas)
    df["days_until_thanksgiving"] = _days_until_next(dates, tday)
    df["days_since_july4"] = _days_since_last(dates, jul4)
    df["days_to_nearest_holiday"] = _days_to_nearest(dates, all_hol)
    return df


def _add_school_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """Days since / until the nearest school-year start."""
    starts = pd.to_datetime(cfg.SCHOOL_STARTS)
    df["days_since_school_start"] = _days_since_last(df["date"], starts)
    df["days_until_school_start"] = _days_until_next(df["date"], starts)
    return df


def _add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Impute weather NaNs (ffill per site → monthly climatology), add temp_range."""
    weather_cols = [c for c in ["temp_min", "temp_max", "precip", "snowfall"] if c in df.columns]
    for col in weather_cols:
        df[col] = df.groupby("site")[col].ffill()
        if df[col].isna().any():
            clim = df.groupby(["site", "month"])[col].transform("mean")
            df[col] = df[col].fillna(clim)
    if "temp_max" in df.columns and "temp_min" in df.columns:
        df["temp_range"] = df["temp_max"] - df["temp_min"]
    return df


def _add_external_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    """Impute CDC ILI rate and AQI NaNs (ffill → monthly climatology), or exclude."""
    for col in ["cdc_ili_rate", "aqi"]:
        if col not in df.columns:
            continue
        # Forward-fill within each site (temporal continuity)
        df[col] = df.groupby("site")[col].ffill()
        # Back-fill any leading NaNs
        df[col] = df.groupby("site")[col].bfill()
        # Final fallback: monthly climatology per site
        if df[col].isna().any():
            clim = df.groupby(["site", "month"])[col].transform("mean")
            df[col] = df[col].fillna(clim)
    return df


def _add_case_mix_shares(df: pd.DataFrame) -> pd.DataFrame:
    """Reason-of-visit shares (top N), lagged by MAX_HORIZON per (site, block)."""
    reason_cols = sorted([c for c in df.columns if c.startswith("count_reason_") and c != "count_reason_other"])
    if not reason_cols:
        return df

    # Pick top N by historical volume
    top = df[reason_cols].sum().nlargest(cfg.TOP_N_SHARE_REASONS).index.tolist()

    # Denominator = sum of ALL reason cols (including other)
    all_reason = [c for c in df.columns if c.startswith("count_reason_")]
    denom = df[all_reason].sum(axis=1).clip(lower=1)

    share_cols: list[str] = []
    for col in top:
        scol = col.replace("count_", "share_")
        df[scol] = df[col] / denom
        share_cols.append(scol)

    # Lag shares by MAX_HORIZON per (site, block)
    for (_s, _b), grp in df.groupby(["site", "block"]):
        idx = grp.index
        for scol in share_cols:
            df.loc[idx, f"{scol}_lag63"] = grp[scol].shift(cfg.MAX_HORIZON).values

    return df


_EMB_SUMMARY_COLS = ["reason_emb_entropy", "reason_emb_norm", "reason_emb_cluster"]
_EMB_ROLLING_COLS = ["reason_emb_entropy", "reason_emb_norm"]  # cluster is categorical → no rolling
_EMB_ROLLING_WINDOWS = [7, 14, 28]


def _add_lagged_embedding_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """Lagged embedding summaries — safe proxies for contemporaneous reason_emb_*.

    Uses the same lag set as Pipeline A (cfg.LAG_DAYS, all >= MAX_HORIZON).
    Rolling means use shift(MAX_HORIZON) then a window, matching _add_rolling_features.
    See master_strategy §10.4 for rationale.
    """
    present = [c for c in _EMB_SUMMARY_COLS if c in df.columns]
    if not present:
        return df

    for (_site, _blk), grp in df.groupby(["site", "block"]):
        idx = grp.index
        for col in present:
            series = grp[col]
            # Point lags
            for lag in cfg.LAG_DAYS:
                df.loc[idx, f"lag_{col.replace('reason_emb_', 'emb_')}_{lag}"] = (
                    series.shift(lag).values
                )
            # Rolling means (entropy & norm only — cluster is categorical)
            if col in _EMB_ROLLING_COLS:
                shifted = series.shift(cfg.ROLLING_SHIFT)
                for w in _EMB_ROLLING_WINDOWS:
                    df.loc[idx, f"roll_{col.replace('reason_emb_', 'emb_')}_{w}"] = (
                        shifted.rolling(w, min_periods=1).mean().values
                    )
    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction terms for LightGBM splits."""
    is_hol = df["is_us_holiday"] if "is_us_holiday" in df.columns else 0
    is_we = df["is_weekend"].astype(int) if "is_weekend" in df.columns else 0
    df["holiday_x_block"] = is_hol * df["block"]
    df["weekend_x_block"] = is_we * df["block"]

    # Site × DOW and Site × Month categorical interactions (from implementation plan §2J)
    if "site_enc" in df.columns and "dow" in df.columns:
        df["site_x_dow"] = df["site_enc"] * 7 + df["dow"]
    if "site_enc" in df.columns and "month" in df.columns:
        df["site_x_month"] = df["site_enc"] * 12 + (df["month"] - 1)
    return df


def _add_sample_weights(df: pd.DataFrame) -> pd.DataFrame:
    """COVID downweighting + volume-based sample weights for A1 & A2."""
    if "is_covid_era" in df.columns:
        df["covid_weight"] = np.where(df["is_covid_era"], cfg.COVID_SAMPLE_WEIGHT, 1.0)
    else:
        print("  WARNING: 'is_covid_era' column missing — defaulting COVID weight to 1.0")
        df["covid_weight"] = 1.0
    df["volume_weight"] = df["total_enc"].clip(lower=1)
    df["sample_weight_a1"] = df["covid_weight"] * df["volume_weight"]
    df["sample_weight_a2"] = df["covid_weight"] * df["admitted_enc"].clip(lower=1)
    return df


def _encode_site(df: pd.DataFrame) -> pd.DataFrame:
    """Integer-encode site (A=0 … D=3) for LightGBM categorical handling."""
    site_map = {s: i for i, s in enumerate(cfg.SITES)}
    df["site_enc"] = df["site"].map(site_map).astype(int)
    return df


def _cast_bools(df: pd.DataFrame) -> pd.DataFrame:
    """Convert bool columns to int — avoids dtype issues in LightGBM."""
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)
    return df


# ── Main entry point ─────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Run all static feature engineering steps. Returns enriched copy."""
    df = df.copy()

    print("  [2a] Lag features ...")
    df = _add_lag_features(df)

    print("  [2b] Rolling features ...")
    df = _add_rolling_features(df)

    print("  [2c] Trend deltas ...")
    df = _add_trend_deltas(df)

    print("  [2d] Cyclical encodings ...")
    df = _add_cyclical_features(df)

    print("  [2e] Holiday proximity ...")
    df = _add_holiday_proximity(df)

    print("  [2f] School proximity ...")
    df = _add_school_proximity(df)

    print("  [2g] Weather features ...")
    df = _add_weather_features(df)

    print("  [2g2] External enrichment (CDC ILI, AQI) ...")
    df = _add_external_enrichment(df)

    print("  [2h] Case-mix shares ...")
    df = _add_case_mix_shares(df)

    print("  [2h2] Lagged embedding summaries ...")
    df = _add_lagged_embedding_summaries(df)

    print("  [2i] Site encoding ...")
    df = _encode_site(df)

    print("  [2j] Interaction features ...")
    df = _add_interaction_features(df)

    print("  [2k] Sample weights ...")
    df = _add_sample_weights(df)

    print("  [2l] Bool cast ...")
    df = _cast_bools(df)

    n_feat = len(get_feature_columns(df))
    print(f"  Feature engineering complete: {n_feat} model features")
    return df


# ── Fold-specific aggregate encodings (called from step_03) ──────────────────

def compute_fold_aggregate_encodings(
    df: pd.DataFrame, train_mask: pd.Series
) -> pd.DataFrame:
    """Compute mean-target encodings from TRAINING data only, map to all rows."""
    df = df.copy()
    train = df.loc[train_mask]
    fallback = float(train["total_enc"].mean())

    for group_cols, col_name in [
        (["site_enc", "month", "block"], "site_month_block_mean"),
        (["site_enc", "dow"],            "site_dow_mean"),
        (["site_enc", "month"],          "site_month_mean"),
    ]:
        means = train.groupby(group_cols)["total_enc"].mean().to_dict()
        keys = list(zip(*(df[c] for c in group_cols)))
        df[col_name] = [means.get(k, fallback) for k in keys]

    return df


# ── Feature column resolver ─────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return columns suitable as LightGBM features (excludes targets/weights/strings)."""
    exclude = {
        # Identifiers & targets
        "site", "date", "total_enc", "admitted_enc", "admit_rate",
        # Weights
        "covid_weight", "volume_weight", "sample_weight_a1", "sample_weight_a2",
        # Sparse string columns
        "event_name", "event_type",
        # Redundant with is_us_holiday
        "is_holiday",
    }
    # Raw reason counts (using lagged shares instead)
    exclude.update(c for c in df.columns if c.startswith("count_reason_"))
    # Current-period shares (leakage) — keep only _lag63 variants
    exclude.update(
        c for c in df.columns
        if c.startswith("share_reason_") and not c.endswith("_lag63")
    )
    # Raw reason embeddings — contemporaneous (target-date leakage, §10).
    # Lagged summaries (lag_emb_*, roll_emb_*) are kept as features.
    exclude.update(c for c in df.columns if c.startswith("reason_emb_"))
    # Safety: exclude any non-numeric columns (object, datetime, timedelta)
    non_numeric = set(
        df.select_dtypes(include=["object", "datetime64", "datetimetz", "timedelta64"]).columns
    )
    exclude.update(non_numeric)
    return sorted(c for c in df.columns if c not in exclude)


if __name__ == "__main__":
    from step_01_data_loading import load_data

    df = load_data()
    df = engineer_features(df)
    print(f"\nFeature columns ({len(get_feature_columns(df))}):")
    for c in get_feature_columns(df):
        print(f"  {c}")
