"""
Step 2: Feature engineering for the daily-level GBDT model.

Produces features for predicting daily total_enc and admit_rate per (site, date).
Structurally similar to Pipeline A but operates on daily aggregates — no block dimension.
All lags >= 63 days (MAX_HORIZON) to prevent future leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays as _holidays_lib
except ImportError:
    _holidays_lib = None

import config as cfg

# ── Datetime helpers (vectorized, same as Pipeline A) ────────────────────────

_EPOCH_D = np.datetime64("1970-01-01", "D")


def _to_days(dt_array) -> np.ndarray:
    return (np.asarray(dt_array, dtype="datetime64[D]") - _EPOCH_D).astype(np.int64)


def _days_since_last(dates: pd.Series, ref_dates) -> np.ndarray:
    d = _to_days(dates.values)
    r = np.sort(_to_days(np.asarray(ref_dates, dtype="datetime64[D]")))
    idx = np.searchsorted(r, d, side="right") - 1
    out = np.full(len(d), np.nan)
    valid = idx >= 0
    out[valid] = d[valid] - r[idx[valid]]
    return out


def _days_until_next(dates: pd.Series, ref_dates) -> np.ndarray:
    d = _to_days(dates.values)
    r = np.sort(_to_days(np.asarray(ref_dates, dtype="datetime64[D]")))
    idx = np.searchsorted(r, d, side="left")
    out = np.full(len(d), np.nan)
    valid = idx < len(r)
    out[valid] = r[idx[valid]] - d[valid]
    return out


def _days_to_nearest(dates: pd.Series, ref_dates) -> np.ndarray:
    return np.fmin(_days_since_last(dates, ref_dates),
                   _days_until_next(dates, ref_dates))


# ── Feature Groups ───────────────────────────────────────────────────────────

def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per (site): lagged total_enc and admit_rate at daily grain."""
    for _site, grp in df.groupby("site"):
        idx = grp.index
        for lag in cfg.LAG_DAYS_DAILY:
            df.loc[idx, f"lag_{lag}"] = grp["total_enc"].shift(lag).values
            df.loc[idx, f"lag_admit_{lag}"] = grp["admit_rate"].shift(lag).values
    return df


def _add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Per (site): rolling stats on total_enc, shifted by ROLLING_SHIFT_DAILY."""
    for _site, grp in df.groupby("site"):
        idx = grp.index
        shifted = grp["total_enc"].shift(cfg.ROLLING_SHIFT_DAILY)
        for w in cfg.ROLLING_WINDOWS_DAILY:
            roll = shifted.rolling(w, min_periods=1)
            df.loc[idx, f"roll_mean_{w}"] = roll.mean().values
            df.loc[idx, f"roll_std_{w}"] = roll.std().values
            df.loc[idx, f"roll_min_{w}"] = roll.min().values
            df.loc[idx, f"roll_max_{w}"] = roll.max().values
    return df


def _add_trend_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum / trend signals from rolling & lag differences."""
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
    """Holiday proximity features via holidays lib."""
    if _holidays_lib is None:
        print("  WARNING: holidays lib missing — skipping holiday proximity")
        df["is_us_holiday"] = df.get("is_holiday", pd.Series(False, index=df.index)).astype(int)
        return df

    us_hol = _holidays_lib.US(years=range(2017, 2027))

    xmas = pd.to_datetime([f"{y}-12-25" for y in range(2017, 2026)])
    jul4 = pd.to_datetime([f"{y}-07-04" for y in range(2017, 2026)])
    tday = pd.to_datetime([d for d, n in sorted(us_hol.items()) if "Thanksgiving" in n])
    all_hol = pd.to_datetime(sorted(us_hol.keys()))

    df["is_us_holiday"] = df["is_holiday"].astype(int) if "is_holiday" in df.columns else 0

    dates = df["date"]
    df["days_since_xmas"] = _days_since_last(dates, xmas)
    df["days_until_thanksgiving"] = _days_until_next(dates, tday)
    df["days_since_july4"] = _days_since_last(dates, jul4)
    df["days_to_nearest_holiday"] = _days_to_nearest(dates, all_hol)
    return df


def _add_school_proximity(df: pd.DataFrame) -> pd.DataFrame:
    """Days since / until nearest school-year start."""
    starts = pd.to_datetime(cfg.SCHOOL_STARTS)
    df["days_since_school_start"] = _days_since_last(df["date"], starts)
    df["days_until_school_start"] = _days_until_next(df["date"], starts)
    return df


def _add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Impute weather NaNs (ffill per site → monthly climatology), add temp_range."""
    weather_cols = [c for c in ["temp_min", "temp_max", "precip", "snowfall"] if c in df.columns]
    for col in weather_cols:
        df[col] = df.groupby("site")[col].ffill()
        df[col] = df.groupby("site")[col].bfill()
        if df[col].isna().any():
            clim = df.groupby(["site", "month"])[col].transform("mean")
            df[col] = df[col].fillna(clim)
    if "temp_max" in df.columns and "temp_min" in df.columns:
        df["temp_range"] = df["temp_max"] - df["temp_min"]
    return df


def _add_target_encodings(df: pd.DataFrame) -> pd.DataFrame:
    """Site-level target encodings for the daily model.

    At the daily level (no block dimension), noise is lower (~71/day vs ~7-26/block).
    The residual correction is more reliable here than in Pipelines A/B/E.
    """
    for _site, grp in df.groupby("site"):
        idx = grp.index

        # 1–2. Site baseline volume (trailing 90-day daily mean, lagged)
        for target_col in ["total_enc", "admitted_enc"]:
            shifted = grp[target_col].shift(cfg.ROLLING_SHIFT_DAILY)
            df.loc[idx, f"te_site_mean_{target_col}"] = (
                shifted.rolling(90, min_periods=30).mean().values
            )

        # 3. Site admit rate (trailing 90-day, lagged)
        shifted_total = grp["total_enc"].shift(cfg.ROLLING_SHIFT_DAILY).rolling(90, min_periods=30).sum()
        shifted_admitted = grp["admitted_enc"].shift(cfg.ROLLING_SHIFT_DAILY).rolling(90, min_periods=30).sum()
        df.loc[idx, "te_site_admit_rate"] = (
            (shifted_admitted / shifted_total.clip(lower=1)).values
        )

    # 4. Site × DOW daily mean (trailing, lagged)
    for (_site, _dow), grp in df.groupby(["site", "dow"]):
        idx = grp.index
        shifted = grp["total_enc"].shift(cfg.ROLLING_SHIFT_DAILY)
        df.loc[idx, "te_site_dow_daily_mean"] = (
            shifted.rolling(90, min_periods=30).mean().values
        )

    return df


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Interaction terms — no block dimension for daily model."""
    is_hol = df.get("is_us_holiday", 0)
    is_we = df["is_weekend"].astype(int) if "is_weekend" in df.columns else 0
    df["holiday_x_site"] = is_hol * df["site_enc"]
    df["weekend_x_site"] = is_we * df["site_enc"]
    return df


def _add_sample_weights(df: pd.DataFrame) -> pd.DataFrame:
    """COVID downweighting + volume-based sample weights."""
    df["covid_weight"] = np.where(df["is_covid_era"], cfg.COVID_SAMPLE_WEIGHT, 1.0)
    df["volume_weight"] = df["total_enc"].clip(lower=1)
    df["sample_weight_total"] = df["covid_weight"] * df["volume_weight"]
    df["sample_weight_rate"] = df["covid_weight"] * df["admitted_enc"].clip(lower=1)
    return df


def _encode_site(df: pd.DataFrame) -> pd.DataFrame:
    """Integer-encode site for LightGBM."""
    site_map = {s: i for i, s in enumerate(cfg.SITES)}
    df["site_enc"] = df["site"].map(site_map).astype(int)
    return df


def _cast_bools(df: pd.DataFrame) -> pd.DataFrame:
    """Convert bool columns to int for LightGBM."""
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)
    return df


# ── Fold-specific aggregate encodings (called from step_04) ─────────────────

def compute_daily_aggregate_encodings(
    df: pd.DataFrame, train_mask: pd.Series
) -> pd.DataFrame:
    """Compute mean-target encodings from TRAINING data only, map to all rows."""
    df = df.copy()
    train = df.loc[train_mask]
    fallback = float(train["total_enc"].mean())

    for group_cols, col_name in [
        (["site_enc", "month"], "site_month_mean"),
        (["site_enc", "dow"], "site_dow_mean"),
    ]:
        means = train.groupby(group_cols)["total_enc"].mean().to_dict()
        keys = list(zip(*(df[c] for c in group_cols)))
        df[col_name] = [means.get(k, fallback) for k in keys]

    return df


# ── Feature column resolver ─────────────────────────────────────────────────

def get_daily_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return columns suitable as LightGBM features for the daily model."""
    exclude = {
        # Identifiers & targets
        "site", "date", "total_enc", "admitted_enc", "admit_rate",
        # Weights
        "covid_weight", "volume_weight", "sample_weight_total", "sample_weight_rate",
        # Sparse string cols
        "event_name", "event_type",
        # Redundant with is_us_holiday
        "is_holiday",
    }
    # Raw reason counts (not used in daily model)
    exclude.update(c for c in df.columns if c.startswith("count_reason_"))
    # Current-period shares (leakage)
    exclude.update(
        c for c in df.columns
        if c.startswith("share_reason_") and not c.endswith("_lag63")
    )
    return sorted(c for c in df.columns if c not in exclude)


# ── Main entry point ─────────────────────────────────────────────────────────

def engineer_daily_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Run all daily feature engineering steps. Returns enriched copy."""
    df = daily_df.copy()

    print("  [2a] Site encoding + bool cast ...")
    df = _encode_site(df)
    df = _cast_bools(df)

    print("  [2b] Lag features ...")
    df = _add_lag_features(df)

    print("  [2c] Rolling features ...")
    df = _add_rolling_features(df)

    print("  [2d] Trend deltas ...")
    df = _add_trend_deltas(df)

    print("  [2e] Cyclical encodings ...")
    df = _add_cyclical_features(df)

    print("  [2f] Holiday proximity ...")
    df = _add_holiday_proximity(df)

    print("  [2g] School proximity ...")
    df = _add_school_proximity(df)

    print("  [2h] Weather features ...")
    df = _add_weather_features(df)

    print("  [2h2] Target encodings (Site D isolation) ...")
    df = _add_target_encodings(df)

    print("  [2i] Interaction features ...")
    df = _add_interaction_features(df)

    print("  [2j] Sample weights ...")
    df = _add_sample_weights(df)

    n_feat = len(get_daily_feature_columns(df))
    print(f"  Daily feature engineering complete: {n_feat} model features")

    # Persist
    cfg.ensure_dirs()
    df.to_parquet(cfg.DATA_DIR / "daily_features.parquet", index=False)

    return df


if __name__ == "__main__":
    from step_01_data_loading import load_data

    _, daily_df, _ = load_data()
    df = engineer_daily_features(daily_df)
    feats = get_daily_feature_columns(df)
    print(f"\nDaily feature columns ({len(feats)}):")
    for c in feats:
        print(f"  {c}")
