"""
Feature engineering for Pipeline E.

Builds the full feature matrix for the final GBDT:
  - Static features: calendar, cyclical, holiday proximity, school, weather,
    interactions (identical to Pipeline B).
  - Target lags & rolling stats: total_enc / admit_rate shifted by >= 63.
  - Factor features (Pipeline E unique): predicted factors, momentum,
    yearly deviation, lagged actuals, rolling factor means.
  - Fold-specific mean-target encodings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays as _holidays_lib
except ImportError:
    _holidays_lib = None

import config as cfg

# ── Datetime helpers (vectorised distance computation) ────────────────────────

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
    since = _days_since_last(dates, ref_dates)
    until = _days_until_next(dates, ref_dates)
    return np.fmin(since, until)


# ══════════════════════════════════════════════════════════════════════════════
#  STATIC FEATURES — computed once on the base DataFrame
# ══════════════════════════════════════════════════════════════════════════════

def add_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all deterministic features derived from the target date."""
    df = df.copy()

    # ── Cyclical encodings ───────────────────────────────────────────────
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"] / 7)
    df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["year_frac"] = df["year"] + (df["day_of_year"] - 1) / 365.25

    # ── Site encoding ────────────────────────────────────────────────────
    site_map = {s: i for i, s in enumerate(cfg.SITES)}
    df["site_enc"] = df["site"].map(site_map).astype(int)

    # ── Holiday proximity ────────────────────────────────────────────────
    if _holidays_lib is not None:
        us_hol = _holidays_lib.US(years=range(2017, 2027))
        xmas = pd.to_datetime([f"{y}-12-25" for y in range(2017, 2026)])
        jul4 = pd.to_datetime([f"{y}-07-04" for y in range(2017, 2026)])
        tday = pd.to_datetime(
            [d for d, n in sorted(us_hol.items()) if "Thanksgiving" in n]
        )
        all_hol = pd.to_datetime(sorted(us_hol.keys()))

        df["is_us_holiday"] = (
            df["is_holiday"].astype(int) if "is_holiday" in df.columns else 0
        )
        dates = df["date"]
        df["days_since_xmas"]        = _days_since_last(dates, xmas)
        df["days_until_thanksgiving"] = _days_until_next(dates, tday)
        df["days_since_july4"]       = _days_since_last(dates, jul4)
        df["days_to_nearest_holiday"] = _days_to_nearest(dates, all_hol)
    else:
        df["is_us_holiday"] = (
            df.get("is_holiday", pd.Series(False, index=df.index)).astype(int)
        )

    # ── School proximity ─────────────────────────────────────────────────
    starts = pd.to_datetime(cfg.SCHOOL_STARTS)
    df["days_since_school_start"] = _days_since_last(df["date"], starts)
    df["days_until_school_start"] = _days_until_next(df["date"], starts)

    # ── Weather derived ──────────────────────────────────────────────────
    if "temp_max" in df.columns and "temp_min" in df.columns:
        df["temp_range"] = df["temp_max"] - df["temp_min"]

    # ── Interaction features ─────────────────────────────────────────────
    is_hol = df["is_us_holiday"] if "is_us_holiday" in df.columns else 0
    is_we  = df["is_weekend"].astype(int) if "is_weekend" in df.columns else 0
    df["holiday_x_block"] = is_hol * df["block"]
    df["weekend_x_block"] = is_we  * df["block"]
    df["site_x_dow"]      = df["site_enc"] * 7  + df["dow"]
    df["site_x_month"]    = df["site_enc"] * 12 + df["month"]

    # ── Cast bools → int (LightGBM compatibility) ────────────────────────
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  TARGET LAG & ROLLING FEATURES — computed once, shifted by MAX_HORIZON
# ══════════════════════════════════════════════════════════════════════════════

def add_target_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add total_enc / admit_rate lags and rolling stats shifted by >= 63."""
    df = df.copy()

    for (_site, _blk), grp in df.groupby(["site", "block"]):
        idx     = grp.index
        s_total = grp["total_enc"]
        s_rate  = grp["admit_rate"]

        # ── Target lags ──────────────────────────────────────────────────
        for lag in cfg.LAG_DAYS:
            df.loc[idx, f"lag_{lag}"]      = s_total.shift(lag).values
            df.loc[idx, f"lag_rate_{lag}"] = s_rate.shift(lag).values

        # ── Rolling stats (shifted by ROLLING_SHIFT = 63) ────────────────
        shifted = s_total.shift(cfg.ROLLING_SHIFT)
        for w in cfg.ROLLING_WINDOWS:
            roll = shifted.rolling(w, min_periods=1)
            df.loc[idx, f"roll_mean_{w}"] = roll.mean().values
            df.loc[idx, f"roll_std_{w}"]  = roll.std().values
            df.loc[idx, f"roll_min_{w}"]  = roll.min().values
            df.loc[idx, f"roll_max_{w}"]  = roll.max().values

    # ── Trend deltas ─────────────────────────────────────────────────────
    if "roll_mean_7" in df.columns and "roll_mean_28" in df.columns:
        df["delta_7_28"] = df["roll_mean_7"] - df["roll_mean_28"]
    if "roll_mean_28" in df.columns and "roll_mean_91" in df.columns:
        df["delta_28_91"] = df["roll_mean_28"] - df["roll_mean_91"]
    if "lag_63" in df.columns and "lag_70" in df.columns:
        df["lag_diff"] = df["lag_63"] - df["lag_70"]

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  FOLD-SPECIFIC MEAN-TARGET ENCODINGS
# ══════════════════════════════════════════════════════════════════════════════

def compute_fold_encodings(
    base_df: pd.DataFrame,
    train_end: str | pd.Timestamp,
) -> tuple[dict[str, tuple], float]:
    """Compute mean-target encodings from training data only.

    Returns (encoding_maps, fallback_value).
    """
    train = base_df[base_df["date"] <= pd.Timestamp(train_end)]
    fallback = float(train["total_enc"].mean())

    encoding_maps: dict[str, tuple] = {}
    for group_cols, col_name in [
        (["site_enc", "month", "block"], "site_month_block_mean"),
        (["site_enc", "dow"],            "site_dow_mean"),
    ]:
        available = [c for c in group_cols if c in train.columns]
        if len(available) == len(group_cols):
            encoding_maps[col_name] = (
                available,
                train.groupby(available)["total_enc"].mean().to_dict(),
            )

    return encoding_maps, fallback


def apply_fold_encodings(
    df: pd.DataFrame,
    encoding_maps: dict,
    fallback: float,
) -> pd.DataFrame:
    """Map pre-computed mean encodings onto a DataFrame."""
    df = df.copy()
    for col_name, (group_cols, means) in encoding_maps.items():
        keys = list(zip(*(df[c] for c in group_cols)))
        df[col_name] = [means.get(k, fallback) for k in keys]
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE COLUMN RESOLVER
# ══════════════════════════════════════════════════════════════════════════════

# Columns that must never be used as features
_EXCLUDE_COLS = {
    "site", "date", "total_enc", "admitted_enc", "admit_rate",
    "sample_weight", "sample_weight_rate",
    "event_name", "event_type", "is_holiday",
}

# Prefixes to exclude (raw data, not features)
_EXCLUDE_PREFIXES = ("count_reason_", "share_", "_")


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return sorted columns suitable as LightGBM features.

    Excludes raw targets, weights, identifiers, share matrix, reason counts,
    and intermediate factor columns (keeps factor_*_pred, factor_*_lag_*,
    factor_*_momentum, factor_*_deviation_yearly, factor_*_roll_mean_*).
    """
    exclude = set(_EXCLUDE_COLS)

    # Exclude raw factor values (we use factor_i_pred / lagged instead)
    for i in range(cfg.N_FACTORS):
        exclude.add(f"factor_{i}")
        exclude.add(f"factor_{i}_lag_427")  # consumed by deviation_yearly

    def _keep(c: str) -> bool:
        if c in exclude:
            return False
        for pref in _EXCLUDE_PREFIXES:
            if c.startswith(pref):
                return False
        return True

    return sorted(c for c in df.columns if _keep(c))


# ══════════════════════════════════════════════════════════════════════════════
#  CONVENIENCE: Add everything at once (used by run_pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def add_all_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add static features + target lag features.  Called once on base_df."""
    df = add_static_features(df)
    df = add_target_lag_features(df)
    return df


if __name__ == "__main__":
    from data_loader import load_data

    df = load_data()
    df = add_all_base_features(df)
    fc = get_feature_columns(df)
    print(f"\n  Feature count (base, no factors): {len(fc)}")
    print(f"  Sample features: {fc[:15]} ...")
    n_nan = df[fc].isna().any(axis=1).sum()
    print(f"  Rows with any NaN: {n_nan:,}")
