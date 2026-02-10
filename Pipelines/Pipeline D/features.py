"""
Deterministic feature engineering for Pipeline D.

Builds the design matrix for Poisson GLM: Fourier terms (weekly + annual),
DOW dummies, linear trend, holiday features, school calendar, COVID indicator,
and optional weather.  NO lagged target features — zero leakage risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays as _holidays_lib
except ImportError:
    _holidays_lib = None

import config as cfg

# ── Datetime helpers (vectorised) ────────────────────────────────────────────

_EPOCH_D = np.datetime64("1970-01-01", "D")


def _to_days(dt_array) -> np.ndarray:
    return (np.asarray(dt_array, dtype="datetime64[D]") - _EPOCH_D).astype(np.int64)


def _days_since_last(dates: pd.Series, ref_dates) -> np.ndarray:
    """Days since the most recent reference date (NaN if before all refs)."""
    d = _to_days(dates.values)
    r = np.sort(_to_days(np.asarray(ref_dates, dtype="datetime64[D]")))
    idx = np.searchsorted(r, d, side="right") - 1
    out = np.full(len(d), np.nan)
    valid = idx >= 0
    out[valid] = d[valid] - r[idx[valid]]
    return out


def _days_until_next(dates: pd.Series, ref_dates) -> np.ndarray:
    """Days until the next reference date (NaN if after all refs)."""
    d = _to_days(dates.values)
    r = np.sort(_to_days(np.asarray(ref_dates, dtype="datetime64[D]")))
    idx = np.searchsorted(r, d, side="left")
    out = np.full(len(d), np.nan)
    valid = idx < len(r)
    out[valid] = r[idx[valid]] - d[valid]
    return out


def _days_to_nearest(dates: pd.Series, ref_dates) -> np.ndarray:
    """Days to nearest reference date in either direction."""
    since = _days_since_last(dates, ref_dates)
    until = _days_until_next(dates, ref_dates)
    return np.fmin(since, until)


# ══════════════════════════════════════════════════════════════════════════════
#  FOURIER TERMS — Core of Pipeline D
# ══════════════════════════════════════════════════════════════════════════════

def make_fourier_features(
    dates: pd.Series,
    period: float,
    order: int,
) -> pd.DataFrame:
    """Generate sin/cos Fourier basis at a given period and order.

    For weekly (period=7):  t = dayofweek (0-6) → ensures Mon always = same phase
    For annual (period=365.25): t = dayofyear (1-366) → ensures Jan 1 = same phase
    """
    if period <= 7.5:
        # Weekly — use day of week
        t = dates.dt.dayofweek.values.astype(float)
    else:
        # Annual — use day of year
        t = dates.dt.dayofyear.values.astype(float)

    features: dict[str, np.ndarray] = {}
    period_tag = f"{period:g}"  # clean string (7, 365.25)
    for k in range(1, order + 1):
        angle = 2.0 * np.pi * k * t / period
        features[f"fourier_{period_tag}_sin_{k}"] = np.sin(angle)
        features[f"fourier_{period_tag}_cos_{k}"] = np.cos(angle)

    return pd.DataFrame(features, index=dates.index)


# ══════════════════════════════════════════════════════════════════════════════
#  HOLIDAY & SCHOOL PROXIMITY
# ══════════════════════════════════════════════════════════════════════════════

def _add_holiday_features(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Add holiday flag + proximity features to design matrix X."""
    # Binary flag from Data Source
    X["is_holiday"] = (
        df["is_holiday"].astype(int) if "is_holiday" in df.columns else 0
    )
    X["is_halloween"] = (
        df["is_halloween"].astype(int) if "is_halloween" in df.columns
        else ((df["date"].dt.month == 10) & (df["date"].dt.day == 31)).astype(int)
    )

    # Proximity features (require holidays lib for reference dates)
    if _holidays_lib is not None:
        us_hol = _holidays_lib.US(years=range(2017, 2027))
        xmas = pd.to_datetime([f"{y}-12-25" for y in range(2017, 2026)])
        jul4 = pd.to_datetime([f"{y}-07-04" for y in range(2017, 2026)])
        tday = pd.to_datetime(
            [d for d, n in sorted(us_hol.items()) if "Thanksgiving" in n]
        )
        all_hol = pd.to_datetime(sorted(us_hol.keys()))

        dates = df["date"]
        X["days_since_xmas"]         = _days_since_last(dates, xmas)
        X["days_until_thanksgiving"]  = _days_until_next(dates, tday)
        X["days_since_july4"]        = _days_since_last(dates, jul4)
        X["days_to_nearest_holiday"] = _days_to_nearest(dates, all_hol)

        # Fill any edge NaNs with large number (no nearby holiday)
        for col in ["days_since_xmas", "days_until_thanksgiving",
                     "days_since_july4", "days_to_nearest_holiday"]:
            X[col] = X[col].fillna(365.0)
    else:
        # Fallback: no proximity, just the binary flags
        for col in ["days_since_xmas", "days_until_thanksgiving",
                     "days_since_july4", "days_to_nearest_holiday"]:
            X[col] = 0.0

    return X


def _add_school_features(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Add school-in-session flag + proximity to school start."""
    X["school_in_session"] = (
        df["school_in_session"].astype(int)
        if "school_in_session" in df.columns else 0
    )

    starts = pd.to_datetime(cfg.SCHOOL_STARTS)
    X["days_since_school_start"] = _days_since_last(df["date"], starts)
    X["days_until_school_start"] = _days_until_next(df["date"], starts)
    for col in ["days_since_school_start", "days_until_school_start"]:
        X[col] = X[col].fillna(365.0)

    return X


# ══════════════════════════════════════════════════════════════════════════════
#  DESIGN MATRIX ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def build_design_matrix(
    df: pd.DataFrame,
    fourier_config: list[dict] | None = None,
) -> pd.DataFrame:
    """Build the full deterministic design matrix X for the GLM.

    Returns a DataFrame aligned to df.index with all features + intercept.
    """
    if fourier_config is None:
        fourier_config = cfg.FOURIER_TERMS

    X = pd.DataFrame(index=df.index)

    # ── 1. Fourier terms ─────────────────────────────────────────────────
    for spec in fourier_config:
        fourier_df = make_fourier_features(df["date"], spec["period"], spec["order"])
        X = pd.concat([X, fourier_df], axis=1)

    # ── 2. DOW dummies (drop Monday=0 as reference) ──────────────────────
    dow_dummies = pd.get_dummies(
        df["date"].dt.dayofweek, prefix="dow", drop_first=True, dtype=int,
    )
    dow_dummies.index = df.index
    X = pd.concat([X, dow_dummies], axis=1)

    # ── 3. Linear trend (scaled to years for numerical stability)
    epoch = pd.Timestamp("2018-01-01")
    X["trend"] = (df["date"] - epoch).dt.days / 365.25

    # ── 4. Holiday features ──────────────────────────────────────────────
    X = _add_holiday_features(df, X)

    # ── 5. School calendar ───────────────────────────────────────────────
    X = _add_school_features(df, X)

    # ── 6. COVID indicator ───────────────────────────────────────────────
    X["is_covid_era"] = (
        df["is_covid_era"].astype(int) if "is_covid_era" in df.columns else 0
    )

    # ── 7. Weather (optional; Pipeline D works without it) ───────────────
    for col in ["temp_min", "temp_max", "precip", "snowfall"]:
        if col in df.columns:
            X[col] = df[col].values
    if "temp_max" in df.columns and "temp_min" in df.columns:
        X["temp_range"] = df["temp_max"].values - df["temp_min"].values

    # ── 8. Intercept (statsmodels GLM needs explicit constant) ───────────
    X.insert(0, "const", 1.0)

    # ── Ensure no object dtypes ──────────────────────────────────────────
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].astype(float)

    return X


def get_feature_names(
    fourier_config: list[dict] | None = None,
) -> str:
    """Return a summary string of expected feature groups."""
    fc = fourier_config or cfg.FOURIER_TERMS
    n_fourier = sum(2 * spec["order"] for spec in fc)
    return (
        f"const(1) + Fourier({n_fourier}) + DOW(6) + trend(1) + "
        f"holiday(6) + school(3) + COVID(1) + weather(<=5)"
    )


if __name__ == "__main__":
    from data_loader import load_data

    df = load_data()
    X = build_design_matrix(df)

    print(f"\n  Design matrix: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"  Feature groups: {get_feature_names()}")
    print(f"  NaN cells: {X.isna().sum().sum()}")
    print(f"  Feature list:\n    {list(X.columns)}")
