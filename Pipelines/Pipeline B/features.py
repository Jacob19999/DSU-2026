"""
Horizon-adaptive feature engineering for Pipeline B.

Core differentiator: shorter-horizon buckets exploit more recent lag data.
Each bucket trains on an expanded dataset where every (site, date, block) row
is replicated across sub-sampled horizons, each with its own lag/rolling
features shifted by (h + k) from the target date.

Static features (calendar, cyclical, holiday, weather, interactions) come from
the TARGET date (deterministic, known in advance).

Lag/rolling features are computed from the perspective of the AS-OF date
(target_date − horizon), ensuring no future leakage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import holidays as _holidays_lib
except ImportError:
    _holidays_lib = None

import config as cfg

# ── Embedding summary config ─────────────────────────────────────────────────

_EMB_SUMMARY_COLS = ["reason_emb_entropy", "reason_emb_norm", "reason_emb_cluster"]
_EMB_ROLLING_COLS = ["reason_emb_entropy", "reason_emb_norm"]  # cluster is categorical
_EMB_ROLLING_WINDOWS = [7, 14, 28]

# ── Datetime helpers ─────────────────────────────────────────────────────────

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
#  STATIC FEATURES — Computed once on the base (non-expanded) DataFrame
# ══════════════════════════════════════════════════════════════════════════════

def add_static_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all deterministic features derived from the target date.

    These are identical across horizons/buckets and computed once.
    """
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
    df["holiday_x_block"]  = is_hol * df["block"]
    df["weekend_x_block"]  = is_we  * df["block"]
    df["site_x_dow"]       = df["site_enc"] * 7  + df["dow"]
    df["site_x_month"]     = df["site_enc"] * 12 + df["month"]

    # ── Cast bools → int (LightGBM compatibility) ────────────────────────
    for col in df.select_dtypes(include=["bool"]).columns:
        df[col] = df[col].astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  HORIZON-ADAPTIVE FEATURES — Per-bucket expansion
# ══════════════════════════════════════════════════════════════════════════════

def build_bucket_data(
    df: pd.DataFrame,
    bucket_id: int,
    horizons: list[int] | None = None,
    target_dates: set | None = None,
) -> pd.DataFrame:
    """Build expanded dataset for one bucket with horizon-adaptive features.

    For each sub-sampled horizon h, copies the base df and computes:
      - lag_total_{k}, lag_rate_{k}  for k in bucket's lag set
      - roll_{stat}_{w}              for w in ROLLING_WINDOWS
      - delta_7_28, delta_28_91, lag_diff
      - days_ahead = h

    Shift math: lag_k from as-of date = series.shift(h + k) from target date.

    Parameters
    ----------
    df           : Base DataFrame (with static features, sorted by site/block/date).
    bucket_id    : 1, 2, or 3.
    horizons     : Override sub-sampled horizons (e.g. exact horizons for prediction).
    target_dates : If provided, only keep rows whose date ∈ target_dates (prediction).
    """
    bucket  = cfg.BUCKETS[bucket_id]
    h_list  = horizons or cfg.BUCKET_HORIZONS[bucket_id]
    lags    = cfg.BUCKET_LAGS[bucket_id]
    min_lag = bucket["min_lag"]

    # Pre-extract Block 3 series per site for cross-block features
    _b3_total_by_site: dict[str, pd.Series] = {}
    for _s in df["site"].unique():
        b3 = df.loc[(df["site"] == _s) & (df["block"] == 3)].sort_values("date")
        _b3_total_by_site[_s] = b3["total_enc"]

    frames: list[pd.DataFrame] = []

    for h in h_list:
        # Compute on full df (need full history for correct shifts)
        expanded = df.copy()
        expanded["days_ahead"] = h
        expanded["bucket"]     = bucket_id

        # Per-(site, block) lag + rolling computation
        for (_site, _blk), grp in df.groupby(["site", "block"]):
            idx      = grp.index
            s_total  = grp["total_enc"]
            s_rate   = grp["admit_rate"]

            # Lag features: shift(h + k) from target-date position
            for k in lags:
                expanded.loc[idx, f"lag_total_{k}"] = s_total.shift(h + k).values
                expanded.loc[idx, f"lag_rate_{k}"]  = s_rate.shift(h + k).values

            # Rolling features: shifted by (h + min_lag), then windowed
            shifted = s_total.shift(h + min_lag)
            for w in cfg.ROLLING_WINDOWS:
                roll = shifted.rolling(w, min_periods=1)
                expanded.loc[idx, f"roll_mean_{w}"] = roll.mean().values
                expanded.loc[idx, f"roll_std_{w}"]  = roll.std().values
                expanded.loc[idx, f"roll_min_{w}"]  = roll.min().values
                expanded.loc[idx, f"roll_max_{w}"]  = roll.max().values

            # Lagged embedding summaries (§10.4) — same shift(h+k) logic
            for emb_col in _EMB_SUMMARY_COLS:
                if emb_col not in grp.columns:
                    continue
                s_emb = grp[emb_col]
                short = emb_col.replace("reason_emb_", "emb_")
                # Point lags
                for k in lags:
                    expanded.loc[idx, f"lag_{short}_{k}"] = s_emb.shift(h + k).values
                # Rolling means (entropy & norm only)
                if emb_col in _EMB_ROLLING_COLS:
                    shifted_emb = s_emb.shift(h + min_lag)
                    for w in _EMB_ROLLING_WINDOWS:
                        expanded.loc[idx, f"roll_{short}_{w}"] = (
                            shifted_emb.rolling(w, min_periods=1).mean().values
                        )

            # ── Cross-block lags: Block 0 ← lagged Block 3 (evening decay) ──
            if _blk == 0 and _site in _b3_total_by_site:
                b3_t = _b3_total_by_site[_site]
                for k in lags:
                    expanded.loc[idx, f"xblock_b3_total_{k}"] = (
                        b3_t.shift(h + k).values
                    )
                b3_shifted = b3_t.shift(h + min_lag)
                for w in [7, 14, 28]:
                    expanded.loc[idx, f"xblock_b3_roll_mean_{w}"] = (
                        b3_shifted.rolling(w, min_periods=1).mean().values
                    )

        # Optionally filter to target dates (prediction efficiency)
        if target_dates is not None:
            expanded = expanded[expanded["date"].isin(target_dates)]

        frames.append(expanded)

    result = pd.concat(frames, ignore_index=True)

    # ── Trend deltas ─────────────────────────────────────────────────────
    if "roll_mean_7" in result.columns and "roll_mean_28" in result.columns:
        result["delta_7_28"] = result["roll_mean_7"] - result["roll_mean_28"]
    if "roll_mean_28" in result.columns and "roll_mean_91" in result.columns:
        result["delta_28_91"] = result["roll_mean_28"] - result["roll_mean_91"]

    # Lag diff (week-over-week change at shortest safe lag)
    if len(lags) >= 2:
        result["lag_diff"] = (
            result[f"lag_total_{lags[0]}"] - result[f"lag_total_{lags[1]}"]
        )

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  TARGET ENCODING — per-bucket min_lag for more current baselines (§11)
# ══════════════════════════════════════════════════════════════════════════════

def add_target_encodings(
    df: pd.DataFrame,
    min_lag: int,
) -> pd.DataFrame:
    """Add site-level target-encoded features using bucket-specific lag.

    Bucket 1 (min_lag=16) gets a 16-day-lagged baseline — more current than
    Pipeline A's 63-day-lagged version.  For Site D where volumes are stable
    (D/A ratio CV < 0.05), even the 16-day-lagged baseline is accurate.

    Parameters
    ----------
    df      : Base DataFrame sorted by (site, block, date).
    min_lag : Bucket-specific minimum lag (16 / 31 / 63).
    """
    for (_site, _blk), grp in df.groupby(["site", "block"]):
        idx = grp.index

        # 1–2. Site baseline volume (trailing 90-day mean, lagged by min_lag)
        for target_col in ["total_enc", "admitted_enc"]:
            shifted = grp[target_col].shift(min_lag)
            df.loc[idx, f"te_site_mean_{target_col}"] = (
                shifted.rolling(90, min_periods=30).mean().values
            )

        # 3–4. Site × Block baseline
        for target_col, tag in [("total_enc", "total"), ("admitted_enc", "admitted")]:
            shifted = grp[target_col].shift(min_lag)
            df.loc[idx, f"te_site_block_mean_{tag}"] = (
                shifted.rolling(90, min_periods=30).mean().values
            )

        # 5. Site admit rate (trailing 90-day ratio, lagged)
        shifted_total = grp["total_enc"].shift(min_lag).rolling(90, min_periods=30).sum()
        shifted_admitted = grp["admitted_enc"].shift(min_lag).rolling(90, min_periods=30).sum()
        df.loc[idx, "te_site_admit_rate"] = (
            (shifted_admitted / shifted_total.clip(lower=1)).values
        )

    # 6. Site × DOW mean (trailing 90-day, lagged by min_lag)
    for (_site, _dow), grp in df.groupby(["site", "dow"]):
        idx = grp.index
        shifted = grp["total_enc"].shift(min_lag)
        df.loc[idx, "te_site_dow_mean"] = (
            shifted.rolling(90, min_periods=30).mean().values
        )

    return df


def add_target_encodings_to_bucket(
    expanded_df: pd.DataFrame,
    base_df: pd.DataFrame,
    bucket_id: int,
) -> pd.DataFrame:
    """Add target encodings to an expanded bucket dataset.

    Computes TE on the base (non-expanded) df with bucket-specific min_lag,
    then joins onto the expanded data by (site, block, date).
    """
    min_lag = cfg.BUCKETS[bucket_id]["min_lag"]
    te_cols = [
        "te_site_mean_total_enc", "te_site_mean_admitted_enc",
        "te_site_block_mean_total", "te_site_block_mean_admitted",
        "te_site_admit_rate", "te_site_dow_mean",
    ]

    # Compute on base df
    base_te = add_target_encodings(base_df.copy(), min_lag)

    # Merge onto expanded df by (site, block, date)
    merge_cols = ["site", "block", "date"]
    te_to_merge = base_te[merge_cols + te_cols].drop_duplicates(subset=merge_cols)
    for col in te_cols:
        if col in expanded_df.columns:
            expanded_df = expanded_df.drop(columns=[col])
    expanded_df = expanded_df.merge(te_to_merge, on=merge_cols, how="left")

    return expanded_df


# ══════════════════════════════════════════════════════════════════════════════
#  FOLD-SPECIFIC AGGREGATE ENCODINGS
# ══════════════════════════════════════════════════════════════════════════════

def compute_fold_encodings(
    base_df: pd.DataFrame,
    train_end: str | pd.Timestamp,
) -> tuple[dict[str, dict], float]:
    """Compute mean-target encodings from non-expanded training data.

    Returns (encoding_maps, fallback_value) for use by apply_fold_encodings.
    """
    train = base_df[base_df["date"] <= pd.Timestamp(train_end)]
    fallback = float(train["total_enc"].mean())

    encoding_maps: dict[str, dict] = {}
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
    """Map pre-computed mean-encodings onto a (possibly expanded) DataFrame."""
    df = df.copy()
    for col_name, (group_cols, means) in encoding_maps.items():
        keys = list(zip(*(df[c] for c in group_cols)))
        df[col_name] = [means.get(k, fallback) for k in keys]
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURE COLUMN RESOLVER
# ══════════════════════════════════════════════════════════════════════════════

_EXCLUDE_COLS = {
    "site", "date", "total_enc", "admitted_enc", "admit_rate",
    "sample_weight", "sample_weight_rate",
    "event_name", "event_type", "is_holiday",
    "bucket",
}


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return columns suitable as LightGBM features."""
    exclude = set(_EXCLUDE_COLS)
    exclude.update(c for c in df.columns if c.startswith("count_reason_"))
    # Raw reason embeddings — contemporaneous (target-date leakage, §10).
    # Lagged summaries (lag_emb_*, roll_emb_*) are kept as features.
    exclude.update(c for c in df.columns if c.startswith("reason_emb_"))
    return sorted(c for c in df.columns if c not in exclude)


if __name__ == "__main__":
    from data_loader import load_data

    df = load_data()
    df = add_static_features(df)

    for bid in [1, 2, 3]:
        bdata = build_bucket_data(df, bid)
        fc = get_feature_columns(bdata)
        n_rows = len(bdata)
        n_nan  = bdata[fc].isna().any(axis=1).sum()
        print(f"\n  Bucket {bid}: {n_rows:,} rows, {len(fc)} features, {n_nan:,} rows with NaN")
        print(f"    Feature list: {fc[:10]} ...")
