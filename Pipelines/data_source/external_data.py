"""
External data extraction for the Data Source ingestion layer.

Fetches / generates real data for columns that were previously stubbed:
  - event_name, event_type, event_count, is_holiday  (holidays + regional events)
  - temp_min, temp_max, precip, snowfall              (Open-Meteo Historical API)
  - school_in_session                                 (heuristic school calendar)
  - cdc_ili_rate                                      (CDC FluView ILINet API)
  - aqi                                               (EPA annual AQI downloads)

All API-fetched data is cached locally under `Pipelines/Data Source/Data/cache/`
to avoid repeated network calls.  Deterministic data (events, school calendar)
is generated in-memory each run.

NO IMPUTATION is performed.  Weather / CDC ILI / AQI values that are
unavailable remain NaN, as required by the Data Source contract.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Site Geography  (master_strategy.md §6)
# Site B ≈ Sanford USD Medical Center, Sioux Falls SD
# Sites A/C ≈ Sanford Medical Center Fargo ND
# Site D  → default Fargo (test Sioux Falls as ablation)
# ══════════════════════════════════════════════════════════════════════════════

WEATHER_LOCATIONS: Dict[str, Dict[str, float]] = {
    "fargo":       {"lat": 46.8772, "lon": -96.7898},
    "sioux_falls": {"lat": 43.5446, "lon": -96.7311},
}

SITE_WEATHER_MAP: Dict[str, str] = {
    "A": "fargo",
    "B": "sioux_falls",
    "C": "fargo",
    "D": "fargo",
}

# County identifiers for EPA AQI lookups
SITE_AQI_COUNTY: Dict[str, Tuple[str, str]] = {
    "fargo":       ("North Dakota", "Cass"),
    "sioux_falls": ("South Dakota", "Minnehaha"),
}


# ══════════════════════════════════════════════════════════════════════════════
# Sturgis Rally dates (annual, early August, near Sioux Falls / western SD)
# ══════════════════════════════════════════════════════════════════════════════

STURGIS_RALLY: Dict[int, Tuple[str, str]] = {
    2018: ("2018-08-03", "2018-08-11"),
    2019: ("2019-08-02", "2019-08-11"),
    2020: ("2020-08-07", "2020-08-16"),
    2021: ("2021-08-06", "2021-08-15"),
    2022: ("2022-08-05", "2022-08-14"),
    2023: ("2023-08-04", "2023-08-13"),
    2024: ("2024-08-02", "2024-08-11"),
    2025: ("2025-08-01", "2025-08-10"),
}


# ══════════════════════════════════════════════════════════════════════════════
# School-calendar heuristic  (Sioux Falls / Fargo districts)
# Start ≈ Aug 20-25, End ≈ May 25-30 each academic year
# ══════════════════════════════════════════════════════════════════════════════

SCHOOL_YEARS: List[Tuple[str, str]] = [
    ("2017-08-22", "2018-05-25"),
    ("2018-08-21", "2019-05-24"),
    ("2019-08-20", "2020-03-13"),   # COVID early closure
    ("2020-09-08", "2021-05-28"),   # Late start due to COVID
    ("2021-08-24", "2022-05-27"),
    ("2022-08-23", "2023-05-26"),
    ("2023-08-22", "2024-05-24"),
    ("2024-08-20", "2025-05-23"),
    ("2025-08-19", "2026-05-22"),
]


# ┌──────────────────────────────────────────────────────────────────────────┐
# │  1. EVENTS & HOLIDAYS                                                   │
# └──────────────────────────────────────────────────────────────────────────┘

def build_event_calendar(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Build a per-date event calendar with US Federal Holidays + Sturgis Rally.

    Returns
    -------
    DataFrame  [date, event_name, event_type, is_holiday, event_count]
        - date:        datetime64
        - event_name:  semicolon-joined names (pd.NA if no event)
        - event_type:  semicolon-joined types  (pd.NA if no event)
        - is_holiday:  True if any event is a federal holiday
        - event_count: int ≥ 0
    """
    try:
        import holidays as holidays_lib
    except ImportError:
        logger.error(
            "The `holidays` package is required for event data. "
            "Install with: pip install holidays"
        )
        raise

    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    start_year = pd.Timestamp(start_date).year
    end_year = pd.Timestamp(end_date).year

    # Collect (name, type) per date
    events_by_date: Dict[pd.Timestamp, List[Tuple[str, str]]] = {}

    # ── 1a. US Federal Holidays ──────────────────────────────────────────
    us_holidays = holidays_lib.US(years=range(start_year, end_year + 1))
    for dt, name in sorted(us_holidays.items()):
        ts = pd.Timestamp(dt).normalize()
        if ts >= dates[0] and ts <= dates[-1]:
            events_by_date.setdefault(ts, []).append((name, "holiday"))

    # ── 1b. Sturgis Motorcycle Rally ─────────────────────────────────────
    for year, (rs, re) in STURGIS_RALLY.items():
        for dt in pd.date_range(start=rs, end=re, freq="D"):
            if dt >= dates[0] and dt <= dates[-1]:
                events_by_date.setdefault(dt, []).append(
                    ("Sturgis Motorcycle Rally", "crowd_event")
                )

    # ── Build result ─────────────────────────────────────────────────────
    rows = []
    for dt in dates:
        evts = events_by_date.get(dt, [])
        if evts:
            names = "; ".join(e[0] for e in evts)
            types = "; ".join(sorted(set(e[1] for e in evts)))
            is_hol = any(e[1] == "holiday" for e in evts)
            count = len(evts)
        else:
            names = pd.NA
            types = pd.NA
            is_hol = False
            count = 0
        rows.append(
            {"date": dt, "event_name": names, "event_type": types,
             "is_holiday": is_hol, "event_count": count}
        )

    logger.info(
        "Built event calendar: %d dates, %d with >=1 event",
        len(dates),
        sum(1 for r in rows if r["event_count"] > 0),
    )
    return pd.DataFrame(rows)


# ┌──────────────────────────────────────────────────────────────────────────┐
# │  2. WEATHER  (Open-Meteo Historical API — free, no key)                 │
# └──────────────────────────────────────────────────────────────────────────┘

def fetch_weather_data(
    start_date: str,
    end_date: str,
    sites: List[str],
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Fetch daily weather from the Open-Meteo Historical API for each site.

    Maps sites to weather stations via SITE_WEATHER_MAP, then returns:
        [site, date, temp_min, temp_max, precip, snowfall]

    Temperatures in °F, precipitation/snowfall in inches.
    Cached to ``cache_dir / weather_cache.csv`` after first successful fetch.
    """
    cache_path = cache_dir / "weather_cache.csv" if cache_dir else None
    if cache_path and cache_path.exists():
        logger.info("Loading cached weather from %s", cache_path)
        return pd.read_csv(cache_path, parse_dates=["date"])

    try:
        import requests
    except ImportError:
        logger.error("The `requests` package is required. pip install requests")
        raise

    # Fetch per unique location (avoid duplicate calls for Fargo)
    location_data: Dict[str, pd.DataFrame] = {}
    for loc_key, coords in WEATHER_LOCATIONS.items():
        logger.info(
            "Fetching weather for %s (%.4f, %.4f) ...",
            loc_key, coords["lat"], coords["lon"],
        )
        location_data[loc_key] = _fetch_open_meteo_weather(
            requests_mod=requests,
            lat=coords["lat"],
            lon=coords["lon"],
            start_date=start_date,
            end_date=end_date,
        )

    # Map locations → sites
    frames = []
    for site in sites:
        loc = SITE_WEATHER_MAP.get(site, "fargo")
        df = location_data[loc].copy()
        df["site"] = site
        frames.append(df)

    result = pd.concat(frames, ignore_index=True)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache_path, index=False)
        logger.info("Cached weather data -> %s (%d rows)", cache_path, len(result))

    return result


def _fetch_open_meteo_weather(
    requests_mod,
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Call Open-Meteo Historical Archive API, chunked by year."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    all_frames: List[pd.DataFrame] = []
    chunk_start = start

    while chunk_start <= end:
        chunk_end = min(
            chunk_start + pd.DateOffset(years=1) - pd.DateOffset(days=1),
            end,
        )
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": chunk_start.strftime("%Y-%m-%d"),
            "end_date": chunk_end.strftime("%Y-%m-%d"),
            "daily": (
                "temperature_2m_max,temperature_2m_min,"
                "precipitation_sum,snowfall_sum"
            ),
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "inch",
            "timezone": "America/Chicago",
        }

        resp = requests_mod.get(url, params=params, timeout=60)
        resp.raise_for_status()
        body = resp.json()
        daily = body.get("daily", {})

        df = pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            "temp_max": pd.to_numeric(daily["temperature_2m_max"], errors="coerce"),
            "temp_min": pd.to_numeric(daily["temperature_2m_min"], errors="coerce"),
            "precip": pd.to_numeric(daily["precipitation_sum"], errors="coerce"),
            "snowfall": pd.to_numeric(daily["snowfall_sum"], errors="coerce"),
        })
        # Open-Meteo returns snowfall in cm even with inch precip unit → convert
        df["snowfall"] = df["snowfall"] / 2.54

        all_frames.append(df)
        chunk_start = chunk_end + pd.DateOffset(days=1)
        time.sleep(0.3)  # polite rate-limiting

    return pd.concat(all_frames, ignore_index=True)


# ┌──────────────────────────────────────────────────────────────────────────┐
# │  3. SCHOOL CALENDAR  (heuristic — no API needed)                        │
# └──────────────────────────────────────────────────────────────────────────┘

def build_school_calendar(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Build ``school_in_session`` flag using heuristic Sioux Falls / Fargo dates.

    Returns
    -------
    DataFrame  [date, school_in_session]   (bool)
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    in_session = pd.Series(False, index=dates, dtype=bool)
    is_weekday = dates.weekday < 5

    # Mark school-year ranges as True (weekdays only) — vectorised
    for school_start_str, school_end_str in SCHOOL_YEARS:
        ss = pd.Timestamp(school_start_str)
        se = pd.Timestamp(school_end_str)
        mask = (dates >= ss) & (dates <= se) & is_weekday
        in_session[mask] = True

    start_year = pd.Timestamp(start_date).year
    end_year = pd.Timestamp(end_date).year

    # Carve out major breaks — vectorised
    # Start from (start_year - 1) so winter break spanning Dec→Jan is covered
    for year in range(start_year - 1, end_year + 2):
        _carve_break(dates, in_session, f"{year}-12-22", f"{min(year + 1, end_year + 1)}-01-02")  # winter
        _carve_break(dates, in_session, f"{year}-03-14", f"{year}-03-21")  # spring break
        # Thanksgiving break: Wed before Thanksgiving through following Sunday
        tg = _thanksgiving(year)
        if tg is not None:
            _carve_break(
                dates, in_session,
                (tg - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                (tg + pd.Timedelta(days=2)).strftime("%Y-%m-%d"),
            )

    result = pd.DataFrame({"date": dates, "school_in_session": in_session.values})
    logger.info(
        "School calendar: %d school days / %d total days",
        int(in_session.sum()),
        len(dates),
    )
    return result


def _carve_break(
    dates: pd.DatetimeIndex,
    in_session: pd.Series,
    break_start: str,
    break_end: str,
) -> None:
    """Set in_session = False for a date range (vectorised)."""
    try:
        bs = pd.Timestamp(break_start)
        be = pd.Timestamp(break_end)
    except Exception:
        return
    mask = (dates >= bs) & (dates <= be)
    in_session[mask] = False


def _thanksgiving(year: int) -> pd.Timestamp | None:
    """Return the 4th Thursday of November for *year*."""
    try:
        nov_first = pd.Timestamp(year=year, month=11, day=1)
    except Exception:
        return None
    # Find all Thursdays in November
    thursdays = pd.date_range(nov_first, f"{year}-11-30", freq="W-THU")
    return thursdays[3] if len(thursdays) >= 4 else None


# ┌──────────────────────────────────────────────────────────────────────────┐
# │  4. CDC ILI RATE                                                        │
# │     Primary:  CDC FluView ILINet (public, no key)                       │
# │     Fallback: CMU Delphi Epidata API (mirrors ILINet, more reliable)    │
# └──────────────────────────────────────────────────────────────────────────┘

def fetch_cdc_ili_data(
    start_date: str,
    end_date: str,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Fetch weekly weighted ILI % (HHS Region 8: SD + ND) and linearly
    interpolate to daily.

    Strategy: try CDC FluView first; on failure fall back to CMU Delphi
    Epidata which mirrors the same ILINet data with better uptime.

    Returns
    -------
    DataFrame  [date, cdc_ili_rate]   (float, daily)
    """
    cache_path = cache_dir / "cdc_ili_cache.csv" if cache_dir else None
    if cache_path and cache_path.exists():
        logger.info("Loading cached CDC ILI data from %s", cache_path)
        return pd.read_csv(cache_path, parse_dates=["date"])

    try:
        import requests  # noqa: F811
    except ImportError:
        logger.error("The `requests` package is required. pip install requests")
        raise

    empty = pd.DataFrame(columns=["date", "cdc_ili_rate"])

    # ── Try CDC FluView first ─────────────────────────────────────────
    weekly = _fetch_fluview_weekly(requests, start_date, end_date)

    # ── Fallback: CMU Delphi Epidata ──────────────────────────────────
    if weekly.empty:
        logger.info("CDC FluView unavailable — trying Delphi Epidata fallback …")
        weekly = _fetch_delphi_weekly(requests, start_date, end_date)

    if weekly.empty:
        logger.warning("Both CDC FluView and Delphi Epidata failed. cdc_ili_rate stays NaN.")
        return empty

    # ── Interpolate weekly → daily ────────────────────────────────────
    daily = _interpolate_ili_to_daily(weekly, start_date, end_date)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(cache_path, index=False)
        logger.info("Cached CDC ILI data -> %s", cache_path)

    return daily


# ──────────────────────────────────────────────────────────────────────────
#  Source A:  CDC FluView ILINet  (original endpoint)
# ──────────────────────────────────────────────────────────────────────────

def _fetch_fluview_weekly(
    requests_mod,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Return DataFrame[date, cdc_ili_rate] at weekly grain, or empty."""
    empty = pd.DataFrame(columns=["date", "cdc_ili_rate"])
    start_year = pd.Timestamp(start_date).year
    end_year = pd.Timestamp(end_date).year

    seasons = [
        {"ID": yr - 1960, "Name": f"{yr}-{(yr + 1) % 100:02d}"}
        for yr in range(start_year - 1, end_year + 1)
    ]
    payload = {
        "AppVersion": "Public",
        "DatasourceDT": [{"ID": 1, "Name": "ILINet"}],
        "RegionTypeId": 1,
        "SubRegionsDT": [{"ID": 8, "Name": "Region 8"}],
        "SeasonsDT": seasons,
    }

    logger.info("Fetching CDC FluView ILI (HHS Region 8, seasons %d-%d) …", start_year - 1, end_year)

    try:
        resp = requests_mod.post(
            "https://gis.cdc.gov/grasp/flu2/PostPhase02DataDownload",
            json=payload,
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("CDC FluView request failed: %s", exc)
        return empty

    csv_text = data.get("datadownload", "")
    if not csv_text:
        logger.warning("CDC FluView returned empty datadownload.")
        return empty

    try:
        ili_df = pd.read_csv(io.StringIO(csv_text))
    except Exception as exc:
        logger.warning("Failed to parse CDC CSV: %s", exc)
        return empty

    ili_df.columns = ili_df.columns.str.strip()

    # Identify ILI rate column
    ili_col = None
    for candidate in ("%WEIGHTED ILI", "% WEIGHTED ILI", "%UNWEIGHTED ILI", "% UNWEIGHTED ILI"):
        if candidate in ili_df.columns:
            ili_col = candidate
            break

    if ili_col is None:
        if "ILITOTAL" in ili_df.columns and "TOTAL PATIENTS" in ili_df.columns:
            ili_df["_ili_rate"] = (
                pd.to_numeric(ili_df["ILITOTAL"], errors="coerce")
                / pd.to_numeric(ili_df["TOTAL PATIENTS"], errors="coerce").clip(lower=1)
                * 100
            )
            ili_col = "_ili_rate"
        else:
            logger.warning("CDC response has no ILI rate column.")
            return empty

    if "YEAR" not in ili_df.columns or "WEEK" not in ili_df.columns:
        logger.warning("CDC response missing YEAR/WEEK columns.")
        return empty

    ili_df["date"] = ili_df.apply(
        lambda r: _mmwr_week_to_saturday(int(r["YEAR"]), int(r["WEEK"])),
        axis=1,
    )
    weekly = ili_df[["date", ili_col]].rename(columns={ili_col: "cdc_ili_rate"})
    weekly["cdc_ili_rate"] = pd.to_numeric(weekly["cdc_ili_rate"], errors="coerce")
    weekly = weekly.dropna(subset=["cdc_ili_rate"]).drop_duplicates(subset=["date"])

    logger.info("CDC FluView returned %d weekly ILI observations.", len(weekly))
    return weekly.sort_values("date").reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────
#  Source B:  CMU Delphi Epidata  (mirrors ILINet — more reliable uptime)
#  API docs: https://cmu-delphi.github.io/delphi-epidata/
# ──────────────────────────────────────────────────────────────────────────

def _fetch_delphi_weekly(
    requests_mod,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """
    Fetch weekly weighted ILI % from the CMU Delphi Epidata ``fluview``
    endpoint (HHS Region 8).  No API key required for public data.

    Returns DataFrame[date, cdc_ili_rate] at weekly grain, or empty.
    """
    empty = pd.DataFrame(columns=["date", "cdc_ili_rate"])
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    # Delphi epiweeks: YYYYWW — start from flu-season onset (week 40 of prior year)
    start_ew = _date_to_epiweek(start_ts - pd.DateOffset(months=3))
    end_ew = _date_to_epiweek(end_ts)

    url = "https://api.delphi.cmu.edu/epidata/fluview/"
    params = {
        "regions": "hhs8",
        "epiweeks": f"{start_ew}-{end_ew}",
    }

    logger.info(
        "Fetching Delphi Epidata FluView (HHS Region 8, epiweeks %s-%s) …",
        start_ew, end_ew,
    )

    try:
        resp = requests_mod.get(url, params=params, timeout=90)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("Delphi Epidata request failed: %s", exc)
        return empty

    # Delphi returns {"result": 1, "epidata": [...], "message": "success"}
    result_code = data.get("result", -1)
    epidata = data.get("epidata", [])

    if result_code != 1 or not epidata:
        logger.warning(
            "Delphi Epidata returned result=%s, message='%s', %d rows.",
            result_code, data.get("message", ""), len(epidata),
        )
        return empty

    rows = []
    for obs in epidata:
        ew = obs.get("epiweek")
        # Prefer weighted ILI (wili); fall back to unweighted (ili)
        ili_val = obs.get("wili") if obs.get("wili") is not None else obs.get("ili")
        if ew is not None and ili_val is not None:
            year = int(str(ew)[:4])
            week = int(str(ew)[4:])
            rows.append({
                "date": _mmwr_week_to_saturday(year, week),
                "cdc_ili_rate": float(ili_val),
            })

    if not rows:
        logger.warning("Delphi Epidata returned observations but none had ILI values.")
        return empty

    weekly = pd.DataFrame(rows)
    weekly = weekly.dropna(subset=["cdc_ili_rate"]).drop_duplicates(subset=["date"])
    logger.info("Delphi Epidata returned %d weekly ILI observations.", len(weekly))
    return weekly.sort_values("date").reset_index(drop=True)


def _date_to_epiweek(ts: pd.Timestamp) -> int:
    """Convert a Timestamp to MMWR epiweek integer YYYYWW."""
    # ISO week is close to MMWR week for practical purposes
    iso_year, iso_week, _ = ts.isocalendar()
    return int(f"{iso_year}{iso_week:02d}")


# ──────────────────────────────────────────────────────────────────────────
#  Shared helpers for ILI data
# ──────────────────────────────────────────────────────────────────────────

def _interpolate_ili_to_daily(
    weekly: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Linearly interpolate weekly ILI observations to daily grain."""
    all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    daily = pd.DataFrame({"date": all_dates}).merge(weekly, on="date", how="left")
    daily["cdc_ili_rate"] = daily["cdc_ili_rate"].interpolate(method="linear")

    logger.info(
        "CDC ILI: %d weekly obs -> %d daily rows (%.1f%% non-NaN)",
        len(weekly),
        len(daily),
        daily["cdc_ili_rate"].notna().mean() * 100,
    )
    return daily


def _mmwr_week_to_saturday(year: int, week: int) -> pd.Timestamp:
    """
    Approximate conversion of MMWR (year, week) to the Saturday of that
    epi-week.  MMWR week 1 contains January 4.
    """
    jan4 = pd.Timestamp(year=year, month=1, day=4)
    # dayofweek: Mon=0 … Sun=6.  MMWR weeks start Sunday.
    days_since_sunday = (jan4.dayofweek + 1) % 7
    week1_sunday = jan4 - pd.Timedelta(days=days_since_sunday)
    return week1_sunday + pd.Timedelta(weeks=week - 1, days=6)


# ┌──────────────────────────────────────────────────────────────────────────┐
# │  5. AQI  (EPA Annual County AQI files — public, no key)                 │
# └──────────────────────────────────────────────────────────────────────────┘

def fetch_aqi_data(
    start_date: str,
    end_date: str,
    sites: List[str],
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """
    Fetch daily AQI from EPA annual ``daily_aqi_by_county`` ZIP archives
    for the relevant counties (Cass County ND / Minnehaha County SD).

    Returns
    -------
    DataFrame  [site, date, aqi]   (float)
    """
    cache_path = cache_dir / "aqi_cache.csv" if cache_dir else None
    if cache_path and cache_path.exists():
        logger.info("Loading cached AQI data from %s", cache_path)
        return pd.read_csv(cache_path, parse_dates=["date"])

    try:
        import requests
    except ImportError:
        logger.error("The `requests` package is required. pip install requests")
        raise

    start_year = pd.Timestamp(start_date).year
    end_year = pd.Timestamp(end_date).year

    # Download EPA annual ZIPs
    all_years: List[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        url = (
            f"https://aqs.epa.gov/aqsweb/airdata/"
            f"daily_aqi_by_county_{year}.zip"
        )
        logger.info("Fetching EPA AQI for %d ...", year)
        try:
            resp = requests.get(url, timeout=120)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = zf.namelist()[0]
                with zf.open(csv_name) as f:
                    year_df = pd.read_csv(f)
            all_years.append(year_df)
            time.sleep(0.5)  # polite pacing
        except Exception as exc:
            logger.warning("EPA AQI download failed for %d: %s", year, exc)

    if not all_years:
        logger.warning("No AQI data fetched. Column will stay NaN.")
        return pd.DataFrame(columns=["site", "date", "aqi"])

    aqi_all = pd.concat(all_years, ignore_index=True)
    # Normalise column names (EPA uses title-case with spaces)
    aqi_all.columns = aqi_all.columns.str.strip()

    # Filter to our counties and map to sites
    frames: List[pd.DataFrame] = []
    for site in sites:
        loc = SITE_WEATHER_MAP.get(site, "fargo")
        state_name, county_name = SITE_AQI_COUNTY[loc]
        mask = (
            (aqi_all["State Name"].str.strip() == state_name)
            & (aqi_all["county Name"].str.strip() == county_name)
        )
        county_df = aqi_all.loc[mask, ["Date", "AQI"]].copy()
        if county_df.empty:
            logger.warning(
                "No AQI rows for site %s (%s, %s)", site, state_name, county_name,
            )
            continue

        county_df["date"] = pd.to_datetime(county_df["Date"])
        county_df["aqi"] = pd.to_numeric(county_df["AQI"], errors="coerce")
        county_df["site"] = site
        county_df = (
            county_df[["site", "date", "aqi"]]
            .sort_values("date")
            .drop_duplicates(subset=["site", "date"])
        )
        frames.append(county_df)

    if not frames:
        return pd.DataFrame(columns=["site", "date", "aqi"])

    result = pd.concat(frames, ignore_index=True)
    logger.info("AQI data: %d rows across %d sites", len(result), result["site"].nunique())

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(cache_path, index=False)
        logger.info("Cached AQI data -> %s", cache_path)

    return result


# ┌──────────────────────────────────────────────────────────────────────────┐
# │  ORCHESTRATOR — called by ingestion.py                                  │
# └──────────────────────────────────────────────────────────────────────────┘

def add_external_features(
    df: pd.DataFrame,
    sites: List[str],
    start_date: str,
    end_date: str,
    cache_dir: Path | None = None,
    fetch_apis: bool = True,
) -> pd.DataFrame:
    """
    Replace stub external columns with real data.

    Parameters
    ----------
    df : DataFrame
        The master block-level frame (must contain ``date`` and ``site``).
    sites : list[str]
        Site identifiers (e.g. ["A","B","C","D"]).
    start_date, end_date : str
        Grid date boundaries (YYYY-MM-DD).
    cache_dir : Path or None
        Where to cache API-fetched data (weather, CDC ILI, AQI).
    fetch_apis : bool
        If False, skip network-dependent fetches (weather, CDC ILI, AQI)
        and leave those columns as NaN.  Deterministic columns (events,
        school calendar) are always populated.
    """
    df = df.copy()

    # ── 1. Events & holidays ────────────────────────────────────────────
    events = build_event_calendar(start_date, end_date)
    # Drop stubs if present, then merge real data
    for col in ("event_name", "event_type", "is_holiday", "event_count"):
        if col in df.columns:
            df = df.drop(columns=[col])
    df = df.merge(events, on="date", how="left")
    df["event_count"] = df["event_count"].fillna(0).astype(int)
    df["is_holiday"] = df["is_holiday"].fillna(False)

    # ── 2. School calendar ──────────────────────────────────────────────
    school = build_school_calendar(start_date, end_date)
    if "school_in_session" in df.columns:
        df = df.drop(columns=["school_in_session"])
    df = df.merge(school, on="date", how="left")

    # ── 3. Weather  (API) ───────────────────────────────────────────────
    for col in ("temp_min", "temp_max", "precip", "snowfall"):
        if col in df.columns:
            df = df.drop(columns=[col])

    if fetch_apis:
        try:
            weather = fetch_weather_data(
                start_date, end_date, sites, cache_dir=cache_dir,
            )
            df = df.merge(weather, on=["site", "date"], how="left")
        except Exception as exc:
            logger.warning("Weather fetch failed (%s). Columns stay NaN.", exc)
            for col in ("temp_min", "temp_max", "precip", "snowfall"):
                df[col] = np.nan
    else:
        for col in ("temp_min", "temp_max", "precip", "snowfall"):
            df[col] = np.nan

    # ── 4. CDC ILI  (API) ──────────────────────────────────────────────
    if "cdc_ili_rate" in df.columns:
        df = df.drop(columns=["cdc_ili_rate"])

    if fetch_apis:
        try:
            ili = fetch_cdc_ili_data(start_date, end_date, cache_dir=cache_dir)
            if not ili.empty:
                df = df.merge(ili, on="date", how="left")
            else:
                df["cdc_ili_rate"] = np.nan
        except Exception as exc:
            logger.warning("CDC ILI fetch failed (%s). Column stays NaN.", exc)
            df["cdc_ili_rate"] = np.nan
    else:
        df["cdc_ili_rate"] = np.nan

    # ── 5. AQI  (API) ──────────────────────────────────────────────────
    if "aqi" in df.columns:
        df = df.drop(columns=["aqi"])

    if fetch_apis:
        try:
            aqi = fetch_aqi_data(
                start_date, end_date, sites, cache_dir=cache_dir,
            )
            if not aqi.empty:
                df = df.merge(aqi, on=["site", "date"], how="left")
            else:
                df["aqi"] = np.nan
        except Exception as exc:
            logger.warning("AQI fetch failed (%s). Column stays NaN.", exc)
            df["aqi"] = np.nan
    else:
        df["aqi"] = np.nan

    return df
