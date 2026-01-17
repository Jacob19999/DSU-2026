from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import pandas as pd
import requests

from dsu_forecast import paths
from dsu_forecast.utils.cache import read_json_cache, write_json_cache


Mode = Literal["archive", "forecast"]


@dataclass(frozen=True)
class OpenMeteoHourlyRequest:
    latitude: float
    longitude: float
    start_date: date
    end_date: date
    timezone: str
    mode: Mode


def _endpoint(mode: Mode) -> str:
    if mode == "archive":
        return "https://archive-api.open-meteo.com/v1/archive"
    return "https://api.open-meteo.com/v1/forecast"


def fetch_openmeteo_hourly(req: OpenMeteoHourlyRequest, *, use_cache: bool = True, timeout_s: int = 60) -> pd.DataFrame:
    hourly_vars = [
        "temperature_2m",
        "relative_humidity_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "snowfall",
        "wind_speed_10m",
        "wind_gusts_10m",
        "cloud_cover",
    ]

    params = {
        "latitude": req.latitude,
        "longitude": req.longitude,
        "start_date": req.start_date.isoformat(),
        "end_date": req.end_date.isoformat(),
        "timezone": req.timezone,
        "hourly": ",".join(hourly_vars),
    }

    cache_key = f"openmeteo:{req.mode}:{params}"
    cache_dir = paths.cache_dir() / "openmeteo"
    if use_cache:
        cached = read_json_cache(cache_dir, cache_key)
        if cached is not None:
            return _payload_to_hourly_df(cached)

    r = requests.get(_endpoint(req.mode), params=params, timeout=timeout_s, headers={"User-Agent": "DSU-2026-forecast/1.0"})
    r.raise_for_status()
    payload: dict[str, Any] = r.json()
    if use_cache:
        write_json_cache(cache_dir, cache_key, payload)
    return _payload_to_hourly_df(payload)


def _payload_to_hourly_df(payload: dict[str, Any]) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame(columns=["dt"])

    df = pd.DataFrame(hourly)
    df = df.rename(columns={"time": "dt"})
    df["dt"] = pd.to_datetime(df["dt"])
    return df


def aggregate_hourly_to_blocks(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert hourly weather rows to (date, block) features aligned to the DSU 6-hour blocks:
      block 0: 00-05, 1: 06-11, 2: 12-17, 3: 18-23
    """
    if hourly_df.empty:
        return pd.DataFrame(columns=["Date", "Block"])

    df = hourly_df.copy()
    df["Date"] = df["dt"].dt.date
    df["Hour"] = df["dt"].dt.hour
    df["Block"] = (df["Hour"] // 6).astype(int)

    agg = {
        "temperature_2m": "mean",
        "relative_humidity_2m": "mean",
        "apparent_temperature": "mean",
        "precipitation": "sum",
        "rain": "sum",
        "snowfall": "sum",
        "wind_speed_10m": "mean",
        "wind_gusts_10m": "max",
        "cloud_cover": "mean",
    }

    out = df.groupby(["Date", "Block"], as_index=False).agg(agg)
    out = out.rename(
        columns={
            "temperature_2m": "wx_temp_mean",
            "relative_humidity_2m": "wx_rh_mean",
            "apparent_temperature": "wx_apparent_temp_mean",
            "precipitation": "wx_precip_sum",
            "rain": "wx_rain_sum",
            "snowfall": "wx_snow_sum",
            "wind_speed_10m": "wx_wind_mean",
            "wind_gusts_10m": "wx_wind_gust_max",
            "cloud_cover": "wx_cloud_mean",
        }
    )
    out["Date"] = pd.to_datetime(out["Date"])
    return out

