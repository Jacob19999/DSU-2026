from __future__ import annotations

from datetime import datetime

import pandas as pd

from dsu_forecast.config import SiteMeta, load_events_config
from dsu_forecast.external.nws import NWSAlertsRequest, alerts_payload_to_daily_features, fetch_nws_alerts
from dsu_forecast.external.openmeteo import OpenMeteoHourlyRequest, aggregate_hourly_to_blocks, fetch_openmeteo_hourly
from dsu_forecast.features.events import events_to_daily_features
from dsu_forecast.paths import artifacts_dir


def join_external_covariates(
    base: pd.DataFrame,
    *,
    site_meta: dict[str, SiteMeta],
    train_end: str | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    base must have: Site, Date, Block
    Returns base + weather + NWS + event features.
    If a site has missing lat/lon, external features will be NaN/0 and the pipeline still works.
    """
    df = base.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    min_date = df["Date"].min().date()
    max_date = df["Date"].max().date()
    train_end_dt = pd.to_datetime(train_end).date() if train_end else max_date

    # Events are global (not site-specific in this first pass)
    events = load_events_config()
    ev_daily = events_to_daily_features(events, min_date.isoformat(), max_date.isoformat())
    df = df.merge(ev_daily, on="Date", how="left")
    df["event_intensity"] = df["event_intensity"].fillna(0.0)
    df["event_count"] = df["event_count"].fillna(0).astype(int)

    # Weather + alerts are site-specific
    weather_parts: list[pd.DataFrame] = []
    alerts_parts: list[pd.DataFrame] = []

    for site, meta in site_meta.items():
        sub = df[df["Site"] == site][["Site", "Date", "Block"]].drop_duplicates()
        if meta.latitude is None or meta.longitude is None:
            # Site coords not configured yet
            sub_wx = sub.copy()
            sub_wx["wx_temp_mean"] = pd.NA
            sub_wx["wx_rh_mean"] = pd.NA
            sub_wx["wx_apparent_temp_mean"] = pd.NA
            sub_wx["wx_precip_sum"] = pd.NA
            sub_wx["wx_rain_sum"] = pd.NA
            sub_wx["wx_snow_sum"] = pd.NA
            sub_wx["wx_wind_mean"] = pd.NA
            sub_wx["wx_wind_gust_max"] = pd.NA
            sub_wx["wx_cloud_mean"] = pd.NA
            weather_parts.append(sub_wx)

            sub_alert = sub[["Site", "Date"]].drop_duplicates().copy()
            sub_alert["nws_alerts_started"] = 0
            sub_alert["nws_alerts_active"] = 0
            sub_alert["nws_severity_index"] = 0.0
            alerts_parts.append(sub_alert)
            continue

        # Weather without leakage:
        # - Fetch archive only up to train_end
        # - For dates after train_end, fill from climatology (avg by month/day/block).
        wx_end = min(train_end_dt, max_date)
        wx_req = OpenMeteoHourlyRequest(
            latitude=float(meta.latitude),
            longitude=float(meta.longitude),
            start_date=min_date,
            end_date=wx_end,
            timezone=meta.timezone,
            mode="archive",
        )
        hourly = fetch_openmeteo_hourly(wx_req, use_cache=use_cache)
        wx_block = aggregate_hourly_to_blocks(hourly)
        wx_block["Site"] = site

        # Climatology by (month, day, block)
        if not wx_block.empty:
            wx_block["month"] = wx_block["Date"].dt.month
            wx_block["day"] = wx_block["Date"].dt.day
            clim = (
                wx_block.groupby(["Site", "month", "day", "Block"], as_index=False)
                .mean(numeric_only=True)
                .drop(columns=["Date"], errors="ignore")
            )
        else:
            clim = pd.DataFrame()

        sub_wx = sub.copy()
        sub_wx["month"] = sub_wx["Date"].dt.month
        sub_wx["day"] = sub_wx["Date"].dt.day
        sub_wx = sub_wx.merge(wx_block.drop(columns=["month", "day"], errors="ignore"), on=["Site", "Date", "Block"], how="left")
        if not clim.empty:
            sub_wx = sub_wx.merge(clim, on=["Site", "month", "day", "Block"], how="left", suffixes=("", "_clim"))
            # Fill missing per-row wx_* with climatology values
            for c in list(sub_wx.columns):
                if c.endswith("_clim"):
                    base_c = c[: -len("_clim")]
                    if base_c in sub_wx.columns:
                        sub_wx[base_c] = sub_wx[base_c].fillna(sub_wx[c])
            sub_wx = sub_wx.drop(columns=[c for c in sub_wx.columns if c.endswith("_clim")], errors="ignore")

        sub_wx = sub_wx.drop(columns=["month", "day"], errors="ignore")
        weather_parts.append(sub_wx)

        # Alerts
        # Alerts (no leakage): only compute from requests up to train_end.
        al_end = min(train_end_dt, max_date)
        alerts_payload = fetch_nws_alerts(
            NWSAlertsRequest(
                latitude=float(meta.latitude),
                longitude=float(meta.longitude),
                start=datetime(min_date.year, min_date.month, min_date.day),
                end=datetime(al_end.year, al_end.month, al_end.day),
            ),
            use_cache=use_cache,
        )
        daily_alerts = alerts_payload_to_daily_features(alerts_payload)
        daily_alerts["Site"] = site
        alerts_parts.append(daily_alerts)

    wx = pd.concat(weather_parts, ignore_index=True) if weather_parts else pd.DataFrame()
    al = pd.concat(alerts_parts, ignore_index=True) if alerts_parts else pd.DataFrame()

    df = df.merge(wx, on=["Site", "Date", "Block"], how="left")
    df = df.merge(al, on=["Site", "Date"], how="left")
    df["nws_alerts_started"] = df["nws_alerts_started"].fillna(0).astype(int)
    df["nws_alerts_active"] = df["nws_alerts_active"].fillna(0).astype(int)
    df["nws_severity_index"] = df["nws_severity_index"].fillna(0.0)

    # Optional: LLM-structured NWS features (global daily; merged by Date)
    llm_path = artifacts_dir() / "llm_nws_daily.parquet"
    if llm_path.exists():
        llm_daily = pd.read_parquet(llm_path)
        if "Date" in llm_daily.columns:
            llm_daily["Date"] = pd.to_datetime(llm_daily["Date"])
            df = df.merge(llm_daily, on="Date", how="left")
            for c in [
                "llm_alert_count",
                "llm_severity_index",
                "llm_confidence_mean",
                "llm_expected_volume_uplift",
                "llm_expected_admit_uplift",
            ]:
                if c in df.columns:
                    df[c] = df[c].fillna(0.0)

    return df

