from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import requests

from dsu_forecast import paths
from dsu_forecast.utils.cache import read_json_cache, write_json_cache


@dataclass(frozen=True)
class NWSAlertsRequest:
    latitude: float
    longitude: float
    start: datetime | None = None
    end: datetime | None = None


def fetch_nws_alerts(req: NWSAlertsRequest, *, use_cache: bool = True, timeout_s: int = 60) -> dict[str, Any]:
    """
    NWS API. We request by point (lat,lon). If start/end provided we pass them through;
    the API will ignore unsupported params gracefully.
    """
    params: dict[str, str] = {"point": f"{req.latitude:.5f},{req.longitude:.5f}"}
    if req.start is not None:
        params["start"] = req.start.isoformat()
    if req.end is not None:
        params["end"] = req.end.isoformat()

    cache_key = f"nws_alerts:{params}"
    cache_dir = paths.cache_dir() / "nws"
    if use_cache:
        cached = read_json_cache(cache_dir, cache_key)
        if cached is not None:
            return cached

    r = requests.get(
        "https://api.weather.gov/alerts",
        params=params,
        timeout=timeout_s,
        headers={"User-Agent": "DSU-2026-forecast/1.0", "Accept": "application/geo+json"},
    )
    r.raise_for_status()
    payload: dict[str, Any] = r.json()
    if use_cache:
        write_json_cache(cache_dir, cache_key, payload)
    return payload


def alerts_payload_to_daily_features(payload: dict[str, Any]) -> pd.DataFrame:
    """
    Rough, robust features. We’re not assuming perfect historical coverage from NWS API.
    We extract:
      - count of alerts starting that day
      - count of alerts active that day (approx using onset/ends when present)
    """
    feats: list[dict[str, Any]] = []

    for f in payload.get("features", []) or []:
        props = f.get("properties", {}) or {}
        onset = props.get("onset") or props.get("effective")
        ends = props.get("ends") or props.get("expires")
        event = props.get("event")
        severity = props.get("severity")
        headline = props.get("headline")

        def parse_dt(v: Any) -> datetime | None:
            if not v:
                return None
            try:
                return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
            except Exception:
                return None

        onset_dt = parse_dt(onset)
        ends_dt = parse_dt(ends)
        if onset_dt is None:
            continue

        feats.append(
            {
                "onset": onset_dt,
                "ends": ends_dt,
                "event": event,
                "severity": severity,
                "headline": headline,
            }
        )

    if not feats:
        return pd.DataFrame(columns=["Date", "nws_alerts_started", "nws_alerts_active", "nws_severity_index"])

    df = pd.DataFrame(feats)
    df["Date"] = pd.to_datetime(df["onset"].dt.date)

    sev_map = {"Extreme": 3.0, "Severe": 2.0, "Moderate": 1.0, "Minor": 0.5, "Unknown": 0.25, None: 0.25}
    df["sev_w"] = df["severity"].map(sev_map).fillna(0.25)

    started = df.groupby("Date", as_index=False).agg(nws_alerts_started=("event", "count"), nws_severity_index=("sev_w", "sum"))
    # Active is approximated as started count (better than nothing when ends missing).
    started["nws_alerts_active"] = started["nws_alerts_started"]
    return started

