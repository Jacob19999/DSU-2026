from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


def events_to_daily_features(events: list[dict[str, Any]], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Returns daily event intensity features on [start_date, end_date] inclusive.
    """
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    all_days = pd.date_range(start, end, freq="D")
    out = pd.DataFrame({"Date": all_days})
    out["event_intensity"] = 0.0
    out["event_count"] = 0

    for e in events:
        try:
            s = pd.to_datetime(e["start_date"])
            t = pd.to_datetime(e["end_date"])
        except Exception:
            continue
        w = float(e.get("weight", 1.0))
        mask = (out["Date"] >= s) & (out["Date"] <= t)
        out.loc[mask, "event_intensity"] += w
        out.loc[mask, "event_count"] += 1

    return out

