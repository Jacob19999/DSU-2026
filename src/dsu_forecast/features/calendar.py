from __future__ import annotations

from datetime import date

import holidays
import numpy as np
import pandas as pd


def add_calendar_features(df: pd.DataFrame, *, date_col: str = "Date") -> pd.DataFrame:
    out = df.copy()
    dts = pd.to_datetime(out[date_col])

    out["dow"] = dts.dt.dayofweek  # 0=Mon
    out["month"] = dts.dt.month
    out["day"] = dts.dt.day
    out["is_weekend"] = (out["dow"] >= 5).astype(int)

    doy = dts.dt.dayofyear.astype(float)
    out["doy"] = doy.astype(int)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # US holidays (works without state/county specificity)
    years = sorted(set(dts.dt.year.tolist()))
    us_holidays = holidays.UnitedStates(years=years)
    out["is_us_holiday"] = dts.dt.date.map(lambda x: 1 if x in us_holidays else 0)

    # End/beginning of month flags (often matters operationally)
    out["is_month_start"] = dts.dt.is_month_start.astype(int)
    out["is_month_end"] = dts.dt.is_month_end.astype(int)
    return out

