"""Time features: trading-day time and scheduled macro events between quote date and expiry.

Why: implied vol is quoted on calendar time, but variance accrues on trading days and jumps on
scheduled announcements. A smooth function of calendar maturity cannot represent either effect.

The event dates below were entered by hand. Check them against the primary sources before relying
on them: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm and
https://www.bls.gov/schedule/news_release/cpi.htm
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# FOMC decision days (second day of each scheduled meeting)
FOMC = pd.to_datetime([
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
])
# CPI release days
CPI = pd.to_datetime([
    "2023-01-12", "2023-02-14", "2023-03-14", "2023-04-12", "2023-05-10", "2023-06-13", "2023-07-12", "2023-08-10",
    "2023-09-13", "2023-10-12", "2023-11-14", "2023-12-12",
    "2024-01-11", "2024-02-13", "2024-03-12", "2024-04-10", "2024-05-15", "2024-06-12", "2024-07-11", "2024-08-14",
    "2024-09-11", "2024-10-10", "2024-11-13", "2024-12-11",
])
# NYSE full-day closures
HOLIDAYS = pd.to_datetime([
    "2023-01-02", "2023-01-16", "2023-02-20", "2023-04-07", "2023-05-29", "2023-06-19", "2023-07-04", "2023-09-04",
    "2023-11-23", "2023-12-25",
    "2024-01-01", "2024-01-15", "2024-02-19", "2024-03-29", "2024-05-27", "2024-06-19", "2024-07-04", "2024-09-02",
    "2024-11-28", "2024-12-25",
])


def _count_between(events: pd.DatetimeIndex, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Number of events e with start < e <= end, vectorised."""
    ev = np.sort(events.values.astype("datetime64[D]"))
    return np.searchsorted(ev, end, side="right") - np.searchsorted(ev, start, side="right")


def add_time_features(q: pd.DataFrame) -> pd.DataFrame:
    """Add columns:
    T_trading      trading days from quote date to expiry, in years (/252)
    time_ratio     T_trading / T  (below 1 when the period holds many weekends or holidays)
    fomc_per_year  FOMC decisions before expiry divided by T
    cpi_per_year   CPI releases before expiry divided by T

    Events are divided by T because an event adds a fixed chunk of variance v, so in annualised terms
    sigma^2 = sigma_base^2 + v * n / T. The effect is large for short expiries and fades for long ones.
    """
    q = q.copy()
    start = q.quote_date.values.astype("datetime64[D]")
    end = q.expire_date.values.astype("datetime64[D]")
    hol = HOLIDAYS.values.astype("datetime64[D]")
    # busday_count counts [start, end); shift by one day to count (start, end]
    one = np.timedelta64(1, "D")
    q["T_trading"] = np.busday_count(start + one, end + one, holidays=hol) / 252.0
    q["time_ratio"] = q.T_trading / q["T"]
    q["fomc_per_year"] = _count_between(FOMC, start, end) / q["T"]
    q["cpi_per_year"] = _count_between(CPI, start, end) / q["T"]
    return q


TIME_FEATURES = ["time_ratio", "fomc_per_year", "cpi_per_year"]
