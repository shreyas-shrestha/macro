"""Daily series from FRED, stamped by the day FRED posts them."""

import os

import numpy as np
import pandas as pd
from fredapi import Fred
from pandas.tseries.holiday import USFederalHolidayCalendar

from macro.config import load_env


def api_key() -> str:
    load_env()
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY is not set; add it to .env (see .env.example)")
    return key


def fetch(series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Values of ``series_id`` stamped by ``posting_days``, ``start`` through ``end``.

    Values posted after ``end`` are dropped.
    """
    observed = Fred(api_key=api_key()).get_series(
        series_id, observation_start=start, observation_end=end
    )
    if observed.empty:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    posted = pd.Series(observed.astype(float).to_numpy(), index=posting_days(observed.index))
    return posted.loc[start:end]


def posting_days(observed: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The Federal Reserve business day after each observation date."""
    last = observed.max() + pd.Timedelta(days=14)
    holidays = USFederalHolidayCalendar().holidays(observed.min(), last)
    days = np.busday_offset(
        observed.to_numpy().astype("datetime64[D]"),
        1,
        roll="backward",
        holidays=holidays.to_numpy().astype("datetime64[D]"),
    )
    return pd.DatetimeIndex(days.astype("datetime64[ns]"))
