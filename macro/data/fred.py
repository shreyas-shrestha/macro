"""Series from FRED: daily ones stamped by the day FRED posts them, weekly claims by release day."""

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


def fetch_claims(series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Weekly claims as published by ``end``, each stamped by its release day.

    The Department of Labor releases the week ending Saturday s on Thursday s + 5 days,
    or the Wednesday before when that Thursday is a holiday. Values released after
    ``end`` are dropped. Reading the vintage at ``end`` instead of the latest one keeps
    revisions published after ``end`` out.
    """
    vintage = {"realtime_start": end.strftime("%Y-%m-%d"), "realtime_end": end.strftime("%Y-%m-%d")}
    observed = Fred(api_key=api_key()).get_series(
        series_id, observation_start=start, observation_end=end, **vintage
    )
    if (observed.index.day_of_week != 5).any():
        raise ValueError(f"{series_id}: expected weeks ending Saturday")
    thursdays = observed.index + pd.Timedelta(days=5)
    return pd.Series(observed.astype(float).to_numpy(), index=thursdays).loc[start:end]


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
