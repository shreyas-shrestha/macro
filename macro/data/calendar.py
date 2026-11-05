"""Shared NYSE session dates, including holidays and exceptional closures."""

from functools import lru_cache

import exchange_calendars as xcals
import pandas as pd


@lru_cache(maxsize=32)
def _nyse_calendar(start_year: int, end_year: int):
    return xcals.get_calendar("XNYS", start=f"{start_year}-01-01", end=f"{end_year}-12-31")


def nyse_sessions(start_date, end_date) -> pd.DatetimeIndex:
    """Return timezone-naive NYSE session labels within inclusive date bounds."""
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)
    if start_date > end_date:
        raise ValueError("start_date must be on or before end_date")
    sessions = _nyse_calendar(start_date.year, end_date.year).sessions
    return sessions[(sessions >= start_date) & (sessions <= end_date)]
