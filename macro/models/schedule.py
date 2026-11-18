"""Walk-forward refit dates and training cutoffs shared by every fitted model."""

import pandas as pd


def refit_schedule(sessions: pd.DatetimeIndex, dates: pd.DatetimeIndex):
    """Yield (last session of the previous year, that year's decision dates)."""
    for year, year_dates in dates.to_series().groupby(dates.year):
        before = sessions[sessions <= f"{year - 1}-12-31"]
        if before.empty:
            raise ValueError(f"no session before {year} to refit on")
        yield before[-1], pd.DatetimeIndex(year_dates)


def training_end(sessions: pd.DatetimeIndex, refit: pd.Timestamp, h: int) -> pd.Timestamp:
    """Last session whose h-session target, t+1 ... t+h, ends by ``refit``."""
    pos = sessions.get_loc(refit) - h
    if pos < 0:
        raise ValueError(f"fewer than {h} sessions before {refit.date()}")
    return sessions[pos]
