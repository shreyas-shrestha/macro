"""The forecast target: annualized realized variance of S&P 500 log returns."""

import numpy as np
import pandas as pd

from macro.config import DEV_END, DEV_START

HORIZONS = (5, 21, 63)
ANNUALIZE = 252


def log_returns(close: pd.Series) -> pd.Series:
    """Close-to-close log returns, indexed by the session they end on."""
    close = close.loc[close.first_valid_index() : close.last_valid_index()]
    if close.isna().any():
        raise ValueError("missing closes would turn into multi-session returns")
    return np.log(close).diff().dropna()


def realized_variance(returns: pd.Series, h: int) -> pd.Series:
    """At each session t, annualized realized variance over sessions t+1 ... t+h.

    Not demeaned. NaN where fewer than ``h`` returns follow t.
    """
    return (returns**2).rolling(h).sum().shift(-h) * ANNUALIZE / h


def forecast_grid(returns: pd.Series, h: int, start=DEV_START, end=DEV_END) -> pd.DatetimeIndex:
    """Month-end sessions from ``start`` whose target window ends by ``end``.

    The defaults are the development period; the holdout run passes its own dates.
    """
    sample = returns.loc[:end]
    sessions = sample.index
    month_ends = pd.DatetimeIndex(sessions.to_series().groupby(sessions.to_period("M")).max())
    complete = realized_variance(sample, h).notna()[month_ends].to_numpy()
    return month_ends[complete & (month_ends >= pd.Timestamp(start))]
