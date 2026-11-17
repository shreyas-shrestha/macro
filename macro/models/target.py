"""The forecast target: annualized realized variance of S&P 500 log returns."""

import numpy as np
import pandas as pd

from macro.config import DEV_END, DEV_START

HORIZONS = (5, 21, 63)
ANNUALIZE = 252


def log_returns(close: pd.Series) -> pd.Series:
    """Close-to-close log returns, indexed by the session they end on."""
    return np.log(close).diff().dropna()


def realized_variance(returns: pd.Series, h: int) -> pd.Series:
    """At each session t, annualized realized variance over sessions t+1 ... t+h.

    ``(252 / h) * sum(r**2)``, unadjusted for the mean. NaN where fewer than ``h``
    returns follow t.
    """
    return (returns**2).rolling(h).sum().shift(-h) * ANNUALIZE / h


def forecast_grid(returns: pd.Series, h: int) -> pd.DatetimeIndex:
    """Month-end sessions from ``DEV_START`` whose target window ends by ``DEV_END``.

    The last decision dates of the development period are dropped for longer horizons,
    because their targets would use holdout returns.
    """
    dev = returns.loc[:DEV_END]
    month_ends = pd.DatetimeIndex(dev.index.to_series().groupby(dev.index.to_period("M")).max())
    complete = realized_variance(dev, h).notna()[month_ends].to_numpy()
    return month_ends[complete & (month_ends >= DEV_START)]
