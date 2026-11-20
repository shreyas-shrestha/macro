"""Slow realized variance of the S&P 500, the reference model for #14's long-horizon test.

    log_rv_126  log annualized realized variance over the 126 sessions ending at t
    log_rv_252  the same over 252 sessions

Engle, Ghysels & Sohn (2013) find macro data predicts the slow component of volatility,
so claims are tested against a model that already contains it.
"""

import numpy as np
import pandas as pd

from macro.data.series import load
from macro.features.base import INDEX_NAME, Signal
from macro.models.target import ANNUALIZE, log_returns

WINDOWS = (126, 252)


class LongRV(Signal):
    name = "long_rv"
    source = "Yahoo (^GSPC)"
    start_date = pd.Timestamp("1990-01-02")
    release_lag = 0

    def build(self, end_date: pd.Timestamp) -> pd.DataFrame:
        squared = log_returns(load("SPX", end_date)) ** 2 * ANNUALIZE
        out = pd.DataFrame({f"log_rv_{w}": np.log(squared.rolling(w).mean()) for w in WINDOWS})
        return out.rename_axis(INDEX_NAME).loc[self.start_date :]
