"""Rates and yield-curve features (issue #12).

Inputs, read with ``macro.data.series.load`` (FRED daily, lag 1 session, carry 3):
    T10Y2Y  10-year minus 2-year Treasury constant-maturity yield, percentage points
    DGS10   10-year Treasury constant-maturity yield, percent

Features (decision date t):
    t10y2y_level      T10Y2Y
    t10y2y_change_20  T10Y2Y minus its value 20 sessions earlier
    t10y2y_z_252      (level - mean) / std over the 252 sessions before t
    dgs10_vol_20      standard deviation of session-to-session DGS10 changes, last 20

The z-score window ends the session before t, so it is fit on past data only. Both
series start in 1990, so every window is full by ``start_date``.

T10Y3M is registered for this issue but not used: it tracks T10Y2Y closely, and with the
3-month rate near zero from late 2008 to late 2015 it was mostly the 10-year yield.

A session where ``load`` carries the previous value forward (Columbus Day, Veterans Day)
shows a zero DGS10 change followed by a two-day change. That is what was knowable on
those sessions, so ``dgs10_vol_20`` keeps it.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from macro.data.series import load
from macro.features.base import INDEX_NAME, Signal

COLUMNS = ("t10y2y_level", "t10y2y_change_20", "t10y2y_z_252", "dgs10_vol_20")
CHANGE_WINDOW = 20
Z_WINDOW = 252
Z_MIN_PERIODS = 245
VOL_WINDOW = 20
VOL_MIN_PERIODS = 15

Loader = Callable[[str, pd.Timestamp], pd.Series]


class YieldCurve(Signal):
    name = "yield_curve"
    source = "FRED (T10Y2Y, DGS10)"
    start_date = pd.Timestamp("2000-01-03")
    release_lag = 1

    def __init__(self, loader: Loader = load):
        self.loader = loader

    def build(self, end_date: pd.Timestamp) -> pd.DataFrame:
        if end_date < self.start_date:
            index = pd.DatetimeIndex([], name=INDEX_NAME)
            return pd.DataFrame(index=index, columns=COLUMNS, dtype=np.float64)

        slope = self.loader("T10Y2Y", end_date)
        ten_year = self.loader("DGS10", end_date)

        past = slope.shift(1).rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS)
        std = past.std()
        daily_change = ten_year.diff()

        out = pd.DataFrame(
            {
                "t10y2y_level": slope,
                "t10y2y_change_20": slope - slope.shift(CHANGE_WINDOW),
                "t10y2y_z_252": (slope - past.mean()) / std.where(std > 0),
                "dgs10_vol_20": daily_change.rolling(VOL_WINDOW, min_periods=VOL_MIN_PERIODS).std(),
            }
        ).astype(np.float64)
        out.index.name = INDEX_NAME
        return out.loc[self.start_date : end_date, list(COLUMNS)]
