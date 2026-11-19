"""Corporate credit spread features (issue #11).

Inputs, read with ``macro.data.series.load`` (FRED, stamped by posting day and used one
session later, so a value observed on s normally enters at s + 2; see data/README.md):
    BAA10Y  Moody's seasoned Baa yield minus the 10-year Treasury constant maturity yield
    DBAA    Moody's seasoned Baa yield
    DAAA    Moody's seasoned Aaa yield

Features at decision date t:
    baa10y_level      BAA10Y, percentage points
    baa10y_change_20  BAA10Y minus its value 20 sessions earlier
    baa10y_z_252      (level - mean) / std over the 252 sessions before t
    baa_aaa_level     DBAA minus DAAA (default-risk spread net of Treasury moves)

The z-score window ends the session before t, so it is fit on past data only.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from macro.data.calendar import INDEX_NAME
from macro.data.series import load
from macro.features.base import Signal

SERIES = ("BAA10Y", "DBAA", "DAAA")
COLUMNS = ("baa10y_level", "baa10y_change_20", "baa10y_z_252", "baa_aaa_level")
CHANGE_WINDOW = 20
Z_WINDOW = 252
Z_MIN_PERIODS = 245

# Moody's yields observed 2016-10-07 to 2017-03-15 first reached FRED on 2017-03-22,
# after the Fed dropped them from H.15. ``load`` treats them as on time (data/README.md
# keeps them as market data). ``exclude_backfill=True`` blanks the sessions those values
# enter (posting day + 1: 2016-10-12 to 2017-03-17) before computing features, so the
# 20-session change and the z-score window drop them too. data/README.md asks #11 to
# report this run.
BACKFILL_SESSIONS = (pd.Timestamp("2016-10-12"), pd.Timestamp("2017-03-17"))

Loader = Callable[[str, pd.Timestamp], pd.Series]


class Credit(Signal):
    name = "credit"
    source = "FRED (BAA10Y, DBAA, DAAA)"
    start_date = pd.Timestamp("2000-01-03")
    release_lag = 2  # observation to first use, in sessions: posting day + 1

    def __init__(self, loader: Loader = load, exclude_backfill: bool = False):
        self.loader = loader
        self.exclude_backfill = exclude_backfill

    def build(self, end_date: pd.Timestamp) -> pd.DataFrame:
        level = {sid: self.loader(sid, end_date) for sid in SERIES}
        sessions = level["BAA10Y"].index
        if end_date < self.start_date or sessions.empty:
            index = pd.DatetimeIndex([], name=INDEX_NAME)
            return pd.DataFrame(index=index, columns=list(COLUMNS), dtype=float)

        if self.exclude_backfill:
            first, last = BACKFILL_SESSIONS
            level = {
                sid: series.mask((series.index >= first) & (series.index <= last))
                for sid, series in level.items()
            }
        spread = level["BAA10Y"]
        past = spread.shift(1).rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS)
        std = past.std()
        out = pd.DataFrame(
            {
                "baa10y_level": spread,
                "baa10y_change_20": spread - spread.shift(CHANGE_WINDOW),
                "baa10y_z_252": (spread - past.mean()) / std.where(std > 0),
                "baa_aaa_level": level["DBAA"] - level["DAAA"].reindex(sessions),
            },
            index=sessions,
        ).astype(np.float64)
        return out.loc[self.start_date :]
