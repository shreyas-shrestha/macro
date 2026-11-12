"""Corporate credit spread features (issue #11).

Inputs, all FRED daily series observed at the close of day s:
    BAA10Y  Moody's seasoned Baa yield minus the 10-year Treasury constant maturity yield
    DBAA    Moody's seasoned Baa yield
    DAAA    Moody's seasoned Aaa yield

Features (decision date t, release lag L = 2 NYSE sessions, see data/README.md):
    baa10y_level      BAA10Y, percentage points
    baa10y_change_20  BAA10Y minus its value 20 sessions earlier
    baa10y_z_252      (level - mean) / std over the 252 sessions before t
    baa_aaa_level     DBAA minus DAAA (default-risk spread net of Treasury moves)

The z-score window ends the session before t, so it is fit on past data only.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from macro.config import DEV_END
from macro.data.calendar import nyse_sessions
from macro.features.base import INDEX_NAME, Signal

SERIES = ("BAA10Y", "DBAA", "DAAA")
COLUMNS = ("baa10y_level", "baa10y_change_20", "baa10y_z_252", "baa_aaa_level")
CHANGE_WINDOW = 20
Z_WINDOW = 252
Z_MIN_PERIODS = 245
# Bond-market holidays that are NYSE sessions (Columbus Day, Veterans Day) leave FRED
# gaps of one session; carry the last released value across at most this many sessions.
MAX_CARRY = 3

Loader = Callable[[str, pd.Timestamp, pd.Timestamp], pd.Series]


def load_fred(series_id: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    """Download a FRED series by observation date, inclusive bounds. Needs FRED_API_KEY."""
    import os

    from macro.config import load_env

    load_env()
    from fredapi import Fred

    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
    series = series.loc[start_date:end_date].astype(float)
    if series.dropna().empty:
        raise RuntimeError(f"FRED returned no observations for {series_id}")
    return series


def align(raw: pd.Series, sessions: pd.DatetimeIndex, lag: int) -> pd.Series:
    """Stamp each observation with the session it becomes usable and carry it forward.

    An observation dated s becomes usable ``lag`` sessions after the first NYSE session
    on or after s. Observations on non-sessions (e.g. Good Friday) are therefore never
    earlier than the following session's own observation would be. The last usable value
    is carried for at most ``MAX_CARRY`` sessions.
    """
    raw = raw.dropna().sort_index()
    raw = raw[raw.index <= sessions[-1]] if len(sessions) else raw.iloc[:0]
    first_session = sessions.searchsorted(raw.index, side="left")
    usable_pos = first_session + lag
    keep = usable_pos < len(sessions)
    stamped = pd.Series(raw.to_numpy()[keep], index=sessions[usable_pos[keep]])
    # Several observations can land on one session; the latest observation wins.
    stamped = stamped[~stamped.index.duplicated(keep="last")]
    return stamped.reindex(sessions).ffill(limit=MAX_CARRY)


class Credit(Signal):
    name = "credit"
    source = "FRED (BAA10Y, DBAA, DAAA)"
    start_date = pd.Timestamp("2000-01-03")
    release_lag = 2

    # Pull a year of history before start_date so the first z-scores are not empty.
    history_start = pd.Timestamp("1998-06-01")

    def __init__(self, loader: Loader = load_fred):
        self.loader = loader

    def build(self, end_date: pd.Timestamp) -> pd.DataFrame:
        if end_date > DEV_END:
            raise ValueError("credit holdout data (2018 onward) is locked")
        if end_date < self.start_date:
            return pd.DataFrame(
                index=pd.DatetimeIndex([], name=INDEX_NAME), columns=COLUMNS, dtype=float
            )

        sessions = nyse_sessions(self.history_start, end_date).rename(INDEX_NAME)
        # Observations after this date cannot be usable by end_date.
        last_obs = sessions[-1 - self.release_lag]
        level = {
            sid: align(self.loader(sid, self.history_start, last_obs), sessions, self.release_lag)
            for sid in SERIES
        }

        spread = level["BAA10Y"]
        past = spread.shift(1).rolling(Z_WINDOW, min_periods=Z_MIN_PERIODS)
        std = past.std()
        z_score = (spread - past.mean()) / std.where(std > 0)

        out = pd.DataFrame(
            {
                "baa10y_level": spread,
                "baa10y_change_20": spread - spread.shift(CHANGE_WINDOW),
                "baa10y_z_252": z_score,
                "baa_aaa_level": level["DBAA"] - level["DAAA"],
            },
            index=sessions,
        ).astype(np.float64)
        return out.loc[self.start_date :]
