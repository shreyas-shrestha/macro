"""Treasury implied volatility features for NYSE close forecasts."""

import pandas as pd

from macro.config import DEV_END
from macro.data.calendar import nyse_sessions
from macro.features.base import INDEX_NAME, Signal

COLUMNS = ("move_level", "move_change_20", "move_z_252")


def load_move(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    """Request daily Yahoo closes through the inclusive end date."""
    import yfinance as yf

    prices = yf.download(
        "^MOVE",
        start=start_date,
        end=end_date + pd.Timedelta(days=1),
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )
    if prices.empty or prices["Close"].dropna().empty:
        raise RuntimeError("Yahoo returned no MOVE closes for the requested dates")
    return prices["Close"]


class Move(Signal):
    name = "move"
    source = "Yahoo Finance (^MOVE)"
    start_date = pd.Timestamp("2003-01-01")
    release_lag = 1

    def __init__(self, loader=load_move):
        self.loader = loader

    def build(self, end_date: pd.Timestamp) -> pd.DataFrame:
        if end_date > DEV_END:
            raise ValueError("MOVE holdout data is locked")
        if end_date < self.start_date:
            return pd.DataFrame(
                index=pd.DatetimeIndex([], name=INDEX_NAME), columns=COLUMNS, dtype=float
            )

        sessions = nyse_sessions(self.start_date, end_date).rename(INDEX_NAME)
        # The final session's close cannot enter the requested decision window.
        if len(sessions) <= self.release_lag:
            return pd.DataFrame(index=sessions, columns=COLUMNS, dtype=float)
        closes = self.loader(self.start_date, sessions[-1 - self.release_lag])
        level = closes.astype(float).reindex(sessions).shift(self.release_lag).ffill(limit=1)
        past = level.shift(1).rolling(252, min_periods=245)
        std = past.std()
        z_score = (level - past.mean()) / std.where(std > 0)
        z_score.iloc[:252] = float("nan")
        return pd.DataFrame(
            {
                "move_level": level,
                "move_change_20": level.diff(20),
                "move_z_252": z_score,
            },
            index=sessions,
        )
