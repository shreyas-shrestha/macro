"""Series registry and ``load``, the one way to read a series (see data/README.md)."""

from dataclasses import dataclass

import pandas as pd

from macro.config import DEV_END
from macro.data import yahoo
from macro.data.cache import cached
from macro.data.calendar import INDEX_NAME, nyse_sessions
from macro.data.pit import align


@dataclass(frozen=True)
class SeriesSpec:
    source: str
    source_id: str
    start: pd.Timestamp
    lag: int
    max_carry: int


SERIES = {
    "SPX": SeriesSpec("yahoo", "^GSPC", pd.Timestamp("1985-01-02"), 0, 0),
    "VIX": SeriesSpec("yahoo", "^VIX", pd.Timestamp("1990-01-02"), 0, 0),
    "VIX3M": SeriesSpec("yahoo", "^VIX3M", pd.Timestamp("2006-07-17"), 0, 0),
    "MOVE": SeriesSpec("yahoo", "^MOVE", pd.Timestamp("2002-11-12"), 1, 3),
}

# A fetcher takes (source_id, start, end) and returns floats indexed by stamp date:
# tz-naive midnight dates within [start, end], NaN for missing, rows sharing a stamp in
# observation order.
FETCHERS = {"yahoo": yahoo.fetch}


def load(name: str, end_date=DEV_END, *, holdout: bool = False, refresh: bool = False) -> pd.Series:
    """Return ``name`` on NYSE sessions from its start through ``end_date``.

    Raises ``ValueError`` for an ``end_date`` after ``DEV_END`` unless ``holdout`` is
    set. Holdout reads bypass the cache, so holdout data never lands on disk by accident.
    Before ``spec.start`` the result is empty.
    """
    spec = SERIES[name]
    end_date = pd.Timestamp(end_date)
    if end_date > DEV_END and not holdout:
        raise ValueError(f"{name}: {end_date.date()} is in the holdout; pass holdout=True")
    if end_date < spec.start:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], name=INDEX_NAME), name=name)

    def fetch(end):
        raw = FETCHERS[spec.source](spec.source_id, spec.start, end)
        check_raw(name, raw, spec.start, end)
        return raw

    if holdout:
        raw = fetch(end_date)
    else:
        raw = cached(f"{spec.source}/{spec.source_id}", lambda: fetch(DEV_END), refresh)
    sessions = nyse_sessions(spec.start, end_date).rename(INDEX_NAME)
    return align(raw.loc[:end_date], sessions, spec.lag, spec.max_carry).rename(name)


def check_raw(name: str, raw: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Raise ``ValueError`` if a fetcher broke its contract (see ``FETCHERS``)."""
    index = raw.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is not None:
        raise ValueError(f"{name}: fetcher must return a tz-naive DatetimeIndex")
    if not (index == index.normalize()).all():
        raise ValueError(f"{name}: fetcher returned stamps with a time of day")
    if not pd.api.types.is_float_dtype(raw):
        raise ValueError(f"{name}: fetcher must return floats, got {raw.dtype}")
    if len(index) and (index.min() < start or index.max() > end):
        raise ValueError(f"{name}: fetcher returned stamps outside {start.date()}..{end.date()}")
