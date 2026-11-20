"""Series registry and ``load``, the one way to read a series (see data/README.md)."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import pandas as pd

from macro.config import DEV_END
from macro.data import alfred, bcrp, claims, fred, yahoo
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
    "BAA10Y": SeriesSpec("fred", "BAA10Y", pd.Timestamp("1990-01-02"), 1, 3),
    "DBAA": SeriesSpec("fred", "DBAA", pd.Timestamp("1990-01-02"), 1, 3),
    "DAAA": SeriesSpec("fred", "DAAA", pd.Timestamp("1990-01-02"), 1, 3),
    "DGS10": SeriesSpec("fred", "DGS10", pd.Timestamp("1990-01-02"), 1, 3),
    "T10Y2Y": SeriesSpec("fred", "T10Y2Y", pd.Timestamp("1990-01-02"), 1, 3),
    "T10Y3M": SeriesSpec("fred", "T10Y3M", pd.Timestamp("1990-01-02"), 1, 3),
    "CPFF": SeriesSpec("fred", "CPFF", pd.Timestamp("1997-01-02"), 1, 3),
    "NFCI": SeriesSpec("alfred", "NFCI", pd.Timestamp("1990-01-02"), 1, 10),
    "ANFCI": SeriesSpec("alfred", "ANFCI", pd.Timestamp("1990-01-02"), 1, 10),
    "ICSA": SeriesSpec("alfred", "ICSA", pd.Timestamp("1990-01-02"), 1, 10),
    "ICNSA": SeriesSpec("claims", "ICNSA", pd.Timestamp("1988-01-02"), 1, 10),
    "EMBIG": SeriesSpec("bcrp", "PD04707XD", pd.Timestamp("1998-01-02"), 2, 3),
    "EMBIG_LATAM": SeriesSpec("bcrp", "PD04708XD", pd.Timestamp("1998-01-02"), 2, 3),
    "EMBIG_PERU": SeriesSpec("bcrp", "PD04709XD", pd.Timestamp("1998-01-02"), 2, 3),
    "EMBIG_ARGENTINA": SeriesSpec("bcrp", "PD04710XD", pd.Timestamp("1998-01-02"), 2, 3),
    "EMBIG_BRAZIL": SeriesSpec("bcrp", "PD04711XD", pd.Timestamp("1998-01-02"), 2, 3),
    "EMBIG_ECUADOR": SeriesSpec("bcrp", "PD04712XD", pd.Timestamp("1998-01-02"), 2, 3),
    "EMBIG_MEXICO": SeriesSpec("bcrp", "PD04713XD", pd.Timestamp("1998-01-02"), 2, 3),
    "EMBIG_VENEZUELA": SeriesSpec("bcrp", "PD04714XD", pd.Timestamp("1998-01-02"), 2, 3),
    "EMBIG_COLOMBIA": SeriesSpec("bcrp", "PD04715XD", pd.Timestamp("1998-01-02"), 2, 3),
}

# A fetcher takes (source_id, start, end) and returns floats indexed by stamp date:
# tz-naive midnight dates within [start, end], NaN for missing, rows sharing a stamp in
# observation order.
FETCHERS = {
    "yahoo": yahoo.fetch,
    "fred": fred.fetch,
    "claims": claims.fetch,
    "alfred": alfred.fetch,
    "bcrp": bcrp.fetch,
}

_HOLDOUT_READS = ContextVar("holdout_reads", default=False)


@contextmanager
def holdout_reads():
    """Let every ``load`` inside the block read the holdout, as if ``holdout=True``.

    For the final run only (``run_signal.py --holdout``): signals call ``load`` inside
    ``build`` and cannot be handed the flag themselves. Every read in the block is a
    fresh download, including pre-2018 history, so it can differ from the cached files
    the development runs used if a source revised its history.
    """
    token = _HOLDOUT_READS.set(True)
    try:
        yield
    finally:
        _HOLDOUT_READS.reset(token)


def load(name: str, end_date=DEV_END, *, holdout: bool = False, refresh: bool = False) -> pd.Series:
    """Return ``name`` on NYSE sessions from its start through ``end_date``.

    Raises ``ValueError`` for an ``end_date`` after ``DEV_END`` unless ``holdout`` is
    set or the call is inside ``holdout_reads()``. Holdout reads bypass the cache, so
    holdout data never lands on disk by accident. Before ``spec.start`` the result is
    empty.
    """
    values = raw(name, end_date, holdout=holdout, refresh=refresh)
    return on_sessions(name, values, end_date).rename(name)


def raw(name: str, end_date=DEV_END, *, holdout: bool = False, refresh: bool = False) -> pd.Series:
    """``name``'s values by stamp date through ``end_date``, before alignment.

    For computing on a series' native frequency before ``on_sessions``. Same holdout and
    cache rules as ``load``.
    """
    spec = SERIES[name]
    holdout = holdout or _HOLDOUT_READS.get()
    end_date = pd.Timestamp(end_date)
    if end_date > DEV_END and not holdout:
        raise ValueError(f"{name}: {end_date.date()} is in the holdout; pass holdout=True")
    if end_date < spec.start:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], dtype="datetime64[ns]"))

    def fetch(end):
        values = FETCHERS[spec.source](spec.source_id, spec.start, end)
        _check_raw(name, values, spec.start, end)
        return values

    if holdout:
        values = fetch(end_date)
    else:
        values = cached(f"{spec.source}/{spec.source_id}", lambda: fetch(DEV_END), refresh)
        # Also check what was read: the file may predate these checks or be edited.
        _check_raw(name, values, spec.start, DEV_END)
    return values.loc[:end_date]


def on_sessions(name: str, values: pd.Series, end_date) -> pd.Series:
    """Put ``values``, stamped like ``name``'s raw data, on NYSE sessions with its lag and carry."""
    spec = SERIES[name]
    end_date = pd.Timestamp(end_date)
    if end_date < spec.start:
        index = pd.DatetimeIndex([], dtype="datetime64[ns]", name=INDEX_NAME)
        return pd.Series(dtype=float, index=index)
    sessions = nyse_sessions(spec.start, end_date).rename(INDEX_NAME)
    return align(values.loc[:end_date], sessions, spec.lag, spec.max_carry)


def _check_raw(name: str, raw: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> None:
    """Raise ``ValueError`` if raw data breaks the fetcher contract (see ``FETCHERS``)."""
    index = raw.index
    if not isinstance(index, pd.DatetimeIndex) or index.tz is not None:
        raise ValueError(f"{name}: index must be a tz-naive DatetimeIndex")
    if not (index == index.normalize()).all():
        raise ValueError(f"{name}: stamps have a time of day")
    if raw.dropna().empty:
        raise ValueError(f"{name}: no data; if this is a cached file, load with refresh=True")
    if not pd.api.types.is_float_dtype(raw):
        raise ValueError(f"{name}: values must be floats, got {raw.dtype}")
    if index.min() < start or index.max() > end:
        raise ValueError(f"{name}: stamps outside {start.date()}..{end.date()}")
