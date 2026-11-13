"""Registry of every series the team uses, and ``load``, the one way to read them.

Each entry fixes where a series comes from and when its values become usable. ``load``
fetches the raw series (cached through ``DEV_END``), then aligns it onto NYSE sessions
with ``pit.align``, so the value at decision date t is the latest one usable by the
close on t. ``data/README.md`` documents every entry; a test keeps the two in step.
"""

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
    description: str


SERIES = {
    "SPX": SeriesSpec("yahoo", "^GSPC", pd.Timestamp("1985-01-02"), 0, 0, "S&P 500 close"),
    "VIX": SeriesSpec("yahoo", "^VIX", pd.Timestamp("1990-01-02"), 0, 0, "CBOE VIX close"),
    "VIX3M": SeriesSpec(
        "yahoo", "^VIX3M", pd.Timestamp("2006-07-17"), 0, 0, "CBOE 3-month VIX close"
    ),
    "MOVE": SeriesSpec(
        "yahoo", "^MOVE", pd.Timestamp("2002-11-12"), 1, 3, "ICE BofA MOVE index close"
    ),
}

FETCHERS = {"yahoo": yahoo.fetch}


def load(name: str, end_date=DEV_END, *, holdout: bool = False, refresh: bool = False) -> pd.Series:
    """Return ``name`` on NYSE sessions from its start through ``end_date``.

    Raises ``ValueError`` for an ``end_date`` after ``DEV_END`` unless ``holdout`` is
    set. Holdout reads bypass the cache, so holdout data never lands on disk by accident.
    """
    spec = SERIES[name]
    end_date = pd.Timestamp(end_date)
    if end_date > DEV_END and not holdout:
        raise ValueError(f"{name}: {end_date.date()} is in the holdout; pass holdout=True")
    fetch = FETCHERS[spec.source]
    if holdout:
        raw = fetch(spec.source_id, spec.start, end_date)
    else:
        raw = cached(
            f"{spec.source}/{spec.source_id}",
            lambda: fetch(spec.source_id, spec.start, DEV_END),
            refresh,
        )
    sessions = nyse_sessions(spec.start, end_date).rename(INDEX_NAME)
    return align(raw.loc[:end_date], sessions, spec.lag, spec.max_carry).rename(name)
