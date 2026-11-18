"""Weekly jobless claims, first releases where ALFRED has them, stamped by release day."""

import pandas as pd
from fredapi import Fred

from macro.data import alfred
from macro.data.fred import api_key

# The week ending Saturday s is released on Thursday s + 5 (see data/README.md).
RELEASE = pd.Timedelta(days=5)


def fetch(series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Weekly values of ``series_id`` known by ``end``, each stamped Thursday s + 5.

    Weeks with an ALFRED first release take that value; earlier weeks take the vintage
    published on ``end``. Raises ``ValueError`` if a first release came after its stamp.
    """
    day = end.strftime("%Y-%m-%d")
    vintage = Fred(api_key=api_key()).get_series(
        series_id,
        observation_start=start,
        observation_end=end,
        realtime_start=day,
        realtime_end=day,
    )
    first = alfred.first_releases(series_id, end)
    late = first["date"][first["realtime_start"] > first["date"] + RELEASE]
    if not late.empty:
        raise ValueError(f"{series_id}: released after the stamp for weeks {late.tolist()}")
    first = pd.Series(first["value"].to_numpy(), index=pd.DatetimeIndex(first["date"]))
    # A missing first release must not fall back to the revised vintage value.
    released = vintage.index[vintage.index + RELEASE <= end]
    covered = released[released >= first.index.min()] if len(first) else []
    missing = first.index[first.isna()].union(pd.DatetimeIndex(covered).difference(first.index))
    if not missing.empty:
        raise ValueError(f"{series_id}: no first release for weeks {missing.date.tolist()}")
    weekly = first.combine_first(vintage.astype(float))
    if (weekly.index.day_of_week != 5).any():
        raise ValueError(f"{series_id}: expected weeks ending Saturday")
    return pd.Series(weekly.to_numpy(), index=weekly.index + RELEASE).loc[start:end]
