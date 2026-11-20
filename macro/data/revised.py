"""Revised history of a weekly Chicago Fed index, for exploratory runs only.

Not point in time: every value is the one in the vintage published on ``end``, so it
includes revisions made after its release. See data/README.md.
"""

import pandas as pd
from fredapi import Fred

from macro.data.fred import api_key

# The week ending Friday f is released Wednesday f + 5, or Thursday f + 6 in holiday
# weeks, so every week is stamped f + 6. One week, ending 2012-10-26, was released late
# (data/README.md); this exploratory series ignores it.
RELEASE = pd.Timedelta(days=6)


def fetch(series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """``series_id`` in the vintage published on ``end``, each week stamped f + 6."""
    day = end.strftime("%Y-%m-%d")
    vintage = (
        Fred(api_key=api_key())
        .get_series(
            series_id,
            observation_start=start,
            observation_end=end,
            realtime_start=day,
            realtime_end=day,
        )
        .astype(float)
    )
    if (vintage.index.day_of_week != 4).any():
        raise ValueError(f"{series_id}: weeks must end on a Friday")
    stamped = pd.Series(vintage.to_numpy(), index=vintage.index + RELEASE)
    return stamped.loc[start:end]
