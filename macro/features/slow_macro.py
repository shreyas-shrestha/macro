"""Slow macro features: weekly jobless claims and financial conditions (issue #14).

Computed on weekly values indexed by release date (``series.raw``), then put on NYSE
sessions with the input's lag and carry (``series.on_sessions``). Sources and timing,
including which values are first releases: data/README.md.

    claims_yoy        log(4-week average of ICNSA / the same average 52 weeks earlier)
    claims_vs_low     log(4-week average / its minimum over the prior 52 weeks)
    anfci_level       ANFCI first release (or revised value, see FinancialConditions)
    anfci_change_13w  that value minus the value 13 releases earlier

The year-ago week is matched by date (364 days back), not by position. ANFCI first
releases are indexed by publication date, so its change counts releases; ANFCI has no
missing weeks 2011–2017.
"""

import numpy as np
import pandas as pd

from macro.data.series import on_sessions, raw
from macro.features.base import INDEX_NAME, Signal

WEEK = pd.Timedelta(weeks=1)


def claims_features(weekly: pd.Series) -> pd.DataFrame:
    """Claims features on a weekly series indexed by release date."""
    average = weekly.rolling(4 * WEEK, min_periods=4).mean()
    year_ago = average.reindex(average.index - 52 * WEEK).to_numpy()
    prior_low = average.rolling(52 * WEEK, closed="left", min_periods=52).min()
    return pd.DataFrame(
        {"claims_yoy": np.log(average / year_ago), "claims_vs_low": np.log(average / prior_low)}
    )


def conditions_features(weekly: pd.Series) -> pd.DataFrame:
    """ANFCI features on first releases in publication order."""
    return pd.DataFrame({"anfci_level": weekly, "anfci_change_13w": weekly - weekly.shift(13)})


def weekly_on_sessions(series: str, features, end_date: pd.Timestamp) -> pd.DataFrame:
    """``features`` of ``series``'s released values, each column put on sessions."""
    weekly = features(raw(series, end_date))
    out = pd.DataFrame({c: on_sessions(series, weekly[c], end_date) for c in weekly})
    out.index.name = INDEX_NAME
    return out


class Claims(Signal):
    name = "claims"
    source = "FRED (ICNSA)"
    # ICNSA is read from 1988, so both features are complete from here.
    start_date = pd.Timestamp("1990-01-02")
    release_lag = 1

    def build(self, end_date: pd.Timestamp) -> pd.DataFrame:
        return weekly_on_sessions("ICNSA", claims_features, end_date).loc[self.start_date :]


class FinancialConditions(Signal):
    """ANFCI features from first releases, or with ``revised=True`` from the revised
    history (exploratory: not point in time, see data/README.md)."""

    name = "financial_conditions"
    source = "ALFRED (ANFCI)"
    start_date = pd.Timestamp("2011-06-03")
    release_lag = 1

    def __init__(self, revised: bool = False):
        self.series = "ANFCI_REVISED" if revised else "ANFCI"
        if revised:
            self.source = "FRED (ANFCI, revised history)"
            self.start_date = pd.Timestamp("1990-01-02")

    def build(self, end_date: pd.Timestamp) -> pd.DataFrame:
        frame = weekly_on_sessions(self.series, conditions_features, end_date)
        return frame.loc[self.start_date :]
