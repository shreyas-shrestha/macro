"""Slow macro features: weekly jobless claims and financial conditions (issue #14).

Both inputs are weekly. Features are computed on the weekly values, each indexed by the
date it was released (``macro.data.series.raw``), and then put on NYSE sessions with
the input's own lag and carry (``series.on_sessions``). A weekly feature uses only its
own release and earlier ones, so its value on a session is what was knowable then.

Claims (ICNSA, not seasonally adjusted, stamped by release Thursday):
    claims_yoy     log of the 4-week average over the same average 52 weeks earlier
    claims_vs_low  log of the 4-week average over its minimum in the prior 52 weeks

The year-over-year ratio removes seasonality using past data only. Weeks are matched by
date (364 days back), not by position, so a missing week cannot shift the comparison.

Financial conditions (ANFCI first releases from ALFRED, real time from 2011-06-02):
    anfci_level       ANFCI as first released
    anfci_change_13w  that value minus the first release 13 weeks earlier

The Chicago Fed re-estimates the whole ANFCI history every week, so the 13-week change
mixes two vintages; both values were public when the later one was released. ALFRED
first releases are indexed by publication date, not by week, so the change counts 13
releases back. ANFCI had no missing weeks 2011–2017, and the one week released late (the
week of 2012-10-26, published with the next on 2012-11-07) keeps its order.
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
    name = "financial_conditions"
    source = "ALFRED (ANFCI)"
    start_date = pd.Timestamp("2011-06-03")
    release_lag = 1

    def build(self, end_date: pd.Timestamp) -> pd.DataFrame:
        return weekly_on_sessions("ANFCI", conditions_features, end_date).loc[self.start_date :]
