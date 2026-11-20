"""Funding and liquidity: 3-month commercial paper minus fed funds (issue #15).

CPFF is read with ``macro.data.series.load``: FRED, stamped by posting day and used one
session later, carried at most 3 sessions; its 16 missing sessions in Nov 2008–Jan 2009
stay NaN (data/README.md).

    cpff_level       CPFF, percentage points
    cpff_change_20d  CPFF minus CPFF 20 sessions earlier
    cpff_z_252       (CPFF - mean) / sample s.d. (ddof = 1) of the 252 sessions ending at
                     t, at least 200 of them non-missing

Windows count sessions, missing ones included, and are not extended to make up for
them. Features are computed on CPFF from 1997, so the z-score is complete from the
first row.
"""

import pandas as pd

from macro.data.series import load
from macro.features.base import Signal


class Funding(Signal):
    name = "funding"
    source = "FRED (CPFF)"
    start_date = pd.Timestamp("2000-01-03")
    release_lag = 1

    def build(self, end_date: pd.Timestamp) -> pd.DataFrame:
        cpff = load("CPFF", end_date)
        window = cpff.rolling(252, min_periods=200)
        out = pd.DataFrame(
            {
                "cpff_level": cpff,
                "cpff_change_20d": cpff - cpff.shift(20),
                "cpff_z_252": (cpff - window.mean()) / window.std(),
            }
        )
        return out.loc[self.start_date :]
