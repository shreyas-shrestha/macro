"""pure random noise - sanity check for the harness (#10)

a fake signal w/ zero info. through the harness it should show NO real improvement.
if it does, the harness is leaking future data or overfitting
"""

import numpy as np
import pandas as pd

from macro.data.calendar import nyse_sessions
from macro.features.base import INDEX_NAME, Signal


class Noise(Signal):
    name = "noise"
    source = "numpy random, fixed seed"
    start_date = pd.Timestamp("2000-01-03")
    release_lag = 0

    def __init__(self, seed: int = 0, columns: int = 3):
        self.seed = seed
        self.columns = columns

    def build(self, end_date: pd.Timestamp) -> pd.DataFrame:
        sessions = nyse_sessions(self.start_date, end_date).rename(INDEX_NAME)
        # same seed + same start -> earlier rows never change when end_date moves
        values = np.random.default_rng(self.seed).normal(size=(len(sessions), self.columns))
        return pd.DataFrame(
            values, index=sessions, columns=[f"noise_{i}" for i in range(self.columns)]
        )
