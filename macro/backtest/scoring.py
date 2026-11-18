"""Forecast losses."""

import numpy as np
import pandas as pd


def qlike(forecast: pd.Series, realized: pd.Series) -> float:
    """Mean QLIKE loss of variance forecasts (Patton 2011); 0 for a perfect forecast."""
    ratio = realized / forecast
    return float(np.mean(ratio - np.log(ratio) - 1))
