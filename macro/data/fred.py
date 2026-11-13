"""Daily and weekly series from FRED, by observation date."""

import os

import pandas as pd
from fredapi import Fred

from macro.config import load_env


def api_key() -> str:
    load_env()
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY is not set; add it to .env (see .env.example)")
    return key


def fetch(series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Values of ``series_id`` observed ``start`` through ``end`` inclusive."""
    series = Fred(api_key=api_key()).get_series(
        series_id, observation_start=start, observation_end=end
    )
    return series.astype(float).loc[start:end]
