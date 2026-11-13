"""On-disk parquet cache for raw downloads."""

import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "cache"


def cached(key: str, fetch: Callable[[], pd.Series], refresh: bool = False) -> pd.Series:
    """Return the series stored under ``key``, calling ``fetch`` on a miss or ``refresh``."""
    path = CACHE_DIR / f"{key}.parquet"
    if path.exists() and not refresh:
        return pd.read_parquet(path)["value"]
    series = fetch()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename, so an interrupted run never leaves a half-written file.
    tmp = path.with_suffix(".tmp")
    series.rename("value").to_frame().to_parquet(tmp)
    os.replace(tmp, path)
    return series
