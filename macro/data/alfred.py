"""First releases from ALFRED, stamped by the date each value was first published."""

import json
import urllib.parse
import urllib.request

import pandas as pd

from macro.data.fred import api_key

API = "https://api.stlouisfed.org/fred/series/observations"
# ALFRED's earliest real-time date, so every vintage is included.
EARLIEST = "1776-07-04"


def fetch(series_id: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """First-release values of ``series_id`` published ``start`` through ``end`` inclusive.

    The index is the publication (vintage) date, not the observation date. Values
    published on the same date are ordered by observation date.
    """
    params = {
        "series_id": series_id,
        "api_key": api_key(),
        "file_type": "json",
        "output_type": 4,  # initial release only
        "realtime_start": EARLIEST,
        "realtime_end": end.strftime("%Y-%m-%d"),
        "observation_end": end.strftime("%Y-%m-%d"),
        "limit": 100000,
    }
    with urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(params)}", timeout=60) as r:
        rows = json.load(r)["observations"]
    return parse(rows).loc[start:end]


def parse(rows: list[dict]) -> pd.Series:
    """Values indexed by first-release date, ties in observation order; "." is missing."""
    frame = pd.DataFrame(rows, columns=["date", "realtime_start", "value"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["realtime_start"] = pd.to_datetime(frame["realtime_start"])
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.sort_values(["realtime_start", "date"])
    return pd.Series(frame["value"].to_numpy(), index=pd.DatetimeIndex(frame["realtime_start"]))
