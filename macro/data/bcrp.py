"""Daily series from the Central Reserve Bank of Peru (BCRP) statistics API."""

import json
import urllib.request

import pandas as pd

API = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"
MONTHS = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Set", "Oct", "Nov", "Dic"]


def fetch(code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Values of series ``code`` dated ``start`` through ``end`` inclusive."""
    url = f"{API}/{code}/json/{start:%Y-%m-%d}/{end:%Y-%m-%d}"
    with urllib.request.urlopen(url, timeout=60) as r:
        periods = json.load(r)["periods"]
    return parse(periods).loc[start:end]


def parse(periods: list[dict]) -> pd.Series:
    """Parse BCRP periods: dates like ``01.Set.17``, missing values as ``n.d.``."""
    dates = []
    for period in periods:
        day, month, year = period["name"].split(".")
        dates.append(f"{day}.{MONTHS.index(month) + 1}.{year}")
    values = [period["values"][0] for period in periods]
    return pd.Series(
        pd.to_numeric(values, errors="coerce"),
        index=pd.to_datetime(dates, format="%d.%m.%y"),
        dtype=float,
    )
