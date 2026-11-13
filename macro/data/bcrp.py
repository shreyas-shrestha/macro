"""Daily series from the Central Reserve Bank of Peru (BCRP) statistics API."""

import json
import urllib.request

import pandas as pd

API = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"
MONTHS = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Set", "Oct", "Nov", "Dic"]


USER_AGENT = "macro-research (https://github.com/shreyas-shrestha/macro)"


def fetch(code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Values of series ``code`` dated ``start`` through ``end`` inclusive."""
    url = f"{API}/{code}/json/{start:%Y-%m-%d}/{end:%Y-%m-%d}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as r:
        body = r.read()
    # BCRP sometimes answers 200 with an HTML bot-check page instead of JSON.
    if not body.lstrip().startswith(b"{"):
        raise RuntimeError(f"BCRP returned a non-JSON page for {code}; try again later")
    return parse(json.loads(body)["periods"]).loc[start:end]


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
