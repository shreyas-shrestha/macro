"""Treasury constant-maturity curve (FRED / Federal Reserve H.15) and maturity-matched spreads.

Source: FRED daily series DGS1MO ... DGS30, the Board's H.15 Treasury constant-maturity
yields (percent, investment basis). Cached as one CSV per series under
``data/cache/fred/`` (public data; no API key is needed for the CSV endpoint).

Timing: a daily FRED value for day s is published the next morning, so its release lag
is L = 1 trading day (the CONTRIBUTING.md convention for FRED daily series). A bond
spread for day s pairs TRACE yields from day s with the curve for day s; the spread is
an observation dated s and must enter a signal no earlier than s + 1. The HY signal
applies that one-session lag to every input, so no same-day publication lookahead arises.

Interpolation uses only the curve of the same observation date: linear in maturity
between the nodes available that day, flat beyond the shortest and longest available
node. No value is carried across dates.

DGS30 was discontinued on 2002-02-18 and reintroduced on 2006-02-09 (FRED series notes).
FRED now shows values inside that gap; they are not the regular 30-year constant-maturity
series, so they are treated as missing and the curve is flat beyond 20 years there.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from macro.config import DEV_END

CMT_SERIES = {
    "DGS1MO": 1 / 12,
    "DGS3MO": 0.25,
    "DGS6MO": 0.5,
    "DGS1": 1.0,
    "DGS2": 2.0,
    "DGS3": 3.0,
    "DGS5": 5.0,
    "DGS7": 7.0,
    "DGS10": 10.0,
    "DGS20": 20.0,
    "DGS30": 30.0,
}
RELEASE_LAG = 1
DGS30_GAP = (pd.Timestamp("2002-02-18"), pd.Timestamp("2006-02-08"))
FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}&coed={end}"


def fetch_cmt_curve(directory: str | Path, start="2000-01-01", end=DEV_END) -> None:
    """Download each CMT series as CSV into ``directory``; ``end`` may not pass DEV_END."""
    import urllib.request

    end = pd.Timestamp(end)
    if end > DEV_END:
        raise ValueError(f"end {end.date()} is after DEV_END; the holdout is locked")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for series in CMT_SERIES:
        url = FRED_CSV_URL.format(series=series, start=pd.Timestamp(start).date(), end=end.date())
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 (fixed https host)
            (directory / f"{series}.csv").write_bytes(response.read())


def load_cmt_curve(directory: str | Path, end=DEV_END) -> pd.DataFrame:
    """Curve indexed by observation date, columns = maturity in years, values in percent.

    Rows after ``end`` (default DEV_END) are dropped; a cached file containing holdout
    dates raises rather than being silently truncated.
    """
    directory = Path(directory)
    columns = {}
    for series, maturity in CMT_SERIES.items():
        path = directory / f"{series}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        raw = pd.read_csv(path, dtype=str, keep_default_na=False)
        dates = pd.to_datetime(raw.iloc[:, 0], errors="raise")
        values = pd.to_numeric(raw.iloc[:, 1].replace({"": None, ".": None}), errors="raise")
        series_values = pd.Series(values.to_numpy(dtype="float64"), index=dates)
        if series == "DGS30":
            in_gap = (series_values.index >= DGS30_GAP[0]) & (series_values.index <= DGS30_GAP[1])
            series_values[in_gap] = np.nan
        columns[maturity] = series_values
    curve = pd.DataFrame(columns).sort_index()
    if (curve.index > pd.Timestamp(DEV_END)).any():
        raise ValueError("cached Treasury curve contains holdout dates (after DEV_END)")
    curve = curve.loc[: pd.Timestamp(end)]
    curve.index.name = "date"
    return curve


def interpolate_curve(
    curve: pd.DataFrame, dates: pd.Series, maturities: pd.Series, min_nodes: int = 2
) -> pd.Series:
    """Treasury yield (percent) for each (date, maturity in years), same-date curve only.

    Missing when the date is not in ``curve`` or has fewer than ``min_nodes`` nodes, or
    when the maturity is missing. Index follows ``dates``.
    """
    dates = pd.to_datetime(pd.Series(dates))
    maturities = pd.Series(maturities, index=dates.index).astype("float64")
    out = pd.Series(np.nan, index=dates.index, dtype="float64")
    nodes = curve.columns.to_numpy(dtype="float64")
    order = np.argsort(nodes)
    nodes = nodes[order]
    values = curve.to_numpy(dtype="float64")[:, order]
    row_of = pd.Series(np.arange(len(curve)), index=curve.index)
    positions = dates.map(row_of)
    for row, idx in dates.index.groupby(positions).items():
        if pd.isna(row):
            continue
        available = ~np.isnan(values[int(row)])
        if available.sum() < min_nodes:
            continue
        mats = maturities.loc[idx].to_numpy()
        result = np.interp(mats, nodes[available], values[int(row)][available])
        result[np.isnan(mats)] = np.nan
        out.loc[idx] = result
    return out


def treasury_spread_bp(
    frame: pd.DataFrame,
    curve: pd.DataFrame,
    *,
    date_col: str = "trd_exctn_dt",
    maturity_col: str = "remaining_maturity_y",
    yield_col: str = "yield_pct",
) -> pd.DataFrame:
    """Add ``treasury_yield_pct`` and ``spread_bp`` = 100 * (yield - Treasury yield)."""
    out = frame.copy()
    out["treasury_yield_pct"] = interpolate_curve(curve, out[date_col], out[maturity_col])
    out["spread_bp"] = 100.0 * (out[yield_col] - out["treasury_yield_pct"])
    return out
