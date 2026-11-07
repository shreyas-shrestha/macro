"""Local WRDS extract helpers for the high-yield credit signal.

Raw WRDS data is licensed. Export CSVs from WRDS into data/cache/ and load
them locally; never commit raw extracts to the public repository.
"""

from pathlib import Path

import pandas as pd

TRACE_REQUIRED = {
    "trd_exctn_dt",
    "cusip_id",
    "rptd_pr",
    "yld_pt",
}

RATINGS_REQUIRED = {
    "gvkey",
    "datadate",
    "splticrm",
}

MASTER_REQUIRED = {
    "cusip_id",
    "issuer_nm",
    "mtrty_dt",
}


def _read_csv(path: str | Path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return frame


def load_trace_trades(path: str | Path) -> pd.DataFrame:
    """Load a WRDS TRACE BTDS CSV export with basic type normalization."""
    frame = _read_csv(path, TRACE_REQUIRED).copy()
    frame["trd_exctn_dt"] = pd.to_datetime(frame["trd_exctn_dt"], errors="raise")
    frame["rptd_pr"] = pd.to_numeric(frame["rptd_pr"], errors="coerce")
    frame["yld_pt"] = pd.to_numeric(frame["yld_pt"], errors="coerce")

    if "ascii_rptd_vol_tx" in frame:
        frame["ascii_rptd_vol_tx"] = pd.to_numeric(
            frame["ascii_rptd_vol_tx"], errors="coerce"
        )

    return frame


def load_trace_master(path: str | Path) -> pd.DataFrame:
    """Load the TRACE corporate-bond master file."""
    frame = _read_csv(path, MASTER_REQUIRED).copy()
    frame["mtrty_dt"] = pd.to_datetime(frame["mtrty_dt"], errors="coerce")
    if "cpn_rt" in frame:
        frame["cpn_rt"] = pd.to_numeric(frame["cpn_rt"], errors="coerce")
    return frame


def load_compustat_ratings(path: str | Path) -> pd.DataFrame:
    """Load the legacy Compustat North America Ratings extract."""
    frame = _read_csv(path, RATINGS_REQUIRED).copy()
    frame["datadate"] = pd.to_datetime(frame["datadate"], errors="raise")
    frame["gvkey"] = frame["gvkey"].astype(str)
    return frame


def basic_clean_trace(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply only uncontroversial filters before TRACE-specific cleaning.

    Cancellation/correction logic is deliberately handled later, after the
    observed code values in our WRDS extract are documented.
    """
    out = frame.copy()
    out = out[
        out["cusip_id"].notna()
        & out["rptd_pr"].gt(0)
        & out["yld_pt"].notna()
    ]

    if "wis_fl" in out:
        out = out[out["wis_fl"].fillna("N").eq("N")]
    if "spcl_trd_fl" in out:
        out = out[out["spcl_trd_fl"].fillna("N").eq("N")]

    return out.sort_values(["trd_exctn_dt", "cusip_id"]).reset_index(drop=True)


def to_bond_day(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse cleaned TRACE trades to one robust observation per bond-day.

    Median yield/price prevents frequently traded bonds from receiving extra
    cross-sectional weight simply because they printed more transactions.
    """
    grouped = frame.groupby(["trd_exctn_dt", "cusip_id"], observed=True)

    out = grouped.agg(
        yield_pct=("yld_pt", "median"),
        price=("rptd_pr", "median"),
        n_trades=("yld_pt", "size"),
    )

    if "ascii_rptd_vol_tx" in frame:
        volume = grouped["ascii_rptd_vol_tx"].sum(min_count=1).rename("volume")
        out = out.join(volume)

    return out.reset_index()
