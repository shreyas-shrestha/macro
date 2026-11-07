"""Pure transformations for the TRACE high-yield credit signal."""

import numpy as np
import pandas as pd

RATING_ORDER = {
    "AAA": 1,
    "AA+": 2,
    "AA": 3,
    "AA-": 4,
    "A+": 5,
    "A": 6,
    "A-": 7,
    "BBB+": 8,
    "BBB": 9,
    "BBB-": 10,
    "BB+": 11,
    "BB": 12,
    "BB-": 13,
    "B+": 14,
    "B": 15,
    "B-": 16,
    "CCC+": 17,
    "CCC": 18,
    "CCC-": 19,
    "CC": 20,
    "C": 21,
    "D": 22,
}

IG_MAX = RATING_ORDER["BBB-"]


def normalize_sp_rating(series: pd.Series) -> pd.Series:
    """Normalize S&P letter ratings and drop non-rating values."""
    out = series.astype("string").str.upper().str.strip()
    return out.where(out.isin(RATING_ORDER))


def add_rating_class(frame: pd.DataFrame, rating_col: str = "splticrm") -> pd.DataFrame:
    """Add ordinal rating and IG/HY labels using the BBB-/BB+ boundary."""
    out = frame.copy()
    out["rating"] = normalize_sp_rating(out[rating_col])
    out["rating_ord"] = out["rating"].map(RATING_ORDER)
    # The comparison yields a nullable boolean mask (NA for unrated rows); map keeps
    # NA as missing, whereas np.where would call bool(pd.NA) and raise.
    out["rating_class"] = (
        out["rating_ord"].le(IG_MAX).map({True: "IG", False: "HY"}).astype("string")
    )
    return out


def add_remaining_maturity(
    frame: pd.DataFrame,
    date_col: str = "trd_exctn_dt",
    maturity_col: str = "mtrty_dt",
) -> pd.DataFrame:
    """Compute remaining maturity in years from execution and maturity dates."""
    out = frame.copy()
    out["remaining_maturity_y"] = (
        pd.to_datetime(out[maturity_col]) - pd.to_datetime(out[date_col])
    ).dt.days / 365.25
    return out


def issuer_day_spread(
    frame: pd.DataFrame,
    issuer_col: str = "issuer_id",
    spread_col: str = "spread_bp",
) -> pd.DataFrame:
    """Collapse bond-level spreads so each issuer counts once per day."""
    required = {"trd_exctn_dt", issuer_col, "rating_class", spread_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return (
        frame.groupby(["trd_exctn_dt", issuer_col, "rating_class"], observed=True)[spread_col]
        .median()
        .rename("issuer_spread_bp")
        .reset_index()
    )


def aggregate_hy_ig(issuer_day: pd.DataFrame) -> pd.DataFrame:
    """Build equal-issuer-weighted daily HY and IG spread series."""
    wide = (
        issuer_day.groupby(["trd_exctn_dt", "rating_class"], observed=True)["issuer_spread_bp"]
        .median()
        .unstack("rating_class")
        .rename(columns={"HY": "hy_spread_bp", "IG": "ig_spread_bp"})
    )

    for column in ("hy_spread_bp", "ig_spread_bp"):
        if column not in wide:
            wide[column] = np.nan

    wide["hy_ig_gap_bp"] = wide["hy_spread_bp"] - wide["ig_spread_bp"]
    return wide.sort_index()


def expanding_zscore(series: pd.Series, min_periods: int = 60) -> pd.Series:
    """Point-in-time z-score using only observations available through t."""
    mean = series.expanding(min_periods=min_periods).mean()
    std = series.expanding(min_periods=min_periods).std()
    return (series - mean) / std.replace(0, np.nan)


def build_simple_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Create the initial preregistered TRACE credit feature set."""
    out = pd.DataFrame(index=daily.index)
    out["hy_level"] = daily["hy_spread_bp"]
    out["hy_change_20d"] = daily["hy_spread_bp"].diff(20)
    out["hy_z"] = expanding_zscore(daily["hy_spread_bp"])
    out["hy_ig_gap"] = daily["hy_ig_gap_bp"]
    out["hy_ig_gap_change_20d"] = daily["hy_ig_gap_bp"].diff(20)
    return out
