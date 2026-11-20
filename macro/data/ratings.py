"""S&P long-term issuer ratings from Legacy Compustat: normalization, IG/HY, as-of join.

Point-in-time rule: an observation on date t uses the latest rating record whose
``datadate`` <= t for the same issuer. A later record never fills an earlier date. If
the latest record has no rating, the observation is unrated; the join does not skip
back to an older rating. Unrated observations are missing, never HY.
"""

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
    """Uppercase and strip S&P letter ratings; anything outside ``RATING_ORDER`` is missing.

    Use ``unrecognized_ratings`` to see which raw values were set to missing.
    """
    out = series.astype("string").str.upper().str.strip()
    return out.where(out.isin(RATING_ORDER))


def unrecognized_ratings(series: pd.Series) -> pd.Series:
    """Counts of non-blank raw values that ``normalize_sp_rating`` sets to missing."""
    raw = series.astype("string").str.upper().str.strip()
    raw = raw[raw.notna() & raw.ne("")]
    return raw[~raw.isin(RATING_ORDER)].value_counts()


def rating_class(rating: pd.Series) -> pd.Series:
    """'IG' for BBB- and better, 'HY' for BB+ and worse, missing when unrated."""
    order = normalize_sp_rating(rating).map(RATING_ORDER)
    return order.le(IG_MAX).map({True: "IG", False: "HY"}).astype("string")


def asof_ratings(
    observations: pd.DataFrame,
    ratings: pd.DataFrame,
    *,
    issuer_key: str,
    date_col: str,
    rating_date_col: str = "datadate",
    rating_col: str = "splticrm",
    tolerance: pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Attach the latest rating with ``rating_date_col`` <= the observation date.

    ``issuer_key`` is any column shared by both frames (e.g. a GVKEY once a crosswalk
    exists). ``tolerance``, if set, treats a rating record older than that as missing,
    so an issuer that leaves the ratings file is not carried forward forever.

    Adds ``rating_date``, ``rating`` (normalized) and ``rating_class``. Rows and index
    order of ``observations`` are preserved. Raises on duplicate (issuer, date) rating
    records, since choosing between them would be arbitrary.
    """
    for column in ("rating_date", "rating", "rating_class"):
        if column in observations:
            raise ValueError(f"observations already has a {column!r} column")
    # One datetime resolution on both sides: Parquet date32 reads back as datetime64[s],
    # other sources as [us]/[ns], and merge_asof rejects mismatched resolutions.
    obs_dates = pd.to_datetime(observations[date_col]).astype("datetime64[ns]")
    if obs_dates.isna().any():
        raise ValueError(f"{date_col} has missing values")

    right = ratings[[issuer_key, rating_date_col, rating_col]].rename(
        columns={rating_date_col: "rating_date", rating_col: "_rating_raw"}
    )
    right["rating_date"] = pd.to_datetime(right["rating_date"]).astype("datetime64[ns]")
    if right["rating_date"].isna().any():
        raise ValueError(f"ratings {rating_date_col} has missing values")
    duplicated = right.duplicated([issuer_key, "rating_date"], keep=False)
    if duplicated.any():
        raise ValueError(
            f"{int(duplicated.sum())} ratings rows share an ({issuer_key}, {rating_date_col}) "
            "pair; resolve duplicates before joining"
        )
    right = right.loc[right[issuer_key].notna()].sort_values("rating_date", kind="stable")

    left = observations.reset_index(drop=True).assign(_row=lambda f: range(len(f)))
    left["_date"] = obs_dates.to_numpy()
    keyed = left.loc[left[issuer_key].notna()].sort_values("_date", kind="stable")
    merged = pd.merge_asof(
        keyed,
        right,
        left_on="_date",
        right_on="rating_date",
        by=issuer_key,
        direction="backward",
        allow_exact_matches=True,
        tolerance=tolerance,
    )
    attached = merged.set_index("_row")[["rating_date", "_rating_raw"]]
    out = left.join(attached, on="_row")
    out["rating"] = normalize_sp_rating(out.pop("_rating_raw"))
    out["rating_class"] = rating_class(out["rating"])
    out = out.drop(columns=["_row", "_date"])
    out.index = observations.index
    return out
