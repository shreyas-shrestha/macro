"""Point-in-time industry classification for issuers (#24 residual extension).

Source of record: Compustat historical GICS (``comp.co_hgic``), one row per GVKEY and
classification spell with effective dates ``indfrom`` .. ``indthru`` (inclusive; a missing
``indthru`` means still in force). An observation on date t gets the spell with
``indfrom`` <= t <= ``indthru``. A later reclassification never reaches back to earlier
dates, and a date before an issuer's first spell stays unclassified (no backfill).
"""

import pandas as pd

GICS_SECTORS = {
    "10": "Energy",
    "15": "Materials",
    "20": "Industrials",
    "25": "Consumer Discretionary",
    "30": "Consumer Staples",
    "35": "Health Care",
    "40": "Financials",
    "45": "Information Technology",
    "50": "Communication Services",  # "Telecommunication Services" before 2018-09
    "55": "Utilities",
    "60": "Real Estate",  # split from Financials 2016-09
}


def asof_interval(
    observations: pd.DataFrame,
    spells: pd.DataFrame,
    *,
    key: str,
    date_col: str,
    start_col: str,
    end_col: str,
    value_cols: list[str],
) -> pd.DataFrame:
    """Attach the spell in force on each observation date; missing if none.

    Spells of one key must not overlap (raises). Rows and index of ``observations`` are
    preserved. Adds ``value_cols`` and ``spell_start``.
    """
    for column in [*value_cols, "spell_start"]:
        if column in observations:
            raise ValueError(f"observations already has a {column!r} column")
    right = spells[[key, start_col, end_col, *value_cols]].rename(
        columns={start_col: "spell_start", end_col: "_spell_end"}
    )
    right["spell_start"] = pd.to_datetime(right["spell_start"]).astype("datetime64[ns]")
    right["_spell_end"] = pd.to_datetime(right["_spell_end"]).astype("datetime64[ns]")
    right[key] = right[key].astype("string")
    if right["spell_start"].isna().any():
        raise ValueError(f"spells have a missing {start_col}")
    right = right.sort_values([key, "spell_start"], kind="stable")
    next_start = right.groupby(key)["spell_start"].shift(-1)
    overlap = next_start.notna() & (
        right["_spell_end"].isna() | (right["_spell_end"] >= next_start)
    )
    if overlap.any():
        raise ValueError(f"{int(overlap.sum())} spells overlap the next spell of the same {key}")
    right = right.sort_values("spell_start", kind="stable")

    left = observations.reset_index(drop=True).assign(_row=lambda f: range(len(f)))
    left["_date"] = pd.to_datetime(left[date_col]).astype("datetime64[ns]")
    left["_key"] = left[key].astype("string")
    keyed = left.loc[left["_key"].notna(), ["_row", "_date", "_key"]].sort_values("_date")
    merged = pd.merge_asof(
        keyed,
        right.rename(columns={key: "_key"}),
        left_on="_date",
        right_on="spell_start",
        by="_key",
        direction="backward",
    )
    expired = merged["_spell_end"].notna() & (merged["_date"] > merged["_spell_end"])
    merged.loc[expired, [*value_cols, "spell_start"]] = pd.NA
    attached = merged.set_index("_row")[[*value_cols, "spell_start"]]
    out = left.join(attached, on="_row").drop(columns=["_row", "_date", "_key"])
    out.index = observations.index
    return out


def gics_sector_asof(
    observations: pd.DataFrame, hgic: pd.DataFrame, *, date_col: str = "trd_exctn_dt"
) -> pd.Series:
    """Two-digit GICS sector (string code) in force for each (gvkey, date) row."""
    spells = hgic.assign(gsector=hgic["gsector"].astype("string").str.strip().str.slice(0, 2))
    out = asof_interval(
        observations[["gvkey", date_col]],
        spells,
        key="gvkey",
        date_col=date_col,
        start_col="indfrom",
        end_col="indthru",
        value_cols=["gsector"],
    )
    return out["gsector"].astype("string")
