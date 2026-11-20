"""Build the TRACE high-yield credit panel from cached WRDS extracts, logging attrition.

Stages (every stage records its row count in an ``Attrition`` log):

raw Enhanced trades -> lifecycle-cleaned -> documented condition filters ->
disseminated only -> sanity filters -> bond-days -> TRACE master join -> bond universe
-> remaining maturity -> Treasury spread -> issuer link (GVKEY) -> point-in-time S&P
rating -> issuer-day medians -> daily HY / IG aggregates.

Conventions fixed before seeing results (see ``research/hy_credit.md``):

- only disseminated trades (``dissem_fl == 'Y'``), so every input was public when it
  printed; sample starts 2006-01-09, when FINRA began disseminating all public
  TRACE-eligible transactions immediately;
- bond universe from the FINRA master-file codes: ``sub_prdct_type`` CORP, not
  convertible, coupon type FXPV (plain vanilla fixed), not 144A;
- Gilchrist & Zakrajsek (2012) screens: remaining maturity in [1, 30] years and spread
  in [5, 3500] bp;
- issuer link: a GVKEY only when the exact-identifier methods agree
  (``issuer_link.consensus_link``); links use information dated before the bond-day;
- ratings: latest ``comp.adsprate`` record with ``datadate`` <= bond-day, expiring after
  ``RATING_TOLERANCE``;
- aggregation: issuer-day median, then cross-issuer median; a day's HY or IG value needs
  at least ``MIN_ISSUERS`` issuers.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from macro.data.issuer_link import consensus_link
from macro.data.ratings import asof_ratings
from macro.data.trace import (
    add_remaining_maturity,
    basic_clean_trace,
    clean_trace_lifecycle,
    filter_enhanced_conditions,
    join_master,
    signed_yield,
    to_bond_day,
)
from macro.data.treasury import treasury_spread_bp
from macro.features.hy_credit import aggregate_hy_ig, issuer_day_spread

SAMPLE_START = pd.Timestamp("2006-01-09")
SUB_PRODUCT = "CORP"
FIXED_COUPON_CODES = ("FXPV",)
MATURITY_RANGE_Y = (1.0, 30.0)
SPREAD_RANGE_BP = (5.0, 3500.0)
RATING_TOLERANCE = pd.Timedelta(days=93)
MIN_ISSUERS = 10
CCM_LINKTYPES = ("LU", "LC")
CCM_LINKPRIM = ("P", "C")
# Master fields the pipeline uses; overlapping versions that agree on these are usable.
MASTER_FIELDS = ["sub_prdct_type", "cnvrb_fl", "cpn_type_cd", "ind_144a", "mtrty_dt"]


@dataclass
class Attrition:
    """Ordered row counts per pipeline stage."""

    rows: list[dict] = field(default_factory=list)

    def add(self, stage: str, n: int, **detail) -> None:
        self.rows.append({"stage": stage, "rows": int(n), **detail})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _codes(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip().str.upper()
    return out.mask(out.eq(""))


# --------------------------------------------------------------------------------------
# Trades -> bond-days
# --------------------------------------------------------------------------------------


def clean_trades(raw: pd.DataFrame, log: Attrition, require_disseminated: bool = True):
    """Lifecycle, documented condition filters, dissemination and sanity filters."""
    log.add("raw_records", len(raw))
    trades, lifecycle = clean_trace_lifecycle(raw)
    log.add("after_lifecycle", len(trades), **lifecycle)
    trades, removed = filter_enhanced_conditions(trades)
    log.add(
        "after_condition_filters", len(trades), **{f"removed_{k}": v for k, v in removed.items()}
    )
    if require_disseminated:
        if "dissem_fl" not in trades:
            raise ValueError("dissem_fl is required to keep only disseminated trades")
        disseminated = _codes(trades["dissem_fl"]).eq("Y").fillna(False)
        trades = trades.loc[disseminated.to_numpy()]
        log.add("after_disseminated_only", len(trades))
    trades = basic_clean_trace(trades)
    log.add("after_sanity_filters", len(trades))
    trades = trades.assign(yield_pct=signed_yield(trades))
    return trades


def trades_to_bond_days(trades: pd.DataFrame, log: Attrition) -> pd.DataFrame:
    """One row per CUSIP and execution date (median signed yield and price)."""
    bond_days = to_bond_day(trades, yield_col="yield_pct")
    log.add("bond_days", len(bond_days), n_bonds=int(bond_days["cusip_id"].nunique()))
    return bond_days


def bond_universe_mask(frame: pd.DataFrame) -> tuple[pd.Series, dict[str, int]]:
    """Documented FINRA master codes: CORP, not convertible, plain fixed coupon, not 144A."""
    keep = pd.Series(True, index=frame.index)
    excluded: dict[str, int] = {}
    rules = {
        "not_corp": ("sub_prdct_type", lambda s: _codes(s).ne(SUB_PRODUCT)),
        "convertible": ("cnvrb_fl", lambda s: _codes(s).eq("Y")),
        "not_plain_fixed": ("cpn_type_cd", lambda s: ~_codes(s).isin(FIXED_COUPON_CODES)),
        "rule_144a": ("ind_144a", lambda s: _codes(s).eq("Y")),
    }
    for name, (column, rule) in rules.items():
        if column not in frame:
            continue
        hit = rule(frame[column]).fillna(name == "not_corp" or name == "not_plain_fixed")
        hit = hit.astype(bool) & keep
        excluded[name] = int(hit.sum())
        keep &= ~hit
    return keep, excluded


def add_master_and_spreads(
    bond_days: pd.DataFrame,
    master: pd.DataFrame,
    curve: pd.DataFrame,
    log: Attrition,
    validity: tuple[str, str] | None = ("stdt", "enddt"),
) -> pd.DataFrame:
    """Master join, bond universe, maturity screens and maturity-matched Treasury spread."""
    joined, counts = join_master(bond_days, master, validity=validity, agree_on=MASTER_FIELDS)
    log.add("master_join", len(joined), **{f"master_{k}": int(v) for k, v in counts.items()})
    joined = joined.loc[joined["master_match"].isin(["unique", "consistent"])]
    log.add("after_master_unique", len(joined))

    keep, excluded = bond_universe_mask(joined)
    joined = joined.loc[keep]
    log.add("after_bond_universe", len(joined), **{f"excluded_{k}": v for k, v in excluded.items()})

    joined = add_remaining_maturity(joined)
    remaining = joined["remaining_maturity_y"]
    unknown, matured = remaining.isna(), remaining.le(0).fillna(False)
    in_range = remaining.between(*MATURITY_RANGE_Y).fillna(False)
    joined = joined.loc[in_range]
    log.add(
        "after_maturity_screen",
        len(joined),
        unknown_maturity=int(unknown.sum()),
        matured_or_nonpositive=int(matured.sum()),
        outside_1_30y=int((~in_range & ~unknown & ~matured).sum()),
    )

    joined = treasury_spread_bp(joined, curve)
    no_treasury = joined["treasury_yield_pct"].isna()
    no_yield = joined["yield_pct"].isna()
    in_bounds = joined["spread_bp"].between(*SPREAD_RANGE_BP).fillna(False)
    out = joined.loc[in_bounds]
    log.add(
        "after_spread_screen",
        len(out),
        no_yield=int(no_yield.sum()),
        no_treasury=int((no_treasury & ~no_yield).sum()),
        outside_5_3500bp=int((~in_bounds & ~no_yield & ~no_treasury).sum()),
    )
    return out


# --------------------------------------------------------------------------------------
# Issuer links
# --------------------------------------------------------------------------------------


def _previous_month_end(dates: pd.Series) -> pd.Series:
    return (pd.to_datetime(dates) - pd.offsets.MonthEnd(1)).dt.normalize()


def link_matches(
    bond_days: pd.DataFrame,
    *,
    fang: pd.DataFrame | None = None,
    bondcrsp: pd.DataFrame | None = None,
    ccm: pd.DataFrame | None = None,
    security: pd.DataFrame | None = None,
    crsp_names: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Exact-identifier GVKEY candidates per (cusip_id, trd_exctn_dt), by method.

    - ``comp_cusip6``: the bond's issuer CUSIP6 equals the first six characters of a
      ``comp.security`` CUSIP. CUSIP6 identifies the issuing entity, so a match means the
      same legal issuer; finance subsidiaries with their own CUSIP6 stay unmatched. A
      CUSIP6 shared by several GVKEYs yields several candidates and is left unlinked.
    - ``crsp_ncusip6``: bond CUSIP -> PERMNO (``bondcrsp_link``, link dates inclusive) ->
      historical CRSP CUSIP (``crsp_names``: ``permno``, ``ncusip``, ``namedt``,
      ``nameenddt``, valid on the bond-day) -> its CUSIP6 -> ``comp.security`` CUSIP6.

    - ``fang``: issuer CUSIP6 -> GVKEY from ``contrib_bond_firm_link.fang_link``, using the
      month-end (``ldom``) strictly before the bond-day, so no later-dated link is used.
    - ``crsp_ccm``: bond CUSIP -> PERMNO (``wrdsapps_link_crsp_bond.bondcrsp_link``, link
      dates inclusive) -> GVKEY (``crsp.ccmxpf_lnkhist``, link types LU/LC, primary P/C,
      link dates inclusive, missing end = open).
    Returns columns ``cusip_id``, ``trd_exctn_dt``, ``gvkey``, ``method``.
    """
    keys = bond_days[["cusip_id", "trd_exctn_dt"]].drop_duplicates()
    keys = keys.assign(trd_exctn_dt=pd.to_datetime(keys["trd_exctn_dt"]))
    parts = []
    if fang is not None and len(fang):
        f = fang.assign(
            issuer_cusip=fang["issuer_cusip"].astype("string").str.upper(),
            ldom=pd.to_datetime(fang["ldom"]),
            gvkey=fang["gvkey"].astype("string"),
        )
        k = keys.assign(
            issuer_cusip=keys["cusip_id"].astype("string").str.upper().str.slice(0, 6),
            ldom=_previous_month_end(keys["trd_exctn_dt"]),
        )
        m = k.merge(f[["issuer_cusip", "ldom", "gvkey"]].dropna(), on=["issuer_cusip", "ldom"])
        parts.append(m[["cusip_id", "trd_exctn_dt", "gvkey"]].assign(method="fang"))
    if bondcrsp is not None and ccm is not None and len(bondcrsp) and len(ccm):
        b = bondcrsp.assign(cusip_id=bondcrsp["cusip"].astype("string").str.upper())
        m = keys.merge(b[["cusip_id", "permno", "link_startdt", "link_enddt"]], on="cusip_id")
        d = m["trd_exctn_dt"]
        ok = (
            pd.to_datetime(m["link_startdt"]).isna() | (pd.to_datetime(m["link_startdt"]) <= d)
        ) & (pd.to_datetime(m["link_enddt"]).isna() | (d <= pd.to_datetime(m["link_enddt"])))
        m = m.loc[ok]
        c = ccm.loc[ccm["linktype"].isin(CCM_LINKTYPES) & ccm["linkprim"].isin(CCM_LINKPRIM)]
        c = c.assign(gvkey=c["gvkey"].astype("string"), permno=c["lpermno"])
        m = m.merge(c[["permno", "gvkey", "linkdt", "linkenddt"]], on="permno")
        d = m["trd_exctn_dt"]
        ok = (pd.to_datetime(m["linkdt"]).isna() | (pd.to_datetime(m["linkdt"]) <= d)) & (
            pd.to_datetime(m["linkenddt"]).isna() | (d <= pd.to_datetime(m["linkenddt"]))
        )
        parts.append(m.loc[ok, ["cusip_id", "trd_exctn_dt", "gvkey"]].assign(method="crsp_ccm"))
    if security is not None and len(security):
        sec = security.loc[security["cusip"].notna() & security["gvkey"].notna()]
        sec = sec.assign(
            issuer_cusip=sec["cusip"].astype("string").str.upper().str.slice(0, 6),
            gvkey=sec["gvkey"].astype("string"),
        )[["issuer_cusip", "gvkey"]].drop_duplicates()
        k = keys.assign(issuer_cusip=keys["cusip_id"].astype("string").str.upper().str.slice(0, 6))
        m = k.merge(sec, on="issuer_cusip")
        parts.append(m[["cusip_id", "trd_exctn_dt", "gvkey"]].assign(method="comp_cusip6"))
    if (
        bondcrsp is not None and crsp_names is not None and security is not None
        and len(bondcrsp) and len(crsp_names) and len(security)
    ):  # fmt: skip
        b = bondcrsp.assign(cusip_id=bondcrsp["cusip"].astype("string").str.upper())
        m = keys.merge(b[["cusip_id", "permno", "link_startdt", "link_enddt"]], on="cusip_id")
        d = m["trd_exctn_dt"]
        ok = (
            pd.to_datetime(m["link_startdt"]).isna() | (pd.to_datetime(m["link_startdt"]) <= d)
        ) & (pd.to_datetime(m["link_enddt"]).isna() | (d <= pd.to_datetime(m["link_enddt"])))
        m = m.loc[ok]
        n = crsp_names.assign(
            permno=pd.to_numeric(crsp_names["permno"]),
            ncusip6=crsp_names["ncusip"].astype("string").str.upper().str.slice(0, 6),
        )
        m = m.assign(permno=pd.to_numeric(m["permno"])).merge(
            n[["permno", "ncusip6", "namedt", "nameenddt"]], on="permno"
        )
        d = m["trd_exctn_dt"]
        ok = (pd.to_datetime(m["namedt"]).isna() | (pd.to_datetime(m["namedt"]) <= d)) & (
            pd.to_datetime(m["nameenddt"]).isna() | (d <= pd.to_datetime(m["nameenddt"]))
        )
        m = m.loc[ok]
        sec = security.loc[security["cusip"].notna() & security["gvkey"].notna()]
        sec = sec.assign(
            ncusip6=sec["cusip"].astype("string").str.upper().str.slice(0, 6),
            gvkey=sec["gvkey"].astype("string"),
        )[["ncusip6", "gvkey"]].drop_duplicates()
        m = m.merge(sec, on="ncusip6")
        parts.append(m[["cusip_id", "trd_exctn_dt", "gvkey"]].assign(method="crsp_ncusip6"))
    if not parts:
        empty = pd.DataFrame({"cusip_id": pd.Series(dtype="string"),
                              "trd_exctn_dt": pd.Series(dtype="datetime64[ns]"),
                              "gvkey": pd.Series(dtype="string"),
                              "method": pd.Series(dtype="string")})  # fmt: skip
        return empty
    out = pd.concat(parts, ignore_index=True).drop_duplicates()
    out["trd_exctn_dt"] = pd.to_datetime(out["trd_exctn_dt"])
    return out


def attach_issuers(bond_days: pd.DataFrame, matches: pd.DataFrame, log: Attrition) -> pd.DataFrame:
    """Attach a consensus GVKEY per bond-day; conflicts and unmatched stay unlinked."""
    frame = bond_days.assign(trd_exctn_dt=pd.to_datetime(bond_days["trd_exctn_dt"]))
    frame["_bd"] = (
        frame["cusip_id"].astype(str) + "|" + frame["trd_exctn_dt"].dt.strftime("%Y-%m-%d")
    )
    m = matches.assign(
        _bd=matches["cusip_id"].astype(str)
        + "|"
        + pd.to_datetime(matches["trd_exctn_dt"]).dt.strftime("%Y-%m-%d")
    )
    links = consensus_link(m, frame["_bd"], left_id="_bd", right_id="gvkey")
    frame = frame.merge(links[["_bd", "gvkey", "status", "methods"]], on="_bd", how="left")
    frame = frame.rename(columns={"status": "link_status", "methods": "link_methods"})
    counts = frame["link_status"].value_counts().to_dict()
    log.add("issuer_link", len(frame), **{f"link_{k}": int(v) for k, v in counts.items()})
    linked = frame.loc[frame["link_status"].eq("linked")].drop(columns="_bd")
    log.add("after_issuer_link", len(linked), n_issuers=int(linked["gvkey"].nunique()))
    return linked


# --------------------------------------------------------------------------------------
# Ratings and aggregation
# --------------------------------------------------------------------------------------


def attach_ratings(linked: pd.DataFrame, ratings: pd.DataFrame, log: Attrition) -> pd.DataFrame:
    """Point-in-time S&P issuer rating per bond-day (latest datadate <= day, expiring)."""
    r = ratings.assign(gvkey=ratings["gvkey"].astype("string"))
    out = asof_ratings(
        linked.assign(gvkey=linked["gvkey"].astype("string")),
        r,
        issuer_key="gvkey",
        date_col="trd_exctn_dt",
        tolerance=RATING_TOLERANCE,
    )
    age = (pd.to_datetime(out["trd_exctn_dt"]) - out["rating_date"]).dt.days
    out["rating_age_days"] = age
    counts = out["rating_class"].value_counts(dropna=False)
    log.add(
        "ratings",
        len(out),
        hy=int(counts.get("HY", 0)),
        ig=int(counts.get("IG", 0)),
        unrated=int(out["rating_class"].isna().sum()),
        median_rating_age_days=float(age.median()) if age.notna().any() else np.nan,
    )
    return out


def daily_aggregates(rated: pd.DataFrame, log: Attrition, min_issuers: int = MIN_ISSUERS):
    """Issuer-day medians, then daily HY/IG cross-issuer medians with a minimum count."""
    issuer = issuer_day_spread(rated, issuer_col="gvkey", date_col="trd_exctn_dt")
    log.add("issuer_days", len(issuer), n_issuers=int(issuer["gvkey"].nunique()))
    daily = aggregate_hy_ig(issuer, date_col="trd_exctn_dt")
    daily.loc[daily["n_hy_issuers"] < min_issuers, "hy_spread_bp"] = np.nan
    daily.loc[daily["n_ig_issuers"] < min_issuers, "ig_spread_bp"] = np.nan
    daily["hy_ig_gap_bp"] = daily["hy_spread_bp"] - daily["ig_spread_bp"]
    unrated = rated.loc[rated["rating_class"].isna()].groupby("trd_exctn_dt")["gvkey"].nunique()
    daily["n_unrated_issuers"] = unrated.reindex(daily.index).fillna(0).astype("int64")
    bonds = rated.groupby("trd_exctn_dt")["cusip_id"].nunique()
    daily["n_bonds"] = bonds.reindex(daily.index).fillna(0).astype("int64")
    log.add("daily", len(daily), days_with_hy=int(daily["hy_spread_bp"].notna().sum()))
    return issuer, daily
