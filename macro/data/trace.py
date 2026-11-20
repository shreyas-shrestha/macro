"""Pure TRACE transformations: identifiers, trade lifecycle, filters, master join, bond-days.

Code semantics come from FINRA's "Historic TRACE Data: Enhanced Historic Time and Sales
Trade Record File Layout":

- pre-migration layout (FINRA p120989, Nov 2010): reports before 2012-02-06;
- post-migration layout (FINRA p353157): reports from 2012-02-06, when TRACE moved to a
  new platform and the file format changed.

Only rules those documents state are implemented. Anything that depends on how WRDS
names or encodes the fields, or on layouts the documents do not cover (the standard
disseminated BTDS file with ``chng_cd``/``frmt_cd`` and capped ``ascii_rptd_vol_tx``), is
isolated in a clearly named function and listed in ``research/hy_credit.md`` as something
to verify on real rows; the disseminated-BTDS lifecycle raises ``NotImplementedError``.

Order of operations on raw Enhanced trades:
``clean_trace_lifecycle`` -> ``filter_enhanced_conditions`` -> ``basic_clean_trace`` ->
``join_master`` -> ``add_remaining_maturity`` -> ``require_live_bonds`` -> ``to_bond_day``.
Lifecycle cleaning must run first: cancellation records may lack yields or prices and
would otherwise be dropped before they can remove the trades they cancel.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

PLATFORM_MIGRATION = pd.Timestamp("2012-02-06")

# --------------------------------------------------------------------------------------
# CUSIPs
# --------------------------------------------------------------------------------------

_CUSIP_CHARS = r"^[0-9A-Z*@#]{9}$"
_CUSIP_SPECIAL = {"*": 36, "@": 37, "#": 38}


def normalize_cusip(series: pd.Series) -> pd.Series:
    """Return CUSIPs as stripped, uppercase strings; blanks become missing.

    Never pads, truncates or otherwise alters an identifier. Numeric input is rejected
    because reading CUSIPs as numbers has already dropped leading zeros.
    """
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        raise TypeError(
            f"CUSIP column {series.name!r} has numeric dtype {series.dtype}; read identifiers "
            "as strings (leading zeros may already be lost)"
        )
    if series.dtype == object:
        bad = series[~series.map(lambda v: isinstance(v, str) or pd.isna(v))]
        if len(bad):
            raise TypeError(
                f"CUSIP column {series.name!r} contains non-string values, e.g. "
                f"{bad.head(3).tolist()}; read identifiers as strings"
            )
    out = series.astype("string").str.strip().str.upper()
    return out.mask(out.eq(""))


def cusip_check_digit(cusip8: str) -> int:
    """Return the CUSIP check digit for the first eight characters (modulus-10 double-add)."""
    total = 0
    for i, ch in enumerate(cusip8[:8]):
        if ch.isdigit():
            value = int(ch)
        elif "A" <= ch <= "Z":
            value = ord(ch) - ord("A") + 10
        else:
            value = _CUSIP_SPECIAL[ch]
        if i % 2 == 1:
            value *= 2
        total += value // 10 + value % 10
    return (10 - total % 10) % 10


def cusip_status(series: pd.Series) -> pd.Series:
    """Classify each CUSIP as ok, missing, bad_length, bad_chars or bad_check_digit.

    Diagnostic only: nothing is dropped or repaired.
    """
    cusips = normalize_cusip(series)
    status = pd.Series("ok", index=series.index, dtype="string")
    missing = cusips.isna()
    length_ok = cusips.str.len().eq(9).fillna(False)
    chars_ok = cusips.str.match(_CUSIP_CHARS).fillna(False)
    status[~length_ok] = "bad_length"
    status[length_ok & ~chars_ok] = "bad_chars"
    candidates = cusips[length_ok & chars_ok]
    check_ok = candidates.map(lambda c: c[8].isdigit() and cusip_check_digit(c) == int(c[8]))
    status[check_ok.index[~check_ok.astype(bool)]] = "bad_check_digit"
    status[missing] = "missing"
    return status


def issuer_cusip6(series: pd.Series) -> pd.Series:
    """First six characters (issuer number) of well-formed 9-character CUSIPs.

    Diagnostic only. CUSIP6 identifies the issuing entity, not the rated parent, so it
    is not on its own a sufficient key for the TRACE -> Compustat issuer join.
    """
    cusips = normalize_cusip(series)
    well_formed = cusips.str.match(_CUSIP_CHARS).fillna(False)
    return cusips.str.slice(0, 6).where(well_formed)


# --------------------------------------------------------------------------------------
# Field decoding
# --------------------------------------------------------------------------------------


def _codes(series: pd.Series) -> pd.Series:
    out = series.astype("string").str.strip().str.upper()
    return out.mask(out.eq(""))


def signed_yield(
    frame: pd.DataFrame, yield_col: str = "yld_pt", sign_col: str = "yld_sign_cd"
) -> pd.Series:
    """Apply the Yield Direction code to the Yield field.

    Both FINRA layouts: '-' marks a negative yield. Pre-2012 '+' and post-2012 blank mark
    a positive or zero yield. Works whether or not ``yield_col`` is already signed, and
    raises on undocumented codes or on a negative yield whose code is not '-'.
    """
    yld = pd.to_numeric(frame[yield_col])
    if sign_col not in frame:
        return yld.astype("float64")
    sign = _codes(frame[sign_col]).fillna("+")
    unknown = sorted(set(sign.unique()) - {"-", "+"})
    if unknown:
        raise ValueError(f"Undocumented {sign_col} values: {unknown}")
    contradiction = yld.lt(0).fillna(False) & sign.ne("-")
    if contradiction.any():
        raise ValueError(
            f"{int(contradiction.sum())} negative {yield_col} values whose {sign_col} is not '-'"
        )
    magnitude = yld.abs().astype("float64")
    return magnitude.where(sign.ne("-"), -magnitude)


def parse_reported_volume(series: pd.Series) -> pd.Series:
    """Parse a reported-quantity text field; non-numeric entries become missing.

    The disseminated BTDS file caps large sizes with text markers. Their encoding is
    not in the FINRA layouts reviewed, so they are left missing rather than mapped to
    a number. Count the missing values before relying on volume.
    """
    text = series.astype("string").str.strip()
    numeric = text.str.fullmatch(r"\d+(\.\d+)?").fillna(False)
    return pd.to_numeric(text.where(numeric), errors="raise").astype("float64")


# --------------------------------------------------------------------------------------
# Trade lifecycle (Enhanced historic layout)
# --------------------------------------------------------------------------------------

PRE2012_STATUSES = {"T", "C", "W"}

# Basic trade details FINRA says to match a reversal on (both layouts, note on ASOF_CD).
REVERSAL_MATCH_COLS = (
    "cusip_id",
    "entrd_vol_qt",
    "rptd_pr",
    "trd_exctn_dt",
    "trd_exctn_tm",
    "rpt_side_cd",
    "cntra_mp_id",
)


def _require(frame: pd.DataFrame, columns, purpose: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{purpose} needs columns {missing}")


def apply_pre2012_cancels_corrections(
    frame: pd.DataFrame,
    *,
    status_col: str = "trc_st",
    report_date_col: str = "trd_rpt_dt",
    seq_col: str = "msg_seq_nb",
    orig_seq_col: str = "orig_msg_seq_nb",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply same-day cancellations (C) and corrections (W) in the pre-2012 layout.

    Documented (FINRA p120989): TRC_ST is T = Trade Report, C = Trade Cancellation,
    W = Trade Correction. On C and W records ORIG_MSG_SEQ_NB holds the MSG_SEQ_NB of the
    underlying trade record, and MSG_SEQ_NB is unique within each Trade Report Date.

    Applied here: C records are removed together with the record they reference; a
    record referenced by a W is removed and the W record is kept as the trade's current
    version. Chains (a C or W referencing an earlier W) resolve because every referenced
    record is removed.

    To confirm on real rows: (1) the link uses the C/W record's own report date, since
    the layout has no prior-report-date field and prior-day cancellations are reversals;
    the unmatched-reference count should be near zero. (2) W rows carry the full,
    corrected trade details.
    """
    _require(frame, [status_col, report_date_col, seq_col, orig_seq_col], "Pre-2012 lifecycle")
    status = _codes(frame[status_col])
    unknown = sorted(set(status.dropna().unique()) - PRE2012_STATUSES)
    if unknown or status.isna().any():
        raise ValueError(f"Statuses outside the pre-2012 layout (T/C/W): {unknown or ['<NA>']}")
    report_date = pd.to_datetime(frame[report_date_col])
    if report_date.isna().any() or report_date.ge(PLATFORM_MIGRATION).any():
        raise ValueError(f"Pre-2012 rules apply only to reports before {PLATFORM_MIGRATION.date()}")

    seq = pd.to_numeric(frame[seq_col]).astype("Int64")
    orig = pd.to_numeric(frame[orig_seq_col]).astype("Int64")
    key = pd.MultiIndex.from_arrays([report_date, seq])
    if key.has_duplicates:
        raise ValueError(f"({report_date_col}, {seq_col}) is not unique; cannot link reliably")

    is_ref = status.isin(["C", "W"]).to_numpy()
    refs = pd.MultiIndex.from_arrays([report_date[is_ref], orig[is_ref]])
    referenced = key.isin(refs)
    matched_refs = refs.isin(key)
    drop = status.eq("C").to_numpy() | referenced

    diagnostics = {
        "n_input": len(frame),
        "n_cancel_records": int(status.eq("C").sum()),
        "n_correction_records": int(status.eq("W").sum()),
        "n_referenced_removed": int(referenced.sum()),
        "n_unmatched_references": int((~matched_refs).sum()),
    }
    out = frame.loc[~drop]
    diagnostics["n_output"] = len(out)
    return out, diagnostics


def _time_seconds(series: pd.Series) -> pd.Series:
    """Execution time as seconds after midnight (accepts 'HH:MM:SS' text or time objects)."""
    return pd.to_timedelta(series.astype("string")).dt.total_seconds()


def apply_reversals(
    frame: pd.DataFrame,
    *,
    match_cols: Sequence[str] = REVERSAL_MATCH_COLS,
    asof_col: str = "asof_cd",
    report_date_col: str = "trd_rpt_dt",
    time_col: str = "trd_exctn_tm",
    order_cols: Sequence[str] = ("trd_rpt_dt", "msg_seq_nb"),
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Remove pre-2012 reversals (ASOF_CD = 'R') and the earlier trades they reverse.

    Documented (both layouts): a reversal cancels a trade report submitted on an earlier
    day. FINRA's procedure is to match the reversal's basic trade details (bond, quantity,
    price, execution date and time, buy/sell, contra party) to earlier non-reversal
    transactions. The original may have been reported as-of, so its report date need
    not equal the execution date.

    Pass 1 matches every detail exactly. Pass 2 handles a precision difference observed
    in the WRDS 2010 sample: reversal execution times often carry no seconds (``HH:MM:00``)
    while the original carries seconds. A still-unmatched reversal whose time has zero
    seconds may match an original executed within that same minute, all other details
    exact. Pass-2 matches are counted separately.

    Each reversal removes one matching non-reversal record with a strictly earlier
    report date, the most recently reported first. Candidates identical on every match
    column cannot be told apart; how often that happens is counted, not hidden.
    Unmatched reversals (original reported before the extract starts) are counted and
    dropped.
    """
    _require(frame, [*match_cols, asof_col, report_date_col, *order_cols], "Reversal matching")
    work = frame.reset_index(drop=True)
    is_rev = _codes(work[asof_col]).eq("R").fillna(False).to_numpy()
    report_date = pd.to_datetime(work[report_date_col]).to_numpy()

    # Sort keys as values, not text: WRDS stores msg_seq_nb as varchar.
    sort_frame = work[list(order_cols)].copy()
    for column in order_cols:
        if not pd.api.types.is_datetime64_any_dtype(sort_frame[column]):
            numeric = pd.to_numeric(sort_frame[column], errors="coerce")
            if numeric.notna().sum() == sort_frame[column].notna().sum():
                sort_frame[column] = numeric

    drop = is_rev.copy()
    used = np.zeros(len(work), dtype=bool)
    matched = np.zeros(len(work), dtype=bool)
    n_multi = 0

    def run_pass(keys: pd.DataFrame, reversal_ok: np.ndarray) -> int:
        nonlocal n_multi
        n = 0
        for rows in keys.groupby(list(keys.columns), dropna=False, sort=False).indices.values():
            if not (reversal_ok[rows]).any():
                continue
            ordered = sort_frame.iloc[rows].sort_values(list(order_cols)).index.to_numpy()
            stack: list[int] = []
            for row in ordered:
                if not is_rev[row]:
                    if not used[row]:
                        stack.append(row)
                    continue
                if not reversal_ok[row]:
                    continue
                eligible = [r for r in stack if report_date[r] < report_date[row]]
                if not eligible:
                    continue
                if len(eligible) > 1:
                    n_multi += 1
                chosen = eligible[-1]
                stack.remove(chosen)
                used[chosen] = drop[chosen] = matched[row] = True
                n += 1
        return n

    exact = run_pass(work[list(match_cols)], is_rev.copy())
    n_minute = 0
    if time_col in match_cols:
        seconds = _time_seconds(work[time_col])
        truncated = (seconds % 60 == 0).fillna(False).to_numpy()
        keys = work[list(match_cols)].copy()
        keys[time_col] = (seconds // 60).astype("Int64")
        n_minute = run_pass(keys, is_rev & ~matched & truncated)

    n_rev = int(is_rev.sum())
    out = frame.loc[~drop]
    diagnostics = {
        "n_reversals": n_rev,
        "n_reversals_matched": exact + n_minute,
        "n_reversals_matched_exact": exact,
        "n_reversals_matched_same_minute": n_minute,
        "n_reversals_unmatched": n_rev - exact - n_minute,
        "n_reversals_multiple_candidates": n_multi,
        "n_output": len(out),
    }
    return out, diagnostics


POST2012_STATUSES = {"T", "X", "C", "R", "Y"}


def apply_post2012_lifecycle(
    frame: pd.DataFrame,
    *,
    status_col: str = "trc_st",
    report_date_col: str = "trd_rpt_dt",
    seq_col: str = "msg_seq_nb",
    prior_date_col: str = "pr_trd_dt",
    prior_seq_col: str = "orig_msg_seq_nb",
    asof_col: str = "asof_cd",
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply cancels, corrections and reversals reported from 2012-02-06.

    Documented (FINRA p353157): Trade Status T = Trade Report, X = Trade Cancel,
    C = Cancelled Correction, R = New Correction, Y = Reversal; As-Of 'R' = reversal;
    Prior Trade Report Date and Prior Reference Number link to the earlier report.

    Encoding observed in WRDS ``trace_enhanced.trace_enhanced`` (2013-01-07/08 sample):

    - X and C records carry the *original's* reference number in ``msg_seq_nb`` (their
      prior fields are mostly blank, and when present are copied from the original). All
      X/C records whose original is in the sample match it on (CUSIP, ``msg_seq_nb``,
      execution date) with the same price and an earlier-or-same report date.
    - R and Y records carry ``pr_trd_dt`` and ``orig_msg_seq_nb`` pointing at the record
      they supersede (a correction arrives as a C copy of the old version plus an R with
      the new values; 399 of 401 C records had their R).
    - (report date, ``msg_seq_nb``) is unique among non-X/C records only.

    Applied rule (union of removals, never revival):
    1. X/C -> remove the non-X/C record with the same CUSIP, ``msg_seq_nb`` and execution
       date reported on or before it (the latest such record if several);
    2. R/Y -> remove the record at (``pr_trd_dt``, ``orig_msg_seq_nb``);
    3. remove X, C and Y records and any record flagged As-Of 'R' (reversal records,
       including corrected reversals); T and R records otherwise stand.

    Only records reported on or after 2012-02-06 act as post-2012 records; the records
    they remove may be earlier (cross-era links are counted in diagnostics).
    """
    _require(
        frame,
        [status_col, report_date_col, seq_col, prior_date_col, prior_seq_col, "cusip_id",
         "trd_exctn_dt"],
        "Post-2012 lifecycle",
    )  # fmt: skip
    work = frame.reset_index(drop=True)
    report_date = pd.to_datetime(work[report_date_col])
    if report_date.isna().any():
        raise ValueError(f"{report_date_col} has missing values")
    post = report_date.ge(PLATFORM_MIGRATION).to_numpy()
    status = _codes(work[status_col])
    post_status = status[post]
    unknown = sorted(set(post_status.dropna().unique()) - POST2012_STATUSES)
    if unknown or post_status.isna().any():
        raise ValueError(f"Post-2012 statuses outside T/X/C/R/Y: {unknown or ['<NA>']}")

    seq = pd.to_numeric(work[seq_col]).astype("Int64")
    exec_date = pd.to_datetime(work["trd_exctn_dt"])
    is_xc = post & status.isin(["X", "C"]).fillna(False).to_numpy()
    key = pd.MultiIndex.from_arrays([report_date, seq])
    if key[~is_xc].has_duplicates:
        raise ValueError(
            f"({report_date_col}, {seq_col}) is not unique among non-X/C records; "
            "cannot link reliably"
        )

    drop = np.zeros(len(work), dtype=bool)

    # 1. X/C -> original by (CUSIP, reference number, execution date).
    originals = pd.DataFrame(
        {"cusip": work["cusip_id"], "seq": seq, "exec": exec_date, "rpt": report_date,
         "row": np.arange(len(work))}
    ).loc[~is_xc]  # fmt: skip
    cancels = pd.DataFrame(
        {"cusip": work["cusip_id"], "seq": seq, "exec": exec_date, "rpt": report_date,
         "xrow": np.arange(len(work))}
    ).loc[is_xc]  # fmt: skip
    pairs = cancels.merge(originals, on=["cusip", "seq", "exec"], suffixes=("", "_o"))
    pairs = pairs.loc[pairs["rpt_o"] <= pairs["rpt"]]
    n_multi = int(pairs.groupby("xrow").size().gt(1).sum())
    chosen = pairs.sort_values(["rpt_o", "row"]).groupby("xrow").tail(1)
    drop[chosen["row"].to_numpy()] = True
    n_xc_matched = int(chosen["xrow"].nunique())

    # 2. R/Y -> record at (prior report date, prior reference number).
    is_ry = post & status.isin(["R", "Y"]).fillna(False).to_numpy()
    prior_date = pd.to_datetime(work[prior_date_col])
    prior_seq = pd.to_numeric(work[prior_seq_col]).astype("Int64")
    has_prior = (prior_date.notna() & prior_seq.notna()).to_numpy()
    refs = pd.MultiIndex.from_arrays([prior_date[is_ry & has_prior], prior_seq[is_ry & has_prior]])
    target = key.isin(refs) & ~is_xc
    drop |= target
    matched_refs = refs.isin(key[~is_xc])

    # 3. Non-trade records.
    reversal_flag = (
        _codes(work[asof_col]).eq("R").fillna(False).to_numpy() if asof_col in work else False
    )
    not_trade = post & (status.isin(["X", "C", "Y"]).fillna(False).to_numpy() | reversal_flag)
    drop |= not_trade

    out = frame.loc[~drop]
    diagnostics = {
        "n_input": len(frame),
        "n_post2012_records": int(post.sum()),
        **{f"n_status_{s}": int(post_status.eq(s).sum()) for s in sorted(POST2012_STATUSES)},
        "n_xc_matched": n_xc_matched,
        "n_xc_unmatched": int(is_xc.sum()) - n_xc_matched,
        "n_xc_multiple_candidates": n_multi,
        "n_ry_missing_prior_fields": int((is_ry & ~has_prior).sum()),
        "n_ry_refs_matched": int(matched_refs.sum()),
        "n_ry_refs_unmatched": int((~matched_refs).sum()),
        "n_removed_as_superseded": int((drop & ~not_trade).sum()),
        "n_cross_era_removed": int((drop & ~not_trade & ~post).sum()),
        "n_output": len(out),
    }
    return out, diagnostics


def apply_btds_lifecycle(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Unresolved: lifecycle rules for the standard disseminated BTDS file.

    The FINRA documents reviewed describe the Enhanced historic file only. The
    disseminated file's ``chng_cd``, ``frmt_cd`` and message-linking conventions are not
    in them, so no cleaning rule is assumed. Prefer the Enhanced file when WRDS exposes it.
    """
    raise NotImplementedError(apply_btds_lifecycle.__doc__)


def clean_trace_lifecycle(
    frame: pd.DataFrame, report_date_col: str = "trd_rpt_dt"
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply the documented lifecycle rules to Enhanced historic trade records.

    1. Reports before 2012-02-06: same-day C/W (``apply_pre2012_cancels_corrections``),
       then As-Of 'R' reversals by basic-detail matching (``apply_reversals``).
    2. Reports from 2012-02-06: ``apply_post2012_lifecycle`` (X/C by reference number,
       R/Y by prior fields), which may also remove pre-2012 records they point at.
    Diagnostics are prefixed ``cw_``, ``rev_`` and ``post_``.
    """
    report_date = pd.to_datetime(frame[report_date_col])
    if report_date.isna().any():
        raise ValueError(f"{report_date_col} has missing values")
    pre = report_date.lt(PLATFORM_MIGRATION)
    diagnostics: dict[str, int] = {"n_input": len(frame)}
    out = frame
    if pre.any():
        cleaned_pre, first = apply_pre2012_cancels_corrections(
            frame.loc[pre], report_date_col=report_date_col
        )
        diagnostics.update({f"cw_{k}": v for k, v in first.items()})
        cleaned_pre, third = apply_reversals(cleaned_pre, report_date_col=report_date_col)
        diagnostics.update({f"rev_{k}": v for k, v in third.items()})
        out = pd.concat([cleaned_pre, frame.loc[~pre]])
    if (~pre).any():
        out, second = apply_post2012_lifecycle(out, report_date_col=report_date_col)
        diagnostics.update({f"post_{k}": v for k, v in second.items()})
    diagnostics["n_output"] = len(out)
    return out, diagnostics


# --------------------------------------------------------------------------------------
# Condition filters
# --------------------------------------------------------------------------------------


def filter_enhanced_conditions(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop trades whose documented codes say the price is not a regular market price.

    Codes from the Enhanced layouts, applied only when the column is present:

    - ``wis_fl == 'Y'``: traded on a when-issued basis.
    - ``spcl_trd_fl == 'Y'``: special trade condition that impacted the execution price.
    - ``sale_cndtn_cd == 'W'``: weighted average price (pre-2012 layout).
    - ``trd_mod_4 == 'W'``: weighted average price (post-2012 Trade Modifier 4).
    - ``trdg_mkt_cd == 'P1'``: primary-market list/fixed-offering-price or takedown trade.
    - ``cntra_mp_id == 'D'`` with ``rpt_side_cd == 'B'``: inter-dealer buy. Inter-dealer
      buys and sells are two sides of one trade, and only the sell side is disseminated.

    Returns the kept rows and the number removed by each rule, in the order listed.
    Not valid for the disseminated BTDS file until its codes are confirmed.
    """
    rules = {
        "when_issued": ("wis_fl", lambda f: _codes(f["wis_fl"]).eq("Y")),
        "special_price": ("spcl_trd_fl", lambda f: _codes(f["spcl_trd_fl"]).eq("Y")),
        "weighted_average_price": ("sale_cndtn_cd", lambda f: _codes(f["sale_cndtn_cd"]).eq("W")),
        "weighted_average_price_mod4": ("trd_mod_4", lambda f: _codes(f["trd_mod_4"]).eq("W")),
        "primary_market_p1": ("trdg_mkt_cd", lambda f: _codes(f["trdg_mkt_cd"]).eq("P1")),
        "interdealer_buy_side": (
            "cntra_mp_id",
            lambda f: _codes(f["cntra_mp_id"]).eq("D") & _codes(f["rpt_side_cd"]).eq("B"),
        ),
    }
    out = frame
    removed: dict[str, int] = {}
    for name, (column, rule) in rules.items():
        if column not in out or (name == "interdealer_buy_side" and "rpt_side_cd" not in out):
            continue
        hit = rule(out).fillna(False).to_numpy(dtype=bool)
        removed[name] = int(hit.sum())
        out = out.loc[~hit]
    return out, removed


def basic_clean_trace(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no CUSIP or a non-positive price, then sort.

    Sanity filters only; run after ``clean_trace_lifecycle``.
    """
    out = frame.loc[frame["cusip_id"].notna() & frame["rptd_pr"].gt(0).fillna(False)]
    return out.sort_values(["trd_exctn_dt", "cusip_id"], kind="stable").reset_index(drop=True)


# --------------------------------------------------------------------------------------
# Bond master join and maturity
# --------------------------------------------------------------------------------------


def master_duplicates(master: pd.DataFrame, cusip_col: str = "cusip_id") -> pd.DataFrame:
    """Master rows whose CUSIP appears more than once, for inspection."""
    return master.loc[master[cusip_col].duplicated(keep=False)].sort_values(cusip_col)


def join_master(
    trades: pd.DataFrame,
    master: pd.DataFrame,
    *,
    date_col: str = "trd_exctn_dt",
    cusip_col: str = "cusip_id",
    validity: tuple[str, str] | None = None,
    agree_on: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Attach master-file fields to each trade by CUSIP, never guessing among duplicates.

    ``validity`` names the master's (start, end) date columns when the master holds
    dated versions of a bond's record. A version applies when start <= trade date <= end,
    with a missing bound treated as open. Confirm the inclusive bounds and the meaning
    of missing bounds on real rows before relying on them.

    Every trade is kept. ``master_match`` is 'unique' (fields attached), 'none' or
    'ambiguous' (more than one applicable row; fields left missing). With ``agree_on``,
    a trade with several applicable rows that all hold identical values in those columns
    is 'consistent': only the ``agree_on`` columns are attached (nothing is chosen, since
    the candidates agree). Master columns that clash with trade columns get a ``_master``
    suffix. Returns the joined frame and the count of trades per status.

    In WRDS ``camasterfile`` versions are contiguous with inclusive bounds (the next
    ``stdt`` is usually ``enddt`` + 1 day); a few overlap by a day or more.
    """
    left = trades.reset_index(drop=True)
    left["_row"] = np.arange(len(left))
    candidates = left[["_row", cusip_col, date_col]].merge(master, on=cusip_col, how="inner")
    if validity is not None:
        start, end = validity
        date = pd.to_datetime(candidates[date_col])
        start_ok = candidates[start].isna() | (pd.to_datetime(candidates[start]) <= date)
        end_ok = candidates[end].isna() | (date <= pd.to_datetime(candidates[end]))
        candidates = candidates.loc[start_ok & end_ok]

    n_matches = candidates.groupby("_row").size()
    status = pd.Series("none", index=left["_row"], dtype="string")
    status[n_matches.index[n_matches.eq(1)]] = "unique"
    status[n_matches.index[n_matches.gt(1)]] = "ambiguous"

    unique = candidates.loc[candidates["_row"].isin(n_matches.index[n_matches.eq(1)])]
    fields = unique.drop(columns=[cusip_col, date_col])
    if agree_on:
        agree_on = [c for c in agree_on if c in candidates]
        multi = candidates.loc[candidates["_row"].isin(n_matches.index[n_matches.gt(1)])]
        if len(multi) and agree_on:
            distinct = multi.groupby("_row")[agree_on].nunique(dropna=False)
            consistent_rows = distinct.index[distinct.le(1).all(axis=1)]
            status[consistent_rows] = "consistent"
            agreed = multi.loc[multi["_row"].isin(consistent_rows)].drop_duplicates("_row")
            fields = pd.concat([fields, agreed[["_row", *agree_on]]], ignore_index=True)
    out = left.merge(fields, on="_row", how="left", suffixes=("", "_master"))
    out["master_match"] = status.to_numpy()
    out = out.drop(columns="_row")
    out.index = trades.index
    return out, out["master_match"].value_counts()


def add_remaining_maturity(
    frame: pd.DataFrame,
    date_col: str = "trd_exctn_dt",
    maturity_col: str = "mtrty_dt",
) -> pd.DataFrame:
    """Remaining maturity in years (actual days / 365.25) from trade date to maturity."""
    out = frame.copy()
    days = (pd.to_datetime(out[maturity_col]) - pd.to_datetime(out[date_col])).dt.days
    out["remaining_maturity_y"] = days / 365.25
    return out


def require_live_bonds(
    frame: pd.DataFrame, maturity_col: str = "remaining_maturity_y"
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep only trades with a known, strictly positive remaining maturity."""
    remaining = frame[maturity_col]
    unknown = remaining.isna()
    matured = remaining.le(0).fillna(False)
    out = frame.loc[~unknown & ~matured]
    return out, {"unknown_maturity": int(unknown.sum()), "matured": int(matured.sum())}


# --------------------------------------------------------------------------------------
# Bond-day aggregation
# --------------------------------------------------------------------------------------


def to_bond_day(
    frame: pd.DataFrame,
    *,
    date_col: str = "trd_exctn_dt",
    cusip_col: str = "cusip_id",
    yield_col: str = "yld_pt",
    price_col: str = "rptd_pr",
    volume_col: str | None = None,
) -> pd.DataFrame:
    """Collapse cleaned trades to one observation per CUSIP and day.

    Median yield and price keep frequently traded bonds from gaining weight just because
    they printed more trades. ``yield_col`` should already be signed (``signed_yield``).
    Diagnostics: ``n_trades`` (raw trades collapsed), ``n_yield_obs``. When
    ``volume_col`` is given, ``volume`` is the day's sum only if every trade has a valid
    volume (otherwise missing, never a partial sum), and ``n_volume_valid`` counts the
    valid ones.
    """
    grouped = frame.groupby([date_col, cusip_col], observed=True, sort=True)
    out = grouped.agg(
        yield_pct=(yield_col, "median"),
        price=(price_col, "median"),
        n_trades=(price_col, "size"),
        n_yield_obs=(yield_col, "count"),
    )
    if volume_col is not None:
        volume = grouped[volume_col].agg(["sum", "count"])
        out["volume"] = volume["sum"].where(volume["count"].eq(out["n_trades"]))
        out["n_volume_valid"] = volume["count"]
    return out.reset_index()
