"""Diagnostics for linking TRACE issuers to Compustat GVKEYs. Diagnosis, not matching.

Every function here reports exact-key candidates and their conflicts. Nothing chooses
among ambiguous candidates and there is no fuzzy matching. A production crosswalk
should be built only after these reports have been reviewed on real data.
"""

import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

LEGAL_SUFFIXES = (
    "INCORPORATED",
    "INC",
    "CORPORATION",
    "CORP",
    "COMPANY",
    "CO",
    "LIMITED",
    "LTD",
    "LLC",
    "LP",
    "PLC",
    "HOLDINGS",
    "HLDGS",
)


def normalize_issuer_name(series: pd.Series, suffixes: Iterable[str] = LEGAL_SUFFIXES) -> pd.Series:
    """Deterministic name key: uppercase, '&' -> 'AND', drop punctuation and trailing suffixes.

    Only trailing legal-form words in ``suffixes`` are removed, repeatedly (so
    "ACME HOLDINGS INC" -> "ACME"). Two issuers that share a key are reported as
    ambiguous downstream, never merged.
    """
    suffix_set = {s.upper() for s in suffixes}

    def key(name):
        if pd.isna(name):
            return pd.NA
        text = str(name).upper().replace("&", " AND ")
        words = re.sub(r"[^A-Z0-9 ]+", " ", text).split()
        while len(words) > 1 and words[-1] in suffix_set:
            words.pop()
        return " ".join(words) or pd.NA

    return series.map(key).astype("string")


def exact_matches(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    left_id: str,
    right_id: str,
    left_key: str,
    right_key: str,
    method: str,
) -> pd.DataFrame:
    """All (left_id, right_id) pairs whose keys are equal. Missing keys never match.

    Keys are compared as given; normalize them first (e.g. ``normalize_issuer_name``,
    ``macro.data.trace.issuer_cusip6``, stripped uppercase tickers).
    """
    lhs = left[[left_id, left_key]].dropna().drop_duplicates()
    rhs = right[[right_id, right_key]].dropna().drop_duplicates()
    pairs = lhs.merge(rhs, left_on=left_key, right_on=right_key)
    pairs = pairs[[left_id, right_id]].drop_duplicates()
    return pairs.assign(method=method).reset_index(drop=True)


def link_status(
    matches: pd.DataFrame, left_ids: pd.Series, *, left_id: str, right_id: str
) -> pd.DataFrame:
    """Per left issuer and method: number of candidates, status and candidate list.

    ``status`` is 'unmatched', 'unique' or 'ambiguous'. Every left id appears once per
    method seen in ``matches``, including ids that no method matched.
    """
    ids = pd.Index(pd.unique(left_ids.dropna()), name=left_id)
    methods = sorted(matches["method"].unique()) if len(matches) else []
    rows = []
    for method in methods:
        subset = matches.loc[matches["method"].eq(method)]
        candidates = {
            lid: tuple(sorted(str(v) for v in g[right_id].dropna().unique()))
            for lid, g in subset.groupby(left_id, sort=False)
        }
        for lid in ids:
            found = candidates.get(lid, ())
            n = len(found)
            status = "unmatched" if n == 0 else "unique" if n == 1 else "ambiguous"
            rows.append((lid, method, n, status, found))
    return pd.DataFrame(rows, columns=[left_id, "method", "n_candidates", "status", "candidates"])


def method_conflicts(matches: pd.DataFrame, *, left_id: str, right_id: str) -> pd.DataFrame:
    """Left issuers whose methods disagree, i.e. more than one distinct right id overall.

    Includes issuers ambiguous within one method. Columns: the left id, every distinct
    candidate, and each method's own candidates.
    """
    if matches.empty:
        return pd.DataFrame(columns=[left_id, "candidates", "by_method"])
    rows = []
    for lid, group in matches.groupby(left_id):
        distinct = tuple(sorted(group[right_id].unique()))
        if len(distinct) > 1:
            by_method = {
                m: tuple(sorted(g[right_id])) for m, g in group.groupby("method", sort=True)
            }
            rows.append((lid, distinct, by_method))
    return pd.DataFrame(rows, columns=[left_id, "candidates", "by_method"])


def link_coverage(status: pd.DataFrame) -> pd.DataFrame:
    """Count of issuers by method and status (unmatched / unique / ambiguous)."""
    table = pd.crosstab(status["method"], status["status"])
    for column in ("unmatched", "unique", "ambiguous"):
        if column not in table:
            table[column] = 0
    table = table[["unique", "ambiguous", "unmatched"]]
    table["total"] = table.sum(axis=1)
    return table


def consensus_link(
    matches: pd.DataFrame, left_ids: pd.Series, *, left_id: str, right_id: str
) -> pd.DataFrame:
    """Production link: a right id only when every matching method agrees on exactly one.

    ``matches`` should contain only exact-identifier methods (not ticker or name). Per
    left id: ``status`` is 'linked' (one distinct candidate across all methods),
    'conflict' (more than one distinct candidate, within or across methods) or
    'unmatched'. ``methods`` lists the methods that produced the linked candidate. No
    priority order is applied, so a disagreement is never resolved silently.
    """
    ids = pd.Index(pd.unique(left_ids.dropna()), name=left_id)
    # Build tuples explicitly: groupby().agg may coerce tuples for string-dtype columns.
    candidates: dict = {}
    methods: dict = {}
    for lid, group in matches.groupby(left_id, sort=False):
        candidates[lid] = tuple(sorted({str(v) for v in group[right_id].dropna()}))
        methods[lid] = tuple(sorted({str(v) for v in group["method"].dropna()}))
    out = pd.DataFrame(index=ids)
    out["candidates"] = [candidates.get(i, ()) for i in ids]
    out["methods"] = [methods.get(i, ()) for i in ids]
    n = out["candidates"].map(len)
    out["status"] = np.select([n.eq(1), n.gt(1)], ["linked", "conflict"], default="unmatched")
    out[right_id] = out["candidates"].map(lambda c: c[0] if len(c) == 1 else pd.NA)
    return out.reset_index()
