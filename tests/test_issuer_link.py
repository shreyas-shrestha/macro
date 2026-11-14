"""Issuer-link diagnostics surface ambiguity instead of resolving it."""

import pandas as pd

from macro.data.issuer_link import (
    exact_matches,
    link_coverage,
    link_status,
    method_conflicts,
    normalize_issuer_name,
)


def test_normalize_issuer_name_is_deterministic():
    names = pd.Series(["Acme Holdings, Inc.", "AT&T Inc", "  ford motor co ", "CO", None])
    assert list(normalize_issuer_name(names).astype(object)) == [
        "ACME",
        "AT AND T",
        "FORD MOTOR",
        "CO",
        pd.NA,
    ]


def _frames():
    trace = pd.DataFrame(
        {
            "trace_issuer": ["T1", "T2", "T3", "T4"],
            "ticker": ["AAA", "BBB", "CCC", None],
            "name_key": ["ALPHA", "BETA", "GAMMA", "DELTA"],
        }
    )
    comp = pd.DataFrame(
        {
            "gvkey": ["001", "002", "003", "004", "005"],
            "tic": ["AAA", "BBB", "BBB", "CCC", "ZZZ"],
            "name_key": ["ALPHA", "BETA", "BETA", "GAMMA2", "DELTA"],
        }
    )
    return trace.astype("string"), comp.astype("string")


def _matches():
    trace, comp = _frames()
    keys = dict(left_id="trace_issuer", right_id="gvkey")
    by_ticker = exact_matches(
        trace, comp, left_key="ticker", right_key="tic", method="ticker", **keys
    )
    by_name = exact_matches(
        trace, comp, left_key="name_key", right_key="name_key", method="name", **keys
    )
    return trace, pd.concat([by_ticker, by_name], ignore_index=True)


def test_ambiguous_links_are_reported_with_all_candidates():
    trace, matches = _matches()
    status = link_status(matches, trace["trace_issuer"], left_id="trace_issuer", right_id="gvkey")
    ticker = status.loc[status["method"].eq("ticker")].set_index("trace_issuer")
    assert ticker.loc["T1", "status"] == "unique"
    assert ticker.loc["T2", "status"] == "ambiguous"
    assert ticker.loc["T2", "candidates"] == ("002", "003")
    assert ticker.loc["T4", "status"] == "unmatched"  # missing ticker never matches


def test_cross_method_conflicts_are_surfaced():
    trace, matches = _matches()
    conflicts = method_conflicts(matches, left_id="trace_issuer", right_id="gvkey").set_index(
        "trace_issuer"
    )
    assert set(conflicts.index) == {"T2"}
    assert conflicts.loc["T2", "by_method"] == {"name": ("002", "003"), "ticker": ("002", "003")}


def test_disagreeing_methods_are_conflicts():
    matches = pd.DataFrame(
        {"trace_issuer": ["T1", "T1"], "gvkey": ["001", "009"], "method": ["ticker", "name"]}
    )
    conflicts = method_conflicts(matches, left_id="trace_issuer", right_id="gvkey")
    assert conflicts["candidates"].tolist() == [("001", "009")]


def test_link_coverage_counts_every_issuer():
    trace, matches = _matches()
    status = link_status(matches, trace["trace_issuer"], left_id="trace_issuer", right_id="gvkey")
    table = link_coverage(status)
    assert table.loc["ticker"].to_dict() == {
        "unique": 2,
        "ambiguous": 1,
        "unmatched": 1,
        "total": 4,
    }
    assert table.loc["name"].to_dict() == {"unique": 2, "ambiguous": 1, "unmatched": 1, "total": 4}


def test_consensus_link_never_resolves_disagreement():
    from macro.data.issuer_link import consensus_link

    matches = pd.DataFrame(
        {
            "cusip": ["B1", "B1", "B2", "B2", "B3", "B3"],
            "gvkey": ["001", "001", "002", "003", "004", "005"],
            "method": ["fang", "ccm", "fang", "ccm", "fang", "fang"],
        }
    ).astype("string")
    out = consensus_link(
        matches, pd.Series(["B1", "B2", "B3", "B4"]), left_id="cusip", right_id="gvkey"
    ).set_index("cusip")
    assert out.loc["B1", "status"] == "linked"
    assert out.loc["B1", "gvkey"] == "001"
    assert out.loc["B1", "methods"] == ("ccm", "fang")
    assert out.loc["B2", "status"] == "conflict"  # methods disagree
    assert out.loc["B3", "status"] == "conflict"  # one method, two candidates
    assert out.loc["B4", "status"] == "unmatched"
    assert out.loc[["B2", "B3", "B4"], "gvkey"].isna().all()
