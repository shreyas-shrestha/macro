"""Synthetic tests for macro.data.trace. Lifecycle cases follow FINRA's Enhanced layouts."""

import numpy as np
import pandas as pd
import pytest

from macro.data.trace import (
    add_remaining_maturity,
    apply_post2012_lifecycle,
    apply_pre2012_cancels_corrections,
    apply_reversals,
    basic_clean_trace,
    clean_trace_lifecycle,
    cusip_check_digit,
    cusip_status,
    filter_enhanced_conditions,
    issuer_cusip6,
    join_master,
    master_duplicates,
    normalize_cusip,
    parse_reported_volume,
    require_live_bonds,
    signed_yield,
    to_bond_day,
)

# --------------------------------------------------------------------------------------
# CUSIPs
# --------------------------------------------------------------------------------------


def test_normalize_cusip_preserves_leading_zeros_and_uppercases():
    raw = pd.Series([" 037833100", "38259p508", "", None])
    out = normalize_cusip(raw)
    assert out.iloc[0] == "037833100"
    assert out.iloc[1] == "38259P508"
    assert out.iloc[2:].isna().all()


def test_normalize_cusip_never_pads_or_truncates():
    out = normalize_cusip(pd.Series(["37833100", "0378331000"]))
    assert out.tolist() == ["37833100", "0378331000"]


def test_normalize_cusip_rejects_numeric_input():
    with pytest.raises(TypeError, match="leading zeros"):
        normalize_cusip(pd.Series([37833100]))
    with pytest.raises(TypeError, match="non-string"):
        normalize_cusip(pd.Series(["037833100", 37833100], dtype=object))


@pytest.mark.parametrize("cusip", ["037833100", "594918104", "38259P508", "459200101"])
def test_cusip_check_digit_on_known_cusips(cusip):
    assert cusip_check_digit(cusip[:8]) == int(cusip[8])


def test_cusip_status_classifies_without_altering():
    raw = pd.Series(["037833100", "037833101", "37833100", "03783310!", None])
    assert cusip_status(raw).tolist() == [
        "ok",
        "bad_check_digit",
        "bad_length",
        "bad_chars",
        "missing",
    ]


def test_issuer_cusip6_only_for_well_formed_cusips():
    out = issuer_cusip6(pd.Series(["037833100", "0378331", None]))
    assert out.iloc[0] == "037833"
    assert out.iloc[1:].isna().all()


# --------------------------------------------------------------------------------------
# Field decoding
# --------------------------------------------------------------------------------------


def test_signed_yield_applies_documented_direction_codes():
    frame = pd.DataFrame({"yld_pt": [5.0, 1.0, 2.0, -3.0], "yld_sign_cd": ["+", "-", None, "-"]})
    assert signed_yield(frame).tolist() == [5.0, -1.0, 2.0, -3.0]


def test_signed_yield_rejects_contradictions_and_unknown_codes():
    with pytest.raises(ValueError, match="negative"):
        signed_yield(pd.DataFrame({"yld_pt": [-1.0], "yld_sign_cd": ["+"]}))
    with pytest.raises(ValueError, match="Undocumented"):
        signed_yield(pd.DataFrame({"yld_pt": [1.0], "yld_sign_cd": ["?"]}))


def test_parse_reported_volume_leaves_capped_text_missing():
    out = parse_reported_volume(pd.Series(["5000", "250.5", "1MM+", None]))
    assert out.iloc[:2].tolist() == [5000.0, 250.5]
    assert out.iloc[2:].isna().all()


# --------------------------------------------------------------------------------------
# Lifecycle: pre-2012 cancellations (C) and corrections (W)
# --------------------------------------------------------------------------------------


def _pre2012(rows):
    columns = ["trc_st", "trd_rpt_dt", "msg_seq_nb", "orig_msg_seq_nb", "rptd_pr"]
    frame = pd.DataFrame(rows, columns=columns)
    frame["trd_rpt_dt"] = pd.to_datetime(frame["trd_rpt_dt"])
    return frame


def test_cancellation_removes_itself_and_original():
    frame = _pre2012(
        [
            ("T", "2010-01-04", 1, None, 100.0),
            ("C", "2010-01-04", 2, 1, 100.0),
            ("T", "2010-01-04", 3, None, 99.0),
        ]
    )
    out, diag = apply_pre2012_cancels_corrections(frame)
    assert out["msg_seq_nb"].tolist() == [3]
    assert diag["n_referenced_removed"] == 1
    assert diag["n_unmatched_references"] == 0


def test_correction_replaces_original():
    frame = _pre2012(
        [
            ("T", "2010-01-04", 1, None, 100.0),
            ("W", "2010-01-04", 2, 1, 101.0),
        ]
    )
    out, _ = apply_pre2012_cancels_corrections(frame)
    assert out["rptd_pr"].tolist() == [101.0]
    assert out["trc_st"].tolist() == ["W"]


def test_correction_chain_keeps_only_latest_version():
    frame = _pre2012(
        [
            ("T", "2010-01-04", 1, None, 100.0),
            ("W", "2010-01-04", 2, 1, 101.0),
            ("W", "2010-01-04", 3, 2, 102.0),
        ]
    )
    out, _ = apply_pre2012_cancels_corrections(frame)
    assert out["rptd_pr"].tolist() == [102.0]


def test_cancelling_a_correction_removes_the_trade_entirely():
    frame = _pre2012(
        [
            ("T", "2010-01-04", 1, None, 100.0),
            ("W", "2010-01-04", 2, 1, 101.0),
            ("C", "2010-01-04", 3, 2, 101.0),
        ]
    )
    out, _ = apply_pre2012_cancels_corrections(frame)
    assert out.empty


def test_sequence_numbers_link_only_within_a_report_date():
    # MSG_SEQ_NB is unique only within a Trade Report Date.
    frame = _pre2012(
        [
            ("T", "2010-01-04", 1, None, 100.0),
            ("T", "2010-01-05", 1, None, 98.0),
            ("C", "2010-01-05", 2, 1, 98.0),
        ]
    )
    out, diag = apply_pre2012_cancels_corrections(frame)
    assert out["rptd_pr"].tolist() == [100.0]
    assert diag["n_unmatched_references"] == 0


def test_unmatched_reference_is_counted():
    frame = _pre2012(
        [
            ("T", "2010-01-04", 1, None, 100.0),
            ("C", "2010-01-05", 2, 1, 100.0),
        ]
    )
    out, diag = apply_pre2012_cancels_corrections(frame)
    assert out["msg_seq_nb"].tolist() == [1]
    assert diag["n_unmatched_references"] == 1


def test_pre2012_rejects_other_statuses_and_duplicate_keys():
    with pytest.raises(ValueError, match="T/C/W"):
        apply_pre2012_cancels_corrections(_pre2012([("X", "2010-01-04", 1, None, 1.0)]))
    duplicate = _pre2012([("T", "2010-01-04", 1, None, 1.0), ("T", "2010-01-04", 1, None, 2.0)])
    with pytest.raises(ValueError, match="not unique"):
        apply_pre2012_cancels_corrections(duplicate)


def test_pre2012_rules_refuse_post_migration_reports():
    with pytest.raises(ValueError, match="before 2012-02-06"):
        apply_pre2012_cancels_corrections(_pre2012([("T", "2012-02-06", 1, None, 1.0)]))


def _post2012(rows):
    """Rows: (status, report date, msg_seq_nb, pr_trd_dt, orig_msg_seq_nb, price, cusip, asof)."""
    columns = ["trc_st", "trd_rpt_dt", "msg_seq_nb", "pr_trd_dt", "orig_msg_seq_nb", "rptd_pr",
               "cusip_id", "asof_cd"]  # fmt: skip
    frame = pd.DataFrame([(*r, *(("B1", None)[len(r) - 6 :])) if len(r) < 8 else r for r in rows],
                         columns=columns)  # fmt: skip
    for column in ("trd_rpt_dt", "pr_trd_dt"):
        frame[column] = pd.to_datetime(frame[column])
    frame["trd_exctn_dt"] = pd.Timestamp("2013-01-04")
    return frame


def test_post2012_cancel_carries_the_originals_reference_number():
    # WRDS encoding: an X record's msg_seq_nb is the original's; its prior fields are blank.
    frame = _post2012(
        [
            ("T", "2013-01-04", 216, None, None, 100.0),
            ("X", "2013-01-07", 216, None, None, 100.0),
            ("T", "2013-01-07", 216, None, None, 99.0, "B2", None),  # unrelated, same number
        ]
    )
    out, diag = apply_post2012_lifecycle(frame)
    assert out["rptd_pr"].tolist() == [99.0]
    assert diag["n_xc_matched"] == 1
    assert diag["n_xc_unmatched"] == 0


def test_post2012_correction_pair_keeps_only_new_version():
    # C is a copy of the old version (old msg_seq_nb); R carries the new values and points
    # at the old version through (pr_trd_dt, orig_msg_seq_nb).
    frame = _post2012(
        [
            ("T", "2013-01-04", 10, None, None, 100.0),
            ("C", "2013-01-07", 10, None, None, 100.0),
            ("R", "2013-01-07", 12, "2013-01-04", 10, 101.0),
        ]
    )
    out, diag = apply_post2012_lifecycle(frame)
    assert out["rptd_pr"].tolist() == [101.0]
    assert out["trc_st"].tolist() == ["R"]
    assert diag["n_ry_refs_matched"] == 1


def test_post2012_correction_of_a_correction():
    frame = _post2012(
        [
            ("T", "2013-01-04", 10, None, None, 100.0),
            ("C", "2013-01-07", 10, None, None, 100.0),
            ("R", "2013-01-07", 12, "2013-01-04", 10, 101.0),
            ("C", "2013-01-08", 12, "2013-01-04", 10, 101.0),  # prior fields copied from R
            ("R", "2013-01-08", 30, "2013-01-07", 12, 102.0),
        ]
    )
    out, _ = apply_post2012_lifecycle(frame)
    assert out["rptd_pr"].tolist() == [102.0]


def test_post2012_reversal_and_corrected_reversal_are_not_trades():
    frame = _post2012(
        [
            ("T", "2012-10-10", 70, None, None, 132.0),
            ("Y", "2012-12-11", 50, "2012-10-10", 70, 132.0, "B1", "R"),
            # The reversal is later corrected: C copies the Y, R is a corrected reversal.
            ("C", "2013-01-07", 50, "2012-10-10", 70, 132.0, "B1", "R"),
            ("R", "2013-01-07", 59, "2012-12-11", 50, 131.0, "B1", "R"),
        ]
    )
    frame["trd_exctn_dt"] = pd.Timestamp("2012-10-10")
    out, _ = apply_post2012_lifecycle(frame)
    assert out.empty


def test_post2012_unmatched_links_are_counted():
    frame = _post2012(
        [
            ("X", "2013-01-07", 5, None, None, 100.0),
            ("Y", "2013-01-07", 6, "2012-11-01", 99, 100.0, "B1", "R"),
        ]
    )
    out, diag = apply_post2012_lifecycle(frame)
    assert out.empty
    assert diag["n_xc_unmatched"] == 1
    assert diag["n_ry_refs_unmatched"] == 1


def test_post2012_rejects_undocumented_statuses_and_duplicate_trade_keys():
    with pytest.raises(ValueError, match="T/X/C/R/Y"):
        apply_post2012_lifecycle(_post2012([("W", "2013-01-07", 1, None, None, 1.0)]))
    dup = _post2012(
        [("T", "2013-01-07", 1, None, None, 1.0), ("T", "2013-01-07", 1, None, None, 2.0)]
    )
    with pytest.raises(ValueError, match="not unique"):
        apply_post2012_lifecycle(dup)


def test_mixed_era_lifecycle_links_post2012_cancel_to_pre2012_record():
    frame = pd.DataFrame(
        [
            _trade("2012-02-03", 1, price=100.0),
            _trade("2012-02-03", 2, price=97.0, qty=3.0),
            _trade("2012-02-07", 1, price=100.0, trc_st="X"),
        ]
    )
    frame["pr_trd_dt"] = pd.to_datetime(frame["pr_trd_dt"])
    out, diag = clean_trace_lifecycle(frame)
    assert out["rptd_pr"].tolist() == [97.0]
    assert diag["post_n_cross_era_removed"] == 1


def test_reversal_same_minute_precision_pass():
    # Observed in WRDS 2010 rows: reversal times often lack seconds.
    frame = pd.DataFrame(
        [
            _trade("2010-01-04", 1, trd_exctn_tm="10:07:23"),
            _trade("2010-01-06", 7, asof="R", trd_exctn_tm="10:07:00"),
            _trade("2010-01-04", 2, price=90.0, trd_exctn_tm="11:00:41"),
            _trade("2010-01-06", 8, price=90.0, asof="R", trd_exctn_tm="11:01:00"),  # other minute
        ]
    )
    out, diag = apply_reversals(frame)
    assert out["msg_seq_nb"].tolist() == [2]
    assert diag["n_reversals_matched_exact"] == 0
    assert diag["n_reversals_matched_same_minute"] == 1
    assert diag["n_reversals_unmatched"] == 1


def test_reversal_ordering_treats_text_sequence_numbers_as_numbers():
    frame = pd.DataFrame(
        [
            _trade("2010-01-04", "45", price=100.0, trc_st="T"),
            _trade("2010-01-04", "123", price=100.0, trc_st="T"),
            _trade("2010-01-06", "7", asof="R"),
        ]
    )
    out, diag = apply_reversals(frame)
    # The most recently reported candidate (seq 123, not lexically-later "45") is removed.
    assert out["msg_seq_nb"].tolist() == ["45"]
    assert diag["n_reversals_multiple_candidates"] == 1


# --------------------------------------------------------------------------------------
# Lifecycle: reversals (ASOF_CD = 'R')
# --------------------------------------------------------------------------------------


def _trade(rpt, seq, price=100.0, asof=None, qty=10.0, side="S", **extra):
    row = {
        "cusip_id": "037833100",
        "entrd_vol_qt": qty,
        "rptd_pr": price,
        "trd_exctn_dt": pd.Timestamp("2010-01-04"),
        "trd_exctn_tm": "10:15:00",
        "rpt_side_cd": side,
        "cntra_mp_id": "C",
        "asof_cd": asof,
        "trd_rpt_dt": pd.Timestamp(rpt),
        "msg_seq_nb": seq,
        "trc_st": "T",
        "orig_msg_seq_nb": None,
        "pr_trd_dt": None,
    }
    row.update(extra)
    return row


def test_reversal_removes_matching_earlier_trade():
    frame = pd.DataFrame(
        [
            _trade("2010-01-04", 1),
            _trade("2010-01-04", 2, price=99.0),
            _trade("2010-01-06", 7, asof="R"),
        ]
    )
    out, diag = apply_reversals(frame)
    assert out["rptd_pr"].tolist() == [99.0]
    assert diag["n_reversals_matched"] == 1


def test_reversal_requires_all_basic_details_to_match():
    frame = pd.DataFrame([_trade("2010-01-04", 1), _trade("2010-01-06", 7, asof="R", side="B")])
    out, diag = apply_reversals(frame)
    assert out["msg_seq_nb"].tolist() == [1]
    assert diag["n_reversals_unmatched"] == 1


def test_reversal_matches_only_strictly_earlier_reports():
    frame = pd.DataFrame([_trade("2010-01-06", 1), _trade("2010-01-06", 7, asof="R")])
    out, diag = apply_reversals(frame)
    assert out["msg_seq_nb"].tolist() == [1]
    assert diag["n_reversals_unmatched"] == 1


def test_reversal_with_identical_candidates_removes_one_and_reports_it():
    frame = pd.DataFrame(
        [_trade("2010-01-04", 1), _trade("2010-01-05", 3), _trade("2010-01-06", 7, asof="R")]
    )
    out, diag = apply_reversals(frame)
    assert len(out) == 1
    assert diag["n_reversals_multiple_candidates"] == 1


def test_reversal_can_match_an_as_of_original():
    # The original may itself be reported as-of, so its report date differs from execution.
    frame = pd.DataFrame([_trade("2010-01-05", 1, asof="A"), _trade("2010-01-07", 7, asof="R")])
    out, _ = apply_reversals(frame)
    assert out.empty


def test_full_lifecycle_applies_corrections_before_reversals():
    frame = pd.DataFrame(
        [
            _trade("2010-01-04", 1, price=100.0),
            _trade("2010-01-04", 2, price=101.0, trc_st="W", orig_msg_seq_nb=1),
            _trade("2010-01-08", 9, price=101.0, asof="R"),
            _trade("2010-01-04", 3, price=95.0, qty=5.0),
        ]
    )
    out, diag = clean_trace_lifecycle(frame)
    assert out["rptd_pr"].tolist() == [95.0]
    assert diag["cw_n_referenced_removed"] == 1
    assert diag["rev_n_reversals_matched"] == 1


# --------------------------------------------------------------------------------------
# Condition filters
# --------------------------------------------------------------------------------------


def test_enhanced_condition_filters_use_documented_codes():
    frame = pd.DataFrame(
        {
            "wis_fl": ["N", "Y", "N", "N", "N", "N", "N", "N"],
            "spcl_trd_fl": [None, None, "Y", None, None, None, None, None],
            "sale_cndtn_cd": ["@", "@", "@", "W", "@", "@", "@", "C"],
            "trdg_mkt_cd": ["S1", "S1", "S1", "S1", "P1", "S1", "S1", "S1"],
            "cntra_mp_id": ["C", "C", "C", "C", "C", "D", "D", "C"],
            "rpt_side_cd": ["B", "B", "B", "B", "B", "B", "S", "B"],
            "id": range(8),
        }
    )
    out, removed = filter_enhanced_conditions(frame)
    assert out["id"].tolist() == [0, 6, 7]
    assert removed == {
        "when_issued": 1,
        "special_price": 1,
        "weighted_average_price": 1,
        "primary_market_p1": 1,
        "interdealer_buy_side": 1,
    }


def test_post2012_weighted_average_modifier_is_filtered():
    frame = pd.DataFrame({"trd_mod_4": ["W", None, ""], "id": [0, 1, 2]})
    out, removed = filter_enhanced_conditions(frame)
    assert out["id"].tolist() == [1, 2]
    assert removed == {"weighted_average_price_mod4": 1}


def test_condition_filters_skip_absent_columns():
    out, removed = filter_enhanced_conditions(pd.DataFrame({"wis_fl": ["N", "Y"]}))
    assert len(out) == 1
    assert removed == {"when_issued": 1}


def test_basic_clean_drops_missing_cusip_and_nonpositive_price():
    frame = pd.DataFrame(
        {
            "trd_exctn_dt": pd.to_datetime(["2010-01-04"] * 4),
            "cusip_id": ["A", None, "B", "C"],
            "rptd_pr": [100.0, 100.0, 0.0, np.nan],
        }
    )
    assert basic_clean_trace(frame)["cusip_id"].tolist() == ["A"]


# --------------------------------------------------------------------------------------
# Master join and maturity
# --------------------------------------------------------------------------------------


def _trades():
    return pd.DataFrame(
        {
            "trd_exctn_dt": pd.to_datetime(
                ["2010-01-04", "2010-01-04", "2010-01-05", "2010-06-01"]
            ),
            "cusip_id": ["AAA", "BBB", "ZZZ", "AAA"],
            "company_symbol": ["A", "B", "Z", "A"],
        },
        index=[10, 11, 12, 13],
    )


def test_join_master_unique_none_and_ambiguous():
    master = pd.DataFrame(
        {
            "cusip_id": ["AAA", "BBB", "BBB"],
            "company_symbol": ["A", "B", "B2"],
            "mtrty_dt": pd.to_datetime(["2015-01-01", "2012-01-01", "2013-01-01"]),
        }
    )
    out, counts = join_master(_trades(), master)
    assert out.index.tolist() == [10, 11, 12, 13]
    assert out["master_match"].tolist() == ["unique", "ambiguous", "none", "unique"]
    assert out.loc[10, "mtrty_dt"] == pd.Timestamp("2015-01-01")
    assert pd.isna(out.loc[11, "mtrty_dt"])  # never picks one of the duplicates
    assert out.loc[10, "company_symbol_master"] == "A"
    assert counts.to_dict() == {"unique": 2, "ambiguous": 1, "none": 1}
    assert master_duplicates(master)["cusip_id"].tolist() == ["BBB", "BBB"]


def test_join_master_validity_intervals_select_version_in_force():
    master = pd.DataFrame(
        {
            "cusip_id": ["AAA", "AAA"],
            "cpn_rt": [5.0, 6.0],
            "stdt": pd.to_datetime(["2009-01-01", "2010-03-01"]),
            "enddt": pd.to_datetime(["2010-02-28", None]),
        }
    )
    trades = _trades().loc[[10, 13]]
    out, _ = join_master(trades, master, validity=("stdt", "enddt"))
    assert out["cpn_rt"].tolist() == [5.0, 6.0]
    assert out["master_match"].tolist() == ["unique", "unique"]


def test_join_master_overlapping_versions_that_agree_are_consistent():
    master = pd.DataFrame(
        {
            "cusip_id": ["AAA", "AAA", "BBB", "BBB"],
            "mtrty_dt": pd.to_datetime(["2015-01-01", "2015-01-01", "2015-01-01", "2016-01-01"]),
            "issuer_nm": ["OLD NAME", "NEW NAME", "X", "X"],
            "stdt": pd.to_datetime(["2009-01-01", "2010-01-04"] * 2),
            "enddt": pd.to_datetime(["2010-01-04", "2011-01-01"] * 2),
        }
    )
    trades = pd.DataFrame(
        {"trd_exctn_dt": pd.to_datetime(["2010-01-04", "2010-01-04"]), "cusip_id": ["AAA", "BBB"]}
    )
    out, _ = join_master(trades, master, validity=("stdt", "enddt"), agree_on=["mtrty_dt"])
    assert out["master_match"].tolist() == ["consistent", "ambiguous"]
    assert out.loc[0, "mtrty_dt"] == pd.Timestamp("2015-01-01")
    assert pd.isna(out.loc[0, "issuer_nm"])  # not in agree_on: nothing chosen
    assert pd.isna(out.loc[1, "mtrty_dt"])


def test_remaining_maturity_and_live_bond_filter():
    frame = pd.DataFrame(
        {
            "trd_exctn_dt": pd.to_datetime(["2010-01-04"] * 4),
            "mtrty_dt": pd.to_datetime(["2015-01-04", "2010-01-04", "2009-12-31", None]),
        }
    )
    frame = add_remaining_maturity(frame)
    assert frame["remaining_maturity_y"].iloc[0] == pytest.approx(1826 / 365.25)
    live, dropped = require_live_bonds(frame)
    assert len(live) == 1
    assert dropped == {"unknown_maturity": 1, "matured": 2}


# --------------------------------------------------------------------------------------
# Bond-day aggregation
# --------------------------------------------------------------------------------------


def test_bond_day_volume_is_never_a_partial_sum():
    trades = pd.DataFrame(
        {
            "trd_exctn_dt": pd.to_datetime(["2010-01-04"] * 5),
            "cusip_id": ["A", "A", "B", "B", "B"],
            "yld_pt": [5.0, 6.0, 7.0, np.nan, 9.0],
            "rptd_pr": [100.0, 101.0, 90.0, 91.0, 92.0],
            "volume": [10.0, 20.0, 5.0, np.nan, 5.0],
        }
    )
    out = to_bond_day(trades, volume_col="volume").set_index("cusip_id")
    assert out.loc["A", "volume"] == 30.0
    assert pd.isna(out.loc["B", "volume"])
    assert out.loc["B", "n_volume_valid"] == 2
    assert out.loc["B", "n_trades"] == 3
    assert out.loc["B", "n_yield_obs"] == 2
    assert out.loc["B", "yield_pct"] == 8.0
    assert out.loc["B", "price"] == 91.0
