import numpy as np
import pandas as pd
import pytest

from macro.data.trace import to_bond_day
from macro.features.hy_credit import (
    add_rating_class,
    aggregate_hy_ig,
    align_to_sessions,
    build_simple_features,
    expanding_zscore,
    issuer_day_spread,
)


def test_trade_frequency_does_not_overweight_bond_day():
    trades = pd.DataFrame(
        {
            "trd_exctn_dt": pd.to_datetime(["2010-01-04"] * 4),
            "cusip_id": ["A", "A", "A", "B"],
            "yld_pt": [8.0, 8.2, 20.0, 5.0],
            "rptd_pr": [100.0, 99.0, 50.0, 101.0],
        }
    )
    out = to_bond_day(trades).set_index("cusip_id")
    assert out.loc["A", "yield_pct"] == 8.2
    assert out.loc["A", "n_trades"] == 3
    assert out.loc["B", "n_trades"] == 1


def test_rating_boundary():
    frame = pd.DataFrame({"splticrm": ["BBB-", "BB+", "CCC", None]})
    out = add_rating_class(frame)
    assert list(out["rating_class"].astype("string")) == ["IG", "HY", "HY", pd.NA]


def test_equal_issuer_weighting():
    frame = pd.DataFrame(
        {
            "trd_exctn_dt": pd.to_datetime(["2010-01-04"] * 4),
            "issuer_id": ["A", "A", "B", "C"],
            "rating_class": ["HY", "HY", "HY", "IG"],
            "spread_bp": [400.0, 600.0, 300.0, 150.0],
        }
    )
    issuer = issuer_day_spread(frame)
    daily = aggregate_hy_ig(issuer)

    # A first collapses to 500; HY market median across A and B is 400.
    assert daily.loc[pd.Timestamp("2010-01-04"), "hy_spread_bp"] == 400.0
    assert daily.loc[pd.Timestamp("2010-01-04"), "hy_ig_gap_bp"] == 250.0


def test_features_use_backward_differences():
    idx = pd.bdate_range("2010-01-01", periods=80)
    daily = pd.DataFrame(
        {
            "hy_spread_bp": range(80),
            "ig_spread_bp": range(80),
            "hy_ig_gap_bp": [0.0] * 80,
        },
        index=idx,
    )
    features = build_simple_features(daily)
    assert features.iloc[20]["hy_change_20d"] == 20
    assert features.iloc[20]["hy_ig_gap_change_20d"] == 0


def _bond_day_panel():
    return pd.DataFrame(
        {
            "trd_exctn_dt": pd.to_datetime(["2010-01-04"] * 7),
            "cusip_id": ["A1", "A2", "A3", "B1", "C1", "D1", "E1"],
            "issuer_id": ["A", "A", "A", "B", "C", "D", "E"],
            "rating_class": ["HY", "HY", "HY", "HY", "IG", "IG", None],
            "spread_bp": [900.0, 950.0, 1000.0, 300.0, 100.0, 140.0, 5000.0],
        }
    )


def test_issuer_with_many_bonds_counts_once():
    issuer = issuer_day_spread(_bond_day_panel())
    assert issuer.set_index("issuer_id").loc["A", "n_bonds"] == 3
    daily = aggregate_hy_ig(issuer)
    row = daily.loc[pd.Timestamp("2010-01-04")]
    # Bond-weighted median of HY would be 950; issuer-weighted is median(950, 300) = 625.
    assert row["hy_spread_bp"] == 625.0
    assert row["ig_spread_bp"] == 120.0
    assert row["n_hy_issuers"] == 2
    assert row["n_ig_issuers"] == 2


def test_unrated_bonds_are_excluded_not_treated_as_hy():
    issuer = issuer_day_spread(_bond_day_panel())
    assert "E" not in set(issuer["issuer_id"])


def test_issuer_with_conflicting_classes_raises():
    panel = _bond_day_panel()
    panel.loc[1, "rating_class"] = "IG"
    with pytest.raises(ValueError, match="both IG and HY"):
        issuer_day_spread(panel)


def test_missing_class_column_in_aggregate_is_filled():
    issuer = issuer_day_spread(_bond_day_panel().query("rating_class == 'HY'"))
    daily = aggregate_hy_ig(issuer)
    assert daily["ig_spread_bp"].isna().all()
    assert daily["n_ig_issuers"].eq(0).all()
    assert daily["hy_ig_gap_bp"].isna().all()


def test_align_to_sessions_adds_gaps_and_drops_non_sessions():
    # 2010-01-18 is Martin Luther King Jr. Day (NYSE closed); 2010-01-19 has no data.
    daily = pd.DataFrame(
        {"hy_spread_bp": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2010-01-15", "2010-01-18", "2010-01-20"]),
    )
    aligned, dropped = align_to_sessions(daily)
    assert list(aligned.index.strftime("%Y-%m-%d")) == ["2010-01-15", "2010-01-19", "2010-01-20"]
    assert pd.isna(aligned.loc["2010-01-19", "hy_spread_bp"])
    assert list(dropped) == [pd.Timestamp("2010-01-18")]


def test_expanding_zscore_has_no_future_dependence():
    rng = np.random.default_rng(1)
    series = pd.Series(rng.normal(size=300).cumsum())
    full = expanding_zscore(series)
    for cutoff in (80, 150, 299):
        pd.testing.assert_series_equal(expanding_zscore(series.iloc[:cutoff]), full.iloc[:cutoff])


def test_features_have_no_future_dependence():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2010-01-04", periods=200, freq="B")
    hy = pd.Series(500 + rng.normal(size=200).cumsum(), index=idx)
    daily = pd.DataFrame({"hy_spread_bp": hy, "hy_ig_gap_bp": hy - 150})
    full = build_simple_features(daily)
    for cutoff in (70, 130):
        pd.testing.assert_frame_equal(
            build_simple_features(daily.iloc[:cutoff]), full.iloc[:cutoff]
        )


def test_features_reject_unsorted_index():
    daily = pd.DataFrame(
        {"hy_spread_bp": [1.0, 2.0], "hy_ig_gap_bp": [0.0, 0.0]},
        index=pd.to_datetime(["2010-01-05", "2010-01-04"]),
    )
    with pytest.raises(ValueError, match="sorted"):
        build_simple_features(daily)


# --------------------------------------------------------------------------------------
# HYCredit signal
# --------------------------------------------------------------------------------------


def _synthetic_daily(start="2006-01-03", end="2017-12-29", seed=3):
    from macro.data.calendar import nyse_sessions

    rng = np.random.default_rng(seed)
    days = nyse_sessions(start, end)
    hy = 500 + rng.normal(size=len(days)).cumsum()
    ig = 150 + 0.2 * rng.normal(size=len(days)).cumsum()
    daily = pd.DataFrame({"hy_spread_bp": hy, "ig_spread_bp": ig}, index=days)
    daily["hy_ig_gap_bp"] = daily["hy_spread_bp"] - daily["ig_spread_bp"]
    daily = daily.drop(daily.index[100:103])  # a short data gap
    return daily


def test_hy_signal_has_no_lookahead():
    from macro.features.hy_credit import HYCredit
    from macro.features.testing import assert_no_lookahead

    assert_no_lookahead(HYCredit(_synthetic_daily()), cutoffs=["2008-10-15", "2012-06-29"])


def test_hy_signal_lags_observations_by_one_session():
    from macro.features.hy_credit import HYCredit

    daily = _synthetic_daily()
    out = HYCredit(daily).run("2010-12-31")
    # Observation on Fri 2010-01-08 is usable on Mon 2010-01-11, not before.
    assert out.loc["2010-01-11", "hy_spread"] == daily.loc["2010-01-08", "hy_spread_bp"]
    assert out.loc["2010-01-08", "hy_spread"] == daily.loc["2010-01-07", "hy_spread_bp"]
    assert list(out.columns) == [
        "hy_spread",
        "hy_spread_change_20",
        "hy_spread_expanding_z",
        "hy_ig_gap",
        "hy_ig_gap_change_20",
    ]


def test_hy_signal_change_is_20_sessions():
    from macro.data.calendar import nyse_sessions
    from macro.features.hy_credit import HYCredit

    out = HYCredit(_synthetic_daily()).run("2011-12-30")
    sessions = nyse_sessions("2011-01-03", "2011-12-30")
    t, t20 = sessions[60], sessions[40]
    expected = out.loc[t, "hy_spread"] - out.loc[t20, "hy_spread"]
    assert out.loc[t, "hy_spread_change_20"] == pytest.approx(expected)


def test_non_session_observation_rolls_to_next_session():
    from macro.features.hy_credit import to_decision_dates

    # Good Friday 2010-04-02: NYSE closed. Thursday's and Friday's rows both map to Monday;
    # the later observation wins.
    daily = pd.DataFrame(
        {"hy_spread_bp": [1.0, 2.0], "hy_ig_gap_bp": [0.0, 0.0]},
        index=pd.to_datetime(["2010-04-01", "2010-04-02"]),
    )
    out = to_decision_dates(daily)
    assert list(out.index) == [pd.Timestamp("2010-04-05")]
    assert out["hy_spread_bp"].iloc[0] == 2.0
