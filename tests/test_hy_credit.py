import pandas as pd

from macro.features.hy_credit import (
    add_rating_class,
    aggregate_hy_ig,
    build_simple_features,
    issuer_day_spread,
)
from macro.data.wrds import to_bond_day


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
