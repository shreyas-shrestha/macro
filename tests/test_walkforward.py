import numpy as np
import pandas as pd
import pytest

from macro.backtest.report import compare, nw_lag, update_scorecard
from macro.backtest.walkforward import decision_grid, fit_ridge, walk_forward
from macro.config import DEV_END
from macro.data.calendar import nyse_sessions
from macro.features.noise import Noise
from macro.features.testing import assert_no_lookahead
from macro.models import baselines
from macro.models.target import forecast_grid


def synthetic_returns(seed: int = 0) -> pd.Series:
    """fake daily returns 1995-2004 w/ slowly changing vol (so there's something to
    forecast). no internet needed"""
    sessions = nyse_sessions("1995-01-03", "2004-12-31")
    rng = np.random.default_rng(seed)
    vol = 0.01 * (1 + 0.5 * np.sin(np.arange(len(sessions)) / 150))
    return pd.Series(rng.normal(size=len(sessions)) * vol, index=sessions)


# w/ no penalty, our ridge on the HAR inputs should give EXACTLY shreyas's #9 HAR
# baseline -> proves same refit dates, same embargo, same bias fix
def test_ridge_with_no_penalty_reproduces_har_baseline():
    returns = synthetic_returns()
    for h in (5, 21):
        grid = forecast_grid(returns, h)
        expected = baselines.har(returns, grid, h)
        ours = walk_forward(baselines.har_features(returns), returns, grid, h, alpha=0.0)
        pd.testing.assert_series_equal(ours, expected, check_names=False, rtol=1e-8)


# messing w/ returns after 2002 should NOT change any 2002-or-earlier forecast
def test_no_lookahead_in_forecasts():
    returns = synthetic_returns()
    later = returns.index > "2002-12-31"
    shocked = returns.where(~later, returns * 3)
    grid = forecast_grid(returns, 21)

    def run(r):
        return walk_forward(baselines.har_features(r), r, grid, 21)

    before = run(returns).loc[:"2002-12-31"]
    after = run(shocked).loc[:"2002-12-31"]
    pd.testing.assert_series_equal(before, after)


def test_decision_grid_matches_target_grid_in_development():
    returns = synthetic_returns()
    for h in (5, 21, 63):
        assert decision_grid(returns, h, DEV_END).equals(forecast_grid(returns, h))


# ridge w/ alpha = 0 is just plain least squares
def test_fit_ridge_alpha_zero_is_ols():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(200, 2))
    y = 1.0 + x @ np.array([0.5, -2.0]) + 0.1 * rng.normal(size=200)
    predict, _ = fit_ridge(x, y, alpha=0.0)
    ols, *_ = np.linalg.lstsq(np.column_stack([np.ones(200), x]), y, rcond=None)
    x_new = np.array([[0.3, -1.0]])
    assert predict(x_new)[0] == pytest.approx(ols[0] + x_new[0] @ ols[1:])


def test_noise_signal_has_no_lookahead():
    assert_no_lookahead(Noise(), ["2005-06-30", "2012-12-31"], end_date="2017-12-29")


def test_compare_reports_improvement_for_a_better_model():
    dates = pd.date_range("2000-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(2)
    target = pd.Series(np.exp(rng.normal(-3, 0.5, size=120)), index=dates)
    baseline = pd.Series(np.exp(-3.0), index=dates)
    model = np.exp(0.7 * np.log(target) + 0.3 * -3.0)
    row = compare(target, baseline, model, h=21)
    assert row["n"] == 120
    assert row["qlike_reduction"] > 0.05
    assert row["cw_p"] < 0.10


def test_nw_lag_counts_overlapping_monthly_rows():
    assert [nw_lag(h) for h in (5, 21, 63)] == [1, 1, 3]


# rerunning a signal swaps its rows + leaves other signals alone
def test_update_scorecard_replaces_only_that_signal(tmp_path):
    path = tmp_path / "scorecard.csv"
    update_scorecard(path, "a", [{"horizon": 21, "qlike_reduction": 0.01}])
    update_scorecard(path, "b", [{"horizon": 21, "qlike_reduction": 0.02}])
    table = update_scorecard(path, "a", [{"horizon": 21, "qlike_reduction": 0.03}])
    assert sorted(table["signal"]) == ["a", "b"]
    assert table.loc[table["signal"] == "a", "qlike_reduction"].item() == 0.03
