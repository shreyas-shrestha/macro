import numpy as np
import pandas as pd
import pytest

from macro.backtest.report import breakdown, compare, nw_lag, update_scorecard
from macro.backtest.scoring import clark_west
from macro.backtest.walkforward import fit_ridge, walk_forward
from macro.config import HOLDOUT_START
from macro.data.calendar import nyse_sessions
from macro.features.noise import Noise
from macro.features.testing import assert_no_lookahead
from macro.models import baselines
from macro.models.target import forecast_grid


def synthetic_returns(start="1995-01-03", end="2004-12-31", seed=0) -> pd.Series:
    """Daily returns with slowly varying volatility, so there is something to forecast."""
    sessions = nyse_sessions(start, end)
    rng = np.random.default_rng(seed)
    vol = 0.01 * (1 + 0.5 * np.sin(np.arange(len(sessions)) / 150))
    return pd.Series(rng.normal(size=len(sessions)) * vol, index=sessions)


def frames(log: np.ndarray, shift: float, index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"log": log, "variance": np.exp(log + shift)}, index=index)


def test_ridge_with_no_penalty_reproduces_har_baseline():
    # Same refit dates, embargo and bias correction as the #9 HAR baseline.
    returns = synthetic_returns()
    for h in (5, 21):
        grid = forecast_grid(returns, h)
        expected = baselines.har(returns, grid, h)
        ours = walk_forward(baselines.har_features(returns), returns, grid, h, alpha=0.0)
        pd.testing.assert_series_equal(ours["variance"], expected, check_names=False, rtol=1e-8)


def test_log_column_is_the_forecast_before_bias_correction():
    returns = synthetic_returns()
    grid = forecast_grid(returns, 21)
    out = walk_forward(baselines.har_features(returns), returns, grid, 21)
    shift = np.log(out["variance"]) - out["log"]
    # One residual variance per yearly refit, positive.
    assert (shift > 0).all()
    assert shift.round(12).groupby(shift.index.year).nunique().eq(1).all()


def test_no_lookahead_in_forecasts():
    returns = synthetic_returns()
    shocked = returns.where(returns.index <= "2002-12-31", returns * 3)
    grid = forecast_grid(returns, 21)

    def run(r):
        return walk_forward(baselines.har_features(r), r, grid, 21)

    pd.testing.assert_frame_equal(run(returns).loc[:"2002"], run(shocked).loc[:"2002"])


def test_short_training_years_are_skipped_with_a_warning():
    returns = synthetic_returns()
    features = baselines.har_features(returns).loc["2001":]
    grid = forecast_grid(returns, 21)
    # 2000 to 2003 are skipped; the 2003 refit has 479 rows through the end of 2002.
    with pytest.warns(UserWarning, match="skipping 2003, 479 training rows"):
        out = walk_forward(features, returns, grid, 21)
    assert out.index[0].year == 2004


def test_holdout_grid_starts_in_2018_and_ends_with_the_targets():
    # Returns run past ``end``; the grid must still stop with targets ending by it.
    returns = synthetic_returns("2015-01-02", "2019-12-31")
    grid = forecast_grid(returns, 21, HOLDOUT_START, "2019-06-28")
    assert grid[0] == pd.Timestamp("2018-01-31")
    # June 2019 has 20 sessions, so a 21-session target from 2019-05-31 ends in July.
    assert grid[-1] == pd.Timestamp("2019-04-30")


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
    log_target = rng.normal(-3, 0.5, size=120)
    target = pd.Series(np.exp(log_target), index=dates)
    baseline = frames(np.full(120, -3.0), 0.0, dates)
    model = frames(0.7 * log_target + 0.3 * -3.0, 0.0, dates)
    row = compare(target, baseline, model, h=21)
    assert row["n"] == 120
    assert row["qlike_reduction"] > 0.05
    assert row["cw_p"] < 0.10


def test_clark_west_ignores_the_bias_correction():
    # Different residual variances shift each model's variance by a different constant.
    # Clark–West must see only the log forecasts.
    dates = pd.date_range("2000-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(3)
    log_target = rng.normal(-3, 0.5, size=120)
    target = pd.Series(np.exp(log_target), index=dates)
    base_log = np.full(120, -3.0)
    model_log = 0.9 * base_log + 0.1 * log_target + rng.normal(0, 0.1, size=120)
    row = compare(target, frames(base_log, 0.13, dates), frames(model_log, 0.05, dates), h=21)
    expected = clark_west(log_target, base_log, model_log, nw_lag(21))
    assert (row["cw_t"], row["cw_p"]) == pytest.approx(expected)


def test_breakdown_counts_forecasts_per_year_and_episode():
    dates = pd.date_range("2008-01-31", "2009-12-31", freq="ME")
    target = pd.Series(0.04, index=dates)
    flat = frames(np.full(len(dates), np.log(0.04)), 0.0, dates)
    table = breakdown(target, flat, flat)
    assert table.loc["2008", "n"] == 12
    assert table.loc["2008 crisis", "n"] == 7  # 2008-09 through 2009-03


def test_nw_lag_counts_overlapping_monthly_rows():
    assert [nw_lag(h) for h in (5, 21, 63)] == [1, 1, 3]


def test_update_scorecard_replaces_only_that_signal(tmp_path):
    path = tmp_path / "scorecard.csv"
    update_scorecard(path, "a", [{"horizon": 21, "qlike_reduction": 0.01}])
    update_scorecard(path, "b", [{"horizon": 21, "qlike_reduction": 0.02}])
    table = update_scorecard(path, "a", [{"horizon": 21, "qlike_reduction": 0.03}])
    assert sorted(table["signal"]) == ["a", "b"]
    assert table.loc[table["signal"] == "a", "qlike_reduction"].item() == 0.03
