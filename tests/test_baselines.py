import numpy as np
import pandas as pd
import pytest

from macro.data.calendar import nyse_sessions
from macro.models import baselines
from macro.models.target import forecast_grid, realized_variance

SESSIONS = nyse_sessions("1996-01-02", "2003-12-31")


OMEGA, ALPHA, BETA = 2e-6, 0.08, 0.9


def simulated_returns(seed=0):
    """GARCH(1,1) returns, and each session's true variance for the next return."""
    rng = np.random.default_rng(seed)
    var = OMEGA / (1 - ALPHA - BETA)
    returns, next_var = np.empty(len(SESSIONS)), np.empty(len(SESSIONS))
    for i in range(len(SESSIONS)):
        returns[i] = np.sqrt(var) * rng.standard_normal()
        var = OMEGA + ALPHA * returns[i] ** 2 + BETA * var
        next_var[i] = var
    return pd.Series(returns, index=SESSIONS), pd.Series(next_var, index=SESSIONS)


RETURNS, NEXT_VARIANCE = simulated_returns()
GRID = forecast_grid(RETURNS, 21)


def test_trailing_rv_matches_hand_computation():
    t = GRID[5]
    window = RETURNS.loc[:t].iloc[-21:]
    assert baselines.trailing_rv(RETURNS, GRID)[t] == pytest.approx(252 * np.mean(window**2))


def test_vix_is_squared_and_scaled():
    vix_close = pd.Series(20.0, index=SESSIONS)
    assert baselines.vix(vix_close, GRID).to_numpy() == pytest.approx(0.04)


@pytest.mark.parametrize("model", [baselines.har, baselines.garch])
def test_forecasts_cover_the_grid(model):
    out = model(RETURNS, GRID, 21)
    assert out.index.equals(GRID)
    assert out.notna().all() and (out > 0).all()


@pytest.mark.parametrize("model", [baselines.har, baselines.garch])
@pytest.mark.parametrize("t", ["2001-01-31", "2002-06-28"])
def test_forecasts_up_to_t_ignore_returns_after_t(model, t):
    full = model(RETURNS, GRID, 21)
    dates = GRID[GRID <= t]
    pd.testing.assert_series_equal(model(RETURNS.loc[:t], dates, 21), full.loc[:t])


def test_har_coefficients_use_only_targets_ending_by_the_refit():
    # January 2002 returns are outside the 22-session features at the end of April and
    # after every training target, which must end by the last 2001 session.
    shocked = RETURNS.copy()
    shocked.loc["2002-01"] *= 5
    t = pd.Timestamp("2002-04-30")
    dates = pd.DatetimeIndex([t])
    assert baselines.har(shocked, dates, 5)[t] == baselines.har(RETURNS, dates, 5)[t]


def test_garch_matches_the_true_h_step_variance():
    persistence = ALPHA + BETA
    long_run = OMEGA / (1 - persistence)
    steps = np.arange(21)
    expected = [
        252 / 21 * np.sum(long_run + persistence**steps * (NEXT_VARIANCE[t] - long_run))
        for t in GRID
    ]
    # Fitted parameters differ from the true ones; the forecast scale must not.
    assert baselines.garch(RETURNS, GRID, 21).to_numpy() == pytest.approx(expected, rel=0.2)


def test_har_forecasts_are_unbiased_in_level():
    target = realized_variance(RETURNS, 21).loc[GRID]
    assert baselines.har(RETURNS, GRID, 21).mean() / target.mean() == pytest.approx(1, abs=0.2)
