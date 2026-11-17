import numpy as np
import pandas as pd
import pytest

from macro.backtest.scoring import qlike
from macro.data.calendar import nyse_sessions
from macro.models import baselines
from macro.models.target import forecast_grid

SESSIONS = nyse_sessions("1996-01-02", "2003-12-31")


def simulated_returns(seed=0):
    """GARCH(1,1) returns so the fits have something to find."""
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 2e-6, 0.08, 0.9
    var = omega / (1 - alpha - beta)
    out = np.empty(len(SESSIONS))
    for i in range(len(out)):
        out[i] = np.sqrt(var) * rng.standard_normal()
        var = omega + alpha * out[i] ** 2 + beta * var
    return pd.Series(out, index=SESSIONS)


RETURNS = simulated_returns()
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


def test_qlike_is_zero_for_a_perfect_forecast_and_penalizes_under_forecasts_more():
    realized = pd.Series([0.04])
    assert qlike(realized, realized) == 0
    assert qlike(realized / 2, realized) > qlike(realized * 2, realized) > 0
