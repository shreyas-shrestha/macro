import numpy as np
import pandas as pd
import pytest

from macro.data.calendar import nyse_sessions
from macro.models.target import ANNUALIZE, forecast_grid, log_returns, realized_variance


def test_realized_variance_matches_hand_computation():
    returns = pd.Series([0.01, -0.02, 0.03, 0.04], index=nyse_sessions("2010-01-04", "2010-01-07"))
    rv = realized_variance(returns, h=2)
    assert rv.iloc[0] == pytest.approx(ANNUALIZE / 2 * (0.02**2 + 0.03**2))
    assert rv.iloc[1] == pytest.approx(ANNUALIZE / 2 * (0.03**2 + 0.04**2))
    assert rv.iloc[2:].isna().all()


def test_target_excludes_the_decision_day_return():
    returns = pd.Series(0.01, index=nyse_sessions("2010-01-04", "2010-02-26"))
    bumped = returns.copy()
    bumped.iloc[10] = 0.05
    before, after = realized_variance(returns, 5), realized_variance(bumped, 5)
    assert after.iloc[10] == before.iloc[10]
    assert after.iloc[9] != before.iloc[9]


def test_log_returns_are_indexed_by_the_session_they_end_on():
    close = pd.Series([100.0, 110.0], index=nyse_sessions("2010-01-04", "2010-01-05"))
    returns = log_returns(close)
    assert list(returns.index) == [pd.Timestamp("2010-01-05")]
    assert returns.iloc[0] == pytest.approx(np.log(1.1))


def test_missing_close_raises():
    close = pd.Series([100.0, np.nan, 101.0], index=nyse_sessions("2010-01-04", "2010-01-06"))
    with pytest.raises(ValueError, match="missing closes"):
        log_returns(close)


@pytest.mark.parametrize("h, last", [(5, "2017-11-30"), (21, "2017-10-31"), (63, "2017-09-29")])
def test_grid_is_month_ends_whose_targets_end_by_dev_end(h, last):
    sessions = nyse_sessions("1999-06-01", "2018-06-29")
    grid = forecast_grid(pd.Series(0.01, index=sessions), h)
    assert grid[0] == pd.Timestamp("2000-01-31")
    assert grid[-1] == pd.Timestamp(last)
    month_ends = sessions.to_series().groupby(sessions.to_period("M")).max()
    assert grid.isin(month_ends).all()
    assert all(sessions[sessions.get_loc(t) + h] <= pd.Timestamp("2017-12-31") for t in grid)
