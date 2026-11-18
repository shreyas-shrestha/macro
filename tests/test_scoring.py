import pandas as pd

from macro.backtest.scoring import qlike


def test_qlike_is_zero_for_a_perfect_forecast_and_penalizes_under_forecasts_more():
    realized = pd.Series([0.04])
    assert qlike(realized, realized) == 0
    assert qlike(realized / 2, realized) > qlike(realized * 2, realized) > 0
