import numpy as np
import pytest

from macro.backtest.scoring import (
    clark_west,
    mse_log_vol_losses,
    newey_west_se,
    qlike,
    qlike_losses,
    r2_oos,
)


def test_nan_raises():
    with pytest.raises(ValueError):
        qlike_losses(np.array([1, 2, 3]), np.array([1, 2, np.nan]))
    with pytest.raises(ValueError):
        qlike_losses(np.array([1, np.nan, 3]), np.array([1, 2, 3]))


def test_nonpositive_raises():
    with pytest.raises(ValueError):
        qlike_losses([0.04, 0.0], [0.04, 0.04])


def test_perfect_forecast_is_zero():
    realized = np.array([0.04, 0.09, 0.01])
    assert np.allclose(qlike_losses(realized, realized), 0)


def test_too_low_worse_than_too_high():
    too_low = qlike_losses([0.02], [0.04])[0]  # about 0.31
    too_high = qlike_losses([0.08], [0.04])[0]  # about 0.19
    assert too_low > too_high


def test_qlike_by_hand():
    # Ratios 2 and 1/2: (2 - ln 2 - 1) and (1/2 + ln 2 - 1) average to 1/4.
    assert qlike([0.02, 0.08], [0.04, 0.04]) == pytest.approx(0.25)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        qlike_losses([0.04, 0.04], [0.04])


def test_mse_log_vol_is_symmetric():
    assert np.allclose(mse_log_vol_losses([0.04], [0.04]), 0)
    half = mse_log_vol_losses([0.02], [0.04])[0]
    double = mse_log_vol_losses([0.08], [0.04])[0]
    assert half == pytest.approx(double)


def test_r2_oos_hand_cases():
    base = [0.25, 0.25]
    assert r2_oos([0.25, 0.25], base) == pytest.approx(0)
    assert r2_oos([0.20, 0.20], base) == pytest.approx(0.20)
    assert r2_oos([0.30, 0.30], base) == pytest.approx(-0.20)


def test_newey_west_lag_zero_matches_plain_se():
    x = np.array([1.0, 2.0, 4.0, 7.0])
    plain = x.std() / np.sqrt(len(x))
    assert newey_west_se(x, 0) == pytest.approx(plain)


def test_clark_west_separates_signal_from_noise():
    rng = np.random.default_rng(0)
    n = 300
    signal = rng.normal(size=n)
    y = signal + rng.normal(size=n)
    baseline = np.zeros(n)
    good = 0.8 * signal
    noise = 0.3 * rng.normal(size=n)
    _, p_good = clark_west(y, baseline, good, lag=2)
    _, p_noise = clark_west(y, baseline, noise, lag=2)
    assert p_good < 0.01
    assert p_noise > 0.10


def test_clark_west_identical_forecasts_raises():
    y = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        clark_west(y, y * 0, y * 0, lag=1)


def test_newey_west_lag_one_by_hand():
    # d = x - 3.5 = [-2.5, -1.5, 0.5, 3.5]; gamma0 = 21 / 4, gamma1 = 4.75 / 4, weight 1/2.
    x = np.array([1.0, 2.0, 4.0, 7.0])
    assert newey_west_se(x, 1) == pytest.approx(np.sqrt((21 / 4 + 2 * 0.5 * 4.75 / 4) / 4))
