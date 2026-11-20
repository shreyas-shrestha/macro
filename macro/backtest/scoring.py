"""Forecast losses and tests of forecast improvement (#10).

Loss functions take variances in the order (forecast, realized).
"""

import math

import numpy as np
import pandas as pd


def _clean(forecast, realized):
    """Float arrays of equal shape, positive and without NaN; raise ``ValueError`` otherwise."""
    forecast = np.asarray(forecast, dtype=float)
    realized = np.asarray(realized, dtype=float)
    if forecast.shape != realized.shape:
        raise ValueError("forecast and realized must be the same length")
    # A single NaN would make the mean NaN without saying why.
    if np.any(np.isnan(forecast)) or np.any(np.isnan(realized)):
        raise ValueError("forecast and realized must not contain NaN")
    if np.any(forecast <= 0) or np.any(realized <= 0):
        raise ValueError("forecast and realized must be positive variances")
    return forecast, realized


def qlike_losses(forecast, realized) -> np.ndarray:
    """Per-row QLIKE loss (Patton 2011); 0 for a perfect forecast.

    Under-forecasts cost more than over-forecasts by the same factor.
    """
    forecast, realized = _clean(forecast, realized)
    ratio = realized / forecast
    return ratio - np.log(ratio) - 1


def qlike(forecast: pd.Series, realized: pd.Series) -> float:
    """Mean QLIKE loss of variance forecasts."""
    return float(np.mean(qlike_losses(forecast, realized)))


def mse_log_vol_losses(forecast, realized) -> np.ndarray:
    """Per-row squared error of log volatility; symmetric in the ratio of forecast to realized."""
    forecast, realized = _clean(forecast, realized)
    return (0.5 * np.log(forecast) - 0.5 * np.log(realized)) ** 2


def r2_oos(model_losses, baseline_losses) -> float:
    """1 - mean model loss / mean baseline loss, over the same rows; positive is better."""
    model_losses = np.asarray(model_losses, dtype=float)
    baseline_losses = np.asarray(baseline_losses, dtype=float)
    if model_losses.shape != baseline_losses.shape:
        raise ValueError("model and baseline losses must cover the same rows")
    return float(1 - model_losses.mean() / baseline_losses.mean())


def newey_west_se(x, lag: int) -> float:
    """Standard error of the mean of ``x`` with Bartlett-weighted autocovariances to ``lag``.

    Targets of nearby decision dates overlap, so their errors are correlated.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    d = x - x.mean()
    var = d @ d / n
    for k in range(1, lag + 1):
        var += 2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k]) / n
    return math.sqrt(var / n)


def clark_west(realized, baseline_forecast, model_forecast, lag: int) -> tuple[float, float]:
    """Clark–West (2007) test that the larger model improves on the nested baseline.

    Compares squared errors after adjusting for the noise of estimating extra
    parameters. Pass values on the scale the models predict (the harness uses log
    variance). Returns the t-statistic and its one-sided normal p-value.
    """
    y = np.asarray(realized, dtype=float)
    base = np.asarray(baseline_forecast, dtype=float)
    model = np.asarray(model_forecast, dtype=float)
    if not (y.shape == base.shape == model.shape):
        raise ValueError("realized, baseline_forecast and model_forecast must be the same length")
    f = (y - base) ** 2 - ((y - model) ** 2 - (base - model) ** 2)
    se = newey_west_se(f, lag)
    if se == 0:
        raise ValueError("forecasts are identical, nothing to test")
    t_stat = f.mean() / se
    return float(t_stat), float(0.5 * math.erfc(t_stat / math.sqrt(2)))
