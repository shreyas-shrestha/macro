"""scoring for #10 - how good is a vol forecast, and is "better" just luck?

everything takes variances (vol squared), always in the order (forecast, realized)
"""

import math

import numpy as np
import pandas as pd


def _clean(forecast, realized):
    """turn both into float arrays + reject bad data before we score anything"""
    forecast = np.asarray(forecast, dtype=float)
    realized = np.asarray(realized, dtype=float)

    # need the same # of rows on both sides
    if forecast.shape != realized.shape:
        raise ValueError("forecast and realized must be the same length")
    # one nan would quietly make the whole average nan
    if np.any(np.isnan(forecast)) or np.any(np.isnan(realized)):
        raise ValueError("forecast and realized must not contain NaN")
    # variance can't be 0 or negative (+ we divide by it / take the log)
    if np.any(forecast <= 0) or np.any(realized <= 0):
        raise ValueError("forecast and realized must be positive variances")

    return forecast, realized


def qlike_losses(forecast, realized):
    """qlike score for each row - 0 = perfect, bigger = worse

    guessing too LOW hurts more than guessing too high (never underestimate vol)
    """
    # simply: takes in: forecasts + what actually happened
    # out: list of scores (0=perfect, bigger=bad)
    forecast, realized = _clean(forecast, realized)
    ratio = realized / forecast
    return ratio - np.log(ratio) - 1


def qlike(forecast: pd.Series, realized: pd.Series) -> float:
    """one overall qlike score = average of the per-row scores (Patton 2011)"""
    return float(np.mean(qlike_losses(forecast, realized)))


def mse_log_vol_losses(forecast, realized):
    """backup score per row: squared gap between log vols, 0 = perfect

    unlike qlike, too low and too high (by the same factor) cost the same
    """
    forecast, realized = _clean(forecast, realized)
    # log vol = half of log variance
    return (0.5 * np.log(forecast) - 0.5 * np.log(realized)) ** 2


def r2_oos(model_losses, baseline_losses) -> float:
    """how much better than the baseline, as a fraction

    0 = same as baseline, 0.05 = 5% less error, negative = worse
    pass per-row losses from the SAME rows (works w/ qlike or mse)
    """
    model_losses = np.asarray(model_losses, dtype=float)
    baseline_losses = np.asarray(baseline_losses, dtype=float)
    if model_losses.shape != baseline_losses.shape:
        raise ValueError("model and baseline losses must cover the same rows")
    # always divide by the BASELINE - it's the yardstick
    return float(1 - model_losses.mean() / baseline_losses.mean())


def newey_west_se(x, lag: int) -> float:
    """standard error of the average of x, fixed for nearby rows being related

    our targets overlap (a 63 day window spills into the next couple monthly rows),
    so neighbors move together. ignore that = results look more certain than they are
    lag = how many rows apart can still be related
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    d = x - x.mean()

    # start w/ the plain variance
    var = d @ d / n
    # add back how much rows move w/ rows 1..lag steps away (closer = more weight)
    for k in range(1, lag + 1):
        weight = 1 - k / (lag + 1)
        var += 2 * weight * (d[k:] @ d[:-k]) / n

    return math.sqrt(var / n)


def clark_west(realized, baseline_forecast, model_forecast, lag: int):
    """luck check: does model (baseline + signal) actually beat the baseline?

    out: (t_stat, p_value) - team bar is p < 0.10
    pass values on the scale the model predicts (harness uses log variance)
    """
    y = np.asarray(realized, dtype=float)
    base = np.asarray(baseline_forecast, dtype=float)
    model = np.asarray(model_forecast, dtype=float)
    if not (y.shape == base.shape == model.shape):
        raise ValueError("realized, baseline_forecast and model_forecast must be the same length")

    # per row: baseline's error - model's error...
    # ...+ a fix, since extra features add a bit of noise even when useless,
    # which would unfairly make the bigger model look worse (Clark & West 2007)
    f = (y - base) ** 2 - ((y - model) ** 2 - (base - model) ** 2)

    # is the avg of f clearly above 0, or could it just be noise?
    se = newey_west_se(f, lag)
    if se == 0:
        raise ValueError("forecasts are identical, nothing to test")
    t_stat = f.mean() / se

    # one-sided p-value: chance of a t this big if the signal adds nothing
    p_value = 0.5 * math.erfc(t_stat / math.sqrt(2))
    return float(t_stat), float(p_value)
