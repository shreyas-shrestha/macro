"""walk-forward for #10 - refit once a year on PAST data only, then predict that year

simply: train on 2000-2004 -> predict 2005, train on 2000-2005 -> predict 2006, etc
model = ridge regression on log variance
the embargo (from models.schedule, same as the #9 baselines) means we only train on rows
whose answer is fully known by the refit date -> no future data leaks in
"""

import numpy as np
import pandas as pd

from macro.config import DEV_START
from macro.models.schedule import refit_schedule, training_end
from macro.models.target import ANNUALIZE, realized_variance

# less training data than this (~2 years of trading days) -> skip that year
MIN_TRAIN_ROWS = 504
# ridge penalty - small, just keeps correlated features from going crazy
ALPHA = 1.0


def baseline_features(returns: pd.Series, vix_close: pd.Series) -> pd.DataFrame:
    """the no-signal inputs every model gets: log recent RV (21 days) + log VIX²"""
    rv = (returns**2).rolling(21).mean() * ANNUALIZE
    vix_var = (vix_close / 100) ** 2
    return np.log(pd.DataFrame({"log_rv": rv, "log_vix": vix_var}))


def decision_grid(returns: pd.Series, h: int, end_date) -> pd.DatetimeIndex:
    """month-end dates we make forecasts on (only ones whose h-day answer is known)

    same rule as target.forecast_grid, but the end date can move for the holdout run
    """
    sample = returns.loc[:end_date]
    month_ends = pd.DatetimeIndex(
        sample.index.to_series().groupby(sample.index.to_period("M")).max()
    )
    complete = realized_variance(sample, h).notna()[month_ends].to_numpy()
    return month_ends[complete & (month_ends >= DEV_START)]


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float):
    """fit ridge on scaled x. out: a predict function + the leftover (residual) variance

    scaling only uses the training rows (no peeking). intercept isn't penalized
    """
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0] = 1.0  # a constant column has no info, just leave it at 0
    z = (x - mean) / std
    y_mean = y.mean()

    # ridge formula: (Z'Z + alpha*I)^-1 Z'y on centered y
    beta = np.linalg.solve(z.T @ z + alpha * np.eye(z.shape[1]), z.T @ (y - y_mean))

    def predict(x_new: np.ndarray) -> np.ndarray:
        return y_mean + ((x_new - mean) / std) @ beta

    resid_var = np.var(y - predict(x), ddof=x.shape[1] + 1)
    return predict, resid_var


def walk_forward(
    features: pd.DataFrame,
    returns: pd.Series,
    dates: pd.DatetimeIndex,
    h: int,
    alpha: float = ALPHA,
) -> pd.Series:
    """variance forecast for each decision date, refit yearly on past data only

    in: features (one row per trading day), returns, the dates to forecast, horizon h
    out: one forecast per date (rows w/ missing features get skipped)
    """
    # what we're predicting: log of realized variance over the next h days
    target = np.log(realized_variance(returns, h)).rename("y")
    data = pd.concat([features, target], axis=1)
    cols = list(features.columns)
    forecasts = []

    for refit, year_dates in refit_schedule(returns.index, dates):
        # embargo: only train on rows whose h-day answer is fully known by the refit date
        train = data.loc[: training_end(returns.index, refit, h)].dropna()
        if len(train) < MIN_TRAIN_ROWS:
            continue
        predict, resid_var = fit_ridge(train[cols].to_numpy(), train["y"].to_numpy(), alpha)

        # predict every decision date in this year (that has all its features)
        x_now = features.reindex(year_dates)
        x_now = x_now[x_now.notna().all(axis=1)]
        if x_now.empty:
            continue
        log_forecast = predict(x_now.to_numpy())
        # back to variance: exp(log) undershoots on average, + resid_var / 2 fixes that
        forecasts.append(pd.Series(np.exp(log_forecast + resid_var / 2), index=x_now.index))

    if not forecasts:
        return pd.Series(dtype=float, index=pd.DatetimeIndex([], name=dates.name))
    return pd.concat(forecasts).rename("forecast")
