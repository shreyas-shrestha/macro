"""Walk-forward ridge forecasts of log realized variance (#10).

Refits yearly on an expanding window with the refit dates and embargo of
``models.schedule``, the same as the #9 baselines, so training targets end by the refit
session.
"""

import warnings

import numpy as np
import pandas as pd

from macro.models import baselines
from macro.models.schedule import refit_schedule, training_end
from macro.models.target import realized_variance

# A refit with fewer training rows (about two years of sessions) is skipped.
MIN_TRAIN_ROWS = 504
# Not tuned. Shrinks standardized slopes by roughly ALPHA / n.
ALPHA = 1.0


BASELINE_COLUMNS = ["log_rv", "log_vix"]


def baseline_features(returns: pd.Series, vix_close: pd.Series) -> pd.DataFrame:
    """Log trailing 21-session realized variance and log VIX squared, on every session."""
    sessions = returns.index
    rv, vix = baselines.trailing_rv(returns, sessions), baselines.vix(vix_close, sessions)
    return np.log(pd.DataFrame(dict(zip(BASELINE_COLUMNS, (rv, vix), strict=True))))


def fit_ridge(x: np.ndarray, y: np.ndarray, alpha: float):
    """Ridge on features standardized with the training rows; the intercept is unpenalized.

    Returns a predict function and the residual variance of the fit.
    """
    mean, std = x.mean(axis=0), x.std(axis=0)
    std[std == 0] = 1.0  # a constant column standardizes to zero and gets no weight
    z = (x - mean) / std
    y_mean = y.mean()
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
) -> pd.DataFrame:
    """Forecasts at ``dates`` with every feature present.

    Columns: ``log``, the ridge prediction of log realized variance over the next ``h``
    sessions, and ``variance``, its lognormal bias-corrected level
    ``exp(log + resid_var / 2)``. Scores on the log scale (Clark–West) use ``log``.
    """
    target = np.log(realized_variance(returns, h)).rename("y")
    data = pd.concat([features, target], axis=1, sort=True)
    cols = list(features.columns)
    forecasts = []
    for refit, year_dates in refit_schedule(returns.index, dates):
        train = data.loc[: training_end(returns.index, refit, h)].dropna()
        if len(train) < MIN_TRAIN_ROWS:
            warnings.warn(
                f"h={h}: skipping {year_dates[0].year}, {len(train)} training rows "
                f"through {refit.date()} (minimum {MIN_TRAIN_ROWS})",
                stacklevel=2,
            )
            continue
        predict, resid_var = fit_ridge(train[cols].to_numpy(), train["y"].to_numpy(), alpha)
        x_now = features.reindex(year_dates).dropna()
        if x_now.empty:
            continue
        log = predict(x_now.to_numpy())
        forecasts.append(
            pd.DataFrame({"log": log, "variance": np.exp(log + resid_var / 2)}, index=x_now.index)
        )
    if not forecasts:
        return pd.DataFrame(columns=["log", "variance"], index=dates[:0], dtype=float)
    return pd.concat(forecasts)
