"""Baseline forecasts of the target, each made with data through the decision date only.

Every function returns annualized variance forecasts indexed by the decision dates it is
given. HAR and GARCH refit yearly on an expanding window (see ``schedule``).
"""

import numpy as np
import pandas as pd
from arch import arch_model

from macro.models.schedule import refit_schedule, training_end
from macro.models.target import ANNUALIZE, realized_variance

# Unchanged S&P 500 closes (5 since 1990) would make the one-day log variance -inf.
MIN_DAILY_VARIANCE = 1e-10


def trailing_rv(returns: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """Annualized realized variance over the 21 sessions ending at each date."""
    return ((returns**2).rolling(21).mean() * ANNUALIZE).loc[dates]


def vix(vix_close: pd.Series, dates: pd.DatetimeIndex) -> pd.Series:
    """VIX squared as annualized variance.

    Raw 30-day implied variance, used unchanged for every horizon; it carries the
    variance risk premium, so it runs above realized variance on average.
    """
    return ((vix_close / 100) ** 2).loc[dates]


def har_features(returns: pd.Series) -> pd.DataFrame:
    """Log annualized realized variance over the last 1, 5 and 22 sessions (Corsi 2009)."""
    daily = (returns**2).clip(lower=MIN_DAILY_VARIANCE) * ANNUALIZE
    return np.log(
        pd.DataFrame({"d": daily, "w": daily.rolling(5).mean(), "m": daily.rolling(22).mean()})
    )


def har(returns: pd.Series, dates: pd.DatetimeIndex, h: int) -> pd.Series:
    """HAR-RV on log variance, fit by OLS, with a lognormal bias correction."""
    features = har_features(returns)
    target = np.log(realized_variance(returns, h))
    forecasts = []
    for refit, year_dates in refit_schedule(returns.index, dates):
        train_end = training_end(returns.index, refit, h)
        train = pd.concat([features, target.rename("y")], axis=1).loc[:train_end].dropna()
        x = np.column_stack([np.ones(len(train)), train[features.columns]])
        coef, *_ = np.linalg.lstsq(x, train["y"], rcond=None)
        resid_var = np.var(train["y"] - x @ coef, ddof=x.shape[1])
        x_now = np.column_stack([np.ones(len(year_dates)), features.loc[year_dates]])
        forecasts.append(pd.Series(np.exp(x_now @ coef + resid_var / 2), index=year_dates))
    return pd.concat(forecasts)


def garch(returns: pd.Series, dates: pd.DatetimeIndex, h: int) -> pd.Series:
    """GARCH(1,1) with constant mean and normal errors, summed over h steps.

    Each step forecasts E[r**2] = variance + mean**2, matching the undemeaned target.
    """
    scaled = returns * 100  # arch's optimizer is better conditioned on percent returns
    forecasts = []
    for refit, year_dates in refit_schedule(returns.index, dates):
        fit = arch_model(scaled.loc[:refit]).fit(disp="off")
        if fit.convergence_flag != 0:
            raise RuntimeError(f"GARCH fit through {refit.date()} did not converge")
        params = fit.params
        # Fixed parameters filter the conditional variance forward one return at a time,
        # so the forecast at t depends only on returns through t.
        fixed = arch_model(scaled.loc[: year_dates[-1]]).fix(params)
        variance = fixed.forecast(horizon=h, start=year_dates[0], reindex=False).variance
        second_moment = variance.loc[year_dates].sum(axis=1) + h * params["mu"] ** 2
        forecasts.append(second_moment / 100**2 * ANNUALIZE / h)
    return pd.concat(forecasts)
