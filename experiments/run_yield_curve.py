"""Yield-curve signal against the RV + VIX benchmark (#12).

A stand-in until the shared harness (#10) merges; rerun there before drawing conclusions.

Benchmark: OLS of log realized variance over t+1 ... t+h on the HAR terms (log RV over
the last 1, 5 and 22 sessions) and log VIX squared. Augmented: the same plus the four
yield-curve features. Both refit yearly on an expanding window (``macro.models.schedule``)
over the same rows, forecast on the month-end grid, and are scored by QLIKE. Clark-West
tests whether the augmented model's gain in log-variance error is more than noise.

Training starts in 2000, where the features start, so forecasts start in 2004. Writes
results/yield_curve.csv (decision_date, horizon, model, forecast, target) and prints one
row per horizon.

    uv run python experiments/run_yield_curve.py
"""

from math import ceil, erfc, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from macro.backtest.scoring import qlike
from macro.config import DEV_END
from macro.data.series import load
from macro.features.yield_curve import YieldCurve
from macro.models.baselines import har_features
from macro.models.schedule import refit_schedule, training_end
from macro.models.target import HORIZONS, forecast_grid, log_returns, realized_variance

ESTIMATION_START = "1990-01-02"
FIRST_FORECAST_YEAR = 2004
GRID_SPACING = 21  # sessions between month-end decision dates, roughly
OUT = Path(__file__).resolve().parents[1] / "results" / "yield_curve.csv"


def benchmark_features(returns: pd.Series, vix_close: pd.Series) -> pd.DataFrame:
    """HAR terms plus log VIX squared, all in log annualized variance."""
    features = har_features(returns)
    features["vix"] = np.log((vix_close / 100) ** 2)
    return features


def _fit_predict(x: pd.DataFrame, y: pd.Series, x_now: pd.DataFrame) -> np.ndarray:
    """OLS of ``y`` on ``x``; log variance forecasts for the rows of ``x_now``.

    Adds half the residual variance, the lognormal correction ``baselines.har`` uses.
    """
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    resid_var = np.var(y - design @ coef, ddof=design.shape[1])
    return np.column_stack([np.ones(len(x_now)), x_now]) @ coef + resid_var / 2


def walk_forward(
    returns: pd.Series, vix_close: pd.Series, signal: pd.DataFrame, h: int
) -> pd.DataFrame:
    """Target and both models' variance forecasts on the decision grid for horizon ``h``."""
    bench = benchmark_features(returns, vix_close)
    both = bench.join(signal)
    log_target = np.log(realized_variance(returns, h)).rename("y")
    grid = forecast_grid(returns, h)
    grid = grid[grid.year >= FIRST_FORECAST_YEAR]
    parts = []
    for refit, year_dates in refit_schedule(returns.index, grid):
        train_end = training_end(returns.index, refit, h)
        # One set of rows for both models, so the only difference is the extra columns.
        train = both.join(log_target).loc[:train_end].dropna()
        now = both.loc[year_dates]
        parts.append(
            pd.DataFrame(
                {
                    "benchmark": _fit_predict(train[bench.columns], train["y"], now[bench.columns]),
                    "augmented": _fit_predict(train[both.columns], train["y"], now),
                },
                index=year_dates,
            )
        )
    out = np.exp(pd.concat(parts))
    out.insert(0, "target", realized_variance(returns, h).loc[out.index])
    return out.dropna()


def clark_west(target: pd.Series, small: pd.Series, large: pd.Series, lag: int) -> tuple:
    """Clark-West (2007) statistic and one-sided p-value that ``large`` beats ``small``.

    Inputs are in the units the models were fit in (log variance). The variance of the
    adjusted loss difference is Newey-West with ``lag`` lags.
    """
    f = (target - small) ** 2 - (target - large) ** 2 + (small - large) ** 2
    d = (f - f.mean()).to_numpy()
    n = len(d)
    long_run = d @ d / n
    for k in range(1, min(lag, n - 1) + 1):
        long_run += 2 * (1 - k / (lag + 1)) * (d[k:] @ d[:-k]) / n
    stat = f.mean() / sqrt(long_run / n)
    return stat, 0.5 * erfc(stat / sqrt(2))


def evaluate(returns: pd.Series, vix_close: pd.Series, signal: pd.DataFrame):
    """Return (summary with one row per horizon, long table of forecasts)."""
    summary, forecasts = [], []
    for h in HORIZONS:
        result = walk_forward(returns, vix_close, signal, h)
        log_result = np.log(result)
        stat, p_value = clark_west(
            log_result["target"],
            log_result["benchmark"],
            log_result["augmented"],
            lag=ceil(h / GRID_SPACING),
        )
        bench = qlike(result["benchmark"], result["target"])
        augmented = qlike(result["augmented"], result["target"])
        summary.append(
            {
                "horizon": h,
                "n": len(result),
                "qlike_benchmark": bench,
                "qlike_augmented": augmented,
                "improvement_pct": 100 * (bench - augmented) / bench,
                "cw_stat": stat,
                "cw_p": p_value,
            }
        )
        long = result.melt(id_vars="target", var_name="model", value_name="forecast")
        long.index = np.tile(result.index, 2)
        forecasts.append(long.assign(horizon=h))
    table = pd.concat(forecasts).rename_axis("decision_date").reset_index()
    columns = ["decision_date", "horizon", "model", "forecast", "target"]
    return pd.DataFrame(summary).set_index("horizon"), table[columns]


def main() -> None:
    returns = log_returns(load("SPX")).loc[ESTIMATION_START:]
    summary, table = evaluate(returns, load("VIX"), YieldCurve().run(DEV_END))
    table.to_csv(OUT, index=False, float_format="%.8g")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
