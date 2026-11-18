"""Provisional walk-forward evaluation of the slow macro signals (#14).

A stand-in until the shared harness (#10) lands, following its spec: ridge on log
realized variance; baseline features log trailing 21-session RV and log VIX squared;
scaling fit on training rows only; lognormal bias correction back to variance; yearly
expanding refit with an embargo of h sessions; month-end forecast grid. In each block of
configs/slow_macro.yaml, the baseline and the baseline plus the block's signals train
and score on the same rows: sessions where every feature of both models exists.

Writes results/forecasts/slow_macro_h{h}.csv and prints, per block and horizon, QLIKE of
both models, the QLIKE gain, R²_OOS on log variance and the one-sided Clark–West p-value.

    uv run python experiments/run_slow_macro.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml

from macro.backtest.scoring import qlike
from macro.config import DEV_END, DEV_START
from macro.data.series import load
from macro.features import slow_macro
from macro.models import baselines
from macro.models.schedule import refit_schedule, training_end
from macro.models.target import forecast_grid, log_returns, realized_variance

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "slow_macro.yaml"
OUT = ROOT / "results" / "forecasts"
# As in run_baselines.py: estimation starts with the VIX history.
ESTIMATION_START = "1990-01-02"


def baseline_features(returns: pd.Series, vix_close: pd.Series) -> pd.DataFrame:
    sessions = returns.index
    return np.log(
        pd.DataFrame(
            {
                "log_rv": baselines.trailing_rv(returns, sessions),
                "log_vix": baselines.vix(vix_close, sessions),
            }
        )
    )


def ridge(
    features: pd.DataFrame,
    log_target: pd.Series,
    sessions: pd.DatetimeIndex,
    dates: pd.DatetimeIndex,
    h: int,
    alpha: float,
) -> pd.DataFrame:
    """Ridge forecasts at ``dates``: ``log`` (log variance) and ``variance`` (bias corrected).

    Refits each year on rows with every feature and a target ending by the refit session.
    Features are standardized with training means and deviations; the intercept is not
    penalized.
    """
    rows = features.assign(y=log_target).dropna()
    penalty = alpha * np.diag([0.0] + [1.0] * features.shape[1])
    forecasts = []
    for refit, year_dates in refit_schedule(sessions, dates):
        train = rows.loc[: training_end(sessions, refit, h)]
        if train.empty:
            raise ValueError(f"no training rows by {refit.date()}; start forecasts later")
        mean, std = train[features.columns].mean(), train[features.columns].std()
        x = np.column_stack([np.ones(len(train)), (train[features.columns] - mean) / std])
        coef = np.linalg.solve(x.T @ x + penalty, x.T @ train["y"])
        resid_var = np.var(train["y"] - x @ coef, ddof=x.shape[1])
        x_now = np.column_stack([np.ones(len(year_dates)), (features.loc[year_dates] - mean) / std])
        log = x_now @ coef
        forecasts.append(
            pd.DataFrame({"log": log, "variance": np.exp(log + resid_var / 2)}, index=year_dates)
        )
    return pd.concat(forecasts)


def evaluate(
    returns: pd.Series,
    vix_close: pd.Series,
    signal: pd.DataFrame,
    h: int,
    alpha: float,
    forecast_start=DEV_START,
) -> pd.DataFrame:
    """Baseline and baseline-plus-signal forecasts with the target, on identical rows."""
    base = baseline_features(returns, vix_close)
    features = base.join(signal, how="inner").dropna()
    target = realized_variance(returns, h)
    grid = forecast_grid(returns, h)
    dates = grid[grid.isin(features.index) & (grid >= pd.Timestamp(forecast_start))]
    out = pd.DataFrame({"target": target.loc[dates]})
    for model, columns in {"baseline": base.columns, "signal": features.columns}.items():
        fit = ridge(features[columns], np.log(target), returns.index, dates, h, alpha)
        out[model] = fit["variance"]
        out[f"{model}_log"] = fit["log"]
    return out.rename_axis("decision_date")


def overlap_lag(sessions: pd.DatetimeIndex, dates: pd.DatetimeIndex, h: int) -> int:
    """Most later dates whose h-session target windows overlap one date's window.

    The Newey–West lag for losses on ``dates``: #10's "lag h" counts sessions, and the
    month-end grid needs it in grid steps.
    """
    pos = sessions.get_indexer(dates)
    return int(max(np.searchsorted(pos, p + h) - i - 1 for i, p in enumerate(pos)))


def score(forecasts: pd.DataFrame, sessions: pd.DatetimeIndex, h: int) -> dict:
    """QLIKE of both models, QLIKE gain, R²_OOS on log variance, one-sided Clark–West."""
    log_target = np.log(forecasts["target"])
    e_base = log_target - forecasts["baseline_log"]
    e_signal = log_target - forecasts["signal_log"]
    # Clark and West (2007): the signal model's squared error, less the noise from
    # estimating parameters that are zero under the null of no gain.
    adjusted = e_base**2 - e_signal**2 + (forecasts["baseline_log"] - forecasts["signal_log"]) ** 2
    lag = overlap_lag(sessions, forecasts.index, h)
    fit = sm.OLS(adjusted.to_numpy(), np.ones(len(adjusted))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lag}
    )
    t, p_two_sided = fit.tvalues[0], fit.pvalues[0]
    qlike_base = qlike(forecasts["baseline"], forecasts["target"])
    qlike_signal = qlike(forecasts["signal"], forecasts["target"])
    return {
        "n": len(forecasts),
        "qlike_baseline": qlike_base,
        "qlike_signal": qlike_signal,
        "qlike_gain": 1 - qlike_signal / qlike_base,
        "r2_oos": 1 - (e_signal**2).sum() / (e_base**2).sum(),
        "cw_t": t,
        "cw_p": p_two_sided / 2 if t > 0 else 1 - p_two_sided / 2,
        "nw_lag": lag,
    }


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text())
    returns = log_returns(load("SPX")).loc[ESTIMATION_START:]
    vix_close = load("VIX")
    names = {name for block in config["blocks"].values() for name in block["signals"]}
    signals = {name: getattr(slow_macro, name)().run(DEV_END) for name in names}
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for h in config["horizons"]:
        frames = []
        for block, spec in config["blocks"].items():
            signal = pd.concat([signals[name] for name in spec["signals"]], axis=1, join="inner")
            start = spec.get("forecast_start", DEV_START)
            forecasts = evaluate(returns, vix_close, signal, h, config["ridge_alpha"], start)
            frames.append(forecasts.assign(block=block))
            rows.append({"block": block, "horizon": h, **score(forecasts, returns.index, h)})
        out = pd.concat(frames).reset_index()
        out.to_csv(OUT / f"{config['name']}_h{h}.csv", index=False, float_format="%.8g")
    print(pd.DataFrame(rows).set_index(["block", "horizon"]).round(4).to_string())


if __name__ == "__main__":
    main()
