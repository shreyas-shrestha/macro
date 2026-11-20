"""Baseline forecasts for every horizon on the development grid (#9).

Writes results/baselines.csv (decision_date, horizon, model, forecast, target) and prints
mean QLIKE per model and horizon.

    uv run python experiments/run_baselines.py
"""

from pathlib import Path

import pandas as pd

from macro.backtest.scoring import qlike
from macro.config import ESTIMATION_START
from macro.data.series import load
from macro.models import baselines
from macro.models.target import HORIZONS, forecast_grid, log_returns, realized_variance

OUT = Path(__file__).resolve().parents[1] / "results" / "baselines.csv"


def main() -> None:
    returns = log_returns(load("SPX")).loc[ESTIMATION_START:]
    vix_close = load("VIX")
    rows = []
    for h in HORIZONS:
        grid = forecast_grid(returns, h)
        target = realized_variance(returns, h).loc[grid]
        forecasts = {
            "rv": baselines.trailing_rv(returns, grid),
            "har": baselines.har(returns, grid, h),
            "garch": baselines.garch(returns, grid, h),
            "vix": baselines.vix(vix_close, grid),
        }
        for model, forecast in forecasts.items():
            rows.append(
                pd.DataFrame({"horizon": h, "model": model, "forecast": forecast, "target": target})
            )
    out = pd.concat(rows).rename_axis("decision_date").reset_index()
    out.to_csv(OUT, index=False, float_format="%.8g")
    scores = out.groupby(["model", "horizon"]).apply(
        lambda g: qlike(g["forecast"], g["target"]), include_groups=False
    )
    print(scores.unstack("horizon").round(4))


if __name__ == "__main__":
    main()
