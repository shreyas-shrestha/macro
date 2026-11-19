"""score one signal vs the no-signal baseline (#10)

    uv run python experiments/run_signal.py --config configs/noise.yaml

simply: for each horizon, fit 2 ridge models walk-forward on the SAME rows:
    baseline: log RV + log VIX
    model:    log RV + log VIX + your signal's features
then score both + check if the gap is luck

out:
    results/forecasts/<signal>_h<h>.csv            the forecasts
    results/scorecard.csv                          your rows (replaced on rerun)
    results/figures/<signal>_cumulative_qlike.png  the chart
    + prints the scorecard and a by-year / by-crisis table

stops at 2017-12-31. 2018+ is locked until #20 (--holdout unlocks it)

config (yaml):
    signal: credit                    # name for output files + scorecard
    module: macro.features.credit     # where your Signal class lives
    class: Credit
    features: [baa10y_level, ...]     # optional, default = every column
    horizons: [5, 21, 63]             # optional, default = 5, 21, 63
    threshold:                        # post this on your issue BEFORE running
      horizon: 21
      qlike_reduction: 0.05
      clark_west_p: 0.10
"""

import argparse
import importlib
from pathlib import Path

import pandas as pd
import yaml

from macro.backtest.report import breakdown, compare, plot_cumulative, update_scorecard
from macro.backtest.scoring import qlike_losses
from macro.backtest.walkforward import baseline_features, decision_grid, walk_forward
from macro.config import DEV_END
from macro.data.series import load
from macro.models.target import HORIZONS, log_returns, realized_variance

ROOT = Path(__file__).resolve().parents[1]
# same start as the #9 baselines - keeps the 1987 crash out of every fit
ESTIMATION_START = "1990-01-02"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="path to configs/<signal>.yaml")
    parser.add_argument("--holdout", action="store_true", help="final run on 2018+ (#20)")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    name = config["signal"]
    horizons = config.get("horizons", list(HORIZONS))
    threshold = config.get("threshold", {})

    # 2018+ is locked - only the final run (--holdout) can read past DEV_END
    end_date = pd.Timestamp.today().normalize() if args.holdout else DEV_END
    tag = "_holdout" if args.holdout else ""

    # load the signal class named in the config + build its features
    signal_cls = getattr(importlib.import_module(config["module"]), config["class"])
    signal = signal_cls()
    features = signal.run(end_date)
    features = features[config.get("features", list(features.columns))]

    returns = log_returns(load("SPX", end_date, holdout=args.holdout)).loc[ESTIMATION_START:]
    base = baseline_features(returns, load("VIX", end_date, holdout=args.holdout))
    full = base.join(features, how="left")
    # same rows for both models: blank out baseline rows where the signal is missing
    base = base.where(full.notna().all(axis=1))

    rows, curves = [], {}
    for h in horizons:
        # forecast dates, the real answers, then both models' forecasts
        grid = decision_grid(returns, h, end_date)
        target = realized_variance(returns, h).loc[grid].rename("target")
        baseline_fc = walk_forward(base, returns, grid, h).rename("baseline")
        model_fc = walk_forward(full, returns, grid, h).rename("model")

        out = pd.concat([target, baseline_fc, model_fc], axis=1).dropna()
        out_path = ROOT / "results" / "forecasts" / f"{name}_h{h}{tag}.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.rename_axis("decision_date").to_csv(out_path, float_format="%.8g")

        # score it + check the pre-registered bar (only at the threshold horizon)
        row = compare(out["target"], out["baseline"], out["model"], h)
        if threshold and h == threshold.get("horizon"):
            row["meets_threshold"] = bool(
                row["qlike_reduction"] >= threshold["qlike_reduction"]
                and row["cw_p"] < threshold["clark_west_p"]
            )
        rows.append(row)
        # per-row gap for the chart (positive = signal did better that month)
        curves[h] = pd.Series(
            qlike_losses(out["baseline"], out["target"])
            - qlike_losses(out["model"], out["target"]),
            index=out.index,
        )

        print(f"\n=== {name}, h = {h}: QLIKE by year and stress episode ===")
        print(breakdown(out["target"], out["baseline"], out["model"]).round(4).to_string())

    table = update_scorecard(ROOT / "results" / f"scorecard{tag}.csv", name, rows)
    print("\n=== scorecard ===")
    print(table[table["signal"] == name].round(4).to_string(index=False))

    fig = ROOT / "results" / "figures" / f"{name}_cumulative_qlike{tag}.png"
    if plot_cumulative(fig, name, curves):
        print(f"\nSaved {fig.relative_to(ROOT)}")
    else:
        print("\nSkipped figure (matplotlib not installed)")


if __name__ == "__main__":
    main()
