"""Score one signal against the no-signal baseline (#10).

    uv run python experiments/run_signal.py --config configs/noise.yaml

For each horizon, fits two ridge models walk-forward on the same rows: the baseline (log
trailing RV, log VIX squared) and the baseline plus the signal's features. Writes

    results/forecasts/<signal>_h<h>.csv            forecasts of both models and the target
    results/scorecard.csv                          the signal's rows, replaced on rerun
    results/figures/<signal>_cumulative_qlike.png  cumulative QLIKE gain

and prints QLIKE by year and stress episode.

Development runs score 2000 through 2017. ``--holdout END`` is the final run (#20): it
scores decision dates from 2018-01-01 whose targets end by END, fitting on everything
before each refit as usual, and writes to ``*_holdout`` files.

Config (yaml):
    signal: credit                    # name for output files and the scorecard
    module: macro.features.credit     # module holding the Signal class
    class: Credit
    features: [baa10y_level, ...]     # optional; default every column
    horizons: [5, 21, 63]             # optional; default 5, 21, 63
    threshold:                        # posted on the signal's issue before running
      horizon: 21
      qlike_reduction: 0.05
      clark_west_p: 0.10
"""

import argparse
import contextlib
import importlib
from pathlib import Path

import pandas as pd
import yaml

from macro.backtest.report import breakdown, compare, plot_cumulative, qlike_gain, update_scorecard
from macro.backtest.walkforward import baseline_features, walk_forward
from macro.config import DEV_END, DEV_START, ESTIMATION_START, HOLDOUT_START
from macro.data.series import holdout_reads, load
from macro.models.target import HORIZONS, forecast_grid, log_returns, realized_variance

ROOT = Path(__file__).resolve().parents[1]


def run(config: dict, holdout_end: pd.Timestamp | None = None, root: Path = ROOT) -> pd.DataFrame:
    """Score the signal in ``config``; return its scorecard rows. See the module docstring."""
    name = config["signal"]
    horizons = config.get("horizons", list(HORIZONS))
    threshold = config.get("threshold", {})
    if threshold and threshold["horizon"] not in horizons:
        raise ValueError(f"threshold horizon {threshold['horizon']} is not in horizons {horizons}")

    if holdout_end is None:
        start, end, tag, reads = DEV_START, DEV_END, "", contextlib.nullcontext()
    else:
        start, end, tag, reads = HOLDOUT_START, holdout_end, "_holdout", holdout_reads()

    signal = getattr(importlib.import_module(config["module"]), config["class"])()
    with reads:
        features = signal.run(end)
        returns = log_returns(load("SPX", end)).loc[ESTIMATION_START:]
        vix_close = load("VIX", end)
    features = features[config.get("features", list(features.columns))]

    model_features = baseline_features(returns, vix_close).join(features, how="left")
    # Both models train and forecast on the same rows: those where the signal exists.
    base_features = model_features[["log_rv", "log_vix"]].where(model_features.notna().all(axis=1))

    # Fit every horizon before writing anything, so a failure leaves no partial output.
    fitted = {}
    for h in horizons:
        grid = forecast_grid(returns, h, start, end)
        target = realized_variance(returns, h).loc[grid]
        baseline = walk_forward(base_features, returns, grid, h)
        model = walk_forward(model_features, returns, grid, h)

        out = (
            target.rename("target")
            .to_frame()
            .join(baseline.add_prefix("baseline_"))
            .join(model.add_prefix("model_"))
            .dropna()
        )
        if out.empty:
            raise ValueError(f"{name}, h = {h}: no decision date has forecasts from both models")
        fitted[h] = out, baseline, model

    rows, curves = [], {}
    for h, (out, baseline, model) in fitted.items():
        path = root / "results" / "forecasts" / f"{name}_h{h}{tag}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        out.rename_axis("decision_date").to_csv(path, float_format="%.8g")

        target = out["target"]
        row = compare(target, baseline, model, h)
        if threshold and h == threshold["horizon"]:
            row["meets_threshold"] = bool(
                row["qlike_reduction"] >= threshold["qlike_reduction"]
                and row["cw_p"] < threshold["clark_west_p"]
            )
        rows.append(row)
        curves[h] = qlike_gain(target, baseline, model)

        print(f"\n=== {name}, h = {h}: QLIKE by year and stress episode ===")
        print(breakdown(target, baseline, model).round(4).to_string())

    table = update_scorecard(root / "results" / f"scorecard{tag}.csv", name, rows)
    fig = root / "results" / "figures" / f"{name}_cumulative_qlike{tag}.png"
    if plot_cumulative(fig, name, curves):
        print(f"\nSaved {fig}")
    else:
        print("\nSkipped figure (matplotlib not installed)")
    return table[table["signal"] == name]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="path to configs/<signal>.yaml")
    parser.add_argument(
        "--holdout",
        metavar="END",
        type=pd.Timestamp,
        help="final run (#20): score 2018-01-01 onward with targets ending by END",
    )
    args = parser.parse_args()
    if args.holdout is not None and args.holdout <= HOLDOUT_START:
        parser.error(f"--holdout END must be after {HOLDOUT_START.date()}")

    rows = run(yaml.safe_load(Path(args.config).read_text()), args.holdout)
    print("\n=== scorecard ===")
    print(rows.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
