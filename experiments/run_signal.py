"""Score one or more signals, together, against the no-signal baseline (#10).

    uv run python experiments/run_signal.py --config configs/noise.yaml

For each horizon, fits ridge models walk-forward on the same rows: the baseline (log
trailing RV, log VIX squared), the baseline plus every signal's features, and one model
per reference (below). Rows are the sessions where every feature exists. Writes

    results/forecasts/<signal>_h<h>.csv            forecasts of every model and the target
    results/scorecard.csv                          the run's rows, replaced on rerun
    results/figures/<signal>_cumulative_qlike.png  cumulative QLIKE gain over the baseline

and prints QLIKE by year and stress episode against the baseline.

Development runs score 2000 through 2017. ``--holdout END`` is the final run (#20): it
scores decision dates from 2018-01-01 whose targets end by END, fitting on everything
before each refit as usual, and writes to ``*_holdout`` files.

Config (yaml), one signal:
    signal: credit                    # name for output files and the scorecard
    module: macro.features.credit     # module holding the Signal class
    class: Credit
    kwargs: {exclude_backfill: true}  # optional; passed to the class
    features: [baa10y_level, ...]     # optional; default every column
    horizons: [5, 21, 63]             # optional; default 5, 21, 63
    threshold:                        # posted on the signal's issue before running
      horizon: 21
      qlike_reduction: 0.05
      clark_west_p: 0.10

Several signals in one model replace module / class / kwargs / features with a list:
    signal: credit_move
    signals:
      - {module: macro.features.credit, class: Credit, features: [baa10y_level]}
      - {module: macro.features.move, class: Move, name: move}  # name: default class.name
    references: [credit]              # optional; also score against baseline + credit

The scorecard's ``reference`` column names the model each row is scored against:
``baseline`` or ``baseline+<names>``. The threshold applies to the baseline row. A
reference model is nested in the full model, so Clark–West applies to it too.
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
BASELINE_COLUMNS = ["log_rv", "log_vix"]
SIGNAL_KEYS = {"module", "class", "kwargs", "features", "name"}


def signal_entries(config: dict) -> list[dict]:
    """The config's signals as a list of entries, whichever form the config uses."""
    single = SIGNAL_KEYS & config.keys()
    if "signals" in config:
        if single:
            raise ValueError(f"use either signals or {sorted(single)}, not both")
        entries = config["signals"]
    else:
        entries = [{key: config[key] for key in single}]
    for entry in entries:
        unknown = entry.keys() - SIGNAL_KEYS
        if unknown:
            raise ValueError(f"unknown signal keys {sorted(unknown)}")
    return entries


def build_features(entries: list[dict], end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Each entry's selected features through ``end``, keyed by the entry's name."""
    out = {}
    for entry in entries:
        cls = getattr(importlib.import_module(entry["module"]), entry["class"])
        signal = cls(**entry.get("kwargs", {}))
        name = entry.get("name", signal.name)
        if name in out:
            raise ValueError(f"two signals are named {name!r}; set name: on one")
        frame = signal.run(end)
        out[name] = frame[entry.get("features", list(frame.columns))]
    columns = [c for frame in out.values() for c in frame.columns]
    clashes = sorted(
        {c for c in columns if columns.count(c) > 1} | (set(columns) & set(BASELINE_COLUMNS))
    )
    if clashes:
        raise ValueError(f"feature names used twice: {clashes}")
    return out


def run(config: dict, holdout_end: pd.Timestamp | None = None, root: Path = ROOT) -> pd.DataFrame:
    """Score the signals in ``config``; return the run's scorecard rows. See the module
    docstring."""
    name = config["signal"]
    horizons = config.get("horizons", list(HORIZONS))
    threshold = config.get("threshold", {})
    if threshold and threshold["horizon"] not in horizons:
        raise ValueError(f"threshold horizon {threshold['horizon']} is not in horizons {horizons}")
    entries = signal_entries(config)

    if holdout_end is None:
        start, end, tag, reads = DEV_START, DEV_END, "", contextlib.nullcontext()
    else:
        start, end, tag, reads = HOLDOUT_START, holdout_end, "_holdout", holdout_reads()

    with reads:
        signals = build_features(entries, end)
        returns = log_returns(load("SPX", end)).loc[ESTIMATION_START:]
        vix_close = load("VIX", end)

    references = config.get("references", [])
    unknown = set(references) - signals.keys()
    if unknown:
        raise ValueError(f"references {sorted(unknown)} are not signals of this run")
    if set(references) == signals.keys():
        raise ValueError("references must leave out at least one signal")

    model_features = baseline_features(returns, vix_close).join(list(signals.values()), how="left")
    # Every model trains and forecasts on the same rows: those where every feature exists.
    rows_used = model_features.notna().all(axis=1)
    reference_columns = {"baseline": BASELINE_COLUMNS}
    if references:
        reference_columns["baseline+" + "+".join(references)] = BASELINE_COLUMNS + [
            c for ref in references for c in signals[ref].columns
        ]

    # Fit every horizon before writing anything, so a failure leaves no partial output.
    fitted = {}
    for h in horizons:
        grid = forecast_grid(returns, h, start, end)
        target = realized_variance(returns, h).loc[grid]
        model = walk_forward(model_features, returns, grid, h)
        compared = {
            label: walk_forward(model_features[cols].where(rows_used), returns, grid, h)
            for label, cols in reference_columns.items()
        }
        out = target.rename("target").to_frame().join(model.add_prefix("model_"))
        for label, forecast in compared.items():
            out = out.join(forecast.add_prefix(f"{label}_"))
        out = out.dropna()
        if out.empty:
            raise ValueError(f"{name}, h = {h}: no decision date has forecasts from every model")
        fitted[h] = out, model, compared

    rows, curves = [], {}
    for h, (out, model, compared) in fitted.items():
        path = root / "results" / "forecasts" / f"{name}_h{h}{tag}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        out.rename_axis("decision_date").to_csv(path, float_format="%.8g")

        target = out["target"]
        for label, forecast in compared.items():
            row = {"reference": label, **compare(target, forecast, model, h)}
            if threshold and label == "baseline" and h == threshold["horizon"]:
                row["meets_threshold"] = bool(
                    row["qlike_reduction"] >= threshold["qlike_reduction"]
                    and row["cw_p"] < threshold["clark_west_p"]
                )
            rows.append(row)
        baseline = compared["baseline"]
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
