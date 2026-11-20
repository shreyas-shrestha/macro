"""Rescore a run_signal forecast file without a window of decision dates (#15).

    uv run python experiments/score_excluding.py \\
        --forecasts results/forecasts/funding_h21.csv --exclude 2007-08-01 2009-06-30

Prints ``compare`` of the model against every reference in the file on all decision
dates (as in the scorecard), on those outside the window, and on those inside it.
Both ends of the window are included in it. The forecasts are the run's own: the models
were still fitted on data from the window; only its decision dates go unscored. The
horizon, and with it the Newey–West lag, is read from the file name, as run_signal
writes it. Outside rows join across the window in the Newey–West series, and decision
dates just before the window have targets that run into it.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

from macro.backtest.report import compare

FILE_NAME = re.compile(r"_h(\d+)(?:_holdout)?\.csv$")


def score_excluding(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """One row per (rows, reference): rows ``all``, ``outside`` or ``inside`` the window."""
    match = FILE_NAME.search(path.name)
    if match is None:
        raise ValueError(f"{path}: expected a run_signal file named <signal>_h<h>[_holdout].csv")
    h = int(match.group(1))
    out = pd.read_csv(path, index_col="decision_date", parse_dates=True)

    def forecast(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        return frame[[f"{label}_log", f"{label}_variance"]].set_axis(["log", "variance"], axis=1)

    references = [c.removesuffix("_log") for c in out if c.endswith("_log") and c != "model_log"]
    inside = (out.index >= start) & (out.index <= end)
    rows = []
    for name, part in (("all", out), ("outside", out[~inside]), ("inside", out[inside])):
        if part.empty:
            raise ValueError(f"no decision dates {name} {start.date()} to {end.date()}")
        for label in references:
            scores = compare(part["target"], forecast(part, label), forecast(part, "model"), h)
            rows.append({"rows": name, "reference": label, **scores})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--forecasts", required=True, type=Path, help="results/forecasts/*.csv")
    parser.add_argument(
        "--exclude", required=True, nargs=2, type=pd.Timestamp, metavar=("START", "END")
    )
    args = parser.parse_args()
    print(score_excluding(args.forecasts, *args.exclude).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
