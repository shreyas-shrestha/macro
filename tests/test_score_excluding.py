import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macro.backtest.report import compare
from macro.data.calendar import nyse_sessions

SPEC = importlib.util.spec_from_file_location(
    "score_excluding", Path(__file__).resolve().parents[1] / "experiments" / "score_excluding.py"
)
score_excluding = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(score_excluding)

START, END = pd.Timestamp("2007-08-01"), pd.Timestamp("2009-06-30")


def write_forecasts(path: Path) -> pd.DataFrame:
    """A run_signal file for 2005–2012 month-ends and START, where the model beats the
    baseline only inside START..END and loses to it outside."""
    sessions = nyse_sessions("2005-01-01", "2012-12-31")
    dates = pd.DatetimeIndex(sessions.to_series().groupby(sessions.to_period("M")).max())
    dates = dates.union([START])
    rng = np.random.default_rng(0)
    target = pd.Series(np.exp(rng.normal(-3, 0.5, len(dates))), index=dates)
    inside = (dates >= START) & (dates <= END)
    noise = np.exp(rng.normal(0, 0.05, (3, len(dates))))
    forecasts = {
        "baseline": target * 1.5 * noise[0],
        "baseline+other": target * 1.4 * noise[1],
        "model": target * np.where(inside, 1.05, 2.5) * noise[2],
    }
    out = target.rename("target").to_frame()
    for label, variance in forecasts.items():
        out[f"{label}_log"] = np.log(variance)
        out[f"{label}_variance"] = variance
    out.rename_axis("decision_date").to_csv(path, float_format="%.8g")
    return out


def test_window_is_scored_apart_with_the_file_horizon(tmp_path):
    path = tmp_path / "x_h63.csv"
    out = write_forecasts(path)
    rows = score_excluding.score_excluding(path, START, END)
    assert list(rows["reference"]) == ["baseline", "baseline+other"] * 3
    by_rows = rows[rows["reference"] == "baseline"].set_index("rows")
    # START, then the month-ends from August 2007 to June 2009, both ends inside.
    assert by_rows["n"].to_dict() == {"all": 97, "outside": 73, "inside": 24}
    assert by_rows.loc["inside", "start"] == START.date()
    assert by_rows.loc["inside", "end"] == END.date()
    assert by_rows.loc["outside", "qlike_reduction"] < 0 < by_rows.loc["inside", "qlike_reduction"]
    assert (rows["horizon"] == 63).all() and (rows["nw_lag"] == 3).all()
    model, baseline = (
        out[[f"{label}_log", f"{label}_variance"]].set_axis(["log", "variance"], axis=1)
        for label in ("model", "baseline")
    )
    full = compare(out["target"], baseline, model, 63)
    assert by_rows.loc["all"].to_dict() == pytest.approx(full | {"reference": "baseline"})


def test_holdout_file_names_give_the_horizon(tmp_path):
    path = tmp_path / "x_h21_holdout.csv"
    write_forecasts(path)
    rows = score_excluding.score_excluding(path, START, END)
    assert (rows["horizon"] == 21).all() and (rows["nw_lag"] == 1).all()


@pytest.mark.parametrize(
    "name, start, end, message",
    [
        ("x.csv", START, END, "named"),
        ("x_h21.csv", pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31"), "inside"),
    ],
)
def test_bad_inputs_raise(tmp_path, name, start, end, message):
    write_forecasts(tmp_path / name)
    with pytest.raises(ValueError, match=message):
        score_excluding.score_excluding(tmp_path / name, start, end)
