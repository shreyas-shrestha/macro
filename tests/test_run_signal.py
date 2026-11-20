"""End-to-end runs of experiments/run_signal.py on fake sources; no network."""

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macro.backtest import walkforward
from macro.config import DEV_END
from macro.data import cache, series
from macro.features.base import INDEX_NAME, Signal

SPEC = importlib.util.spec_from_file_location(
    "run_signal", Path(__file__).resolve().parents[1] / "experiments" / "run_signal.py"
)
run_signal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_signal)


class Fake(Signal):
    """Reads the fake series ``SIG`` through ``load``, as real signals do."""

    name = "fake"
    source = "test"
    start_date = pd.Timestamp("2000-01-03")
    release_lag = 0

    def build(self, end_date):
        values = series.load("SIG", end_date).loc[self.start_date :]
        return values.to_frame("sig").rename_axis(INDEX_NAME)


class Other(Signal):
    """A second signal on ``SIG``, starting in 2003, scaled by a constructor argument."""

    name = "other"
    source = "test"
    start_date = pd.Timestamp("2003-01-02")
    release_lag = 0

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def build(self, end_date):
        values = series.load("SIG", end_date).loc[self.start_date :]
        return (self.scale * values.abs()).to_frame("other").rename_axis(INDEX_NAME)


@pytest.fixture
def sources(tmp_path, monkeypatch):
    """SPX, VIX and SIG from seeded random data; SIG has a gap in 2005 and ends in 2017."""
    rng = np.random.default_rng(0)
    days = pd.bdate_range("1985-01-01", "2020-12-31")
    vol = 0.01 * (1 + 0.5 * np.sin(np.arange(len(days)) / 150))
    data = {
        "^GSPC": pd.Series(100 * np.exp(np.cumsum(rng.normal(size=len(days)) * vol)), days),
        "^VIX": pd.Series(100 * vol * np.sqrt(252) * (1 + 0.1 * rng.random(len(days))), days),
    }
    sig = pd.Series(rng.normal(size=len(days)), days).loc[:"2017-12-29"]
    data["SIG"] = sig.drop(sig.loc["2005-03":"2005-08"].index)

    def fetch(source_id, start, end):
        return data[source_id].loc[start:end].astype(float)

    monkeypatch.setattr(cache, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setitem(series.FETCHERS, "yahoo", fetch)
    monkeypatch.setitem(series.FETCHERS, "fake", fetch)
    monkeypatch.setitem(
        series.SERIES, "SIG", series.SeriesSpec("fake", "SIG", pd.Timestamp("1999-01-04"), 1, 0)
    )
    module = types.ModuleType("fake_signal")
    module.Fake = Fake
    module.Other = Other
    monkeypatch.setitem(sys.modules, "fake_signal", module)
    return tmp_path


CONFIG = {"signal": "fake", "module": "fake_signal", "class": "Fake", "horizons": [21]}


def test_baseline_and_model_train_on_the_same_rows(sources, monkeypatch):
    fits = []
    fit_ridge = walkforward.fit_ridge

    def spy(x, y, alpha):
        fits.append((len(y), round(float(y.sum()), 6)))
        return fit_ridge(x, y, alpha)

    monkeypatch.setattr(walkforward, "fit_ridge", spy)
    rows = run_signal.run(CONFIG, root=sources)
    # Two walk-forward passes, baseline then model, refit on identical rows.
    assert len(fits) % 2 == 0 and fits[: len(fits) // 2] == fits[len(fits) // 2 :]
    out = pd.read_csv(sources / "results/forecasts/fake_h21.csv", index_col=0, parse_dates=True)
    assert out.loc["2005-03":"2005-08"].empty  # the gap is dropped from both models
    assert out.index[-1] <= DEV_END
    assert rows["n"].item() == len(out)


def test_holdout_run_reads_through_the_signal_and_scores_only_2018_on(sources, monkeypatch):
    data_end = pd.Timestamp("2019-06-28")
    days = pd.bdate_range("2018-01-01", data_end)
    sig = pd.Series(np.random.default_rng(1).normal(size=len(days)), days)
    fetch = series.FETCHERS["fake"]
    monkeypatch.setitem(
        series.FETCHERS, "fake", lambda i, s, e: pd.concat([fetch(i, s, e), sig.loc[s:e]])
    )
    rows = run_signal.run(CONFIG, holdout_end=data_end, root=sources)
    assert pd.Timestamp(rows["start"].item()) == pd.Timestamp("2018-01-31")
    assert pd.Timestamp(rows["end"].item()) == pd.Timestamp("2019-04-30")
    assert not list((sources / "cache").rglob("*.parquet"))
    assert (sources / "results/scorecard_holdout.csv").exists()
    assert not (sources / "results/scorecard.csv").exists()


def test_a_horizon_without_forecasts_fails_before_any_output(sources, monkeypatch):
    # Signal data from 2019-04-01: h = 5 has April and May 2019 forecasts, h = 63 none.
    days = pd.bdate_range("2019-04-01", "2019-06-28")
    sig = pd.Series(np.random.default_rng(2).normal(size=len(days)), days)
    fetch = series.FETCHERS["fake"]
    monkeypatch.setitem(
        series.FETCHERS, "fake", lambda i, s, e: pd.concat([fetch(i, s, e), sig.loc[s:e]])
    )
    config = CONFIG | {"horizons": [5, 63]}
    with pytest.raises(ValueError, match="h = 63: no decision date"):
        run_signal.run(config, holdout_end=pd.Timestamp("2019-06-28"), root=sources)
    assert not (sources / "results").exists()


def test_threshold_horizon_must_be_scored(sources):
    config = CONFIG | {"threshold": {"horizon": 63, "qlike_reduction": 0.05, "clark_west_p": 0.1}}
    with pytest.raises(ValueError, match="threshold horizon 63"):
        run_signal.run(config, root=sources)


TWO = {
    "signal": "both",
    "signals": [
        {"module": "fake_signal", "class": "Fake"},
        {"module": "fake_signal", "class": "Other", "kwargs": {"scale": 2.0}},
    ],
    "references": ["fake"],
    "horizons": [21],
    "threshold": {"horizon": 21, "qlike_reduction": 0.05, "clark_west_p": 0.1},
}


def test_several_signals_share_rows_and_score_against_each_reference(sources, monkeypatch):
    fits = []
    fit_ridge = walkforward.fit_ridge

    def spy(x, y, alpha):
        fits.append((x.shape[1], len(y), round(float(y.sum()), 6)))
        return fit_ridge(x, y, alpha)

    monkeypatch.setattr(walkforward, "fit_ridge", spy)
    rows = run_signal.run(TWO, root=sources)
    assert list(rows["reference"]) == ["baseline", "baseline+fake"]
    assert rows["meets_threshold"].notna().tolist() == [True, False]
    # Full model, baseline and baseline + fake: 4, 2 and 3 columns, the same rows each refit.
    by_width = {}
    for width, n, total in fits:
        by_width.setdefault(width, []).append((n, total))
    assert sorted(by_width) == [2, 3, 4]
    assert by_width[2] == by_width[3] == by_width[4]
    out = pd.read_csv(sources / "results/forecasts/both_h21.csv", index_col=0, parse_dates=True)
    # Other starts in 2003 and SIG skips 2005-03 to 08: 2006 is the first refit with 504 rows.
    assert out.index[0] == pd.Timestamp("2006-01-31")
    assert {"model_log", "baseline_log", "baseline+fake_log"} <= set(out.columns)


def test_kwargs_reach_the_signal(sources):
    entry = {"module": "fake_signal", "class": "Other"}
    scaled = run_signal.build_features([entry | {"kwargs": {"scale": 3.0}}], DEV_END)
    unscaled = run_signal.build_features([entry], DEV_END)
    pd.testing.assert_frame_equal(scaled["other"], 3.0 * unscaled["other"])


@pytest.mark.parametrize(
    "change, message",
    [
        ({"module": "fake_signal"}, "either signals or"),
        (
            {"signals": [{"module": "fake_signal", "class": "Fake", "kwarg": {}}]},
            "unknown signal keys",
        ),
        ({"signals": [{"module": "fake_signal", "class": "Fake"}] * 2}, "named 'fake'"),
        (
            {
                "signals": [
                    {"module": "fake_signal", "class": "Fake"},
                    {"module": "fake_signal", "class": "Fake", "name": "again"},
                ]
            },
            r"feature names clash .*: \['sig'\]",
        ),
        ({"references": ["missing"]}, "not signals of this run"),
        ({"references": ["fake", "other"]}, "leave out at least one"),
    ],
)
def test_bad_signal_configs_raise(sources, change, message):
    with pytest.raises(ValueError, match=message):
        run_signal.run(TWO | change, root=sources)
    assert not (sources / "results").exists()
