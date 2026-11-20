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


def test_holdout_run_without_signal_data_fails_before_writing(sources):
    with pytest.raises(ValueError, match="no decision date"):
        run_signal.run(CONFIG, holdout_end=pd.Timestamp("2019-06-28"), root=sources)
    assert not list((sources / "results").rglob("*.csv"))


def test_threshold_horizon_must_be_scored(sources):
    config = CONFIG | {"threshold": {"horizon": 63, "qlike_reduction": 0.05, "clark_west_p": 0.1}}
    with pytest.raises(ValueError, match="threshold horizon 63"):
        run_signal.run(config, root=sources)
