import runpy
from pathlib import Path

import numpy as np
import pandas as pd

from macro.data.calendar import nyse_sessions
from macro.models.target import realized_variance

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "run_yield_curve.py"
RUN = runpy.run_path(SCRIPT)
SESSIONS = nyse_sessions("1998-01-02", "2012-12-31")
OMEGA, ALPHA, BETA = 2e-6, 0.08, 0.9


def simulated_market(seed=0):
    """GARCH(1,1) returns and a noisy VIX built from each session's true variance."""
    rng = np.random.default_rng(seed)
    var = OMEGA / (1 - ALPHA - BETA)
    returns, next_var = np.empty(len(SESSIONS)), np.empty(len(SESSIONS))
    for i in range(len(SESSIONS)):
        returns[i] = np.sqrt(var) * rng.standard_normal()
        var = OMEGA + ALPHA * returns[i] ** 2 + BETA * var
        next_var[i] = var
    vix = 100 * np.sqrt(252 * next_var) * np.exp(rng.normal(0, 0.05, len(SESSIONS)))
    return pd.Series(returns, index=SESSIONS), pd.Series(vix, index=SESSIONS)


RETURNS, VIX = simulated_market()
SIGNAL_INDEX = SESSIONS[SESSIONS >= "2000-01-03"]


def noise_signal(seed=1):
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((len(SIGNAL_INDEX), 4))
    return pd.DataFrame(values, index=SIGNAL_INDEX, columns=["f1", "f2", "f3", "f4"])


def test_noise_features_do_not_pass():
    summary, _ = RUN["evaluate"](RETURNS, VIX, noise_signal())
    assert list(summary.index) == [5, 21, 63]
    assert (summary["improvement_pct"] < 2).all()
    assert (summary["cw_p"] > 0.05).all()


def test_a_feature_that_knows_the_future_passes():
    """Sanity check on the scoring: leaking the target must show up as a clear win."""
    signal = noise_signal()
    leak = np.log(realized_variance(RETURNS, 21)).reindex(SIGNAL_INDEX)
    signal["f1"] = leak.fillna(leak.mean())
    summary, _ = RUN["evaluate"](RETURNS, VIX, signal)
    assert summary.loc[21, "improvement_pct"] > 50
    assert summary.loc[21, "cw_p"] < 0.01


def test_forecasts_start_in_2004_and_cover_both_models():
    _, table = RUN["evaluate"](RETURNS, VIX, noise_signal())
    assert list(table.columns) == ["decision_date", "horizon", "model", "forecast", "target"]
    assert table["decision_date"].min().year == 2004
    assert set(table["model"]) == {"benchmark", "augmented"}
    assert (table["forecast"] > 0).all()
    counts = table.groupby(["horizon", "model"]).size().unstack()
    assert (counts["benchmark"] == counts["augmented"]).all()


def test_walk_forward_uses_no_data_after_the_decision_date():
    """Forecasts for 2004-2006 are unchanged when everything from 2007 on is removed."""
    signal = noise_signal()
    full = RUN["walk_forward"](RETURNS, VIX, signal, 21)
    cut = "2006-12-29"
    # Keep enough returns after the last 2006 decision date for its 21-session target.
    end = SESSIONS[SESSIONS.get_loc(pd.Timestamp(cut)) + 21]
    early = RUN["walk_forward"](RETURNS[:end], VIX[:end], signal[:end], 21)
    shared = early.index[early.index <= cut]
    pd.testing.assert_frame_equal(early.loc[shared], full.loc[shared])


def test_clark_west_favors_the_model_with_smaller_errors():
    rng = np.random.default_rng(0)
    truth = pd.Series(rng.standard_normal(200))
    worse = truth + rng.standard_normal(200)
    better = truth + 0.1 * rng.standard_normal(200)
    stat, p_value = RUN["clark_west"](truth, worse, better, 1)
    assert stat > 3
    assert p_value < 0.01
    reverse, reverse_p = RUN["clark_west"](truth, better, worse, 1)
    assert reverse < stat
    assert reverse_p > p_value
