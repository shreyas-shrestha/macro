import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from macro.data.calendar import INDEX_NAME, nyse_sessions
from macro.features import slow_macro
from macro.models.schedule import training_end
from macro.models.target import realized_variance

PATH = Path(__file__).resolve().parents[1] / "experiments" / "run_slow_macro.py"
spec = importlib.util.spec_from_file_location("run_slow_macro", PATH)
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)

SESSIONS = nyse_sessions("1994-01-03", "2005-12-30").rename(INDEX_NAME)
OMEGA, ALPHA, BETA = 2e-6, 0.08, 0.9


def simulated(seed=0):
    """GARCH(1,1) returns and a VIX built from the true next-day variance plus noise."""
    rng = np.random.default_rng(seed)
    var = OMEGA / (1 - ALPHA - BETA)
    returns, vix = np.empty(len(SESSIONS)), np.empty(len(SESSIONS))
    for i in range(len(SESSIONS)):
        returns[i] = np.sqrt(var) * rng.standard_normal()
        var = OMEGA + ALPHA * returns[i] ** 2 + BETA * var
        vix[i] = 100 * np.sqrt(252 * var) * np.exp(0.1 * rng.standard_normal())
    return pd.Series(returns, index=SESSIONS), pd.Series(vix, index=SESSIONS)


RETURNS, VIX = simulated()
NOISE = pd.DataFrame(
    np.random.default_rng(1).standard_normal((len(SESSIONS), 2)), index=SESSIONS, columns=["a", "b"]
)


def leak(h):
    """The log target itself, plus a little noise: knows the future."""
    target = np.log(realized_variance(RETURNS, h))
    return (target + 0.1 * NOISE["a"]).to_frame("leak")


def test_noise_feature_shows_no_gain():
    forecasts = run.evaluate(RETURNS, VIX, NOISE, 21, alpha=1.0)
    scores = run.score(forecasts, SESSIONS, 21)
    assert scores["n"] == 71
    assert abs(scores["qlike_gain"]) < 0.02 and scores["r2_oos"] < 0.02
    assert scores["cw_p"] > 0.1


def test_leaking_feature_shows_a_large_gain():
    forecasts = run.evaluate(RETURNS, VIX, leak(21), 21, alpha=1.0)
    scores = run.score(forecasts, SESSIONS, 21)
    assert scores["qlike_gain"] > 0.5 and scores["r2_oos"] > 0.5
    assert scores["cw_p"] < 0.01


def test_both_models_use_the_same_rows():
    signal = NOISE.copy()
    signal.loc[:"2001-06-29"] = np.nan
    with pytest.raises(ValueError, match="no training rows"):
        run.evaluate(RETURNS, VIX, signal, 21, alpha=1.0)
    forecasts = run.evaluate(RETURNS, VIX, signal, 21, alpha=1.0, forecast_start="2002-01-01")
    assert forecasts.index[0] == pd.Timestamp("2002-01-31")
    # The baseline is fit on 2001 rows only, so it differs from a fit on the full history.
    full = run.evaluate(RETURNS, VIX, NOISE, 21, alpha=1.0)
    assert forecasts.loc["2002-01-31", "baseline"] != full.loc["2002-01-31", "baseline"]


@pytest.mark.parametrize("t", ["2001-06-29", "2003-12-31"])
def test_forecasts_up_to_t_ignore_data_after_t(t):
    full = run.evaluate(RETURNS, VIX, NOISE, 21, alpha=1.0)
    truncated = run.evaluate(RETURNS.loc[:t], VIX.loc[:t], NOISE.loc[:t], 21, alpha=1.0)
    assert len(truncated) > 0
    pd.testing.assert_frame_equal(truncated, full.loc[truncated.index])


def test_training_targets_end_by_the_refit():
    # January 2002 returns are outside the features at the end of April and after every
    # training target, which must end by the last 2001 session.
    shocked = RETURNS.copy()
    shocked.loc["2002-01"] *= 5
    t = pd.Timestamp("2002-04-30")
    columns = ["baseline", "signal"]
    expected = run.evaluate(RETURNS, VIX, NOISE, 5, alpha=1.0).loc[t, columns]
    actual = run.evaluate(shocked, VIX, NOISE, 5, alpha=1.0).loc[t, columns]
    pd.testing.assert_series_equal(actual, expected)


def test_forecast_start_drops_earlier_dates():
    forecasts = run.evaluate(RETURNS, VIX, NOISE, 21, alpha=1.0, forecast_start="2003-01-01")
    assert forecasts.index[0] == pd.Timestamp("2003-01-31")


@pytest.mark.parametrize("h, lag", [(5, 0), (21, 1), (63, 3), (252, 12)])
def test_newey_west_lag_counts_overlapping_month_ends(h, lag):
    assert run.overlap_lag(SESSIONS, run.forecast_grid(RETURNS, h), h) == lag


@pytest.mark.parametrize("step, lag", [(5, 0), (4, 1)])
def test_newey_west_lag_counts_only_windows_that_share_a_session(step, lag):
    # Windows t+1 ... t+5 and t+6 ... t+10 do not overlap; t+5 ... t+9 does.
    assert run.overlap_lag(SESSIONS, SESSIONS[[0, step, 2 * step]], 5) == lag


def test_score_by_hand():
    # Log target 0, baseline log forecast 1 and signal log forecast s make the
    # Clark–West series 2 * (1 - s) = d.
    d = np.array([1.0, 3.0, 2.0, 5.0, 4.0, 0.0])
    signal_log = 1 - d / 2
    dates = SESSIONS[::3][:6]  # 3 sessions apart, so with h = 5 the Newey–West lag is 1
    forecasts = pd.DataFrame(
        {
            "target": 1.0,
            "baseline": np.e,
            "baseline_log": 1.0,
            "signal": np.exp(signal_log),
            "signal_log": signal_log,
        },
        index=dates,
    )
    scores = run.score(forecasts, SESSIONS, 5)
    u = d - d.mean()
    bartlett = (u @ u + 2 * 0.5 * (u[1:] @ u[:-1])) / len(d)  # weight 1 - 1/(lag + 1)
    t = d.mean() / math.sqrt(bartlett / len(d))
    assert scores["nw_lag"] == 1
    assert scores["cw_t"] == pytest.approx(t)
    assert scores["cw_p"] == pytest.approx(0.5 * math.erfc(t / math.sqrt(2)))
    assert scores["r2_oos"] == pytest.approx(1 - np.sum(signal_log**2) / len(d))
    # QLIKE with realized variance 1 and forecast exp(f) is exp(-f) + f - 1.
    qlike_base, qlike_signal = np.e**-1, np.mean(np.exp(-signal_log) + signal_log - 1)
    assert scores["qlike_baseline"] == pytest.approx(qlike_base)
    assert scores["qlike_gain"] == pytest.approx(1 - qlike_signal / qlike_base)


def ridge_inputs():
    x = NOISE[["a"]]
    y = 5 + 2 * x["a"] + 0.3 * NOISE["b"]
    dates = SESSIONS[(SESSIONS.year == 2001)][::21]
    train_end = training_end(SESSIONS, pd.Timestamp("2000-12-29"), 5)
    return x, y, dates, train_end


def test_ridge_without_penalty_is_ols_with_a_lognormal_bias_correction():
    x, y, dates, train_end = ridge_inputs()
    out = run.ridge(x, y, SESSIONS, dates, 5, alpha=0.0)
    design = np.column_stack([np.ones(len(x.loc[:train_end])), x.loc[:train_end, "a"]])
    coef, *_ = np.linalg.lstsq(design, y.loc[:train_end], rcond=None)
    resid_var = np.var(y.loc[:train_end] - design @ coef, ddof=2)
    log = coef[0] + coef[1] * x.loc[dates, "a"]
    assert out["log"].to_numpy() == pytest.approx(log.to_numpy())
    assert out["variance"].to_numpy() == pytest.approx(np.exp(log + resid_var / 2).to_numpy())


def test_ridge_does_not_penalize_the_intercept():
    x, y, dates, train_end = ridge_inputs()
    out = run.ridge(x, y, SESSIONS, dates, 5, alpha=1e12)
    assert out["log"].to_numpy() == pytest.approx(y.loc[:train_end].mean())


def test_a_missing_signal_value_drops_that_date_for_both_models():
    signal = NOISE.copy()
    signal.loc["2002-06-28"] = np.nan
    forecasts = run.evaluate(RETURNS, VIX, signal, 21, alpha=1.0)
    full = run.evaluate(RETURNS, VIX, NOISE, 21, alpha=1.0)
    assert full.index.difference(forecasts.index).equals(pd.DatetimeIndex(["2002-06-28"]))
    assert forecasts[["baseline", "signal"]].notna().all().all()


def test_main_writes_one_file_per_horizon(tmp_path, monkeypatch):
    close = 100 * np.exp(RETURNS.cumsum())
    monkeypatch.setattr(run, "load", lambda name: {"SPX": close, "VIX": VIX}[name])
    monkeypatch.setattr(run, "ESTIMATION_START", "1994-01-03")
    monkeypatch.setattr(run, "OUT", tmp_path)

    def stub(columns, start):
        class Stub:
            def run(self, end_date):
                return NOISE.loc[start:].set_axis(columns, axis=1)

        return Stub

    monkeypatch.setattr(slow_macro, "Claims", stub(["c1", "c2"], "1994"))
    monkeypatch.setattr(slow_macro, "FinancialConditions", stub(["f1", "f2"], "2001-06"))
    monkeypatch.setattr(run, "CONFIG", tmp_path / "config.yaml")
    config = run.ROOT / "configs" / "slow_macro.yaml"
    text = config.read_text().replace("2013-01-01", "2003-01-01").replace(", 126, 252", "")
    run.CONFIG.write_text(text)
    run.main()
    written = sorted(p.name for p in tmp_path.glob("slow_macro_h*.csv"))
    assert written == ["slow_macro_h21.csv", "slow_macro_h5.csv", "slow_macro_h63.csv"]
    out = pd.read_csv(tmp_path / "slow_macro_h21.csv", parse_dates=["decision_date"])
    assert set(out["block"]) == {"claims", "anfci", "claims_anfci"}
    assert out.loc[out["block"] == "anfci", "decision_date"].min() >= pd.Timestamp("2003-01-01")
