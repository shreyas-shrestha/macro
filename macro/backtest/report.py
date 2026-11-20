"""Scores a signal model against the no-signal baseline and records the result (#10).

``compare`` and ``breakdown`` take aligned frames from ``walk_forward`` with columns
``log`` and ``variance``, plus the realized variance target.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from macro.backtest.scoring import clark_west, mse_log_vol_losses, qlike_losses, r2_oos

# Stress episodes reported separately, as listed in #10. On the monthly grid the 2010,
# 2011 and 2015 windows hold two or three forecasts each.
EPISODES = {
    "2008 crisis": ("2008-09-01", "2009-03-31"),
    "2010 flash crash": ("2010-05-01", "2010-07-31"),
    "2011 downgrade": ("2011-08-01", "2011-10-31"),
    "2015 China deval": ("2015-08-01", "2015-09-30"),
}


def nw_lag(h: int) -> int:
    """Newey–West lag in monthly grid rows: how many later month-end targets an
    h-session target overlaps, at least 1."""
    return max(1, math.ceil(h / 21))


def _common(target: pd.Series, baseline: pd.DataFrame, model: pd.DataFrame):
    rows = target.index.intersection(baseline.dropna().index).intersection(model.dropna().index)
    return target.loc[rows], baseline.loc[rows], model.loc[rows]


def compare(target: pd.Series, baseline: pd.DataFrame, model: pd.DataFrame, h: int) -> dict:
    """Scores on the rows where both models have a forecast.

    QLIKE and MSE on log volatility use the bias-corrected ``variance``. Clark–West
    compares squared errors of ``log`` against log realized variance: ``log`` is what the
    ridge estimates, and the bias correction would add a different constant to each
    model's errors.
    """
    y, b, m = _common(target, baseline, model)
    base_qlike = qlike_losses(b["variance"], y)
    model_qlike = qlike_losses(m["variance"], y)
    t_stat, p_value = clark_west(np.log(y), b["log"], m["log"], nw_lag(h))
    return {
        "horizon": h,
        "n": len(y),
        "start": y.index[0].date(),
        "end": y.index[-1].date(),
        "qlike_baseline": base_qlike.mean(),
        "qlike_model": model_qlike.mean(),
        # Share of the baseline's QLIKE removed: 0.05 is 5% lower.
        "qlike_reduction": r2_oos(model_qlike, base_qlike),
        "r2_oos_mse_log_vol": r2_oos(
            mse_log_vol_losses(m["variance"], y), mse_log_vol_losses(b["variance"], y)
        ),
        "cw_t": t_stat,
        "cw_p": p_value,
        "nw_lag": nw_lag(h),
    }


def qlike_gain(target: pd.Series, baseline: pd.DataFrame, model: pd.DataFrame) -> pd.Series:
    """Per-row baseline QLIKE minus model QLIKE; positive where the signal helps."""
    y, b, m = _common(target, baseline, model)
    gain = qlike_losses(b["variance"], y) - qlike_losses(m["variance"], y)
    return pd.Series(gain, index=y.index)


def breakdown(target: pd.Series, baseline: pd.DataFrame, model: pd.DataFrame) -> pd.DataFrame:
    """Mean QLIKE of both models and the count of forecasts, by year and stress episode."""
    y, b, m = _common(target, baseline, model)
    losses = pd.DataFrame(
        {"baseline": qlike_losses(b["variance"], y), "model": qlike_losses(m["variance"], y)},
        index=y.index,
    )

    def summarize(group: pd.DataFrame) -> pd.Series:
        return pd.Series({"n": len(group), **group.mean()})

    by_year = pd.DataFrame(
        {str(year): summarize(g) for year, g in losses.groupby(losses.index.year)}
    ).T
    episodes = {
        name: summarize(losses.loc[start:end])
        for name, (start, end) in EPISODES.items()
        if not losses.loc[start:end].empty
    }
    out = pd.concat([by_year, pd.DataFrame(episodes).T])
    out["n"] = out["n"].astype(int)
    out["reduction"] = 1 - out["model"] / out["baseline"]
    return out


def update_scorecard(path: Path, signal: str, rows: list[dict]) -> pd.DataFrame:
    """Replace ``signal``'s rows in the scorecard at ``path``, keeping other signals'."""
    new = pd.DataFrame(rows).assign(signal=signal)
    new = new[["signal"] + [c for c in new.columns if c != "signal"]]
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old["signal"] != signal], new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False, float_format="%.6g")
    return new


def plot_cumulative(path: Path, signal: str, curves: dict[int, pd.Series]) -> bool:
    """Save the cumulative QLIKE gain over the baseline, one line per horizon.

    A rising line means the signal is helping. Returns False if matplotlib is missing.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(10, 4))
    for h, gain in curves.items():
        ax.plot(gain.index, gain.cumsum(), label=f"h = {h}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"{signal}: cumulative QLIKE improvement over the no-signal model")
    ax.set_ylabel("cumulative (baseline - model) QLIKE")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True
