"""report for #10 - compare baseline vs baseline + signal, then save the results"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from macro.backtest.scoring import clark_west, mse_log_vol_losses, qlike_losses, r2_oos

# crisis windows we check separately (from issue #10)
EPISODES = {
    "2008 crisis": ("2008-09-01", "2009-03-31"),
    "2010 flash crash": ("2010-05-01", "2010-07-31"),
    "2011 downgrade": ("2011-08-01", "2011-10-31"),
    "2015 China deval": ("2015-08-01", "2015-09-30"),
}


def nw_lag(h: int) -> int:
    """newey-west lag in rows - grid is monthly (~21 trading days apart), so an h-day
    target overlaps about h / 21 of the next rows"""
    return max(1, math.ceil(h / 21))


def compare(target: pd.Series, baseline: pd.Series, model: pd.Series, h: int) -> dict:
    """score baseline vs model, only on rows where BOTH have a forecast (fair fight)"""
    rows = target.index.intersection(baseline.dropna().index).intersection(model.dropna().index)
    y, b, m = target.loc[rows], baseline.loc[rows], model.loc[rows]

    base_qlike, model_qlike = qlike_losses(b, y), qlike_losses(m, y)
    # luck check on log variance (the scale the ridge model predicts)
    t_stat, p_value = clark_west(np.log(y), np.log(b), np.log(m), nw_lag(h))
    return {
        "horizon": h,
        "n": len(rows),
        "start": rows[0].date(),
        "end": rows[-1].date(),
        "qlike_baseline": base_qlike.mean(),
        "qlike_model": model_qlike.mean(),
        # how much of the baseline's qlike the signal removes (0.05 = 5% lower)
        "qlike_reduction": r2_oos(model_qlike, base_qlike),
        "r2_oos_mse_log_vol": r2_oos(mse_log_vol_losses(m, y), mse_log_vol_losses(b, y)),
        "cw_t": t_stat,
        "cw_p": p_value,
        "nw_lag": nw_lag(h),
    }


def breakdown(target: pd.Series, baseline: pd.Series, model: pd.Series) -> pd.DataFrame:
    """qlike of baseline + model by year and by crisis - did it only help in 2008?"""
    rows = target.index.intersection(baseline.dropna().index).intersection(model.dropna().index)
    losses = pd.DataFrame(
        {
            "baseline": qlike_losses(baseline.loc[rows], target.loc[rows]),
            "model": qlike_losses(model.loc[rows], target.loc[rows]),
        },
        index=rows,
    )
    by_year = losses.groupby(losses.index.year).mean()
    by_year.index = by_year.index.astype(str)
    episodes = {
        name: losses.loc[start:end].mean()
        for name, (start, end) in EPISODES.items()
        if not losses.loc[start:end].empty
    }
    out = pd.concat([by_year, pd.DataFrame(episodes).T])
    out["reduction"] = 1 - out["model"] / out["baseline"]
    return out


def update_scorecard(path: Path, signal: str, rows: list[dict]) -> pd.DataFrame:
    """swap in this signal's rows in the shared scorecard csv (other signals stay)"""
    new = pd.DataFrame(rows).assign(signal=signal)
    new = new[["signal"] + [c for c in new.columns if c != "signal"]]
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old[old["signal"] != signal], new], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(path, index=False, float_format="%.6g")
    return new


def plot_cumulative(path: Path, signal: str, curves: dict[int, pd.Series]) -> bool:
    """save the standard chart: running total of (baseline - model) qlike, 1 line per h

    line going UP = signal is helping. returns False if matplotlib isn't installed
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(10, 4))
    for h, diff in curves.items():
        ax.plot(diff.index, diff.cumsum(), label=f"h = {h}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_title(f"{signal}: cumulative QLIKE improvement over the no-signal model")
    ax.set_ylabel("cumulative (baseline - model) QLIKE")
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True
