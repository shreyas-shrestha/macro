"""Pre-harness diagnostics for the credit signal (#11).

Pulls BAA10Y, DBAA and DAAA through DEV_END, plots levels, 20-session changes and gaps,
measures the FRED release lag from ALFRED vintages, and checks how much history FRED
still serves for BAMLH0A0HYM2. Writes research/credit_diagnostics.png and prints a
summary to paste into the PR.

    uv run python experiments/credit_diagnostics.py

Every request stops at DEV_END, so no holdout observation is downloaded.
"""

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from macro.config import DEV_END, load_env  # noqa: E402
from macro.data.calendar import nyse_sessions  # noqa: E402
from macro.features.credit import Credit, load_fred  # noqa: E402

API = "https://api.stlouisfed.org/fred/series/observations"
OUT = Path(__file__).resolve().parents[1] / "research" / "credit_diagnostics.png"


def fred_json(**params) -> dict:
    params |= {"api_key": os.environ["FRED_API_KEY"], "file_type": "json"}
    with urllib.request.urlopen(f"{API}?{urllib.parse.urlencode(params)}", timeout=60) as r:
        return json.load(r)


def gap_report(name: str, raw: pd.Series) -> None:
    sessions = nyse_sessions(raw.index.min(), DEV_END)
    on_sessions = raw.reindex(sessions)
    missing = on_sessions.isna()
    runs = missing.groupby((~missing).cumsum()).sum()
    off_calendar = raw.index.difference(sessions)
    print(
        f"{name}: {raw.notna().sum()} obs {raw.index.min().date()}..{raw.index.max().date()}; "
        f"{missing.sum()} NYSE sessions missing, longest run {runs.max()}; "
        f"{len(off_calendar)} observations on non-NYSE days"
    )
    print(f"  missing sessions (first 15): {[d.date() for d in sessions[missing][:15]]}")


def release_lag_report(series_id: str) -> None:
    """First ALFRED vintage date minus observation date, in NYSE sessions."""
    data = fred_json(
        series_id=series_id,
        realtime_start="1776-07-04",
        realtime_end="9999-12-31",
        observation_start="1986-01-01",
        observation_end=DEV_END.strftime("%Y-%m-%d"),
    )
    obs = pd.DataFrame(data["observations"])
    obs = obs[obs["value"] != "."]
    obs["date"] = pd.to_datetime(obs["date"])
    obs["realtime_start"] = pd.to_datetime(obs["realtime_start"])
    first = obs.groupby("date")["realtime_start"].min()
    vintages = obs.groupby("date")["realtime_start"].nunique()
    sessions = nyse_sessions("1985-01-01", "2030-12-31")
    lag = sessions.searchsorted(first.to_numpy()) - sessions.searchsorted(first.index)
    lag = pd.Series(lag, index=first.index)
    tracked = lag[first > first.min()]  # drop observations whose first vintage is the series' first
    print(f"{series_id}: ALFRED vintages start {first.min().date()}")
    print(f"  observations revised after first release: {(vintages > 1).sum()} of {len(vintages)}")
    if len(tracked):
        print(f"  first-release lag in NYSE sessions (n={len(tracked)}):")
        print("  " + tracked.value_counts().sort_index().to_string().replace("\n", "\n  "))
    print(
        "  ALFRED stamps dates, not times: a lag of 1 does not show the value was out by 16:00 ET."
    )


def hy_oas_report() -> None:
    try:
        hy = load_fred("BAMLH0A0HYM2", pd.Timestamp("1996-12-01"), DEV_END)
        print(
            f"BAMLH0A0HYM2: {hy.notna().sum()} obs through DEV_END, first {hy.index.min().date()}"
        )
    except Exception as err:
        print(f"BAMLH0A0HYM2: no development-sample data from FRED ({err})")


def main() -> None:
    load_env()
    raw = {
        sid: load_fred(sid, pd.Timestamp("1986-01-01"), DEV_END)
        for sid in ("BAA10Y", "DBAA", "DAAA")
    }
    for sid, series in raw.items():
        gap_report(sid, series)
    for sid in ("BAA10Y", "DBAA", "DGS10"):
        release_lag_report(sid)
    hy_oas_report()

    features = Credit().run(DEV_END)
    print(features.describe().T.to_string())

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    spread = raw["BAA10Y"]
    axes[0].plot(spread.index, spread, lw=0.8, color="#1f4e79")
    axes[0].set_ylabel("BAA10Y, pp")
    missing = nyse_sessions(spread.index.min(), DEV_END).difference(spread.dropna().index)
    axes[0].plot(missing, [spread.min()] * len(missing), "|", color="#c0392b", ms=6, label="gap")
    axes[0].legend(loc="upper left", frameon=False)
    axes[1].plot(features.index, features["baa10y_change_20"], lw=0.8, color="#1f4e79")
    axes[1].set_ylabel("20-session change, pp")
    axes[2].plot(features.index, features["baa10y_z_252"], lw=0.8, color="#1f4e79")
    axes[2].set_ylabel("252-session z")
    axes[3].plot(features.index, features["baa_aaa_level"], lw=0.8, color="#1f4e79")
    axes[3].set_ylabel("Baa - Aaa, pp")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2)
    fig.suptitle("Credit spreads, development sample (features use a 2-session release lag)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
