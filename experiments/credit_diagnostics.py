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
import time
import urllib.error
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


def first_releases(series_id: str) -> pd.DataFrame:
    """Earliest ALFRED vintage date and value per observation, queried a year at a time.

    FRED caps one request at 2000 vintage dates, so the real-time window is cut into
    calendar years. Observations whose first vintage predates the window (the series'
    initial load into ALFRED) come back stamped with the window start and are dropped.
    """
    frames = []
    for year in range(1986, DEV_END.year + 1):
        start = pd.Timestamp(year=year, month=1, day=1)
        end = min(pd.Timestamp(year=year, month=12, day=31), DEV_END)
        try:
            data = fred_json(
                series_id=series_id,
                observation_start=start.strftime("%Y-%m-%d"),
                observation_end=end.strftime("%Y-%m-%d"),
                realtime_start=start.strftime("%Y-%m-%d"),
                realtime_end=(end + pd.Timedelta(days=15)).strftime("%Y-%m-%d"),
                limit=100000,
            )
        except urllib.error.HTTPError as err:
            print(f"  {series_id} {year}: {err} {err.read().decode()[:200]}")
            continue
        finally:
            time.sleep(0.6)  # FRED allows 120 requests a minute
        obs = pd.DataFrame(data.get("observations", []))
        if obs.empty:
            continue
        obs = obs[obs["value"] != "."]
        obs["date"] = pd.to_datetime(obs["date"])
        obs["realtime_start"] = pd.to_datetime(obs["realtime_start"])
        obs = obs[obs["realtime_start"] > obs["date"]]
        frames.append(obs.sort_values("realtime_start").groupby("date").first())
    if not frames:
        return pd.DataFrame(columns=["realtime_start", "value"])
    return pd.concat(frames)[["realtime_start", "value"]]


def release_lag_report(series_id: str, current: pd.Series | None = None) -> None:
    """First-release date minus observation date, in NYSE sessions (ALFRED)."""
    first = first_releases(series_id)
    if first.empty:
        print(f"{series_id}: no ALFRED vintages after the initial load in the development sample")
        return
    sessions = nyse_sessions("1985-01-01", "2030-12-31")
    released = first["realtime_start"].to_numpy()
    lag = sessions.searchsorted(released) - sessions.searchsorted(first.index)
    lag = pd.Series(lag, index=first.index)
    print(
        f"{series_id}: {len(lag)} observations with a tracked first release, "
        f"{first.index.min().date()}..{first.index.max().date()}"
    )
    print("  first-release lag in NYSE sessions:")
    print("  " + lag.value_counts().sort_index().to_string().replace("\n", "\n  "))
    if current is not None:
        initial = pd.to_numeric(first["value"])
        both = pd.concat([initial, current], axis=1, join="inner").dropna()
        revised = (both.iloc[:, 0] - both.iloc[:, 1]).abs() > 1e-9
        print(f"  first release differs from today's value: {revised.sum()} of {len(both)}")
        if revised.any():
            print(f"  e.g. {[d.date() for d in both.index[revised][:10]]}")
    print(
        "  ALFRED stamps dates, not times: a lag of 1 does not prove the value was out by 16:00 ET."
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
        release_lag_report(sid, raw.get(sid))
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
