"""Development-only diagnostic plots for #24 (aggregates only; local data; no WRDS).

    uv run --with matplotlib python plot_hy.py [DAILY_PARQUET] [OUT_DIR]

Run from the repo root (defaults: data/cache/hy_credit/hy_daily.parquet, results/hy_credit).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

HOLDOUT = pd.Timestamp("2018-01-01")


def make_plots(daily_path: Path, out: Path) -> list[Path]:
    from macro.features.hy_credit import HYCredit

    daily = pd.read_parquet(daily_path)
    if len(daily) and pd.DatetimeIndex(daily.index).max() >= HOLDOUT:
        raise SystemExit("daily aggregates contain holdout dates")
    out.mkdir(parents=True, exist_ok=True)
    written = []

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    daily[["hy_spread_bp", "ig_spread_bp"]].plot(ax=ax[0], lw=0.8)
    ax[0].set_ylabel("issuer-median spread (bp)")
    ax[0].legend(["HY", "IG"])
    daily["hy_ig_gap_bp"].plot(ax=ax[1], lw=0.8, color="k")
    ax[1].set_ylabel("HY - IG (bp)")
    fig.suptitle("TRACE issuer-weighted median spreads (observation dates, development sample)")
    fig.tight_layout()
    written.append(out / "hy_ig_series.png")
    fig.savefig(written[-1], dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    counts = daily[["n_hy_issuers", "n_ig_issuers", "n_unrated_issuers"]]
    counts.rolling(20, min_periods=1).median().plot(ax=ax, lw=0.8)
    ax.set_ylabel("issuers per day (20-day median)")
    ax.legend(["HY rated", "IG rated", "unrated (excluded)"])
    fig.tight_layout()
    written.append(out / "issuer_counts.png")
    fig.savefig(written[-1], dpi=120)
    plt.close(fig)

    end = min(
        pd.DatetimeIndex(daily.index).max() + pd.Timedelta(days=7), pd.Timestamp("2017-12-29")
    )
    features = HYCredit(daily).run(end)
    fig, axes = plt.subplots(len(features.columns), 1, figsize=(11, 10), sharex=True)
    for ax, col in zip(axes, features.columns, strict=True):
        features[col].plot(ax=ax, lw=0.7)
        ax.set_ylabel(col, rotation=0, ha="right", fontsize=8)
    fig.suptitle("HYCredit features on decision dates (release lag 1)")
    fig.tight_layout()
    written.append(out / "features.png")
    fig.savefig(written[-1], dpi=120)
    plt.close(fig)
    return written


if __name__ == "__main__":
    daily_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/cache/hy_credit/hy_daily.parquet")
    )
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/hy_credit")
    for path in make_plots(daily_path, out_dir):
        print("wrote", path)
