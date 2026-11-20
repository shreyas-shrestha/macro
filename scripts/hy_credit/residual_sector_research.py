"""RESEARCH ONLY (not adopted): residual stress with past-fitted sector premiums (#24).

Production is M1 (``scripts/hy_credit/residual_research.py``; decision B). This runner
documents why sector residualization was rejected; its outputs are comparisons, not the
production signal.

    uv run --with matplotlib python scripts/hy_credit/residual_sector_research.py   (repo root)

Inputs (read-only): ``data/cache/hy_credit/`` and the point-in-time GICS pull in
``data/cache/hy_residual/sector/co_hgic.parquet`` (``wrds_pull_sector.py``).
Writes issuer-level rows to ``data/cache/hy_residual/past_sector/`` (git-ignored) and
aggregates/plots to ``results/hy_credit/residual_sector/``. The earlier no-sector outputs
(``results/hy_credit/residual/``) are left untouched as the M1 record.

Variants compared (descriptive only; nothing is tuned on any predictive target):
  M1   rating + maturity (no sector)
  M2   + same-day sector effects (diagnostic: absorbs contemporaneous sector repricing)
  M2P  + past-fitted sector premium (research; not adopted)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import residual_research as rr

from macro.config import DEV_END, HOLDOUT_START
from macro.data.sector import GICS_SECTORS, gics_sector_asof
from macro.features.hy_credit import HYCredit
from macro.features.hy_residual import (
    MIN_ISSUERS_PER_DAY,
    SECTOR_MIN_DAYS,
    SECTOR_SHRINK_DAYS,
    SECTOR_START_COVERAGE,
    HYResidualStress,
    _isotonic_increasing,
    add_sector_offset,
    bond_residuals,
    build_residual_features,
    daily_residual_features,
    issuer_residuals,
    past_sector_premium,
    prepare_bond_days,
    production_start,
    sector_effects,
)
from macro.features.testing import assert_no_lookahead

SOURCE = Path("data/cache/hy_credit")
SECTOR_FILE = Path("data/cache/hy_residual/sector/co_hgic.parquet")
CACHE = Path("data/cache/hy_residual/past_sector")
OUT = Path("results/hy_credit/residual_sector")
WINDOWS = {
    "2007-09 financial crisis": ("2007-07-01", "2009-06-30"),
    "2011 credit stress": ("2011-07-01", "2011-12-31"),
    "2012-13 HY residual elevation": ("2012-01-01", "2013-06-30"),
    "2015-16 energy/credit": ("2015-06-01", "2016-03-31"),
}
CALM = ("2013-07-01", "2014-06-30")
FOCAL = {
    "2007-09 financial crisis": "40",
    "2011 credit stress": "40",
    "2015-16 energy/credit": "10",
}
VARIANTS = ("M1", "M2", "M2P")
RESEARCH_SPEC = "M2P"  # past-sector variant studied here; NOT the production spec (M1)
FEATURE_COLS = rr.RESID + ["residual_mean"]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["trd_exctn_dt", "gvkey", "rating", "spread_bp", "remaining_maturity_y"]
    bonds = pd.read_parquet(SOURCE / "hy_bond_days.parquet", columns=cols)
    hgic = pd.read_parquet(SECTOR_FILE)
    if (pd.to_datetime(hgic["indfrom"]) >= HOLDOUT_START).any():
        raise ValueError("sector spells start in the holdout")
    return bonds, hgic


def coverage(prepared: pd.DataFrame) -> dict:
    issuers = prepared.drop_duplicates(["trd_exctn_dt", "gvkey"])
    has = issuers["sector"].notna()
    year = issuers["trd_exctn_dt"].dt.year
    by_class = issuers.groupby(issuers["hy"].map({1: "HY", 0: "IG"}))["sector"].apply(
        lambda s: round(float(s.notna().mean()), 4)
    )
    return {
        "issuer_days": int(len(issuers)),
        "with_pit_sector": round(float(has.mean()), 4),
        "by_year": has.groupby(year).mean().round(4).to_dict(),
        "by_class": by_class.to_dict(),
        "sector_counts_issuer_days": issuers["sector"]
        .fillna("unclassified")
        .value_counts()
        .to_dict(),
    }


def run_variant(name: str, frame: pd.DataFrame, start=None) -> tuple:
    res, stats = bond_residuals(frame, name, implied=(name != "M2"))
    if start is not None:
        res = res.loc[res["trd_exctn_dt"] >= start]
        stats = stats.loc[stats.index >= start]
    issuers = issuer_residuals(res)
    daily = daily_residual_features(issuers)
    for c in ("notch_gap_ge1_share", "notch_gap_ge2_share", "notch_gap_median", "notch_gap_p90"):
        if c not in daily:
            daily[c] = np.nan  # no implied rating under same-day sector effects
    return res, stats, issuers, daily


def separation(stats: pd.DataFrame, daily: pd.DataFrame) -> dict:
    """Adjacent rating-bucket separation relative to residual dispersion, per day."""
    effects = stats.filter(like="effect_")
    ratios = []
    for date, row in effects.iterrows():
        e = row.dropna()
        if len(e) < 3 or date not in daily.index:
            continue
        iso = _isotonic_increasing(e.to_numpy(), np.ones(len(e)))
        ratios.append((date, float(np.mean(np.diff(iso))) / daily.loc[date, "residual_iqr"]))
    r = pd.Series(dict(ratios))
    return {
        "median_adjacent_gap_over_iqr": round(float(r.median()), 4),
        "p10_p90": [round(float(r.quantile(0.1)), 4), round(float(r.quantile(0.9)), 4)],
    }


def sector_means(issuers: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Daily issuer-weighted mean residual by sector (date x sector)."""
    m = issuers.merge(labels, on=["trd_exctn_dt", "gvkey"], how="left")
    m = m.loc[m["sector"].notna()]
    return m.groupby(["trd_exctn_dt", "sector"])["residual"].mean().unstack("sector")


def normal_vs_unusual(issuers_by_variant: dict, labels: pd.DataFrame) -> dict:
    out = {}
    for name, issuers in issuers_by_variant.items():
        means = sector_means(issuers, labels)
        calm = means.loc[CALM[0] : CALM[1]]
        row = {
            "calm_cross_sector_sd_of_mean_residual": round(float(calm.std(axis=1).mean()), 4),
            "calm_sector_mean_residual": calm.mean().round(4).to_dict(),
        }
        for label, sector in FOCAL.items():
            a, b = WINDOWS[label]
            row[f"{label}: sector {sector} mean residual"] = round(
                float(means.loc[a:b, sector].mean()), 4
            )
        out[name] = row
    return out


def tail_composition(issuers: pd.DataFrame, labels: pd.DataFrame) -> dict:
    m = issuers.merge(labels, on=["trd_exctn_dt", "gvkey"], how="left")
    cutoff = m.groupby("trd_exctn_dt")["residual"].transform(lambda s: s.quantile(0.9))
    top = m.loc[m["residual"] > cutoff]
    periods = dict(WINDOWS, calm=CALM)
    return {
        label: top.loc[top["trd_exctn_dt"].between(a, b), "sector"]
        .fillna("unclassified")
        .value_counts(normalize=True)
        .round(3)
        .head(5)
        .to_dict()
        for label, (a, b) in periods.items()
    }


def notch_stability(daily: pd.DataFrame, hy: pd.Series) -> dict:
    share = daily["notch_gap_ge1_share"]
    aligned = pd.concat([share, hy], axis=1, sort=True).dropna()
    return {
        "ge1_share_median": round(float(share.median()), 4),
        "ge1_share_sd_over_time": round(float(share.std()), 4),
        "ge2_share_median": round(float(daily["notch_gap_ge2_share"].median()), 4),
        "corr_ge1_share_with_hy_spread_obs_date": round(
            float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1])), 3
        ),
    }


def rolling_premium(effects: pd.DataFrame, window: int = 504) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sensitivity only: same rule as the M2P expanding premium, but a rolling window."""
    prior_n = effects.notna().rolling(window, min_periods=1).sum().shift(1).fillna(0)
    prior_sum = effects.fillna(0).rolling(window, min_periods=1).sum().shift(1).fillna(0)
    mean = prior_sum / prior_n.where(prior_n > 0)
    premium = (mean * prior_n / (prior_n + SECTOR_SHRINK_DAYS)).where(
        prior_n >= SECTOR_MIN_DAYS, 0.0
    )
    return premium.fillna(0.0), prior_n


def premium_tracking(effects: pd.DataFrame, premium: pd.DataFrame, labels, start) -> dict:
    """How well the past premium predicts each day's same-day sector effect.

    Per day: issuer-weighted RMSE across sectors of (same-day effect - past premium),
    against the no-sector baseline (premium = 0). Production estimator plus a 504-day
    rolling-window variant reported as a descriptive sensitivity only (not adopted).
    """
    counts = labels.dropna(subset=["sector"]).groupby(["trd_exctn_dt", "sector"]).size()
    weights = counts.unstack("sector").reindex(index=effects.index, columns=effects.columns)
    rolling, _ = rolling_premium(effects)

    def rmse(pred):
        err = (effects - pred) ** 2
        ok = err.notna() & weights.notna()
        num = (err.where(ok) * weights.where(ok)).sum(axis=1)
        return np.sqrt(num / weights.where(ok).sum(axis=1))

    frame = pd.DataFrame(
        {
            "none": rmse(0.0),
            "expanding (M2P estimator)": rmse(premium),
            "rolling 504 (sensitivity)": rmse(rolling),
        }
    ).loc[start:]
    by_year = frame.groupby(frame.index.year).mean().round(4)
    return {
        "all_days_mean": frame.mean().round(4).to_dict(),
        "by_year": by_year.to_dict(orient="index"),
    }


def plots(variant_daily: dict, coverage_by_year: dict, premium: pd.DataFrame, comp: dict):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.0,
        }
    )
    names = {
        "M1": "M1 no sector",
        "M2": "M2 same-day sector (diagnostic)",
        "M2P": "M2P past sector (research, not adopted)",
    }
    paths = []

    def shade(ax):
        for a, b in WINDOWS.values():
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#888888", alpha=0.12, lw=0)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, col, label in zip(
        axes,
        ["residual_p90", "residual_iqr", "residual_positive_share"],
        ["residual P90", "residual IQR", "share above hist. P90"],
        strict=True,
    ):
        for color, v in zip(rr.COLORS, VARIANTS, strict=False):
            s = variant_daily[v][col].rolling(20, min_periods=10).median()
            ax.plot(s.index, s, color=color, label=names[v])
        ax.set_ylabel(f"{label} (20-day median)", fontsize=8)
        shade(ax)
    axes[0].legend(fontsize=8)
    axes[0].set_title(
        "Residual stress by sector treatment (observation dates; shaded: annotated windows)"
    )
    fig.tight_layout()
    paths.append(OUT / "variants_p90_iqr_share.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    years = sorted(coverage_by_year)
    axes[0].bar(years, [coverage_by_year[y] for y in years], color=rr.COLORS[0], width=0.6)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Rated issuer-days with a point-in-time GICS sector", fontsize=10)
    big = premium.abs().mean().sort_values(ascending=False).index[:5]
    for color, s in zip(rr.COLORS, big, strict=False):
        axes[1].plot(premium.index, premium[s], color=color, label=f"{s} {GICS_SECTORS.get(s, '')}")
    axes[1].set_title("Past-fitted sector premium used on each date (log pts)", fontsize=10)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    paths.append(OUT / "sector_coverage_and_premium.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    labels = list(comp["M1"])
    width = 0.38
    xs = np.arange(len(labels))
    for k, (v, color) in enumerate((("M1", rr.COLORS[0]), ("M2P", rr.COLORS[2]))):
        top = [
            max(
                ((s, x) for s, x in comp[v][lab].items() if s != "unclassified"),
                key=lambda kv: kv[1],
                default=("", 0),
            )
            if comp[v][lab]
            else ("", 0)
            for lab in labels
        ]
        bars = ax.bar(
            xs + (k - 0.5) * width, [t[1] for t in top], width, color=color, label=names[v]
        )
        for bar, (s, _) in zip(bars, top, strict=True):
            ax.annotate(
                GICS_SECTORS.get(s, s),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(xs, labels, fontsize=8)
    ax.set_ylabel("share of top residual decile")
    ax.set_title("Largest classified sector in the upper residual tail")
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths.append(OUT / "tail_top_sector.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)
    return paths


def main() -> int:
    global SECTOR_FILE, CACHE, OUT
    args = sys.argv[1:]
    for flag in ("--sector-file", "--cache", "--out"):  # overrides, for smoke tests only
        if flag in args:
            value = Path(args[args.index(flag) + 1])
            if flag == "--sector-file":
                SECTOR_FILE = value
            elif flag == "--cache":
                CACHE = value
            else:
                OUT = value
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    bonds, hgic = load()
    sector = gics_sector_asof(bonds, hgic)
    prepared = prepare_bond_days(bonds, sector)
    cov = coverage(prepared)
    print("coverage", cov["with_pit_sector"], flush=True)

    print("same-day sector fits (M2)", flush=True)
    m2_res, m2_stats = bond_residuals(prepared, "M2")
    effects = sector_effects(m2_stats, prepared["trd_exctn_dt"])
    premium, prior_n = past_sector_premium(effects)
    with_offset = add_sector_offset(prepared, premium, prior_n)
    start = production_start(with_offset)
    print("comparison start", start, flush=True)

    variants, variant_issuers, variant_daily, variant_stats = {}, {}, {}, {}
    for v in VARIANTS:
        print(f"variant {v}", flush=True)
        frame = with_offset if v == "M2P" else prepared
        res, stats, issuers, daily = run_variant(v, frame, start)
        variants[v], variant_issuers[v], variant_daily[v], variant_stats[v] = (
            res,
            issuers,
            daily,
            stats,
        )

    prod = variant_daily[RESEARCH_SPEC]
    prod_issuers = variant_issuers[RESEARCH_SPEC]
    checks = {
        "no_holdout_dates": bool(prod.index.max() < HOLDOUT_START),
        "no_duplicate_issuer_days": bool(
            not prod_issuers.duplicated(["trd_exctn_dt", "gvkey"]).any()
        ),
        "max_abs_bond_residual_log": float(variants[RESEARCH_SPEC]["residual"].abs().max()),
        "min_issuers_per_fitted_day": int(prod["n_issuers"].min()),
        "premium_uses_only_earlier_days": "by construction (cumsum shifted one fitted day); tested",
    }
    assert checks["no_holdout_dates"] and checks["no_duplicate_issuer_days"]
    assert checks["min_issuers_per_fitted_day"] >= MIN_ISSUERS_PER_DAY
    assert checks["max_abs_bond_residual_log"] < np.log(3500 / 5)

    prod_issuers.to_parquet(CACHE / "issuer_residuals.parquet")
    premium.to_parquet(CACHE / "sector_premium.parquet")
    prod.to_parquet(CACHE / "residual_daily.parquet")
    premium.round(5).to_csv(OUT / "sector_premium_daily.csv")

    cutoffs = ["2008-09-15", "2008-12-31", "2011-08-08", "2014-01-02", "2016-02-11", "2017-05-31"]
    assert_no_lookahead(
        HYResidualStress(pd.read_parquet(CACHE / "residual_daily.parquet")), cutoffs
    )

    hy = HYCredit().run(DEV_END)
    rr.WINDOWS = WINDOWS
    fit_table, comparisons = [], {}
    labels = prepared.drop_duplicates(["trd_exctn_dt", "gvkey"])[
        ["trd_exctn_dt", "gvkey", "sector"]
    ]
    for v in VARIANTS:
        stats = variant_stats[v].loc[variant_stats[v]["fitted"]]
        daily = variant_daily[v]
        fit_table.append(
            {
                "model": v,
                "days": int(len(stats)),
                "issuer_days": int(daily["n_issuers"].sum()),
                "median_daily_r2": round(float(stats["r2"].median()), 4),
                "p10_daily_r2": round(float(stats["r2"].quantile(0.1)), 4),
                "r2_sd_across_days": round(float(stats["r2"].std()), 4),
                "median_residual_iqr": round(float(daily["residual_iqr"].median()), 4),
                "median_residual_p90": round(float(daily["residual_p90"].median()), 4),
                "median_residual_std": round(float(daily["residual_std"].median()), 4),
            }
        )
        feats = build_residual_features(daily, columns=FEATURE_COLS, end_date=DEV_END)
        joined = hy.join(feats, how="left").loc[: daily.index.max() + pd.Timedelta(days=10)]
        comparisons[v] = rr.compare(joined)
    fit_table = pd.DataFrame(fit_table)
    fit_table.to_csv(OUT / "fit_comparison.csv", index=False)
    r2_by_year = pd.DataFrame(
        {
            v: variant_stats[v]
            .loc[variant_stats[v]["fitted"], "r2"]
            .groupby(lambda d: d.year)
            .median()
            for v in VARIANTS
        }
    ).round(4)
    r2_by_year.to_csv(OUT / "r2_by_year.csv")
    comp = {v: tail_composition(variant_issuers[v], labels) for v in VARIANTS}
    episodes = {
        v: {
            label: {
                "p90_mean": round(float(variant_daily[v].loc[a:b, "residual_p90"].mean()), 4),
                "iqr_mean": round(float(variant_daily[v].loc[a:b, "residual_iqr"].mean()), 4),
                "share_mean": round(
                    float(variant_daily[v].loc[a:b, "residual_positive_share"].mean()), 4
                ),
                "p90_peak_date": str(variant_daily[v].loc[a:b, "residual_p90"].idxmax().date()),
            }
            for label, (a, b) in dict(WINDOWS, calm=CALM).items()
        }
        for v in VARIANTS
    }
    report = {
        "status": "research comparison only; production residual spec is M1 (decision B)",
        "sector_source": "comp.co_hgic (historical GICS; indfrom..indthru, capped 2017-12-31), "
        "two-digit gsector, assigned per issuer-day by macro.data.sector.gics_sector_asof",
        "sector_pull_manifest": json.loads((SECTOR_FILE.parent / "manifest.json").read_text()),
        "sector_coverage": cov,
        "premium_rule": {
            "same_day_effects": "daily WLS M2 sector dummies (>= 10 issuers; thinner/unclassified "
            "pooled as baseline), centred to issuer-weighted mean 0",
            "premium": f"expanding mean over earlier fitted days * n/(n+{SECTOR_SHRINK_DAYS}); "
            f"0 until n >= {SECTOR_MIN_DAYS}; 0 for unclassified issuers",
            "comparison_start_rule": f">= {SECTOR_START_COVERAGE:.0%} of classified issuers "
            "have a usable premium",
            "comparison_start": str(start.date()) if start is not None else None,
        },
        "fit_comparison": fit_table.to_dict(orient="records"),
        "r2_by_year": r2_by_year.to_dict(orient="index"),
        "checks": checks,
        "no_lookahead_real_data_cutoffs": cutoffs,
        "features_m2p_research": {c: rr.q(prod[c]) for c in FEATURE_COLS},
        "episodes": episodes,
        "normal_vs_unusual_sector": normal_vs_unusual(variant_issuers, labels),
        "tail_sector_composition": comp,
        "premium_tracking_rmse": premium_tracking(effects, premium, labels, start),
        "notch_separation": {
            v: separation(variant_stats[v], variant_daily[v]) for v in ("M1", "M2P")
        },
        "notch_stability": {
            v: notch_stability(
                variant_daily[v], pd.read_parquet(SOURCE / "hy_daily.parquet")["hy_spread_bp"]
            )
            for v in ("M1", "M2P")
        },
        "comparison_vs_hycredit": comparisons,
        "last_decision_date": str(
            build_residual_features(prod, end_date=DEV_END)["residual_p90"]
            .last_valid_index()
            .date()
        ),
    }
    roll, roll_n = rolling_premium(effects)
    _, roll_stats, roll_issuers, roll_daily = run_variant(
        "M2P", add_sector_offset(prepared, roll, roll_n), start
    )
    roll_feats = build_residual_features(roll_daily, columns=FEATURE_COLS, end_date=DEV_END)
    roll_joined = hy.join(roll_feats, how="left").dropna(subset=["hy_spread", "residual_p90"])
    report["sensitivity_rolling_504_not_adopted"] = {
        "median_daily_r2": round(float(roll_stats.loc[roll_stats["fitted"], "r2"].median()), 4),
        "normal_vs_unusual": normal_vs_unusual({"rolling": roll_issuers}, labels)["rolling"],
        "corr_with_hy_spread": {
            c: round(float(roll_joined[c].corr(roll_joined["hy_spread"])), 3)
            for c in ("residual_p90", "residual_iqr", "residual_positive_share")
        },
        "p90_mean_by_window": {
            lab: round(float(roll_daily.loc[a:b, "residual_p90"].mean()), 4)
            for lab, (a, b) in dict(WINDOWS, calm=CALM).items()
        },
    }
    (OUT / "residual_sector_report.json").write_text(json.dumps(report, indent=1, default=str))
    for p in plots(variant_daily, cov["by_year"], premium, comp):
        print("plot", p)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
