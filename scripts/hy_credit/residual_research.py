"""Descriptive residual-stress research for #24 (development data only; no WRDS).

    uv run --with matplotlib python scripts/hy_credit/residual_research.py   (repo root)

PRODUCTION residual-stress build (spec M1: rating + maturity, no sector adjustment).
Reads (read-only) ``data/cache/hy_credit/`` and, for interpretation only, the
point-in-time GICS spells in ``data/cache/hy_residual/sector/co_hgic.parquet``.
Writes licensed issuer-level rows only to ``data/cache/hy_residual/``
(git-ignored) and aggregate tables/plots to ``results/hy_credit/residual/``.

No predictive target is used anywhere: specification and thresholds are fixed by the
rules in ``macro.features.hy_residual`` and the descriptive rule ``R2_GAIN_RULE`` below.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from macro.config import DEV_END, HOLDOUT_START
from macro.data.sector import gics_sector_asof
from macro.features.hy_credit import HYCredit, expanding_zscore
from macro.features.hy_residual import (
    CHOSEN_SPEC,
    FEATURES,
    MIN_ISSUERS_PER_DAY,
    HYResidualStress,
    bond_residuals,
    build_residual_features,
    daily_residual_features,
    issuer_residuals,
    prepare_bond_days,
)
from macro.features.testing import assert_no_lookahead

SOURCE = Path("data/cache/hy_credit")
CACHE = Path("data/cache/hy_residual")
OUT = Path("results/hy_credit/residual")
SECTOR_FILE = Path("data/cache/hy_residual/sector/co_hgic.parquet")  # point-in-time GICS
R2_GAIN_RULE = 0.02  # add a model component only if median daily R2 rises by >= 0.02
COVERAGE_MONTHS = ("2007-06", "2010-01", "2016-06")
# Fixed calendar windows, used only to annotate results after construction.
WINDOWS = {
    "2007-09 financial crisis": ("2007-07-01", "2009-06-30"),
    "2011 credit stress": ("2011-07-01", "2011-12-31"),
    "2012-13 HY residual elevation": ("2012-01-01", "2013-06-30"),
    "2015-16 energy/credit": ("2015-06-01", "2016-03-31"),
}
COMPARE = ["hy_spread", "hy_ig_gap", "hy_spread_change_20", "hy_ig_gap_change_20"]
RESID = [
    "residual_p90",
    "residual_p95",
    "residual_iqr",
    "residual_mad",
    "residual_std",
    "residual_positive_share",
    "mean_positive_residual",
    "notch_gap_ge1_share",
    "notch_gap_ge2_share",
]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]  # categorical slots 1-5, fixed


def q(series: pd.Series) -> dict:
    s = series.dropna()
    qs = s.quantile([0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]).round(4)
    return {"n": int(len(s)), **{f"q{int(k * 100):02d}": float(v) for k, v in qs.items()}}


def load_inputs():
    cols = ["trd_exctn_dt", "gvkey", "rating", "spread_bp", "remaining_maturity_y", "grade"]
    bonds = pd.read_parquet(SOURCE / "hy_bond_days.parquet", columns=cols)
    sector = gics_sector_asof(bonds, pd.read_parquet(SECTOR_FILE))  # per row, point in time
    return bonds, sector


def spec_ladder(prepared: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Fit M0, M1, M3 on all issuers and M1 vs M2 on the sector-covered subset."""
    rows, yearly, subset_p90 = [], [], {}

    def run(name, frame, spec, sample):
        res, stats = bond_residuals(frame, spec)
        daily = daily_residual_features(issuer_residuals(res))
        fitted = stats.loc[stats["fitted"]]
        rows.append(
            {
                "model": name,
                "sample": sample,
                "days_fitted": int(len(fitted)),
                "days_not_fitted": int((~stats["fitted"]).sum()),
                "issuer_days": int(daily["n_issuers"].sum()),
                "median_daily_r2": float(fitted["r2"].median()),
                "p10_daily_r2": float(fitted["r2"].quantile(0.1)),
                "median_residual_iqr": float(daily["residual_iqr"].median()),
                "median_residual_std": float(daily["residual_std"].median()),
                "median_residual_p90": float(daily["residual_p90"].median()),
            }
        )
        by_year = fitted.groupby(fitted.index.year)["r2"].median().rename(f"{name} ({sample})")
        yearly.append(by_year)
        subset_p90[name, sample] = daily["residual_p90"]
        return fitted

    for spec in ("M0", "M1", "M3"):
        print(f"fitting {spec} (all rated issuers)", flush=True)
        run(spec, prepared, spec, "all")
    covered = prepared.loc[prepared["sector"].notna()]
    for spec in ("M1", "M2"):
        print(f"fitting {spec} (sector-covered subset)", flush=True)
        run(spec, covered, spec, "PIT-classified")
    table = pd.DataFrame(rows)
    r2 = table.loc[table["sample"].eq("all")].set_index("model")["median_daily_r2"]
    chosen = "M0"
    if r2["M1"] - r2["M0"] >= R2_GAIN_RULE:
        chosen = "M1"
        if r2["M3"] - r2["M1"] >= R2_GAIN_RULE:
            chosen = "M3"
    sector_check = pd.DataFrame(
        {
            "M1_classified": subset_p90["M1", "PIT-classified"],
            "M2_same_day_sector": subset_p90["M2", "PIT-classified"],
        }
    )
    return table, pd.concat(yearly, axis=1), chosen, sector_check


def coverage_bias() -> list[dict]:
    """Linked vs not-linked valid-spread bond-days, by TRACE grade, for a few months."""
    from macro.data.hy_dataset import (
        SAMPLE_START,
        Attrition,
        add_master_and_spreads,
        attach_issuers,
        clean_trades,
        link_matches,
        trades_to_bond_days,
    )
    from macro.data.hy_runs import _type_raw_chunk
    from macro.data.treasury import load_cmt_curve

    master = pd.read_parquet(SOURCE / "master.parquet")
    curve = load_cmt_curve(Path("data/cache/fred"))
    links = {
        n: pd.read_parquet(SOURCE / f"{n}.parquet") for n in ("bondcrsp", "security", "crsp_names")
    }
    out = []
    for month in COVERAGE_MONTHS:
        print(f"coverage check {month}", flush=True)
        log = Attrition()
        raw = _type_raw_chunk(pd.read_parquet(SOURCE / "trace_raw" / f"{month}.parquet"))
        trades = clean_trades(raw, log)
        bd = trades_to_bond_days(trades.loc[trades["trd_exctn_dt"] >= SAMPLE_START], log)
        spreads = add_master_and_spreads(bd, master, curve, log)
        linked = attach_issuers(spreads, link_matches(spreads, **links), log)
        key = spreads["cusip_id"].astype(str) + spreads["trd_exctn_dt"].astype(str)
        is_linked = key.isin(linked["cusip_id"].astype(str) + linked["trd_exctn_dt"].astype(str))
        for flag, part in spreads.groupby(is_linked):
            out.append(
                {
                    "month": month,
                    "linked": bool(flag),
                    "bond_days": int(len(part)),
                    "trace_grade_H_share": float(part["grade"].eq("H").mean()),
                    "trace_grade_missing_share": float(part["grade"].isna().mean()),
                    "median_spread_bp": float(part["spread_bp"].median()),
                    "median_spread_grade_H": float(
                        part.loc[part["grade"].eq("H"), "spread_bp"].median()
                    ),
                    "median_spread_grade_I": float(
                        part.loc[part["grade"].eq("I"), "spread_bp"].median()
                    ),
                }
            )
    return out


def episodes(mask: pd.Series) -> list[list[str]]:
    """Contiguous runs (on the session index) of True, as [start, end, n_sessions]."""
    runs, start, n = [], None, 0
    for date, flag in mask.items():
        if flag and start is None:
            start, n = date, 0
        if flag:
            n += 1
            end = date
        elif start is not None:
            runs.append([str(start.date()), str(end.date()), n])
            start = None
    if start is not None:
        runs.append([str(start.date()), str(end.date()), n])
    return runs


def compare(features: pd.DataFrame) -> dict:
    """Correlations, distinct variance, lead-lag and windows on decision dates."""
    f = features.dropna(subset=["hy_spread", "residual_p90"])
    report = {}
    levels = [
        "residual_p90",
        "residual_p95",
        "residual_iqr",
        "residual_positive_share",
        "mean_positive_residual",
        "notch_gap_ge1_share",
    ]
    for c in levels:
        f[f"{c}_change_20"] = features[c].diff(20)
    cols = levels + [f"{c}_change_20" for c in levels]
    report["correlation_levels_and_changes"] = (
        f[cols + COMPARE].corr().loc[cols, COMPARE].round(3).to_dict(orient="index")
    )
    report["spearman_levels"] = (
        f[levels + COMPARE].corr("spearman").loc[levels, COMPARE].round(3).to_dict(orient="index")
    )
    distinct = {}
    for c in levels:
        for name, regs in (
            ("levels_on_hy_spread_and_gap", ["hy_spread", "hy_ig_gap"]),
            ("change20_on_hy_changes", ["hy_spread_change_20", "hy_ig_gap_change_20"]),
        ):
            y = f[c] if name.startswith("levels") else f[f"{c}_change_20"]
            d = pd.concat([y, f[regs]], axis=1).dropna()
            if len(d) < 60:
                continue
            x = np.column_stack([np.ones(len(d)), d[regs].to_numpy()])
            beta, *_ = np.linalg.lstsq(x, d.iloc[:, 0].to_numpy(), rcond=None)
            resid = d.iloc[:, 0].to_numpy() - x @ beta
            r2 = 1 - resid.var() / d.iloc[:, 0].var(ddof=0)
            distinct.setdefault(c, {})[name] = {
                "r2": round(float(r2), 3),
                "distinct_share": round(float(1 - r2), 3),
            }
    report["variance_not_explained_by_hy_features"] = distinct
    lead = {}
    for c in ("residual_p90", "residual_positive_share", "residual_iqr"):
        rc, hc = f[f"{c}_change_20"], f["hy_spread_change_20"]
        lead[c] = {
            f"k={k}": round(float(rc.corr(hc.shift(-k))), 3) for k in (-40, -20, -10, 0, 10, 20, 40)
        }
    report["lead_lag_corr_change20_residual_t_vs_hy_t_plus_k"] = lead
    z_res = expanding_zscore(f["residual_p90"])
    z_hy = expanding_zscore(f["hy_spread"])
    mask = (z_res > 1) & (z_hy < 0.5)
    report["residual_high_while_hy_moderate"] = {
        "rule": "expanding z(residual_p90) > 1 and expanding z(hy_spread) < 0.5 (min 60 obs)",
        "sessions": int(mask.sum()),
        "by_year": {int(k): int(v) for k, v in mask.groupby(mask.index.year).sum().items() if v},
        "episodes_ge_5_sessions": [e for e in episodes(mask) if e[2] >= 5],
    }
    windows = {}
    for label, (a, b) in WINDOWS.items():
        w = f.loc[a:b]
        outside = f.loc[(f.index < a) | (f.index > b)]
        row = {}
        for c in [
            "hy_spread",
            "hy_ig_gap",
            "residual_p90",
            "residual_iqr",
            "residual_positive_share",
            "notch_gap_ge1_share",
        ]:
            if not w[c].notna().any():
                continue
            # First session in the window above the series' own expanding 90th percentile.
            hist90 = f[c].expanding(60).quantile(0.9).shift(1)
            crossed = w.index[(w[c] > hist90.loc[w.index]).fillna(False)]
            row[c] = {
                "mean_in_window": round(float(w[c].mean()), 4),
                "mean_outside_windows": round(float(outside[c].mean()), 4),
                "peak_date": str(w[c].idxmax().date()),
                "first_above_own_expanding_p90": str(crossed[0].date()) if len(crossed) else None,
            }
        windows[label] = row
    report["stress_windows"] = windows
    return report


def plots(features, fits, ladder_by_year, daily, sector_check):
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
    paths = []

    def shade(ax):
        for a, b in WINDOWS.values():
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#888888", alpha=0.12, lw=0)

    panels = [
        ("hy_spread", "HY median spread (bp)"),
        ("residual_p90", "residual P90 (log pts)"),
        ("residual_iqr", "residual IQR (log pts)"),
        ("residual_positive_share", "share above hist. P90"),
        ("notch_gap_ge1_share", ">=1 notch worse (descriptive)"),
    ]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 11), sharex=True)
    for ax, (c, label) in zip(axes, panels, strict=True):
        ax.plot(features.index, features[c], color=COLORS[0])
        ax.set_ylabel(label, fontsize=8)
        shade(ax)
    axes[0].set_title(
        "Residual credit stress vs HY level (decision dates; shaded: annotated stress windows)"
    )
    fig.tight_layout()
    paths.append(OUT / "residual_series.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    f = features.dropna(subset=["residual_p90", "hy_spread"])
    years = f.index.year
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    sc = axes[0].scatter(f["hy_spread"], f["residual_p90"], c=years, s=4, cmap="Blues", vmin=2003)
    axes[0].set_xlabel("hy_spread (bp)")
    axes[0].set_ylabel("residual_p90 (log pts)")
    axes[0].set_title("Levels", fontsize=10)
    axes[1].scatter(
        f["hy_spread"].diff(20), f["residual_p90"].diff(20), c=years, s=4, cmap="Blues", vmin=2003
    )
    axes[1].set_xlabel("hy_spread_change_20 (bp)")
    axes[1].set_ylabel("residual_p90 change over 20 sessions")
    axes[1].set_title("20-session changes", fontsize=10)
    fig.colorbar(sc, ax=axes, label="year", shrink=0.8)
    paths.append(OUT / "residual_vs_hy_scatter.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for color, col in zip(COLORS, ladder_by_year.columns, strict=False):
        ax.plot(ladder_by_year.index, ladder_by_year[col], marker="o", ms=4, color=color, label=col)
    ax.set_ylabel("median daily R² (issuer-weighted)")
    ax.set_title("Specification ladder: cross-sectional fit by year")
    ax.legend(fontsize=8, ncol=3)
    fig.tight_layout()
    paths.append(OUT / "spec_ladder_r2.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    for color, (c, label) in zip(
        COLORS, [("residual_p90_hy", "HY issuers"), ("residual_p90_ig", "IG issuers")], strict=False
    ):
        axes[0].plot(daily.index, daily[c].rolling(20).median(), color=color, label=label)
    axes[0].set_ylabel("residual P90 (20-day median)")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Residual P90 within rating class (observation dates)")
    axes[1].plot(daily.index, daily["n_issuers"], color=COLORS[0])
    axes[1].set_ylabel("issuers in daily fit")
    shade(axes[0])
    shade(axes[1])
    fig.tight_layout()
    paths.append(OUT / "residual_by_class_and_count.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    monthly = sector_check.resample("ME").mean()
    ax.plot(monthly.index, monthly["M1_classified"], color=COLORS[0], label="M1: rating + maturity")
    ax.plot(
        monthly.index,
        monthly["M2_same_day_sector"],
        color=COLORS[1],
        label="M2: + same-day sector (diagnostic)",
    )
    shade(ax)
    ax.set_ylabel("residual P90 (monthly mean)")
    ax.set_title(
        "Diagnostic only: M1 vs same-day point-in-time GICS sector effects (not production)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths.append(OUT / "sector_check_p90.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 4))
    for color, (c, label) in zip(
        COLORS,
        [
            ("tail_share_financials", "Financials (GICS 40)"),
            ("tail_share_energy", "Energy (GICS 10)"),
        ],
        strict=False,
    ):
        ax.plot(
            daily.index, daily[c].rolling(20, min_periods=10).median(), color=color, label=label
        )
    shade(ax)
    ax.set_ylabel("share of top residual decile (20-day median)")
    ax.set_title("Who is in the upper residual tail (point-in-time GICS; interpretation only)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    paths.append(OUT / "residual_tail_sectors.png")
    fig.savefig(paths[-1], dpi=110)
    plt.close(fig)
    return paths


FOCAL = {
    "2007-09 financial crisis": "40",
    "2011 credit stress": "40",
    "2012-13 HY residual elevation": None,
    "2015-16 energy/credit": "10",
}
CONCENTRATED_RATIO = 2.0  # largest sector's tail share >= 2x its share of all issuers


def tail_labels(issuers: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Issuer residuals with point-in-time sector and a top-decile (per day) flag."""
    m = issuers.merge(labels, on=["trd_exctn_dt", "gvkey"], how="left")
    m["sector"] = m["sector"].fillna("unclassified")
    cutoff = m.groupby("trd_exctn_dt")["residual"].transform(lambda s: s.quantile(0.9))
    m["top_decile"] = m["residual"] > cutoff
    return m


def daily_tail_diagnostics(m: pd.DataFrame) -> pd.DataFrame:
    """Per day: Financials and Energy share of the top decile, and its sector HHI."""
    top = m.loc[m["top_decile"]]
    shares = top.groupby(["trd_exctn_dt", "sector"]).size().unstack("sector", fill_value=0)
    shares = shares.div(shares.sum(axis=1), axis=0)
    return pd.DataFrame(
        {
            "tail_share_financials": shares.get("40", 0.0),
            "tail_share_energy": shares.get("10", 0.0),
            "tail_sector_hhi": (shares.drop(columns="unclassified", errors="ignore") ** 2).sum(
                axis=1
            ),
        }
    )


def sector_diagnostic(m: pd.DataFrame, sector_check: pd.DataFrame) -> dict:
    """Interpretation only: is each episode's upper residual tail broad or sector-concentrated?"""
    periods = dict(WINDOWS, **{"calm reference 2013-07..2014-06": ("2013-07-01", "2014-06-30")})
    out = {
        "sector_source": "comp.co_hgic historical GICS, two-digit gsector in force on each "
        "issuer-day (macro.data.sector.gics_sector_asof)",
        "use": "interpretation only; the production residual is NOT sector-adjusted",
        "concentration_rule": f"largest classified sector's top-decile share >= "
        f"{CONCENTRATED_RATIO}x its share of all issuer-days in the period",
        "periods": {},
    }
    for label, (a, b) in periods.items():
        part = m.loc[m["trd_exctn_dt"].between(a, b)]
        top = part.loc[part["top_decile"]]
        tail = top["sector"].value_counts(normalize=True)
        base = part["sector"].value_counts(normalize=True)
        classified = tail.drop("unclassified", errors="ignore")
        lead = classified.index[0]
        ratio = float(classified.iloc[0] / base[lead])
        row = {
            "top_decile_share": tail.round(3).head(6).to_dict(),
            "all_issuer_share": base.round(3).head(6).to_dict(),
            "largest_tail_sector": lead,
            "largest_tail_share_over_base": round(ratio, 2),
            "classification": "sector-concentrated" if ratio >= CONCENTRATED_RATIO else "broad",
            "m1_vs_same_day_sector_p90": sector_check.loc[a:b].mean().round(4).to_dict(),
        }
        focal = FOCAL.get(label)
        if focal is not None:
            row[f"sector_{focal}_tail_share"] = round(float(tail.get(focal, 0.0)), 3)
            row[f"sector_{focal}_base_share"] = round(float(base.get(focal, 0.0)), 3)
            row[f"sector_{focal}_mean_residual"] = round(
                float(part.loc[part["sector"].eq(focal), "residual"].mean()), 4
            )
        out["periods"][label] = row
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    bonds, sector = load_inputs()
    prepared = prepare_bond_days(bonds, sector)
    excluded = {
        "bond_days_input": int(len(bonds)),
        "bond_days_model_ready": int(len(prepared)),
        "unrated": int(bonds["rating"].isna().sum()),
        "rated_D": int(bonds["rating"].eq("D").sum()),
    }
    print(excluded, flush=True)

    table, ladder_by_year, chosen, sector_check = spec_ladder(prepared)
    sector_check.resample("ME").mean().round(4).to_csv(OUT / "sector_check_monthly_p90.csv")
    table.to_csv(OUT / "spec_comparison.csv", index=False)
    ladder_by_year.round(4).to_csv(OUT / "spec_r2_by_year.csv")
    print(table.to_string(), "\nrule picks:", chosen, flush=True)
    if chosen != CHOSEN_SPEC:
        print(
            f"NOTE: rule picks {chosen}; module CHOSEN_SPEC is {CHOSEN_SPEC}. Stop and reconcile."
        )
        return 1

    res, fits = bond_residuals(prepared, CHOSEN_SPEC, implied=True)
    issuers = issuer_residuals(res)
    issuers = issuers.merge(
        prepared.groupby(["trd_exctn_dt", "gvkey"])["hy"].first().reset_index(),
        on=["trd_exctn_dt", "gvkey"],
    )
    daily = daily_residual_features(issuers)
    for cls, flag in (("hy", 1), ("ig", 0)):
        part = issuers.loc[issuers["hy"].eq(flag)]
        daily[f"residual_p90_{cls}"] = part.groupby("trd_exctn_dt")["residual"].quantile(0.9)
        daily[f"n_{cls}_issuers"] = part.groupby("trd_exctn_dt").size()
    daily["hy_issuer_share"] = daily["n_hy_issuers"] / daily["n_issuers"]
    labels = prepared.drop_duplicates(["trd_exctn_dt", "gvkey"])[
        ["trd_exctn_dt", "gvkey", "sector"]
    ]
    tagged = tail_labels(issuers, labels)
    daily = daily.join(daily_tail_diagnostics(tagged))  # interpretation only

    # Sanity checks (fail loudly).
    checks = {
        "no_holdout_dates": bool(daily.index.max() < HOLDOUT_START),
        "no_duplicate_issuer_days": bool(not issuers.duplicated(["trd_exctn_dt", "gvkey"]).any()),
        "max_abs_bond_residual_log": float(res["residual"].abs().max()),
        "residuals_within_log_spread_range": bool(res["residual"].abs().max() < np.log(3500 / 5)),
        "min_issuers_per_fitted_day": int(daily["n_issuers"].min()),
        "min_issuers_rule": MIN_ISSUERS_PER_DAY,
        "rating_dates_not_after_observation": bool(
            pd.read_parquet(
                SOURCE / "hy_bond_days.parquet", columns=["trd_exctn_dt", "rating_date"]
            )
            .pipe(
                lambda b: (
                    (pd.to_datetime(b["rating_date"]) <= pd.to_datetime(b["trd_exctn_dt"]))
                    | b["rating_date"].isna()
                )
            )
            .all()
        ),
    }
    assert checks["no_holdout_dates"] and checks["no_duplicate_issuer_days"]
    assert checks["residuals_within_log_spread_range"]
    assert checks["min_issuers_per_fitted_day"] >= MIN_ISSUERS_PER_DAY
    assert checks["rating_dates_not_after_observation"]

    issuers.to_parquet(CACHE / "issuer_residuals.parquet")
    fits.to_parquet(CACHE / "fit_stats.parquet")
    daily.to_parquet(CACHE / "residual_daily.parquet")
    keep = (
        ["n_issuers", "n_hy_issuers", "n_ig_issuers"]
        + RESID
        + [
            "residual_mean",
            "residual_median",
            "notch_gap_median",
            "notch_gap_p90",
            "threshold",
            "residual_p90_hy",
            "residual_p90_ig",
            "tail_share_financials",
            "tail_share_energy",
            "tail_sector_hhi",
        ]
    )
    daily[keep].round(5).to_csv(OUT / "residual_daily.csv")

    hy = HYCredit().run(DEV_END)
    resid_cols = RESID + ["residual_mean"]
    res_features = build_residual_features(daily, columns=resid_cols, end_date=DEV_END)
    features = hy.join(res_features, how="left").loc[: daily.index.max() + pd.Timedelta(days=10)]
    comparison = compare(features)

    fitted = fits.loc[fits["fitted"]]
    report = {
        "spec": CHOSEN_SPEC,
        "selection_rule": f"add a component if median daily R2 rises by >= {R2_GAIN_RULE}",
        "spec_comparison": table.round(4).to_dict(orient="records"),
        "excluded": excluded,
        "observation_range": [str(daily.index.min().date()), str(daily.index.max().date())],
        "days_fitted": int(len(fitted)),
        "days_not_fitted": int((~fits["fitted"]).sum()),
        "issuer_days": int(len(issuers)),
        "hy_issuer_days": int(issuers["hy"].eq(1).sum()),
        "ig_issuer_days": int(issuers["hy"].eq(0).sum()),
        "daily_r2": q(fitted["r2"]),
        "r2_by_year": fitted.groupby(fitted.index.year)["r2"].median().round(4).to_dict(),
        "issuers_per_day": q(daily["n_issuers"]),
        "checks": checks,
        "threshold": {
            "rule": "q90 of all issuer residuals on days strictly before t; first 60 days missing",
            "summary": q(daily["threshold"]),
            "first_available": str(daily["threshold"].first_valid_index().date()),
        },
        "features": {
            c: q(daily[c])
            for c in RESID
            + ["residual_mean", "residual_median", "notch_gap_median", "notch_gap_p90"]
        },
        "feature_by_year_mean": daily[RESID]
        .groupby(daily.index.year)
        .mean()
        .round(4)
        .to_dict(orient="index"),
        "extreme_dates_top5": {
            c: {str(k.date()): round(float(v), 4) for k, v in daily[c].nlargest(5).items()}
            for c in (
                "residual_p90",
                "residual_iqr",
                "residual_positive_share",
                "notch_gap_ge1_share",
            )
        },
        "composition": {
            "corr_residual_p90_vs_hy_issuer_share": round(
                float(daily["residual_p90"].corr(daily["hy_issuer_share"])), 3
            ),
            "corr_residual_p90_hy_vs_ig": round(
                float(daily["residual_p90_hy"].corr(daily["residual_p90_ig"])), 3
            ),
            "median_residual_p90_hy": round(float(daily["residual_p90_hy"].median()), 4),
            "median_residual_p90_ig": round(float(daily["residual_p90_ig"].median()), 4),
        },
        "decision_date_rows_with_residual_p90": int(features["residual_p90"].notna().sum()),
        "last_decision_date_with_residual": str(features["residual_p90"].last_valid_index().date()),
        "comparison": comparison,
    }
    report["production_spec"] = CHOSEN_SPEC
    report["production_features"] = list(FEATURES)
    report["descriptive_only"] = [
        "residual_p95",
        "mean_positive_residual",
        "residual_mad",
        "residual_std",
        "notch_gap_*",
        "tail_share_*",
        "tail_sector_hhi",
    ]
    report["sector_diagnostic"] = sector_diagnostic(tagged, sector_check)
    cutoffs = ["2006-06-30", "2008-09-15", "2008-12-31", "2011-08-08", "2014-01-02", "2016-02-11"]
    assert_no_lookahead(HYResidualStress(), cutoffs + ["2017-05-31"])
    report["no_lookahead_real_data_cutoffs"] = cutoffs + ["2017-05-31"]
    if "--skip-coverage" not in sys.argv:
        report["coverage_linked_vs_unlinked"] = coverage_bias()
    (OUT / "residual_report.json").write_text(json.dumps(report, indent=1, default=str))
    for p in plots(features, fits, ladder_by_year, daily, sector_check):
        print("plot", p)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
