"""Final QA of the development build (local data only; no WRDS). Run from the repo root:

    uv run python qa_dev.py [CACHE_ROOT] [RESULTS_DIR]

Writes RESULTS_DIR/qa_report.json (aggregates only) and exits non-zero on a hard failure:
holdout dates, duplicate days, impossible spreads, or a decision row that uses an
observation from the same or a later day.
"""

import json
import sys
from pathlib import Path

import pandas as pd

HOLDOUT = pd.Timestamp("2018-01-01")


def main(cache: Path, results: Path) -> int:
    from macro.data.calendar import nyse_sessions
    from macro.data.hy_dataset import SPREAD_RANGE_BP
    from macro.features.hy_credit import to_decision_dates

    work = cache / "hy_credit"
    daily = pd.read_parquet(work / "hy_daily.parquet")
    rated = pd.read_parquet(work / "hy_bond_days.parquet")
    issuer = pd.read_parquet(work / "hy_issuer_days.parquet")
    attrition = pd.read_csv(results / "attrition.csv")
    stage = attrition.groupby("stage", sort=False)["rows"].sum()
    failures: list[str] = []

    # Holdout: every date column in every development input and output.
    latest = {}
    for path in sorted((work / "trace_raw").glob("*.parquet")):
        frame = pd.read_parquet(path, columns=["trd_exctn_dt", "trd_rpt_dt"])
        latest[path.name] = str(pd.to_datetime(frame["trd_rpt_dt"]).max())
        if pd.to_datetime(frame[["trd_exctn_dt", "trd_rpt_dt"]].stack()).max() >= HOLDOUT:
            failures.append(f"holdout date in {path.name}")
    for name, frame, cols in (
        ("hy_daily", daily.reset_index(), ["trd_exctn_dt"]),
        ("hy_bond_days", rated, ["trd_exctn_dt", "rating_date"]),
        ("hy_issuer_days", issuer, ["trd_exctn_dt"]),
    ):
        for col in cols:
            if col in frame and pd.to_datetime(frame[col]).max() >= HOLDOUT:
                failures.append(f"holdout date in {name}.{col}")
    ratings = pd.read_parquet(work / "ratings.parquet")
    if pd.to_datetime(ratings["datadate"]).max() >= HOLDOUT:
        failures.append("holdout date in ratings.datadate")

    # Duplicates and impossible values.
    if daily.index.has_duplicates:
        failures.append("duplicate days in hy_daily")
    if rated.duplicated(["cusip_id", "trd_exctn_dt"]).any():
        failures.append("duplicate bond-days")
    if issuer.duplicated(["trd_exctn_dt", "gvkey"]).any():
        failures.append("duplicate issuer-days")
    lo, hi = SPREAD_RANGE_BP
    if not rated["spread_bp"].between(lo, hi).all():
        failures.append("bond-day spread outside the screen")
    for col in ("hy_spread_bp", "ig_spread_bp"):
        s = daily[col].dropna()
        if len(s) and not s.between(lo, hi).all():
            failures.append(f"{col} outside [{lo}, {hi}]")

    # Lookahead: every observation must enter on a decision date strictly after it, and the
    # signal's decision index must contain no date earlier than its first observation + 1.
    obs = pd.DatetimeIndex(daily.index).normalize()
    if len(obs):
        sessions = nyse_sessions(obs.min(), obs.max() + pd.Timedelta(days=14))
        decision = sessions[sessions.searchsorted(obs, side="right")]
        if (decision <= obs).any():
            failures.append("an observation maps to a decision date on or before itself")
        decided = to_decision_dates(daily[["hy_spread_bp", "hy_ig_gap_bp"]])
        if decided.index.min() <= obs.min():
            failures.append("decision index starts on or before the first observation")

    usable = daily.dropna(subset=["hy_spread_bp", "ig_spread_bp"])
    linked = int(stage.get("after_issuer_link", 0))
    spread_bd = int(stage.get("after_spread_screen", 0))
    rated_bd = int(rated["rating"].notna().sum())
    hy_issuer_days = int((issuer["rating_class"] == "HY").sum())
    ig_issuer_days = int((issuer["rating_class"] == "IG").sum())
    summary = {
        "raw_trace_rows": int(stage.get("raw_records", 0)),
        "cleaned_trades": int(stage.get("after_sanity_filters", 0)),
        "bond_days": int(stage.get("bond_days", 0)),
        "bond_days_with_valid_spread": spread_bd,
        "linked_bond_days": linked,
        "rated_bond_days": rated_bd,
        "issuer_days_rated": len(issuer),
        "hy_issuer_days": hy_issuer_days,
        "ig_issuer_days": ig_issuer_days,
        "first_usable_date": str(usable.index.min().date()) if len(usable) else None,
        "last_usable_date": str(usable.index.max().date()) if len(usable) else None,
    }
    denominators = {
        "master_usable_share_of_bond_days": round(
            int(stage.get("after_master_unique", 0)) / max(summary["bond_days"], 1), 4),
        "linked_share_of_valid_spread_bond_days": round(linked / max(spread_bd, 1), 4),
        "rated_share_of_linked_bond_days": round(rated_bd / max(linked, 1), 4),
        "note": "issuer-days are formed from rated bond-days only; HY/IG aggregates use rated "
                "issuers only, unrated issuers are counted in n_unrated_issuers",
    }  # fmt: skip
    in_range = daily.loc[usable.index.min() : usable.index.max()] if len(usable) else daily
    gaps = in_range["hy_spread_bp"].isna()
    runs = (gaps != gaps.shift()).cumsum()[gaps]
    longest = runs.value_counts().max() if len(runs) else 0
    report = {
        "failures": failures,
        "summary": summary,
        "denominators": denominators,
        "days_missing_hy_within_usable_range": int(gaps.sum()),
        "longest_missing_hy_run_days": int(longest),
        "latest_report_date_by_chunk_max": max(latest.values()) if latest else None,
        "daily_counts": {
            c: {
                "min": int(daily[c].min()),
                "median": float(daily[c].median()),
                "max": int(daily[c].max()),
            }
            for c in ("n_hy_issuers", "n_ig_issuers", "n_unrated_issuers", "n_bonds")
        },
    }
    results.mkdir(parents=True, exist_ok=True)
    (results / "qa_report.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))
    return 1 if failures else 0


if __name__ == "__main__":
    cache = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/cache")
    results = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results/hy_credit")
    raise SystemExit(main(cache, results))
