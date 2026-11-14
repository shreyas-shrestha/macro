"""Runners for the #24 HY credit data: real-sample validation and the development build.

    uv run python -m macro.data.hy_runs validate-sample
    uv run python -m macro.data.hy_runs build-dev

Inputs are WRDS extracts in ``data/cache/`` (git-ignored; written by the interactive
pull scripts) and the FRED curve in ``data/cache/fred/``. Outputs:

- ``data/cache/hy_credit/``: panels built from licensed rows (never committed);
- ``results/hy_credit/``: aggregated diagnostics only (counts, medians, attrition).

Nothing here reads or writes any date on or after ``macro.config.HOLDOUT_START``; the
loaders raise if a cached extract contains such rows.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from macro.config import HOLDOUT_START
from macro.data.hy_dataset import (
    MASTER_FIELDS,
    SAMPLE_START,
    Attrition,
    add_master_and_spreads,
    attach_issuers,
    attach_ratings,
    clean_trades,
    daily_aggregates,
    link_matches,
    trades_to_bond_days,
)
from macro.data.issuer_link import (
    exact_matches,
    link_coverage,
    link_status,
    method_conflicts,
    normalize_issuer_name,
)
from macro.data.ratings import unrecognized_ratings
from macro.data.trace import clean_trace_lifecycle, cusip_status, issuer_cusip6, join_master
from macro.data.treasury import load_cmt_curve
from macro.data.wrds import load_compustat_ratings, load_trace_master, load_trace_trades

CACHE = Path("data/cache")
RESULTS = Path("results/hy_credit")
CODE_COLUMNS = [
    "trc_st", "wis_fl", "asof_cd", "sale_cndtn_cd", "sale_cndtn2_cd", "spcl_trd_fl", "chng_cd",
    "frmt_cd", "dissem_fl", "trdg_mkt_cd", "cntra_mp_id", "rpt_side_cd", "yld_sign_cd",
    "sub_prdct", "scrty_type_cd", "trd_mod_3", "trd_mod_4", "cmsn_trd", "agu_qsr_id",
    "lckd_in_ind", "ats_indicator", "pull_part",
]  # fmt: skip


def _read_text(path: Path, dates=()) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame = frame.apply(lambda s: s.str.strip().mask(s.str.strip().eq("")))
    for column in dates:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="raise")
    return frame


def _counts(series: pd.Series, top: int = 40) -> dict:
    return {str(k): int(v) for k, v in series.value_counts(dropna=False).head(top).items()}


def _quantiles(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    qs = s.quantile([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {"n": int(len(s)), **{f"q{int(q * 100):02d}": round(float(v), 4) for q, v in qs.items()}}


def _pct(part: int, whole: int) -> float:
    return round(100.0 * part / whole, 2) if whole else float("nan")


def _maybe(path: Path, loader, **kwargs):
    return loader(path, **kwargs) if path.exists() else None


# --------------------------------------------------------------------------------------
# Real-sample validation
# --------------------------------------------------------------------------------------


def server_safe_dissem_mask(raw: pd.DataFrame) -> pd.Series:
    """Rows a server-side filter may keep without changing lifecycle-cleaned output.

    Keeps every disseminated row and every non-T or As-Of 'R' record (cancels,
    corrections and reversals are needed to remove their originals). Mirrors the SQL
    ``dissem_fl = 'Y' or trc_st <> 'T' or asof_cd = 'R'`` in the development pull.
    """
    status = raw["trc_st"].astype("string").str.upper()
    dissem = raw["dissem_fl"].astype("string").str.upper().eq("Y").fillna(False)
    reversal = raw["asof_cd"].astype("string").str.upper().eq("R").fillna(False)
    return dissem | status.ne("T").fillna(True) | reversal


def SERVER_BLANKET_DISSEM(raw: pd.DataFrame) -> pd.Series:  # noqa: N802 (reference rule)
    """Blanket ``dissem_fl = 'Y'`` filter, kept only to show why it is unsafe."""
    return raw["dissem_fl"].astype("string").str.upper().eq("Y").fillna(False)


def _trade_keys(frame: pd.DataFrame) -> set:
    cols = ["trd_rpt_dt", "msg_seq_nb", "cusip_id"]
    return set(map(tuple, frame[cols].astype(str).values))


def _prefilter_diff(raw: pd.DataFrame, rule) -> dict:
    full = _trade_keys(clean_trades(raw, Attrition()))
    keep = rule(raw)
    alt = _trade_keys(clean_trades(raw.loc[keep.to_numpy()], Attrition()))
    return {
        "rows_kept_pct": _pct(int(keep.sum()), len(raw)),
        "only_in_full": len(full - alt),
        "only_in_prefiltered": len(alt - full),
    }


# Sanity gates fixed after the first real-sample validation (2026-11-07), before any
# development data or predictive result exists. They check data health, not performance.
GATES = {
    "cw_unmatched_share_max": 0.01,
    "reversal_match_share_min": 0.80,
    "post2012_ry_match_share_min": 0.95,
    "master_usable_share_min": 0.90,
    "issuer_link_share_min": 0.25,
    "rated_share_of_linked_min": 0.70,
    "min_daily_hy_issuers": 10,
}


def evaluate_gates(report: dict) -> list[dict]:
    """Pass/fail for each sanity gate on a ``validate_sample`` report."""
    lc = report["lifecycle"]
    cw_refs = lc["cw_n_cancel_records"] + lc["cw_n_correction_records"]
    master = report["master"]["with_validity_agree"]
    bond_days = sum(v["n"] for v in master.values())
    usable = sum(master.get(k, {"n": 0})["n"] for k in ("unique", "consistent"))
    links = report["links"]["consensus"]
    rated = report.get("ratings_on_bond_days", {})
    post = report.get("post2012_sample", {}).get("lifecycle", {})
    ry = post.get("post_n_ry_refs_matched", 0) + post.get("post_n_ry_refs_unmatched", 0)
    daily = report.get("daily_window", [])
    checks = [
        ("cw_unmatched_share", lc["cw_n_unmatched_references"] / max(cw_refs, 1), "<=",
         GATES["cw_unmatched_share_max"]),
        ("reversal_match_share", lc["rev_n_reversals_matched"] / max(lc["rev_n_reversals"], 1),
         ">=", GATES["reversal_match_share_min"]),
        ("post2012_ry_match_share", post.get("post_n_ry_refs_matched", 0) / max(ry, 1), ">=",
         GATES["post2012_ry_match_share_min"]),
        ("master_usable_share", usable / max(bond_days, 1), ">=", GATES["master_usable_share_min"]),
        ("issuer_link_share", links.get("linked", 0) / max(sum(links.values()), 1), ">=",
         GATES["issuer_link_share_min"]),
        ("rated_share_of_linked", rated.get("with_rating_pct", 0) / 100, ">=",
         GATES["rated_share_of_linked_min"]),
        ("min_daily_hy_issuers", min((d["n_hy_issuers"] for d in daily), default=0), ">=",
         GATES["min_daily_hy_issuers"]),
        ("ratings_point_in_time_mismatches", report.get("ratings_point_in_time_mismatches", 1),
         "<=", 0),
        ("safe_prefilter_differences",
         report["prefilter_equivalence"]["safe_dissem"]["only_in_full"]
         + report["prefilter_equivalence"]["safe_dissem"]["only_in_prefiltered"]
         + report.get("post2012_sample", {}).get("safe_prefilter_differences", 0), "<=", 0),
    ]  # fmt: skip
    out = []
    for name, value, op, threshold in checks:
        ok = value <= threshold if op == "<=" else value >= threshold
        out.append({"gate": name, "value": round(float(value), 4), "rule": f"{op} {threshold}",
                    "pass": bool(ok)})  # fmt: skip
    return out


def validate_sample(cache: Path = CACHE, out_dir: Path = RESULTS) -> dict:
    """Run every stage on the 2010 sample and report the checks listed in #24."""
    report: dict = {}
    manifest_path = cache / "pull_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        # Keep counts and the first line of any error only: full error text can echo
        # query parameters (licensed identifiers), which must not reach results/.
        report["manifest"] = {
            name: {
                k: (str(v).splitlines()[0][:200] if k == "error" else v)
                for k, v in entry.items()
                if k != "sql"
            }
            for name, entry in manifest["pulls"].items()
        }
        report["table_metadata"] = {
            key: {
                "table": info.get("table"),
                "columns": [c for c, _ in info.get("columns", [])],
                "select_privilege": info.get("select_privilege"),
                "error": str(info["error"]).splitlines()[0][:200] if "error" in info else None,
            }
            for key, info in manifest.get("metadata", {}).items()
        }

    raw = load_trace_trades(cache / "trace_enhanced_sample.csv")
    window = (pd.Timestamp("2010-01-04"), pd.Timestamp("2010-01-08"))
    report["raw"] = {
        "rows": len(raw),
        "by_pull_part": _counts(raw["pull_part"]) if "pull_part" in raw else None,
        "trd_exctn_dt": [
            str(raw["trd_exctn_dt"].min().date()),
            str(raw["trd_exctn_dt"].max().date()),
        ],
        "trd_rpt_dt": [str(raw["trd_rpt_dt"].min().date()), str(raw["trd_rpt_dt"].max().date())],
        "missing_optional_columns": raw.attrs.get("missing_optional", []),
        "cusip_status": _counts(cusip_status(raw["cusip_id"])),
    }
    report["observed_codes"] = {c: _counts(raw[c]) for c in CODE_COLUMNS if c in raw}
    sign = raw["yld_sign_cd"].fillna("<blank>")
    yld = raw["yld_pt"]
    report["yield_sign_vs_value"] = pd.crosstab(
        sign, np.select([yld.isna(), yld < 0, yld == 0], ["missing", "neg", "zero"], "pos")
    ).to_dict()

    # Lifecycle on everything pulled, then restrict to window executions for modeling.
    lifecycle_clean, lifecycle = clean_trace_lifecycle(raw)
    report["lifecycle"] = lifecycle
    log = Attrition()
    trades = clean_trades(raw, log)
    in_window = trades["trd_exctn_dt"].between(*window)
    trades = trades.loc[in_window]
    log.add("window_executions_only", len(trades))

    # Which server-side prefilters preserve the lifecycle-cleaned result exactly?
    report["prefilter_equivalence"] = {
        "blanket_dissem": _prefilter_diff(raw, SERVER_BLANKET_DISSEM),
        "safe_dissem": _prefilter_diff(raw, server_safe_dissem_mask),
    }

    report["prices_and_yields"] = {
        "rptd_pr": _quantiles(trades["rptd_pr"]),
        "yield_pct": _quantiles(trades["yield_pct"]),
        "share_missing_yield": _pct(int(trades["yield_pct"].isna().sum()), len(trades)),
    }
    bond_days = trades_to_bond_days(trades, log)
    report["trades_per_bond_day"] = _quantiles(bond_days["n_trades"])

    # Master join diagnostics, with and without the stdt/enddt validity interval.
    master = load_trace_master(cache / "trace_master_sample.csv")
    dup = master["cusip_id"].duplicated(keep=False)
    report["master"] = {
        "rows": len(master),
        "distinct_cusips": int(master["cusip_id"].nunique()),
        "cusips_with_multiple_rows": int(master.loc[dup, "cusip_id"].nunique()),
        "enddt_missing_share": _pct(int(master["enddt"].isna().sum()), len(master))
        if "enddt" in master else None,
        "codes": {c: _counts(master[c]) for c in
                  ("sub_prdct_type", "cnvrb_fl", "cpn_type_cd", "ind_144a", "grade", "debt_type_cd",
                   "scrty_type_cd", "dissem") if c in master},
    }  # fmt: skip
    for label, validity, agree in (
        ("with_validity_agree", ("stdt", "enddt"), MASTER_FIELDS),
        ("with_validity", ("stdt", "enddt"), None),
        ("without_validity", None, None),
    ):
        _, counts = join_master(bond_days, master, validity=validity, agree_on=agree)
        total = int(counts.sum())
        report["master"][label] = {
            k: {"n": int(v), "pct": _pct(int(v), total)} for k, v in counts.items()
        }

    curve = load_cmt_curve(cache / "fred")
    spreads = add_master_and_spreads(bond_days, master, curve, log)
    report["spreads_bp"] = _quantiles(spreads["spread_bp"])
    report["remaining_maturity_y"] = _quantiles(spreads["remaining_maturity_y"])

    # Issuer links.
    fang = _maybe(cache / "fang_link_sample.csv", _read_text, dates=("ldom",))
    bondcrsp = _maybe(cache / "bondcrsp_link_sample.csv", _read_text,
                      dates=("link_startdt", "link_enddt", "trace_startdt", "trace_enddt",
                             "crsp_startdt", "crsp_enddt"))  # fmt: skip
    ccm = _maybe(cache / "ccm_link_sample.csv", _read_text, dates=("linkdt", "linkenddt"))
    if bondcrsp is not None:
        bondcrsp["permno"] = pd.to_numeric(bondcrsp["permno"]).astype("Int64")
    if ccm is not None:
        ccm["lpermno"] = pd.to_numeric(ccm["lpermno"]).astype("Int64")
    security = _maybe(cache / "comp_security_sample.csv", _read_text)
    matches = link_matches(spreads, fang=fang, bondcrsp=bondcrsp, ccm=ccm, security=security)
    bonds = spreads.assign(
        _bd=spreads["cusip_id"] + "|" + spreads["trd_exctn_dt"].dt.strftime("%F")
    )
    m = matches.assign(_bd=matches["cusip_id"] + "|" + matches["trd_exctn_dt"].dt.strftime("%F"))
    status = link_status(m, bonds["_bd"], left_id="_bd", right_id="gvkey")
    report["links"] = {
        "bond_days": len(bonds),
        "coverage_by_method": link_coverage(status).to_dict(orient="index") if len(status) else {},
        "bond_days_with_method_conflict": len(method_conflicts(m, left_id="_bd", right_id="gvkey")),
    }
    linked = attach_issuers(spreads, matches, log)
    report["links"]["consensus"] = {
        k.removeprefix("link_"): int(v) for k, v in log.rows[-2].items() if k.startswith("link_")
    }

    # Diagnostics only: ticker / name against Compustat identifier tables (the production
    # methods, including comp_cusip6, are appended below from ``matches``).
    names = _maybe(cache / "comp_names_sample.csv", _read_text)
    diag_rows = []
    issuers = (
        spreads[["cusip_id"]]
        .drop_duplicates()
        .assign(cusip6=lambda f: issuer_cusip6(f["cusip_id"]))
    )
    trace_meta = master.drop_duplicates("cusip_id")[["cusip_id", "company_symbol", "issuer_nm"]]
    issuers = issuers.merge(trace_meta, on="cusip_id", how="left")
    issuers["name_key"] = normalize_issuer_name(issuers["issuer_nm"])
    if security is not None and {"cusip", "gvkey"} <= set(security):
        sec = security
        if "tic" in sec:
            diag_rows.append(
                exact_matches(
                    issuers,
                    sec,
                    left_id="cusip_id",
                    right_id="gvkey",
                    left_key="company_symbol",
                    right_key="tic",
                    method="ticker",
                )
            )
    if names is not None and {"conm", "gvkey"} <= set(names):
        nm = names.assign(name_key=normalize_issuer_name(names["conm"]))
        diag_rows.append(
            exact_matches(
                issuers,
                nm,
                left_id="cusip_id",
                right_id="gvkey",
                left_key="name_key",
                right_key="name_key",
                method="name",
            )
        )
    if len(matches):
        prod = matches[["cusip_id", "gvkey", "method"]].drop_duplicates()
        diag_rows.append(prod)
    if diag_rows:
        all_matches = pd.concat(diag_rows, ignore_index=True).astype("string")
        bond_status = link_status(
            all_matches, issuers["cusip_id"], left_id="cusip_id", right_id="gvkey"
        )
        report["links"]["bond_level_by_method"] = link_coverage(bond_status).to_dict(orient="index")
        pairs = {}
        methods = sorted(all_matches["method"].unique())
        uniq = bond_status.loc[bond_status["status"].eq("unique")].set_index(["cusip_id", "method"])
        for i, m1 in enumerate(methods):
            for m2 in methods[i + 1 :]:
                a1 = (
                    uniq.xs(m1, level="method")["candidates"]
                    if m1 in uniq.index.get_level_values(1)
                    else pd.Series(dtype=object)
                )
                a2 = (
                    uniq.xs(m2, level="method")["candidates"]
                    if m2 in uniq.index.get_level_values(1)
                    else pd.Series(dtype=object)
                )
                both = a1.index.intersection(a2.index)
                agree = int((a1.loc[both] == a2.loc[both]).sum())
                pairs[f"{m1}_vs_{m2}"] = {"both_unique": len(both), "agree": agree,
                                          "disagree": len(both) - agree}  # fmt: skip
        report["links"]["pairwise_agreement_bond_level"] = pairs

    # Ratings.
    ratings = load_compustat_ratings(cache / "compustat_ratings_sample.csv")
    report["ratings_file"] = {
        "rows": len(ratings),
        "gvkeys": int(ratings["gvkey"].nunique()),
        "datadates": _counts(ratings["datadate"].dt.strftime("%F")),
        "splticrm_missing_share": _pct(int(ratings["splticrm"].isna().sum()), len(ratings)),
        "unrecognized_values": {
            str(k): int(v) for k, v in unrecognized_ratings(ratings["splticrm"]).items()
        },
    }
    rated = attach_ratings(linked, ratings, log) if len(linked) else linked
    if len(rated):
        report["ratings_on_bond_days"] = {
            "bond_days": len(rated),
            "with_rating_pct": _pct(int(rated["rating"].notna().sum()), len(rated)),
            "rating_distribution": _counts(rated["rating"]),
            "class_counts": _counts(rated["rating_class"]),
            "rating_age_days": _quantiles(rated["rating_age_days"]),
        }
        # Real-data leak check: ratings known at each cutoff reproduce earlier rows.
        mismatches = 0
        for cutoff in sorted(rated["trd_exctn_dt"].unique()):
            known = ratings.loc[ratings["datadate"] <= cutoff]
            sub = linked.loc[linked["trd_exctn_dt"] <= cutoff]
            again = attach_ratings(sub, known, Attrition())
            merged = rated.loc[rated["trd_exctn_dt"] <= cutoff, "rating"].fillna("NA").to_numpy()
            mismatches += int((again["rating"].fillna("NA").to_numpy() != merged).sum())
        report["ratings_point_in_time_mismatches"] = mismatches
        issuer, daily = daily_aggregates(rated, log, min_issuers=1)
        report["daily_window"] = (
            daily.reset_index().astype({"trd_exctn_dt": str}).to_dict(orient="records")
        )

        # Bond Returns benchmark (if pulled).
        bondret = _maybe(cache / "bondret_sample.csv", _read_text, dates=("date", "maturity"))
        if bondret is not None and len(bondret):
            br = bondret.assign(t_spread=pd.to_numeric(bondret["t_spread"], errors="coerce"))
            report["bondret"] = {
                "rows": len(br),
                "columns": list(br.columns),
                "dates": _counts(br["date"].dt.strftime("%F")),
                "rating_class_values": _counts(br["rating_class"])
                if "rating_class" in br
                else None,
                "r_sp_values": _counts(br["r_sp"]) if "r_sp" in br else None,
                "t_spread": _quantiles(br["t_spread"]),
            }
            dec = br.loc[br["date"].dt.strftime("%Y-%m").eq("2009-12")].drop_duplicates("cusip")
            ours = rated.groupby("cusip_id").agg(
                spread_bp=("spread_bp", "median"), rating_class=("rating_class", "last")
            )
            both = ours.join(
                dec.set_index("cusip")[["t_spread", "rating_class"]], rsuffix="_br", how="inner"
            )
            if len(both):
                report["bondret"]["matched_bonds_dec2009"] = len(both)
                report["bondret"]["spearman_ours_vs_t_spread"] = round(
                    float(both["spread_bp"].rank().corr(both["t_spread"].rank())), 4
                )  # Spearman
                report["bondret"]["ratio_median_ours_to_t_spread"] = round(
                    float((both["spread_bp"] / both["t_spread"]).median()), 4
                )
                report["bondret"]["rating_class_crosstab"] = pd.crosstab(
                    both["rating_class"].fillna("NA"), both["rating_class_br"].fillna("NA")
                ).to_dict()

    # Post-2012 lifecycle sample (if pulled).
    post_path = cache / "trace_enhanced_sample_2013.csv"
    if post_path.exists():
        post = load_trace_trades(post_path)
        _, post_diag = clean_trace_lifecycle(post)
        post_safe = _prefilter_diff(post, server_safe_dissem_mask)
        report["post2012_sample"] = {"rows": len(post), "codes": {c: _counts(post[c]) for c in
                                     ("trc_st", "asof_cd", "trd_mod_3", "trd_mod_4") if c in post},
                                     "lifecycle": post_diag,
                                     "safe_prefilter": post_safe,
                                     "safe_prefilter_differences": post_safe["only_in_full"]
                                     + post_safe["only_in_prefiltered"]}  # fmt: skip

    report["attrition"] = log.frame().to_dict(orient="records")
    report["gates"] = evaluate_gates(report)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sample_validation.json").write_text(json.dumps(report, indent=1, default=str))
    return report


# --------------------------------------------------------------------------------------
# Development build
# --------------------------------------------------------------------------------------


RAW_NUMERIC = ("rptd_pr", "yld_pt", "entrd_vol_qt", "msg_seq_nb", "orig_msg_seq_nb")
RAW_DATES = ("trd_exctn_dt", "trd_rpt_dt", "pr_trd_dt")


def _type_raw_chunk(raw: pd.DataFrame) -> pd.DataFrame:
    """Give a raw TRACE chunk consistent types whatever the Parquet writer stored."""
    raw = raw.copy()
    for column in RAW_DATES:
        if column in raw:
            raw[column] = pd.to_datetime(raw[column])
    for column in RAW_NUMERIC:
        if column in raw:
            text = raw[column].astype("string").str.strip()
            raw[column] = pd.to_numeric(text.mask(text.eq("")), errors="raise")
    for column in raw.columns.difference([*RAW_DATES, *RAW_NUMERIC]):
        if raw[column].dtype == object and column != "trd_exctn_tm":
            raw[column] = raw[column].astype("string")
    return raw


def build_development(cache: Path = CACHE, out_dir: Path = RESULTS) -> pd.DataFrame:
    """Chunked build from ``data/cache/hy_credit/trace_raw/YYYY-MM.parquet`` to daily aggregates."""
    work = cache / "hy_credit"
    chunks = sorted((work / "trace_raw").glob("*.parquet"))
    if not chunks:
        raise FileNotFoundError(f"no raw TRACE chunks in {work / 'trace_raw'}")
    attrition, bond_day_parts = [], []
    for path in chunks:
        raw = _type_raw_chunk(pd.read_parquet(path))
        if raw["trd_exctn_dt"].ge(HOLDOUT_START).any() or raw["trd_rpt_dt"].ge(HOLDOUT_START).any():
            raise ValueError(f"{path.name} contains holdout dates")
        log = Attrition()
        trades = clean_trades(raw, log)
        trades = trades.loc[trades["trd_exctn_dt"] >= SAMPLE_START]
        bond_day_parts.append(trades_to_bond_days(trades, log))
        attrition.append(log.frame().assign(chunk=path.stem))
        print(
            f"{path.stem}: {len(raw):>10,} raw -> {len(bond_day_parts[-1]):>8,} bond-days",
            flush=True,
        )
    bond_days = pd.concat(bond_day_parts, ignore_index=True)

    log = Attrition()
    master = pd.read_parquet(work / "master.parquet")
    curve = load_cmt_curve(cache / "fred")
    spreads = add_master_and_spreads(bond_days, master, curve, log)
    link_names = ("fang", "bondcrsp", "ccm", "security", "crsp_names")
    links = {
        name: pd.read_parquet(work / f"{name}.parquet")
        if (work / f"{name}.parquet").exists()
        else None
        for name in link_names
    }
    matches = link_matches(spreads, **links)
    linked = attach_issuers(spreads, matches, log)
    ratings = pd.read_parquet(work / "ratings.parquet")
    rated = attach_ratings(linked, ratings, log)
    issuer, daily = daily_aggregates(rated, log)

    rated.to_parquet(work / "hy_bond_days.parquet")
    issuer.to_parquet(work / "hy_issuer_days.parquet")
    daily.to_parquet(work / "hy_daily.parquet")

    out_dir.mkdir(parents=True, exist_ok=True)
    chunk_log = pd.concat(attrition, ignore_index=True)
    chunk_log.to_csv(out_dir / "attrition_by_month.csv", index=False)
    totals = chunk_log.groupby("stage", sort=False)["rows"].sum().reset_index()
    pd.concat([totals, log.frame()], ignore_index=True).to_csv(
        out_dir / "attrition.csv", index=False
    )
    coverage = daily[[c for c in daily.columns if c.startswith("n_")]]
    coverage.to_csv(out_dir / "daily_coverage.csv")
    return daily


def dev_report(cache: Path = CACHE, out_dir: Path = RESULTS) -> dict:
    """Aggregate diagnostics of the development build (no licensed rows are written)."""
    from macro.features.hy_credit import HYCredit

    work = cache / "hy_credit"
    rated = pd.read_parquet(work / "hy_bond_days.parquet")
    issuer = pd.read_parquet(work / "hy_issuer_days.parquet")
    daily = pd.read_parquet(work / "hy_daily.parquet")
    attrition = pd.read_csv(out_dir / "attrition.csv")
    stage = attrition.groupby("stage", sort=False)["rows"].sum()
    report: dict = {"attrition_totals": {k: int(v) for k, v in stage.items()}}
    usable = daily.dropna(subset=["hy_spread_bp", "ig_spread_bp"])
    report["usable_date_range"] = [str(usable.index.min().date()), str(usable.index.max().date())]
    report["days"] = {"with_any_data": len(daily), "with_hy_and_ig": len(usable)}
    report["bond_days_rated"] = {
        "bond_days": len(rated),
        "with_rating_pct": _pct(int(rated["rating"].notna().sum()), len(rated)),
        "class_counts": _counts(rated["rating_class"]),
        "rating_distribution": _counts(rated["rating"]),
        "rating_age_days": _quantiles(rated["rating_age_days"]),
        "remaining_maturity_y": _quantiles(rated["remaining_maturity_y"]),
    }
    report["issuer_days"] = len(issuer)
    for col in ("hy_spread_bp", "ig_spread_bp", "hy_ig_gap_bp"):
        report[f"summary_{col}"] = _quantiles(daily[col])
    by_year = daily.groupby(daily.index.year).agg(
        hy_median=("hy_spread_bp", "median"), ig_median=("ig_spread_bp", "median"),
        gap_median=("hy_ig_gap_bp", "median"), hy_issuers=("n_hy_issuers", "median"),
        ig_issuers=("n_ig_issuers", "median"), unrated_issuers=("n_unrated_issuers", "median"),
        bonds=("n_bonds", "median"), days=("hy_spread_bp", "size"),
        days_missing_hy=("hy_spread_bp", lambda s: int(s.isna().sum())),
    ).round(1)  # fmt: skip
    report["by_year"] = by_year.reset_index().to_dict(orient="records")
    monthly = daily.resample("ME").agg(
        {"hy_spread_bp": "mean", "ig_spread_bp": "mean", "hy_ig_gap_bp": "mean",
         "n_hy_issuers": "median", "n_ig_issuers": "median", "n_bonds": "median"}
    )  # fmt: skip
    monthly.to_csv(out_dir / "monthly_summary.csv")
    features = HYCredit(daily).run(
        min(daily.index.max() + pd.Timedelta(days=7), pd.Timestamp("2017-12-29"))
    )
    report["features"] = {c: _quantiles(features[c]) for c in features.columns}
    bondret_path = work / "bondret_monthly_by_class.parquet"
    report["bondret_comparison"] = (
        "not accessible" if not bondret_path.exists() else "see bondret_monthly_by_class"
    )
    (out_dir / "dev_report.json").write_text(json.dumps(report, indent=1, default=str))
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "command", choices=["validate-sample", "gate-sample", "build-dev", "report-dev"]
    )
    parser.add_argument("--cache", type=Path, default=CACHE, help="cache root (default data/cache)")
    parser.add_argument("--results", type=Path, default=RESULTS, help="aggregated outputs dir")
    args = parser.parse_args(argv)
    if args.command == "gate-sample":
        report = validate_sample(cache=args.cache, out_dir=args.results)
        for gate in report["gates"]:
            print(
                f"{'PASS' if gate['pass'] else 'FAIL'}  "
                f"{gate['gate']}: {gate['value']} {gate['rule']}"
            )
        if not all(g["pass"] for g in report["gates"]):
            raise SystemExit("Sample gates failed; development pull not started.")
        return
    if args.command == "report-dev":
        print(json.dumps(dev_report(cache=args.cache, out_dir=args.results), indent=1, default=str))
        return
    if args.command == "validate-sample":
        report = validate_sample(cache=args.cache, out_dir=args.results)
        print(
            json.dumps({k: report[k] for k in report if k != "daily_window"}, indent=1, default=str)
        )
    else:
        daily = build_development(cache=args.cache, out_dir=args.results)
        print(daily.describe().T)


if __name__ == "__main__":
    main()
