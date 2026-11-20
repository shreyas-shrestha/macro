"""Minimal real-data sample for issue #24 (trades executed 2010-01-04..2010-01-08).

Run interactively in your own terminal (hidden password prompt, then approve Duo):
    cd <repo> && WRDS_USERNAME=<your WRDS id> uv run --no-project --python 3.12 --with wrds \\
        python -I scripts/hy_credit/wrds_pull_sample.py

Phase 1, metadata only (no table rows):
  - information_schema columns and has_table_privilege for every table used;
  - a one-row access probe of wrdsapps_bondret.bondret (row access vs schema visibility).
Phase 2, minimal rows:
  - TRACE Enhanced records needed for the window and its lifecycle links;
  - a post-2012 lifecycle check: records reported 2013-01-07..08 plus the records their
    prior-reference fields point at;
  - TRACE master rows for the window's CUSIPs only;
  - link rows (bond->CRSP, issuer-CUSIP->GVKEY, CRSP->Compustat) for those CUSIPs;
  - Compustat identifier rows only for issuers tied to the sample (by link GVKEY, issuer
    CUSIP6 or ticker), selecting only columns confirmed in phase 1;
  - Compustat ratings only for those GVKEYs;
  - a tiny Bond Returns sample if row access works (benchmark only).

Everything is pre-2018: every date bound is asserted before any query, and returned
event dates are checked again before saving. Raw rows go only to <repo>/data/cache/
(git-ignored). Metadata and counts go to data/cache/pull_manifest.json.
The password is read with getpass, used in memory only, never printed or saved, and
no .pgpass file is created. Each step is independent: a failure is recorded and the
script continues.
"""

import getpass
import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import wrds

USERNAME = os.environ.get("WRDS_USERNAME", "")
REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "data" / "cache"

HOLDOUT = date(2018, 1, 1)
MIGRATION = date(2012, 2, 6)  # TRACE platform migration; pre-2012 lifecycle rules apply before it
WIN_START, WIN_END = date(2010, 1, 4), date(2010, 1, 8)
EXEC_FLOOR = date(2009, 1, 1)  # lets the report-date query use an execution-date index
RATINGS_START, RATINGS_END = date(2009, 6, 30), date(2010, 2, 28)
LINK_MONTHS = (date(2009, 12, 31), date(2010, 1, 31))
BONDRET_RANGE = (date(2009, 12, 1), date(2010, 1, 31))
BONDRET_MAX_ROWS = 500
POST_START, POST_END = date(2013, 1, 7), date(2013, 1, 8)  # post-migration validation days
POST_EXEC_FLOOR = date(2012, 1, 1)

for bound in (
    WIN_START,
    WIN_END,
    MIGRATION,
    EXEC_FLOOR,
    RATINGS_START,
    RATINGS_END,
    *LINK_MONTHS,
    *BONDRET_RANGE,
    POST_START,
    POST_END,
    POST_EXEC_FLOOR,
):
    assert bound < HOLDOUT, bound

TABLES = {
    "trace": "trace_enhanced.trace_enhanced",
    "master": "trace_enhanced.camasterfile",
    "ratings": "comp.adsprate",
    "company": "comp.company",
    "security": "comp.security",
    "names": "comp.names",
    "bondcrsp": "wrdsapps_link_crsp_bond.bondcrsp_link",
    "fang": "contrib_bond_firm_link.fang_link",
    "ccm": "crsp.ccmxpf_lnkhist",
    "bondret": "wrdsapps_bondret.bondret",
}

# Columns wanted from each identifier table; only those confirmed in phase 1 are selected.
WANTED = {
    "company": ["gvkey", "conm", "cik", "sic", "naics", "gsector", "ggroup", "gind", "gsubind"],
    "security": ["gvkey", "iid", "cusip", "tic", "exchg", "excntry", "tpci", "dldtei"],
    "names": [
        "gvkey",
        "conm",
        "tic",
        "cusip",
        "cik",
        "sic",
        "naics",
        "gsector",
        "gind",
        "year1",
        "year2",
    ],
    "ccm": ["gvkey", "liid", "lpermno", "lpermco", "linktype", "linkprim", "linkdt", "linkenddt"],
}

TRACE_COLS = [
    "cusip_id",
    "bond_sym_id",
    "company_symbol",
    "trd_exctn_dt",
    "trd_exctn_tm",
    "trd_rpt_dt",
    "trd_rpt_tm",
    "msg_seq_nb",
    "trc_st",
    "scrty_type_cd",
    "wis_fl",
    "cmsn_trd",
    "entrd_vol_qt",
    "rptd_pr",
    "yld_sign_cd",
    "yld_pt",
    "asof_cd",
    "days_to_sttl_ct",
    "sale_cndtn_cd",
    "sale_cndtn2_cd",
    "rpt_side_cd",
    "buy_cpcty_cd",
    "sell_cpcty_cd",
    "cntra_mp_id",
    "agu_qsr_id",
    "spcl_trd_fl",
    "trdg_mkt_cd",
    "dissem_fl",
    "orig_msg_seq_nb",
    "sub_prdct",
    "stlmnt_dt",
    "trd_mod_3",
    "trd_mod_4",
    "rptg_party_type",
    "lckd_in_ind",
    "ats_indicator",
    "pr_trd_dt",
]
T = TABLES["trace"]
TCOLS = ", ".join(TRACE_COLS)

SQL = {
    # Every record reported on the window's report dates: complete days for same-day C/W links.
    "trace_reported_in_window": f"""
        select {TCOLS}, 'reported_in_window' as pull_part from {T}
        where trd_rpt_dt between %(ws)s and %(we)s and trd_exctn_dt >= %(exec_floor)s""",
    # Later reports (as-of trades, reversals) of window trades, before the platform migration.
    "trace_later_reports": f"""
        select {TCOLS}, 'later_report_of_window_trade' as pull_part from {T}
        where trd_exctn_dt between %(ws)s and %(we)s
          and trd_rpt_dt > %(we)s and trd_rpt_dt < %(migration)s""",
    # Earlier originals of reversals reported in the window (FINRA basic-detail match).
    "trace_reversal_originals": f"""
        with rev as (
            select distinct cusip_id, trd_exctn_dt, trd_exctn_tm, rptd_pr, entrd_vol_qt,
                   rpt_side_cd, cntra_mp_id
            from {T} where trd_rpt_dt between %(ws)s and %(we)s and asof_cd = 'R')
        select {", ".join("t." + c for c in TRACE_COLS)},
               'earlier_original_of_window_reversal' as pull_part
        from {T} t join rev r
          on t.cusip_id = r.cusip_id and t.trd_exctn_dt = r.trd_exctn_dt
         and t.trd_exctn_tm = r.trd_exctn_tm and t.rptd_pr = r.rptd_pr
         and t.entrd_vol_qt = r.entrd_vol_qt and t.rpt_side_cd = r.rpt_side_cd
         and t.cntra_mp_id = r.cntra_mp_id
        where t.trd_rpt_dt < %(ws)s and coalesce(t.asof_cd, '') <> 'R'""",
    # Aggregate count only: later records of window trades on/after the migration (not pulled).
    "trace_post_migration_count": f"""
        select count(*) as n from {T}
        where trd_exctn_dt between %(ws)s and %(we)s
          and trd_rpt_dt >= %(migration)s and trd_rpt_dt < %(holdout)s""",
    # Post-2012 validation sample: all records reported on two 2013 days, plus the exact
    # earlier records their (pr_trd_dt, orig_msg_seq_nb) point at.
    "trace_2013_reported": f"""
        select {TCOLS}, 'reported_2013_window' as pull_part from {T}
        where trd_rpt_dt between %(p0)s and %(p1)s and trd_exctn_dt >= %(p_floor)s""",
    "trace_2013_priors": f"""
        with refs as (
            select distinct pr_trd_dt, orig_msg_seq_nb from {T}
            where trd_rpt_dt between %(p0)s and %(p1)s and trd_exctn_dt >= %(p_floor)s
              and pr_trd_dt is not null and orig_msg_seq_nb is not null)
        select {", ".join("t." + c for c in TRACE_COLS)}, 'prior_of_2013_window' as pull_part
        from {T} t join refs r
          on t.trd_rpt_dt = r.pr_trd_dt and t.msg_seq_nb = r.orig_msg_seq_nb
        where t.trd_rpt_dt < %(p0)s""",
    "trace_master": f"""
        select * from {TABLES["master"]}
        where cusip_id = any(%(cusips)s) and (stdt is null or stdt < %(holdout)s)""",
    "bondcrsp_link": f"""
        select * from {TABLES["bondcrsp"]}
        where cusip = any(%(cusips)s) and (link_startdt is null or link_startdt < %(holdout)s)""",
    "fang_link": f"""
        select * from {TABLES["fang"]}
        where issuer_cusip = any(%(cusip6)s) and ldom between %(l0)s and %(l1)s""",
    "ratings": f"""
        select gvkey, datadate, splticrm, spsdrm, spsticrm from {TABLES["ratings"]}
        where gvkey = any(%(gvkeys)s) and datadate between %(rs)s and %(re)s""",
}

META_SQL = """
select table_schema, table_name, column_name, data_type, ordinal_position
from information_schema.columns
where table_schema = %(schema)s and table_name = %(table)s
order by ordinal_position
"""

# Attributes fixed at issuance or describing validity; they may lie after 2017.
REFERENCE_DATE_COLS = {
    "mtrty_dt",
    "maturity",
    "enddt",
    "link_enddt",
    "trace_enddt",
    "crsp_enddt",
    "linkenddt",
    "last_interest_date",
}


def connect():
    if not USERNAME:
        sys.exit("Set WRDS_USERNAME=<your WRDS id> before running; nothing was queried.")
    password = getpass.getpass(f"WRDS password for {USERNAME} (input hidden): ")
    print("Connecting... approve the Duo push on your phone.")
    db = wrds.Connection(autoconnect=False, wrds_username=USERNAME, wrds_password=password)
    make = getattr(db, "_Connection__make_sa_engine_conn", None)
    if make is None:
        sys.exit("wrds package internals changed; refusing to use its interactive fallback.")
    try:
        make(raise_err=True)
    except Exception as err:  # noqa: BLE001
        message = str(err)
        for secret in (password, quote(password)):
            if secret:
                message = message.replace(secret, "***")
        sys.exit(f"Login failed: {type(err).__name__}: {message.splitlines()[0][:300]}")
    finally:
        db._password = ""
        del password
    if db.engine is None:
        sys.exit("Login failed (no engine).")
    db.connection.exec_driver_sql("SET statement_timeout = '15min'")
    return db


def _date_columns(frame: pd.DataFrame) -> list[str]:
    out = []
    for column in frame.columns:
        values = frame[column]
        if pd.api.types.is_datetime64_any_dtype(values) or (
            values.dtype == object and values.map(lambda v: isinstance(v, date)).any()
        ):
            out.append(column)
    return out


def enforce_no_holdout(name: str, frame: pd.DataFrame, strict: bool) -> dict:
    """Strict (event-dated) tables raise on any 2018+ date. Reference tables blank such
    dates (e.g. a security's inactivation date), except issuance/validity attributes."""
    masked = {}
    for column in _date_columns(frame):
        if column in REFERENCE_DATE_COLS:
            continue
        late = pd.to_datetime(frame[column], errors="coerce") >= pd.Timestamp(HOLDOUT)
        if late.any():
            if strict:
                raise RuntimeError(f"{name}: {int(late.sum())} rows with {column} >= {HOLDOUT}")
            frame.loc[late, column] = None
            masked[column] = int(late.sum())
    return masked


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"window": [str(WIN_START), str(WIN_END)], "metadata": {}, "pulls": {}}
    db = connect()

    def run(name, sql, params=None, filename=None, strict=True):
        start = time.time()
        print(f"[{name}] querying...", flush=True)
        try:
            frame = db.raw_sql(sql, params=params)
            masked = enforce_no_holdout(name, frame, strict)
        except Exception as err:  # noqa: BLE001
            manifest["pulls"][name] = {"error": f"{type(err).__name__}: {str(err)[:500]}"}
            print(f"[{name}] FAILED: {type(err).__name__}", flush=True)
            try:
                db.connection.rollback()
            except Exception:  # noqa: BLE001
                pass
            return None
        entry = {
            "rows": len(frame),
            "seconds": round(time.time() - start, 1),
            "sql": " ".join(sql.split()),
        }
        if masked:
            entry["blanked_post_2017_reference_dates"] = masked
        if filename:
            frame.to_csv(CACHE / filename, index=False)
            entry["file"] = f"data/cache/{filename}"
        manifest["pulls"][name] = entry
        print(f"[{name}] {len(frame)} rows ({entry['seconds']}s)", flush=True)
        return frame

    try:
        # ---------------- Phase 1: metadata and access only ----------------
        available: dict[str, list[str]] = {}
        for key, qualified in TABLES.items():
            schema, table = qualified.split(".")
            info: dict = {"table": qualified}
            try:
                cols = db.raw_sql(META_SQL, params={"schema": schema, "table": table})
                info["columns"] = [
                    [c, t] for c, t in zip(cols["column_name"], cols["data_type"], strict=True)
                ]
                available[key] = list(cols["column_name"])
                if available[key]:
                    priv = db.raw_sql(
                        "select has_table_privilege(%(t)s, 'SELECT') as ok", params={"t": qualified}
                    )
                    info["select_privilege"] = bool(priv["ok"].iloc[0])
                else:
                    info["note"] = "not visible in information_schema"
            except Exception as err:  # noqa: BLE001
                info["error"] = f"{type(err).__name__}: {str(err)[:300]}"
            manifest["metadata"][key] = info
            print(
                f"[meta] {qualified}: {len(available.get(key, []))} columns, "
                f"select={info.get('select_privilege')}",
                flush=True,
            )

        # One-row probe: proves row-level access (not just schema visibility). Not saved.
        probe = run(
            "bondret_access_probe",
            f"select date, cusip from {TABLES['bondret']} where date < %(h)s limit 1",
            {"h": HOLDOUT},
        )
        bondret_ok = probe is not None and len(probe) == 1
        manifest["metadata"]["bondret"]["row_access"] = bondret_ok

        # ---------------- Phase 2: minimal rows ----------------
        base = {
            "ws": WIN_START,
            "we": WIN_END,
            "migration": MIGRATION,
            "holdout": HOLDOUT,
            "exec_floor": EXEC_FLOOR,
        }
        parts = [
            run(name, SQL[name], base)
            for name in (
                "trace_reported_in_window",
                "trace_later_reports",
                "trace_reversal_originals",
            )
        ]
        run(
            "trace_post_migration_count",
            SQL["trace_post_migration_count"],
            base,
            "trace_post_migration_count.csv",
        )
        parts = [p for p in parts if p is not None]
        if not parts:
            raise RuntimeError("no TRACE rows pulled; stopping before dependent pulls")
        trades = pd.concat(parts, ignore_index=True).drop_duplicates(
            subset=["trd_rpt_dt", "msg_seq_nb", "cusip_id"]
        )
        trades.to_csv(CACHE / "trace_enhanced_sample.csv", index=False)

        post_params = {"p0": POST_START, "p1": POST_END, "p_floor": POST_EXEC_FLOOR}
        post_parts = [
            run(name, SQL[name], post_params)
            for name in ("trace_2013_reported", "trace_2013_priors")
        ]
        post_parts = [p for p in post_parts if p is not None]
        if post_parts:
            post = pd.concat(post_parts, ignore_index=True).drop_duplicates(
                subset=["trd_rpt_dt", "msg_seq_nb", "cusip_id"]
            )
            post.to_csv(CACHE / "trace_enhanced_sample_2013.csv", index=False)
            manifest["pulls"]["trace_enhanced_sample_2013"] = {
                "rows": len(post),
                "file": "data/cache/trace_enhanced_sample_2013.csv",
                "by_part": post["pull_part"].value_counts().to_dict(),
            }
        manifest["pulls"]["trace_enhanced_sample"] = {
            "rows": len(trades),
            "file": "data/cache/trace_enhanced_sample.csv",
            "by_part": trades["pull_part"].value_counts().to_dict(),
        }

        window = trades.loc[
            pd.to_datetime(trades["trd_exctn_dt"]).between(
                pd.Timestamp(WIN_START), pd.Timestamp(WIN_END)
            )
        ]
        cusips = sorted(window["cusip_id"].dropna().astype(str).unique())
        cusip6 = sorted({c[:6] for c in cusips if len(c) == 9})
        tickers = sorted(window["company_symbol"].dropna().astype(str).str.strip().unique())
        manifest["sample_keys"] = {
            "cusips": len(cusips),
            "cusip6": len(cusip6),
            "tickers": len(tickers),
        }

        run(
            "trace_master",
            SQL["trace_master"],
            {"cusips": cusips, "holdout": HOLDOUT},
            "trace_master_sample.csv",
            strict=False,
        )
        bondcrsp = run(
            "bondcrsp_link",
            SQL["bondcrsp_link"],
            {"cusips": cusips, "holdout": HOLDOUT},
            "bondcrsp_link_sample.csv",
            strict=False,
        )
        fang = run(
            "fang_link",
            SQL["fang_link"],
            {"cusip6": cusip6, "l0": LINK_MONTHS[0], "l1": LINK_MONTHS[1]},
            "fang_link_sample.csv",
        )

        permnos = set()
        for frame, col in ((bondcrsp, "permno"), (fang, "permno")):
            if frame is not None:
                permnos |= {int(p) for p in frame[col].dropna()}
        gvkeys = set(fang["gvkey"].dropna().astype(str)) if fang is not None else set()

        ccm_cols = [c for c in WANTED["ccm"] if c in available.get("ccm", [])]
        if permnos and {"gvkey", "lpermno"} <= set(ccm_cols):
            ccm = run(
                "ccm_link",
                f"select {', '.join(ccm_cols)} from {TABLES['ccm']} "
                "where lpermno = any(%(permnos)s) and (linkdt is null or linkdt < %(h)s)",
                {"permnos": sorted(permnos), "h": HOLDOUT},
                "ccm_link_sample.csv",
                strict=False,
            )
            if ccm is not None:
                gvkeys |= set(ccm["gvkey"].dropna().astype(str))
        else:
            manifest["pulls"]["ccm_link"] = {"skipped": "table/columns not available or no permnos"}

        # Compustat identifier rows: only issuers tied to the sample, only confirmed columns.
        for key in ("security", "names"):
            cols = [c for c in WANTED[key] if c in available.get(key, [])]
            if "gvkey" not in cols:
                manifest["pulls"][f"comp_{key}"] = {"skipped": f"columns unavailable: {cols}"}
                continue
            where = ["gvkey = any(%(gvkeys)s)"]
            if "cusip" in cols:
                where.append("substr(cusip, 1, 6) = any(%(cusip6)s)")
            if "tic" in cols:
                where.append("tic = any(%(tickers)s)")
            frame = run(
                f"comp_{key}",
                f"select {', '.join(cols)} from {TABLES[key]} where {' or '.join(where)}",
                {"gvkeys": sorted(gvkeys), "cusip6": cusip6, "tickers": tickers},
                f"comp_{key}_sample.csv",
                strict=False,
            )
            if frame is not None:
                gvkeys |= set(frame["gvkey"].dropna().astype(str))

        company_cols = [c for c in WANTED["company"] if c in available.get("company", [])]
        if "gvkey" in company_cols and gvkeys:
            run(
                "comp_company",
                f"select {', '.join(company_cols)} from {TABLES['company']} "
                "where gvkey = any(%(gvkeys)s)",
                {"gvkeys": sorted(gvkeys)},
                "comp_company_sample.csv",
                strict=False,
            )
        manifest["sample_keys"]["candidate_gvkeys"] = len(gvkeys)

        run(
            "compustat_ratings",
            SQL["ratings"],
            {"gvkeys": sorted(gvkeys), "rs": RATINGS_START, "re": RATINGS_END},
            "compustat_ratings_sample.csv",
        )

        if bondret_ok:
            br_cols = available.get("bondret", [])
            run(
                "bondret",
                f"select {', '.join(chr(34) + c + chr(34) for c in br_cols)} "
                f"from {TABLES['bondret']} "
                "where cusip = any(%(cusips)s) and date between %(b0)s and %(b1)s "
                f"order by cusip, date limit {BONDRET_MAX_ROWS}",
                {"cusips": cusips, "b0": BONDRET_RANGE[0], "b1": BONDRET_RANGE[1]},
                "bondret_sample.csv",
                strict=False,
            )
        else:
            manifest["pulls"]["bondret"] = {"skipped": "no row-level access"}
    finally:
        db.close()
        (CACHE / "pull_manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
        print(f"Manifest: {CACHE / 'pull_manifest.json'}")


if __name__ == "__main__":
    main()
