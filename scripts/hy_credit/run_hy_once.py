"""ONE-LOGIN orchestrator for the #24 development pull (2006-01 .. 2017-12).

    cd <repo> && WRDS_USERNAME=<your WRDS id> uv run --no-project --python 3.12 \\
        --with wrds --with pyarrow python -I scripts/hy_credit/run_hy_once.py

Phase A  all local preflight; exits BEFORE any password prompt if anything fails.
Phase B  one getpass prompt, exactly one database connection (``open_single_connection``);
         after it opens, every connection constructor reachable from this process is
         replaced by a function that raises.
Phase C  smoke test: the first incomplete month on that connection, written and verified.
Phase D  remaining months and reference tables on the SAME connection. A failed statement
         is retried only while that connection is alive; a dead connection is recorded and
         the process exits. There is no reconnect path and no second prompt.
Phase E  connection closed; build, report, plots, QA, tests and lint run as local child
         processes (project environment) that never connect to WRDS.

The connection is a plain psycopg2 connection (no SQLAlchemy engine/pool), so a dropped
connection cannot be silently replaced by a new one. TCP keepalives are enabled.
Nothing dated 2018-01-01 or later is queried or kept. The password is never printed,
logged, written to disk or placed in the environment; no .pgpass is created.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent  # scripts/hy_credit; helper scripts live here
REPO = HERE.parents[1]
USERNAME = os.environ.get("WRDS_USERNAME", "")
BRANCH = "24-high-yield-credit"
HOLDOUT = date(2018, 1, 1)
RATINGS = (date(2005, 6, 30), date(2017, 12, 31))
MIN_FREE_GB = 25
STATEMENT_TIMEOUT = "60min"
FETCH_BATCH = 200_000
MAX_STATEMENT_RETRIES = 2
KEY_BATCH = 20_000

EXIT_OK, EXIT_PREFLIGHT, EXIT_AUTH, EXIT_DEAD, EXIT_DATA, EXIT_LOCAL = 0, 2, 3, 4, 5, 6

RAW_COLUMNS = [
    "cusip_id",
    "trd_exctn_dt",
    "trd_exctn_tm",
    "trd_rpt_dt",
    "msg_seq_nb",
    "orig_msg_seq_nb",
    "pr_trd_dt",
    "trc_st",
    "asof_cd",
    "entrd_vol_qt",
    "rptd_pr",
    "yld_sign_cd",
    "yld_pt",
    "rpt_side_cd",
    "cntra_mp_id",
    "dissem_fl",
    "wis_fl",
    "spcl_trd_fl",
    "sale_cndtn_cd",
    "trdg_mkt_cd",
    "trd_mod_4",
]
_DATE_COLS = {"trd_exctn_dt", "trd_rpt_dt", "pr_trd_dt"}
_FLOAT_COLS = {"entrd_vol_qt", "rptd_pr", "yld_pt"}
RAW_SCHEMA = pa.schema(
    [
        (c, pa.date32() if c in _DATE_COLS else pa.float64() if c in _FLOAT_COLS else pa.string())
        for c in RAW_COLUMNS
    ]
)

UNIVERSE_SQL = """
    select cusip_id from trace_enhanced.camasterfile
    where sub_prdct_type = 'CORP' and coalesce(cnvrb_fl, 'N') <> 'Y'
      and cpn_type_cd = 'FXPV' and coalesce(ind_144a, 'N') <> 'Y'
      and (stdt is null or stdt < %(holdout)s)"""
MONTH_SQL = f"""
select {", ".join(RAW_COLUMNS)} from trace_enhanced.trace_enhanced
where trd_exctn_dt between %(m0)s and %(m1)s and trd_rpt_dt < %(holdout)s
  and (dissem_fl = 'Y' or coalesce(trc_st, '') <> 'T' or coalesce(asof_cd, '') = 'R')
  and cusip_id in ({UNIVERSE_SQL})"""
REFERENCE_SQL = {
    "master": "select * from trace_enhanced.camasterfile "
    "where cusip_id = any(%(keys)s) and (stdt is null or stdt < %(holdout)s)",
    "bondcrsp": "select * from wrdsapps_link_crsp_bond.bondcrsp_link "
    "where cusip = any(%(keys)s) and (link_startdt is null or link_startdt < %(holdout)s)",
    "crsp_names": "select permno, ncusip, ticker, comnam, namedt, nameenddt from {table} "
    "where permno = any(%(keys)s) and (namedt is null or namedt < %(holdout)s)",
    "security": "select gvkey, iid, cusip, tic, tpci, excntry, dldtei from comp.security "
    "where substr(cusip, 1, 6) = any(%(keys)s)",
    "ratings": "select gvkey, datadate, splticrm, spsdrm, spsticrm from comp.adsprate "
    "where gvkey = any(%(keys)s) and datadate between %(r0)s and %(r1)s",
}
CRSP_NAMES_TABLES = ("crsp.stocknames", "crsp_a_stock.stocknames")
REFERENCE_DATE_COLS = {"mtrty_dt", "enddt", "link_enddt", "trace_enddt", "crsp_enddt", "nameenddt"}
SAMPLE_FILES = [
    "data/cache/trace_enhanced_sample.csv",
    "data/cache/trace_enhanced_sample_2013.csv",
    "data/cache/trace_master_sample.csv",
    "data/cache/comp_security_sample.csv",
    "data/cache/bondcrsp_link_sample.csv",
    "data/cache/compustat_ratings_sample.csv",
    "data/cache/pull_manifest.json",
]
FRED_SERIES = [
    "DGS1MO",
    "DGS3MO",
    "DGS6MO",
    "DGS1",
    "DGS2",
    "DGS3",
    "DGS5",
    "DGS7",
    "DGS10",
    "DGS20",
    "DGS30",
]


# --------------------------------------------------------------------------------------
# Configuration, logging, manifest
# --------------------------------------------------------------------------------------


@dataclass
class Config:
    repo: Path = REPO
    cache_root: Path | None = None  # default <repo>/data/cache
    results: Path | None = None  # default <repo>/results/hy_credit
    log_dir: Path | None = None  # default <repo>/data/cache/logs (git-ignored)
    first_month: str = "2006-01"
    last_month: str = "2017-12"
    run_child_checks: bool = True  # gates, pytest, ruff, preflight build
    run_post_pull: bool = True

    def __post_init__(self):
        self.cache_root = self.cache_root or self.repo / "data" / "cache"
        self.results = self.results or self.repo / "results" / "hy_credit"
        self.log_dir = self.log_dir or self.cache_root / "logs"
        self.out = self.cache_root / "hy_credit"
        self.raw = self.out / "trace_raw"
        self.manifest = self.out / "pull_dev_manifest.json"

    def months(self) -> list[pd.Period]:
        return list(pd.period_range(self.first_month, self.last_month, freq="M"))


class Log:
    """Timestamped log to stdout and one log file. Registered secrets are scrubbed."""

    def __init__(self, path: Path):
        self.path = path
        self._secrets: list[str] = []
        self._fh = open(path, "a", buffering=1)  # noqa: SIM115

    def secret(self, value: str | None) -> None:
        self._secrets = [value] if value else []

    def scrub(self, text: str) -> str:
        for s in self._secrets:
            if s:
                from urllib.parse import quote

                text = text.replace(s, "***").replace(quote(s), "***")
        return text

    def __call__(self, message: str, phase: str = "") -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {phase:<7} {self.scrub(str(message))}"
        print(line, flush=True)
        self._fh.write(line + "\n")

    def close(self) -> None:
        self._fh.close()


def fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}
        self.data.setdefault("version", 2)
        self.data.setdefault("months", {})
        self.data.setdefault("reference", {})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(self.path, json.dumps(self.data, indent=1, default=str))


# --------------------------------------------------------------------------------------
# Parquet writing and integrity
# --------------------------------------------------------------------------------------


def _to_text(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if hasattr(value, "strftime") and not isinstance(value, (date, datetime)):  # datetime.time
        return value.strftime("%H:%M:%S")
    text = str(value).strip()
    return text or None


def to_arrow(frame: pd.DataFrame, schema: pa.Schema) -> pa.Table:
    """Convert a DB-API frame to the fixed raw schema (dates, floats, text)."""
    cols = {}
    for f in schema:
        values = frame[f.name]
        if pa.types.is_date32(f.type):
            cols[f.name] = pd.to_datetime(values, errors="raise").dt.date
        elif pa.types.is_float64(f.type):
            cols[f.name] = pd.to_numeric(values, errors="raise").astype("float64")
        else:
            cols[f.name] = values.map(_to_text).astype(object)
    out = pd.DataFrame(cols)
    for f in schema:
        if pa.types.is_date32(f.type):
            out[f.name] = out[f.name].where(out[f.name].notna(), None)
    return pa.Table.from_pandas(out, schema=schema, preserve_index=False)


def write_parquet_frame(frame: pd.DataFrame, path: Path, schema: pa.Schema) -> int:
    """Write via temp file, fsync, atomic rename. Returns rows written."""
    tmp = path.with_name(path.name + ".tmp")
    table = to_arrow(frame, schema)
    pq.write_table(table, tmp)
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)
    return table.num_rows


def server_side_keep(raw: pd.DataFrame) -> pd.Series:
    """Local mirror of the SQL dissemination rule (used by the preflight rehearsal)."""
    status = raw["trc_st"].astype("string").str.upper()
    dissem = raw["dissem_fl"].astype("string").str.upper().eq("Y").fillna(False)
    reversal = raw["asof_cd"].astype("string").str.upper().eq("R").fillna(False)
    return (dissem | status.ne("T").fillna(True) | reversal).to_numpy()


def month_file(cfg: Config, month: pd.Period) -> Path:
    return cfg.raw / f"{month}.parquet"


def check_month_file(path: Path, month: pd.Period, expected_rows: int | None) -> str | None:
    """None if the chunk is complete and sane, else the problem."""
    if not path.exists():
        return "file missing"
    try:
        meta = pq.ParquetFile(path)
        names = meta.schema_arrow.names
        rows = meta.metadata.num_rows
        if names != RAW_COLUMNS:
            return f"schema mismatch: {names}"
        if expected_rows is not None and rows != expected_rows:
            return f"row count {rows} != manifest {expected_rows}"
        if rows == 0:
            return "empty chunk"
        dates = pq.read_table(path, columns=["trd_exctn_dt", "trd_rpt_dt"]).to_pandas()
        ex = pd.to_datetime(dates["trd_exctn_dt"])
        rp = pd.to_datetime(dates["trd_rpt_dt"])
        if ex.min() < month.start_time or ex.max() > month.end_time:
            return "execution dates outside the month"
        if rp.isna().any() or rp.max() >= pd.Timestamp(HOLDOUT):
            return "report date missing or in the holdout"
    except Exception as err:  # noqa: BLE001
        return f"unreadable: {type(err).__name__}: {err}"
    return None


def month_params(month: pd.Period) -> dict:
    m0, m1 = month.start_time.date(), month.end_time.date()
    return {"m0": m0, "m1": m1, "holdout": HOLDOUT}


def validate_query_construction(cfg: Config) -> list[str]:
    """Static checks of every monthly statement (no SQL is executed)."""
    problems = []
    placeholders = set(re.findall(r"%\((\w+)\)s", MONTH_SQL))
    for month in cfg.months():
        params = month_params(month)
        if set(params) != placeholders:
            problems.append(f"{month}: placeholders {placeholders} != params {set(params)}")
        if not (params["m0"] <= params["m1"] < HOLDOUT):
            problems.append(f"{month}: bad bounds {params['m0']}..{params['m1']}")
        if params["m0"].day != 1 or (pd.Timestamp(params["m1"]) + pd.Timedelta(days=1)).day != 1:
            problems.append(f"{month}: not a full calendar month")
    for needle in ("trd_rpt_dt < %(holdout)s", "dissem_fl = 'Y'", "cusip_id in ("):
        if needle not in MONTH_SQL:
            problems.append(f"MONTH_SQL lacks {needle!r}")
    try:  # optional full parse when sqlglot is available (not required at run time)
        import sqlglot

        sqlglot.parse_one(re.sub(r"%\((\w+)\)s", "'x'", MONTH_SQL), read="postgres")
    except ImportError:
        pass
    except Exception as err:  # noqa: BLE001
        problems.append(f"MONTH_SQL does not parse: {err}")
    return problems


# --------------------------------------------------------------------------------------
# Child processes (project environment; never connect to WRDS)
# --------------------------------------------------------------------------------------


def unhide_editable_pth(cfg: Config) -> None:
    """Clear the recurring macOS hidden flag on the editable-install .pth (see memory notes)."""
    for pth in (cfg.repo / ".venv").glob("lib/python*/site-packages/_editable_impl_macro.pth"):
        st = os.stat(pth)
        if st.st_flags & stat.UF_HIDDEN:
            os.chflags(pth, st.st_flags & ~stat.UF_HIDDEN)


def child_env(repo: Path) -> dict:
    """Environment for local children: no PG*/password variables, no inherited venv, and the
    repo on PYTHONPATH so the helper scripts import ``macro`` even if the editable .pth
    is hidden."""
    drop = ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH", "UV_PROJECT_ENVIRONMENT")
    env = {
        k: v
        for k, v in os.environ.items()
        if k not in drop and not k.startswith("PG") and "PASSWORD" not in k.upper()
    }
    env["PYTHONPATH"] = str(repo)
    return env


def run_child(cfg: Config, log: Log, name: str, argv: list[str], phase: str) -> int:
    """Run a local child in the repo with no stdin; capture its output into the log."""
    unhide_editable_pth(cfg)
    start = time.time()
    proc = subprocess.run(
        argv,
        cwd=cfg.repo,
        env=child_env(cfg.repo),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    for line in proc.stdout.splitlines()[-25:]:
        log(f"  [{name}] {line}", phase)
    log(f"child {name} exit={proc.returncode} ({time.time() - start:.0f}s)", phase)
    return proc.returncode


# --------------------------------------------------------------------------------------
# Phase A: preflight
# --------------------------------------------------------------------------------------


def preflight(cfg: Config, log: Log, manifest: Manifest) -> list[pd.Period]:
    """All local checks. Returns the months still to pull; raises SystemExit on failure."""
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> bool:
        log(f"{'PASS' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}", "A")
        if not ok:
            failures.append(name)
        return ok

    if not check("repo exists", (cfg.repo / "pyproject.toml").exists(), str(cfg.repo)):
        raise SystemExit(EXIT_PREFLIGHT)
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=cfg.repo,
        text=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    ).stdout.strip()
    check("branch", branch == BRANCH, branch)
    check("WRDS_USERNAME set", bool(USERNAME), "export WRDS_USERNAME=<your WRDS id>")
    ignored = (
        subprocess.run(
            ["git", "check-ignore", "-q", "data/cache/x.parquet"],
            cwd=cfg.repo,
            stdin=subprocess.DEVNULL,
        ).returncode
        == 0
    )
    check("data/cache exists and is gitignored", (cfg.repo / "data/cache").is_dir() and ignored)
    tracked = subprocess.run(
        ["git", "ls-files", "data/cache"],
        cwd=cfg.repo,
        text=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    ).stdout.strip()
    check("no tracked files under data/cache", tracked == "")
    for d in (cfg.raw, cfg.results, cfg.log_dir):
        try:
            d.mkdir(parents=True, exist_ok=True)
            probe = d / ".write_probe"
            probe.write_text("ok")
            probe.unlink()
            check(f"writable {d}", True)
        except OSError as err:
            check(f"writable {d}", False, str(err))
    free_gb = shutil.disk_usage(cfg.out).free / 1e9
    check("free disk", free_gb >= MIN_FREE_GB, f"{free_gb:.0f} GB (need {MIN_FREE_GB})")
    import importlib.util

    for module in ("psycopg2", "wrds", "pyarrow", "pandas"):
        check(f"import {module} (orchestrator env)", importlib.util.find_spec(module) is not None)
    for rel in SAMPLE_FILES + [f"data/cache/fred/{s}.csv" for s in FRED_SERIES]:
        check(f"local file {rel}", (cfg.repo / rel).exists())
    try:
        last_fred = max(
            pd.read_csv(cfg.repo / f"data/cache/fred/{s}.csv", usecols=[0]).iloc[:, 0].max()
            for s in FRED_SERIES
        )
        check(
            "FRED cache before holdout", pd.Timestamp(last_fred) < pd.Timestamp(HOLDOUT), last_fred
        )
    except Exception as err:  # noqa: BLE001
        check("FRED cache readable", False, str(err))

    # Manifest and existing chunks.
    for tmp in list(cfg.raw.glob("*.tmp")) + list(cfg.out.glob("*.tmp")):
        log(f"removing incomplete temp file {tmp.name}", "A")
        tmp.unlink()
    months = cfg.months()
    remaining = []
    for month in months:
        path = month_file(cfg, month)
        entry = manifest.data["months"].get(str(month))
        if entry:
            problem = check_month_file(path, month, entry.get("rows"))
            if problem:
                check(
                    f"manifest month {month}",
                    False,
                    f"{problem}; delete {path} and its manifest entry to re-pull",
                )
            continue
        if path.exists():  # complete (atomic rename) but not yet recorded: adopt after checks
            problem = check_month_file(path, month, None)
            if problem:
                check(f"orphan chunk {month}", False, f"{problem}; delete {path} to re-pull")
                continue
            rows = pq.ParquetFile(path).metadata.num_rows
            manifest.data["months"][str(month)] = {"rows": rows, "adopted": True}
            log(f"adopted complete chunk {month} ({rows} rows)", "A")
            continue
        remaining.append(month)
    stray = {p.stem for p in cfg.raw.glob("*.parquet")} - {str(m) for m in months}
    check("no chunks outside 2006-01..2017-12", not stray, ", ".join(sorted(stray)))
    for name, info in manifest.data.get("reference", {}).items():
        ok = (cfg.out / f"{name}.parquet").exists() and info.get("rows") is not None
        check(f"reference file {name}", ok)
    if manifest.data.get("status") == "running":
        log("previous run ended without a final status; resuming from verified chunks", "A")

    problems = validate_query_construction(cfg)
    check(f"query construction for {len(months)} months", not problems, "; ".join(problems[:3]))

    # Parquet round trip with fake rows shaped like DB-API output.
    try:
        fake = pd.DataFrame({c: [None, None] for c in RAW_COLUMNS})
        fake["cusip_id"] = ["037833100", "00206RAG7"]
        fake["trd_exctn_dt"] = [date(2010, 1, 4), date(2010, 1, 5)]
        fake["trd_rpt_dt"] = [date(2010, 1, 4), date(2010, 1, 6)]
        fake["trd_exctn_tm"] = [datetime(2010, 1, 4, 10, 7, 23).time(), None]
        fake["msg_seq_nb"] = ["39149", "216"]
        fake["rptd_pr"] = [101.5, 99.0]
        fake["yld_pt"] = [5.25, None]
        probe = cfg.out / ".preflight_probe.parquet"
        write_parquet_frame(fake, probe, RAW_SCHEMA)
        back = pq.read_table(probe)
        ok = back.schema.equals(RAW_SCHEMA) and back.num_rows == 2
        ok = ok and back.column("trd_exctn_tm").to_pylist() == ["10:07:23", None]
        probe.unlink()
        check("parquet schema round trip", ok)
    except Exception as err:  # noqa: BLE001
        check("parquet schema round trip", False, f"{type(err).__name__}: {err}")

    if cfg.run_child_checks and not failures:
        children = [
            (
                "project imports",
                [
                    "uv",
                    "run",
                    "python",
                    "-c",
                    "import macro.data.hy_runs, macro.features.hy_credit, pyarrow",
                ],
            ),
            (
                "sample gates",
                [
                    "uv",
                    "run",
                    "python",
                    "-m",
                    "macro.data.hy_runs",
                    "gate-sample",
                    "--results",
                    str(cfg.log_dir / "gate_sample"),
                ],
            ),
            ("pytest", ["uv", "run", "pytest", "-q"]),
            ("ruff check", ["uv", "run", "ruff", "check", "."]),
            ("ruff format", ["uv", "run", "ruff", "format", "--check", "."]),
            (
                "preflight build/report/plots/QA on real sample",
                ["uv", "run", "--with", "matplotlib", "python", str(HERE / "preflight_build.py")],
            ),
        ]
        for name, argv in children:
            if run_child(cfg, log, name, argv, "A") != 0:
                check(name, False, "child failed")
                break
            check(name, True)

    manifest.save()
    if failures:
        log(f"PREFLIGHT FAILED ({len(failures)}): {failures}. No password requested.", "A")
        raise SystemExit(EXIT_PREFLIGHT)
    done = len(months) - len(remaining)
    log(
        f"PREFLIGHT PASSED. months complete {done}/{len(months)}; to pull {len(remaining)}"
        + (f" starting {remaining[0]}" if remaining else ""),
        "A",
    )
    return remaining


# --------------------------------------------------------------------------------------
# Phase B: the single connection
# --------------------------------------------------------------------------------------

CONNECT_CALLS = 0


class DeadConnection(RuntimeError):
    """The single connection is unusable; the run must stop (no reconnect)."""


class DataSanityError(RuntimeError):
    """Returned rows violate a schema/date/holdout rule."""


def _driver_connect(**kwargs):
    """The ONLY database connection constructor in this execution path."""
    import psycopg2

    return psycopg2.connect(**kwargs)


def _forbidden(*_args, **_kwargs):
    raise RuntimeError("second database connection attempted; this run allows exactly one")


def _seal_connection_constructors() -> None:
    """After the one connection opens, make every reachable constructor raise."""
    global _driver_connect
    _driver_connect = _forbidden
    import psycopg2

    psycopg2.connect = _forbidden
    for module_name, attr in (
        ("sqlalchemy", "create_engine"),
        ("wrds.sql", "Connection"),
        ("wrds", "Connection"),
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, attr):
            setattr(module, attr, _forbidden)


def open_single_connection(log: Log):
    """Prompt once, open exactly one connection, then seal all constructors."""
    global CONNECT_CALLS
    from wrds.sql import WRDS_CONNECT_ARGS, WRDS_POSTGRES_DB, WRDS_POSTGRES_HOST, WRDS_POSTGRES_PORT

    if CONNECT_CALLS:
        raise RuntimeError("open_single_connection called twice")
    password = getpass.getpass(f"WRDS password for {USERNAME} (input hidden): ")
    log.secret(password)
    log("connecting (approve the Duo push)...", "B")
    CONNECT_CALLS += 1
    try:
        conn = _driver_connect(
            host=WRDS_POSTGRES_HOST,
            port=WRDS_POSTGRES_PORT,
            dbname=WRDS_POSTGRES_DB,
            user=USERNAME,
            password=password,
            sslmode=WRDS_CONNECT_ARGS.get("sslmode", "require"),
            application_name="macro-hy24-once",
            connect_timeout=120,
            keepalives=1,
            keepalives_idle=60,
            keepalives_interval=15,
            keepalives_count=8,
        )
    except Exception as err:  # noqa: BLE001
        log(f"login failed: {type(err).__name__}: {str(err).splitlines()[0][:300]}", "B")
        raise SystemExit(EXIT_AUTH) from None
    finally:
        del password
        log.secret(None)
    _seal_connection_constructors()
    assert CONNECT_CALLS == 1, CONNECT_CALLS
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(f"SET statement_timeout = '{STATEMENT_TIMEOUT}'")
        cur.execute("select 1")
        cur.fetchone()
    log("connected (1 connection; constructors sealed; keepalives on)", "B")
    return conn


def connection_alive(conn) -> bool:
    if getattr(conn, "closed", 1):
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("select 1")
            cur.fetchone()
        return True
    except Exception:  # noqa: BLE001
        return False


def with_retries(conn, log: Log, label: str, fn):
    """Run fn(); retry a failed statement only while the same connection is alive."""
    for attempt in range(MAX_STATEMENT_RETRIES + 1):
        try:
            return fn()
        except DataSanityError:
            raise
        except Exception as err:  # noqa: BLE001
            msg = f"{type(err).__name__}: {str(err).splitlines()[0][:300] if str(err) else ''}"
            if not connection_alive(conn):
                raise DeadConnection(f"{label}: {msg}") from None
            if attempt == MAX_STATEMENT_RETRIES:
                raise
            log(f"{label}: statement failed ({msg}); connection alive; retry {attempt + 1}", "D")
            time.sleep(5 * (attempt + 1))


# --------------------------------------------------------------------------------------
# Phases C and D: pulls on the single connection
# --------------------------------------------------------------------------------------


def pull_month(conn, cfg: Config, month: pd.Period, log: Log) -> int:
    """Stream one month through a server-side cursor into a verified Parquet chunk."""
    path = month_file(cfg, month)
    tmp = path.with_name(path.name + ".tmp")
    params = month_params(month)
    lo, hi = pd.Timestamp(params["m0"]), pd.Timestamp(params["m1"])
    rows = 0
    cursor_name = f"hy_{str(month).replace('-', '_')}_{int(time.time())}"
    cur = conn.cursor(name=cursor_name, withhold=True)
    writer = None
    try:
        cur.itersize = FETCH_BATCH
        cur.execute(MONTH_SQL, params)
        while True:
            batch = cur.fetchmany(FETCH_BATCH)
            if not batch:
                break
            names = [d[0] for d in cur.description]
            if names != RAW_COLUMNS:
                raise DataSanityError(f"{month}: unexpected columns {names}")
            frame = pd.DataFrame(batch, columns=names)
            ex = pd.to_datetime(frame["trd_exctn_dt"])
            rp = pd.to_datetime(frame["trd_rpt_dt"])
            if ex.isna().any() or ex.min() < lo or ex.max() > hi:
                raise DataSanityError(f"{month}: execution date outside the month")
            if rp.isna().any() or rp.max() >= pd.Timestamp(HOLDOUT):
                raise DataSanityError(f"{month}: report date missing or >= {HOLDOUT}")
            if pd.to_datetime(frame["pr_trd_dt"]).max() >= pd.Timestamp(HOLDOUT):
                raise DataSanityError(f"{month}: prior report date >= {HOLDOUT}")
            table = to_arrow(frame, RAW_SCHEMA)
            if writer is None:
                writer = pq.ParquetWriter(tmp, RAW_SCHEMA)
            writer.write_table(table)
            rows += table.num_rows
    finally:
        try:
            cur.close()
        except Exception:  # noqa: BLE001
            pass
        if writer is not None:
            writer.close()
    if rows == 0:
        tmp.unlink(missing_ok=True)
        raise DataSanityError(f"{month}: zero rows returned (impossible for 2006-2017 TRACE)")
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)
    problem = check_month_file(path, month, rows)
    if problem:
        raise DataSanityError(f"{month}: written chunk failed verification: {problem}")
    return rows


def fetch_all(conn, sql: str, params: dict) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
        names = [d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=names)


def blank_late_dates(frame: pd.DataFrame) -> dict:
    masked = {}
    for column in frame.columns:
        if column in REFERENCE_DATE_COLS:
            continue
        values = frame[column]
        if values.dtype == object and values.map(lambda v: isinstance(v, date)).any():
            late = pd.to_datetime(values, errors="coerce") >= pd.Timestamp(HOLDOUT)
            if late.any():
                frame.loc[late, column] = None
                masked[column] = int(late.sum())
    return masked


def pull_reference(conn, cfg: Config, manifest: Manifest, log: Log) -> None:
    """Reference tables keyed on the pulled CUSIPs, on the same connection."""
    cusips: set[str] = set()
    for path in sorted(cfg.raw.glob("*.parquet")):
        cusips |= set(pq.read_table(path, columns=["cusip_id"]).column(0).drop_null().to_pylist())
    log(f"{len(cusips):,} distinct CUSIPs in development chunks", "D")

    def pull(name: str, sql: str, keys, extra: dict) -> pd.DataFrame:
        done = manifest.data["reference"].get(name)
        path = cfg.out / f"{name}.parquet"
        if done and path.exists() and done.get("keys_hash") == _keys_hash(keys):
            log(f"{name}: already complete ({done['rows']:,} rows)", "D")
            return pd.read_parquet(path)
        parts = []
        for chunk in _batched(keys):
            parts.append(
                with_retries(
                    conn, log, name, lambda c=chunk: fetch_all(conn, sql, {"keys": c, **extra})
                )
            )
        frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        masked = blank_late_dates(frame)
        tmp = path.with_name(path.name + ".tmp")
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        fsync_dir(path.parent)
        manifest.data["reference"][name] = {
            "rows": len(frame),
            "keys_hash": _keys_hash(keys),
            "blanked_post_2017": masked,
        }
        manifest.save()
        log(f"{name}: COMPLETE rows={len(frame):,}", "D")
        return frame

    pull("master", REFERENCE_SQL["master"], cusips, {"holdout": HOLDOUT})
    bondcrsp = pull("bondcrsp", REFERENCE_SQL["bondcrsp"], cusips, {"holdout": HOLDOUT})
    cusip6 = {c[:6] for c in cusips if len(c) == 9}
    names_table = manifest.data.get("crsp_names_table")
    if names_table is None and "crsp_names_probe" not in manifest.data:
        probes = {}
        for candidate in CRSP_NAMES_TABLES:
            try:
                fetch_all(conn, f"select permno from {candidate} limit 1", {})
                names_table = candidate
                break
            except Exception as err:  # noqa: BLE001  (permission errors keep the connection alive)
                if not connection_alive(conn):
                    raise DeadConnection(f"crsp names probe: {type(err).__name__}") from None
                probes[candidate] = type(err).__name__
        manifest.data["crsp_names_probe"] = probes
        manifest.data["crsp_names_table"] = names_table
        manifest.save()
    log(f"CRSP stock names table: {names_table or 'not readable'}", "D")
    if names_table and len(bondcrsp):
        permnos = {int(p) for p in bondcrsp["permno"].dropna()}
        names = pull(
            "crsp_names",
            REFERENCE_SQL["crsp_names"].format(table=names_table),
            permnos,
            {"holdout": HOLDOUT},
        )
        cusip6 |= {str(c)[:6] for c in names["ncusip"].dropna()}
    security = pull("security", REFERENCE_SQL["security"], cusip6, {})
    gvkeys = set(security["gvkey"].dropna().astype(str))
    ratings = pull(
        "ratings", REFERENCE_SQL["ratings"], gvkeys, {"r0": RATINGS[0], "r1": RATINGS[1]}
    )
    if len(ratings) and pd.to_datetime(ratings["datadate"]).max() >= pd.Timestamp(HOLDOUT):
        raise DataSanityError("ratings contain holdout dates")


def _batched(values, size: int = KEY_BATCH):
    values = sorted(values)
    for i in range(0, len(values), size):
        yield values[i : i + size]


def _keys_hash(keys) -> str:
    return hashlib.sha256("\n".join(sorted(map(str, keys))).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main(cfg: Config | None = None) -> int:
    cfg = cfg or Config()
    cfg.log_dir.mkdir(parents=True, exist_ok=True)
    log = Log(cfg.log_dir / f"hy_once_{datetime.now():%Y%m%d_%H%M%S}.log")
    cfg.out.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(cfg.manifest)
    conn = None
    try:
        log(f"log file {log.path}", "start")
        try:
            remaining = preflight(cfg, log, manifest)
        except SystemExit as stop:
            return int(stop.code or EXIT_PREFLIGHT)

        needs_wrds = bool(remaining) or not manifest.data.get("reference_complete")
        if needs_wrds:
            try:
                conn = open_single_connection(log)
            except SystemExit as stop:
                return int(stop.code or EXIT_AUTH)
            manifest.data["status"] = "running"
            manifest.data["started_at"] = datetime.now().isoformat(timespec="seconds")
            manifest.save()
            for i, month in enumerate(remaining):
                start = time.time()
                rows = with_retries(
                    conn, log, str(month), lambda m=month: pull_month(conn, cfg, m, log)
                )
                manifest.data["months"][str(month)] = {
                    "rows": rows,
                    "seconds": round(time.time() - start),
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                }
                manifest.data["last_completed_month"] = str(month)
                manifest.save()
                if i == 0:
                    log("=" * 64, "C")
                    log(f"FIRST DEVELOPMENT MONTH WRITTEN SUCCESSFULLY: {month}", "C")
                    log("=" * 64, "C")
                total = sum(v["rows"] for v in manifest.data["months"].values())
                log(
                    f"[{month}] COMPLETE rows={rows:,} elapsed={time.time() - start:.0f}s "
                    f"total={total:,} ({len(manifest.data['months'])}/{len(cfg.months())})",
                    "D",
                )
            pull_reference(conn, cfg, manifest, log)
            manifest.data["reference_complete"] = True
            manifest.data["status"] = "pull_complete"
            manifest.save()
            conn.close()
            conn = None
            log("WRDS connection closed; no further connections are possible in this process", "D")
        else:
            log("all months and reference tables already complete; no login needed", "D")

        if cfg.run_post_pull:
            steps = [
                (
                    "build-dev",
                    [
                        "uv",
                        "run",
                        "python",
                        "-m",
                        "macro.data.hy_runs",
                        "build-dev",
                        "--cache",
                        str(cfg.cache_root),
                        "--results",
                        str(cfg.results),
                    ],
                ),
                (
                    "report-dev",
                    [
                        "uv",
                        "run",
                        "python",
                        "-m",
                        "macro.data.hy_runs",
                        "report-dev",
                        "--cache",
                        str(cfg.cache_root),
                        "--results",
                        str(cfg.results),
                    ],
                ),
                (
                    "plots",
                    [
                        "uv",
                        "run",
                        "--with",
                        "matplotlib",
                        "python",
                        str(HERE / "plot_hy.py"),
                        str(cfg.out / "hy_daily.parquet"),
                        str(cfg.results),
                    ],
                ),
                (
                    "final QA",
                    [
                        "uv",
                        "run",
                        "python",
                        str(HERE / "qa_dev.py"),
                        str(cfg.cache_root),
                        str(cfg.results),
                    ],
                ),
                ("pytest", ["uv", "run", "pytest", "-q"]),
                ("ruff check", ["uv", "run", "ruff", "check", "."]),
                ("ruff format", ["uv", "run", "ruff", "format", "--check", "."]),
            ]
            for name, argv in steps:
                code = run_child(cfg, log, name, argv, "E")
                manifest.data.setdefault("post_pull", {})[name] = code
                manifest.save()
                if code != 0:
                    log(
                        f"LOCAL STEP FAILED: {name} (exit {code}). Data is cached; rerun resumes.",
                        "E",
                    )
                    return EXIT_LOCAL
        manifest.data["status"] = "complete"
        manifest.save()
        log("ALL PHASES COMPLETE", "E")
        return EXIT_OK
    except DeadConnection as err:
        manifest.data["status"] = "dead_connection"
        manifest.data["failure"] = {
            "error": log.scrub(str(err)),
            "at": datetime.now().isoformat(timespec="seconds"),
            "last_completed_month": manifest.data.get("last_completed_month"),
        }
        manifest.save()
        log(
            f"CONNECTION LOST: {err}. Not reconnecting. Last completed month: "
            f"{manifest.data.get('last_completed_month')}. Rerun the same command to resume.",
            "D",
        )
        return EXIT_DEAD
    except DataSanityError as err:
        manifest.data["status"] = "data_sanity_failure"
        manifest.data["failure"] = {
            "error": str(err),
            "last_completed_month": manifest.data.get("last_completed_month"),
        }
        manifest.save()
        log(f"DATA SANITY FAILURE: {err}", "D")
        return EXIT_DATA
    except KeyboardInterrupt:
        manifest.data["status"] = "interrupted"
        manifest.save()
        log("interrupted by user; completed months are kept; rerun resumes", "D")
        return 130
    except Exception as err:  # noqa: BLE001
        manifest.data["status"] = "error"
        manifest.data["failure"] = {"error": log.scrub(f"{type(err).__name__}: {err}")}
        manifest.save()
        log(log.scrub(traceback.format_exc()), "error")
        return EXIT_LOCAL
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
        log.close()


if __name__ == "__main__":
    sys.exit(main())
