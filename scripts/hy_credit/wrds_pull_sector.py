"""ONE-LOGIN pull of point-in-time industry classifications for the #24 issuer panel.

    cd <repo> && WRDS_USERNAME=<your WRDS id> uv run --no-project --python 3.12 \\
        --with wrds --with pyarrow python -I scripts/hy_credit/wrds_pull_sector.py

Local preflight first (exits before any password prompt on failure), then exactly one
psycopg2 connection (no SQLAlchemy pool, no reconnect, keepalives on), then up to four
small queries restricted to the GVKEYs / PERMNOs already in the development panel:

1. metadata: column lists and SELECT privilege for the three tables below;
2. ``comp.co_hgic``: historical GICS (``indfrom``/``indthru``) -- preferred source;
3. ``comp.funda``: annual historical SIC/NAICS (``sich``/``naicsh``) -- fallback;
4. ``crsp.stocknames``: dated CRSP SIC (``siccd``, ``namedt``/``nameenddt``) -- fallback.

Nothing dated 2018-01-01 or later is requested: start dates are bounded below 2018 and
end dates are capped at 2017-12-31 in SQL, and returned dates are re-checked. Output goes
only to ``data/cache/hy_residual/sector/`` (git-ignored); ``data/cache/hy_credit/`` is
only read. A permission error on one table is recorded and the others still run. The
password is never printed, logged or saved; no .pgpass is created.
"""

import getpass
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "data" / "cache" / "hy_credit"
OUT = REPO / "data" / "cache" / "hy_residual" / "sector"
USERNAME = os.environ.get("WRDS_USERNAME", "")
BRANCH = "24-high-yield-credit"
HOLDOUT = "2018-01-01"
CAP = "2017-12-31"
EXIT_PREFLIGHT, EXIT_AUTH = 2, 3

META_SQL = """
select table_schema, table_name, column_name, data_type
from information_schema.columns
where (table_schema, table_name) in (('comp','co_hgic'), ('comp','funda'), ('crsp','stocknames'))
  and column_name = any(%(cols)s)
order by table_schema, table_name, ordinal_position
"""
PRIV_SQL = "select has_table_privilege(%(t)s, 'SELECT')"
QUERIES = {
    "co_hgic": (
        "select gvkey, indfrom, least(coalesce(indthru, date %(cap)s), date %(cap)s) as indthru,"
        " indthru is null as open_ended, gsector, ggroup, gind, gsubind"
        " from comp.co_hgic where gvkey = any(%(gvkeys)s) and indfrom < date %(holdout)s"
        " order by gvkey, indfrom",
        ["indfrom", "indthru"],
    ),
    "funda_sic": (
        "select gvkey, datadate, fyear, sich, naicsh from comp.funda"
        " where gvkey = any(%(gvkeys)s) and datadate >= date '2000-01-01'"
        " and datadate < date %(holdout)s and indfmt = 'INDL' and datafmt = 'STD'"
        " and popsrc = 'D' and consol = 'C' order by gvkey, datadate",
        ["datadate"],
    ),
    "crsp_siccd": (
        "select permno, namedt, least(coalesce(nameenddt, date %(cap)s), date %(cap)s)"
        " as nameenddt, siccd from crsp.stocknames"
        " where permno = any(%(permnos)s) and namedt < date %(holdout)s order by permno, namedt",
        ["namedt", "nameenddt"],
    ),
}
META_COLS = [
    "gvkey",
    "indfrom",
    "indthru",
    "gsector",
    "ggroup",
    "gind",
    "gsubind",
    "datadate",
    "fyear",
    "sich",
    "naicsh",
    "indfmt",
    "datafmt",
    "popsrc",
    "consol",
    "permno",
    "namedt",
    "nameenddt",
    "siccd",
]
TABLE_NAMES = {
    "co_hgic": "comp.co_hgic",
    "funda_sic": "comp.funda",
    "crsp_siccd": "crsp.stocknames",
}


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S} {msg}", flush=True)


def preflight() -> tuple[list[str], list[int]]:
    failures = []
    if not USERNAME:
        failures.append("WRDS_USERNAME not set")
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=REPO,
        text=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    ).stdout.strip()
    if branch != BRANCH:
        failures.append(f"branch is {branch!r}, expected {BRANCH}")
    try:
        import psycopg2  # noqa: F401
        import pyarrow  # noqa: F401
        import wrds  # noqa: F401
    except ImportError as err:
        failures.append(f"import: {err}")
    gvkeys, permnos = [], []
    try:
        gvkeys = sorted(
            pd.read_parquet(SOURCE / "hy_bond_days.parquet", columns=["gvkey"])["gvkey"]
            .dropna()
            .astype(str)
            .unique()
        )
        permnos = sorted(
            int(p)
            for p in pd.read_parquet(SOURCE / "crsp_names.parquet")["permno"].dropna().unique()
        )
    except Exception as err:  # noqa: BLE001
        failures.append(f"reading local panel: {err}")
    if not gvkeys:
        failures.append("no GVKEYs in the local panel")
    try:
        OUT.mkdir(parents=True, exist_ok=True)
        probe = OUT / ".write_probe"
        probe.write_text("ok")
        probe.unlink()
    except OSError as err:
        failures.append(f"output dir not writable: {err}")
    if failures:
        for f in failures:
            log(f"PREFLIGHT FAIL: {f}")
        log("No password requested.")
        raise SystemExit(EXIT_PREFLIGHT)
    log(f"preflight ok: {len(gvkeys)} GVKEYs, {len(permnos)} PERMNOs; output {OUT}")
    return gvkeys, permnos


def connect():
    import psycopg2
    from wrds.sql import WRDS_CONNECT_ARGS, WRDS_POSTGRES_DB, WRDS_POSTGRES_HOST, WRDS_POSTGRES_PORT

    password = getpass.getpass(f"WRDS password for {USERNAME} (input hidden): ")
    log("connecting (approve the Duo push)...")
    try:
        conn = psycopg2.connect(
            host=WRDS_POSTGRES_HOST,
            port=WRDS_POSTGRES_PORT,
            dbname=WRDS_POSTGRES_DB,
            user=USERNAME,
            password=password,
            sslmode=WRDS_CONNECT_ARGS.get("sslmode", "require"),
            application_name="macro-hy24-sector",
            connect_timeout=120,
            keepalives=1,
            keepalives_idle=60,
            keepalives_interval=15,
            keepalives_count=8,
        )
    except Exception as err:  # noqa: BLE001
        msg = str(err).splitlines()[0][:300].replace(password, "***") if password else ""
        log(f"login failed: {type(err).__name__}: {msg}")
        raise SystemExit(EXIT_AUTH) from None
    finally:
        del password
    psycopg2.connect = _forbidden  # no second connection from this process
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = '15min'")
    log("connected (1 connection)")
    return conn


def _forbidden(*_a, **_k):
    raise RuntimeError("second database connection attempted")


def fetch(conn, sql: str, params: dict) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def main() -> int:
    gvkeys, permnos = preflight()
    conn = connect()
    manifest = {
        "pulled_at": f"{datetime.now():%Y-%m-%dT%H:%M:%S}",
        "n_gvkeys": len(gvkeys),
        "n_permnos": len(permnos),
        "tables": {},
    }
    params = {"gvkeys": gvkeys, "permnos": permnos, "holdout": HOLDOUT, "cap": CAP}
    try:
        meta = fetch(conn, META_SQL, {"cols": META_COLS})
        manifest["columns"] = {
            f"{s}.{t}": list(g["column_name"])
            for (s, t), g in meta.groupby(["table_schema", "table_name"])
        }
        for name, table in TABLE_NAMES.items():
            try:
                manifest["tables"][name] = {
                    "select_privilege": bool(fetch(conn, PRIV_SQL, {"t": table}).iloc[0, 0])
                }
            except Exception as err:  # noqa: BLE001
                manifest["tables"][name] = {
                    "select_privilege": None,
                    "error": str(err).splitlines()[0][:200],
                }
        for name, (sql, date_cols) in QUERIES.items():
            entry = manifest["tables"][name]
            try:
                frame = fetch(conn, sql, params)
                for c in date_cols:
                    if (pd.to_datetime(frame[c]) >= pd.Timestamp(HOLDOUT)).any():
                        raise RuntimeError(f"{name}.{c} returned a holdout date; not saved")
                tmp = OUT / f"{name}.parquet.tmp"
                frame.to_parquet(tmp)
                os.replace(tmp, OUT / f"{name}.parquet")
                entry.update(rows=len(frame), columns=list(frame.columns))
                log(f"{name}: {len(frame):,} rows")
            except Exception as err:  # noqa: BLE001
                entry["error"] = f"{type(err).__name__}: {str(err).splitlines()[0][:200]}"
                log(f"{name}: FAILED ({entry['error']})")
    finally:
        conn.close()
        log("connection closed")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    ok = any("rows" in v for v in manifest["tables"].values())
    log("SECTOR PULL COMPLETE" if ok else "SECTOR PULL: no table succeeded")
    return 0 if ok else 5


if __name__ == "__main__":
    sys.exit(main())
