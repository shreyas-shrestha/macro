"""Local reliability tests for run_hy_once.py — no WRDS, fake connection, fake getpass.

    cd <repo> && uv run --no-project --python 3.12 --with wrds --with pyarrow --with sqlglot \\
        python -I scripts/hy_credit/test_run_hy_once.py

Not collected by pytest (testpaths = tests): the full-path case itself runs pytest.

The fake connection serves REAL rows from the 2010 sample in data/cache (licensed rows stay
under data/cache/_rehearsal_*, deleted afterwards; data/cache/hy_credit is never touched).
"""

import contextlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.errors

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FAKE_PASSWORD = "Fake-P@ss w0rd!"  # never a real credential
REAL_PSYCOPG2_CONNECT = psycopg2.connect
SAMPLE = pd.read_csv(
    REPO / "data/cache/trace_enhanced_sample.csv", dtype=str, keep_default_na=False
)
SAMPLE = SAMPLE.apply(lambda s: s.str.strip().mask(s.str.strip().eq("")))
MASTER = pd.read_csv(REPO / "data/cache/trace_master_sample.csv", dtype=str, keep_default_na=False)
SECURITY = pd.read_csv(
    REPO / "data/cache/comp_security_sample.csv", dtype=str, keep_default_na=False
)
BONDCRSP = pd.read_csv(
    REPO / "data/cache/bondcrsp_link_sample.csv", dtype=str, keep_default_na=False
)
RATINGS = pd.read_csv(
    REPO / "data/cache/compustat_ratings_sample.csv", dtype=str, keep_default_na=False
)


def fresh_module():
    psycopg2.connect = REAL_PSYCOPG2_CONNECT
    spec = importlib.util.spec_from_file_location("run_hy_once_under_test", HERE / "run_hy_once.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.USERNAME = "test_user"  # never a real account; the connection is faked
    return module


class FakeCursor:
    def __init__(self, conn, name=None, withhold=False):
        self.conn, self.name, self.rows, self.description, self.itersize = conn, name, [], None, 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _result(self, frame):
        frame = frame.astype(object).where(frame.notna(), None)
        self.description = [(c,) for c in frame.columns]
        self.rows = [tuple(r) for r in frame.itertuples(index=False)]

    def execute(self, sql, params=None):
        conn = self.conn
        if conn.closed:
            raise psycopg2.InterfaceError("connection already closed")
        conn.statements.append(sql.split()[0] if sql.strip() else "")
        if "trace_enhanced.trace_enhanced" in sql:
            conn.month_calls += 1
            if conn.dead_on_month_call == conn.month_calls:
                conn.closed = 2
                raise psycopg2.OperationalError("server closed the connection unexpectedly")
            if conn.fail_once_on_month_call == conn.month_calls:
                raise psycopg2.errors.QueryCanceled("canceling statement due to statement timeout")
            m0, m1 = pd.Timestamp(params["m0"]), pd.Timestamp(params["m1"])
            ex = pd.to_datetime(SAMPLE["trd_exctn_dt"])
            self._result(SAMPLE.loc[ex.between(m0, m1), conn.columns])
        elif sql.strip().lower().startswith(("set ", "select 1")):
            self._result(pd.DataFrame({"x": [1]}))
        elif "stocknames" in sql:
            raise psycopg2.errors.InsufficientPrivilege("permission denied for schema crsp_a_stock")
        elif "camasterfile" in sql:
            self._result(MASTER.loc[MASTER["cusip_id"].isin(params["keys"])].replace("", None))
        elif "bondcrsp_link" in sql:
            self._result(BONDCRSP.loc[BONDCRSP["cusip"].isin(params["keys"])].replace("", None))
        elif "comp.security" in sql:
            sec = SECURITY[["gvkey", "iid", "cusip", "tic", "tpci", "excntry", "dldtei"]]
            self._result(sec.loc[sec["cusip"].str[:6].isin(params["keys"])].replace("", None))
        elif "comp.adsprate" in sql:
            self._result(RATINGS.loc[RATINGS["gvkey"].isin(params["keys"])].replace("", None))
        else:
            raise AssertionError(f"unexpected SQL: {sql[:80]}")

    def fetchmany(self, n):
        out, self.rows = self.rows[:n], self.rows[n:]
        return out

    def fetchall(self):
        out, self.rows = self.rows, []
        return out

    def fetchone(self):
        return self.rows.pop(0) if self.rows else None

    def close(self):
        pass


class FakeConn:
    def __init__(self, columns, dead_on_month_call=None, fail_once_on_month_call=None):
        self.columns, self.closed, self.autocommit = columns, 0, False
        self.dead_on_month_call, self.fail_once_on_month_call = (
            dead_on_month_call,
            fail_once_on_month_call,
        )
        self.month_calls, self.statements = 0, []

    def cursor(self, name=None, withhold=False):
        return FakeCursor(self, name, withhold)

    def close(self):
        self.closed = 1


def install_fakes(module, **conn_kwargs):
    state = {"getpass": 0, "connect": 0, "conn": None, "kwargs": None}

    def fake_getpass(prompt=""):
        state["getpass"] += 1
        return FAKE_PASSWORD

    def fake_connect(**kwargs):
        state["connect"] += 1
        state["kwargs"] = {k: ("<redacted>" if k == "password" else v) for k, v in kwargs.items()}
        assert kwargs["password"] == FAKE_PASSWORD
        state["conn"] = FakeConn(module.RAW_COLUMNS, **conn_kwargs)
        return state["conn"]

    module.getpass.getpass = fake_getpass
    module._driver_connect = fake_connect
    return state


def make_cfg(module, rehearsal: Path, results: Path, logs: Path, **kw):
    return module.Config(
        cache_root=rehearsal,
        results=results,
        log_dir=logs,
        first_month="2009-12",
        last_month="2010-01",
        **kw,
    )


def run(module, cfg):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = module.main(cfg)
    return code, out.getvalue(), err.getvalue()


def assert_no_secret(*texts):
    for t in texts:
        assert FAKE_PASSWORD not in t, "password leaked"
        from urllib.parse import quote

        assert quote(FAKE_PASSWORD) not in t, "url-encoded password leaked"


def setup_dirs(tag):
    rehearsal = REPO / "data" / "cache" / f"_rehearsal_{tag}"
    shutil.rmtree(rehearsal, ignore_errors=True)
    rehearsal.mkdir(parents=True)
    (rehearsal / "fred").symlink_to((REPO / "data/cache/fred").resolve())
    results = Path(tempfile.mkdtemp(prefix=f"hy_results_{tag}_"))
    logs = Path(tempfile.mkdtemp(prefix=f"hy_logs_{tag}_"))
    return rehearsal, results, logs


def all_text(logs: Path, cfg) -> str:
    text = "".join(p.read_text() for p in logs.glob("*.log"))
    if cfg.manifest.exists():
        text += cfg.manifest.read_text()
    return text


# --------------------------------------------------------------------------------------


def test_local_failure_happens_before_authentication():
    m = fresh_module()
    state = install_fakes(m)
    m.SAMPLE_FILES = m.SAMPLE_FILES + ["data/cache/this_file_does_not_exist.csv"]
    rehearsal, results, logs = setup_dirs("prefail")
    try:
        cfg = make_cfg(m, rehearsal, results, logs, run_child_checks=False, run_post_pull=False)
        code, out, err = run(m, cfg)
        assert code == m.EXIT_PREFLIGHT, code
        assert state["getpass"] == 0 and state["connect"] == 0
        assert "No password requested" in out
    finally:
        shutil.rmtree(rehearsal, ignore_errors=True)
    print("ok  local failure exits before any password prompt or connection")


def test_dead_connection_never_reconnects():
    m = fresh_module()
    state = install_fakes(m, dead_on_month_call=2)
    rehearsal, results, logs = setup_dirs("dead")
    try:
        cfg = make_cfg(m, rehearsal, results, logs, run_child_checks=False, run_post_pull=False)
        code, out, err = run(m, cfg)
        assert code == m.EXIT_DEAD, (code, out[-2000:])
        assert state["getpass"] == 1 and state["connect"] == 1 and m.CONNECT_CALLS == 1
        manifest = json.loads(cfg.manifest.read_text())
        assert manifest["status"] == "dead_connection"
        assert manifest["failure"]["last_completed_month"] == "2009-12"
        assert "FIRST DEVELOPMENT MONTH WRITTEN SUCCESSFULLY: 2009-12" in out
        assert "Not reconnecting" in out
        # Constructors are sealed: any further attempt raises instead of connecting.
        for fn in (m._driver_connect, psycopg2.connect):
            try:
                fn(host="x")
                raise AssertionError("constructor not sealed")
            except RuntimeError as e:
                assert "exactly one" in str(e)
        assert_no_secret(out, err, all_text(logs, cfg))
        # Rerun resumes from the verified month (here: fresh fakes, a new process would be new).
        m2 = fresh_module()
        state2 = install_fakes(m2)
        code2, out2, err2 = run(m2, cfg)
        assert code2 == m2.EXIT_OK, out2[-2000:]
        assert state2["getpass"] == 1 and state2["connect"] == 1
        assert state2["conn"].month_calls == 1, "completed month was pulled again"
    finally:
        shutil.rmtree(rehearsal, ignore_errors=True)
    print("ok  dead connection: exit 4, manifest records last month, no reconnect; rerun resumes")


def test_statement_error_retries_on_same_connection():
    m = fresh_module()
    state = install_fakes(m, fail_once_on_month_call=1)
    rehearsal, results, logs = setup_dirs("retry")
    try:
        cfg = make_cfg(m, rehearsal, results, logs, run_child_checks=False, run_post_pull=False)
        code, out, err = run(m, cfg)
        assert code == m.EXIT_OK, out[-2000:]
        assert state["connect"] == 1 and state["getpass"] == 1
        assert "connection alive; retry 1" in out
        assert state["conn"].month_calls == 3  # 2 months + 1 retry
        manifest = json.loads(cfg.manifest.read_text())
        assert manifest["crsp_names_table"] is None  # permission error kept the connection alive
        assert_no_secret(out, err, all_text(logs, cfg))
        # Everything complete -> a rerun needs no login at all.
        m2 = fresh_module()
        state2 = install_fakes(m2)
        code2, out2, _ = run(m2, cfg)
        assert code2 == m2.EXIT_OK and state2["getpass"] == 0 and state2["connect"] == 0
    finally:
        shutil.rmtree(rehearsal, ignore_errors=True)
    print(
        "ok  statement failure retried on the same live connection; complete rerun needs no login"
    )


def test_full_path_with_real_preflight_and_local_build():
    m = fresh_module()
    state = install_fakes(m)
    rehearsal, results, logs = setup_dirs("full")
    try:
        cfg = make_cfg(m, rehearsal, results, logs)
        code, out, err = run(m, cfg)
        assert code == m.EXIT_OK, out[-4000:]
        assert state["getpass"] == 1 and state["connect"] == 1 and m.CONNECT_CALLS == 1
        kwargs = state["kwargs"]
        assert kwargs["keepalives"] == 1 and kwargs["sslmode"] == "require"
        for name in (
            "hy_daily.parquet",
            "hy_bond_days.parquet",
            "master.parquet",
            "security.parquet",
            "ratings.parquet",
            "bondcrsp.parquet",
        ):
            assert (cfg.out / name).exists(), name
        for name in (
            "attrition.csv",
            "dev_report.json",
            "qa_report.json",
            "hy_ig_series.png",
            "issuer_counts.png",
            "features.png",
            "monthly_summary.csv",
        ):
            assert (results / name).exists(), name
        qa = json.loads((results / "qa_report.json").read_text())
        assert qa["failures"] == [], qa["failures"]
        assert "PREFLIGHT PASSED" in out and "ALL PHASES COMPLETE" in out
        assert "WRDS connection closed" in out
        assert_no_secret(out, err, all_text(logs, cfg))
        print("    summary:", json.dumps(qa["summary"]))
    finally:
        shutil.rmtree(rehearsal, ignore_errors=True)
        shutil.rmtree(results, ignore_errors=True)
    print(
        "ok  full path: real preflight -> fake months -> manifest -> "
        "real local build/report/plots/QA"
    )


if __name__ == "__main__":
    test_local_failure_happens_before_authentication()
    test_dead_connection_never_reconnects()
    test_statement_error_retries_on_same_connection()
    test_full_path_with_real_preflight_and_local_build()
    print("ALL ORCHESTRATOR RELIABILITY TESTS PASSED")
