"""Fake-connection tests for wrds_pull_sector.py (no WRDS, no password, temp output dir).

    cd <repo> && uv run --no-project --python 3.12 --with wrds --with pyarrow \\
        python -I scripts/hy_credit/test_wrds_pull_sector.py
"""

import importlib.util
import json
import tempfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent


def load():
    spec = importlib.util.spec_from_file_location("pull_sector", HERE / "wrds_pull_sector.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.USERNAME = "test_user"
    return module


class FakeCursor:
    def __init__(self, conn):
        self.conn, self.description, self.rows = conn, None, []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.statements.append(sql)
        frame = self.conn.respond(sql)
        if isinstance(frame, Exception):
            raise frame
        self.description = [(c,) for c in frame.columns]
        self.rows = [tuple(r) for r in frame.itertuples(index=False)]

    def fetchall(self):
        return self.rows


class FakeConn:
    def __init__(self, hgic_dates=("2005-01-01",)):
        self.statements, self.closed, self.autocommit = [], 0, False
        self.hgic_dates = hgic_dates

    def cursor(self):
        return FakeCursor(self)

    def close(self):
        self.closed = 1

    def respond(self, sql):
        if "information_schema" in sql:
            return pd.DataFrame(
                {
                    "table_schema": ["comp"],
                    "table_name": ["co_hgic"],
                    "column_name": ["gvkey"],
                    "data_type": ["text"],
                }
            )
        if "has_table_privilege" in sql:
            return pd.DataFrame({"x": [True]})
        if "co_hgic" in sql:
            assert "indfrom < date" in sql and "least(" in sql
            return pd.DataFrame(
                {
                    "gvkey": ["001004"] * len(self.hgic_dates),
                    "indfrom": pd.to_datetime(list(self.hgic_dates)).date,
                    "indthru": pd.to_datetime(["2017-12-31"] * len(self.hgic_dates)).date,
                    "open_ended": [True] * len(self.hgic_dates),
                    "gsector": ["20"] * len(self.hgic_dates),
                    "ggroup": ["2010"] * len(self.hgic_dates),
                    "gind": ["201010"] * len(self.hgic_dates),
                    "gsubind": ["20101010"] * len(self.hgic_dates),
                }
            )
        if "comp.funda" in sql:
            return RuntimeError("permission denied for schema comp")
        if "stocknames" in sql:
            return pd.DataFrame(
                {
                    "permno": [10001],
                    "namedt": [pd.Timestamp("2004-01-01").date()],
                    "nameenddt": [pd.Timestamp("2017-12-31").date()],
                    "siccd": [4920],
                }
            )
        raise AssertionError(sql)


def run(conn):
    m = load()
    out = Path(tempfile.mkdtemp(prefix="sector_test_"))
    m.OUT = out
    m.connect = lambda: conn
    code = m.main()
    return m, out, code


def test_pull_writes_tables_and_survives_one_permission_error():
    conn = FakeConn()
    _, out, code = run(conn)
    assert code == 0 and conn.closed == 1
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["tables"]["co_hgic"]["rows"] == 1
    assert "error" in manifest["tables"]["funda_sic"]
    assert manifest["tables"]["crsp_siccd"]["rows"] == 1
    assert (out / "co_hgic.parquet").exists() and not (out / "funda_sic.parquet").exists()
    print("ok  tables written; a permission error on one table is recorded, others continue")


def test_holdout_dates_are_never_saved():
    conn = FakeConn(hgic_dates=("2005-01-01", "2018-03-01"))
    _, out, _ = run(conn)
    manifest = json.loads((out / "manifest.json").read_text())
    assert "holdout" in manifest["tables"]["co_hgic"]["error"]
    assert not (out / "co_hgic.parquet").exists()
    print("ok  a returned 2018+ date blocks saving that table")


def test_no_username_exits_before_prompt():
    m = load()
    m.USERNAME = ""
    m.connect = lambda: (_ for _ in ()).throw(AssertionError("connect called"))
    try:
        m.main()
    except SystemExit as err:
        assert err.code == m.EXIT_PREFLIGHT
    print("ok  preflight failure exits before any password prompt")


if __name__ == "__main__":
    test_pull_writes_tables_and_survives_one_permission_error()
    test_holdout_dates_are_never_saved()
    test_no_username_exits_before_prompt()
    print("ALL SECTOR PULL TESTS PASSED")
