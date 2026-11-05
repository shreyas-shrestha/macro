import importlib
import os

import pytest

MODULES = [
    "macro",
    "macro.config",
    "macro.data",
    "macro.data.calendar",
    "macro.features",
    "macro.features.base",
    "macro.features.testing",
    "macro.models",
    "macro.backtest",
    "macro.strategy",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports_with_docstring(name):
    module = importlib.import_module(name)
    assert module.__doc__ and module.__doc__.strip()


def test_periods_do_not_overlap():
    from macro import config

    assert config.DEV_START < config.DEV_END < config.HOLDOUT_START


def test_load_env_reads_working_directory(tmp_path, monkeypatch):
    from macro.config import load_env

    (tmp_path / ".env").write_text("MACRO_TEST_VAR=from_cwd\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("MACRO_TEST_VAR", raising=False)
    assert load_env()
    assert os.environ["MACRO_TEST_VAR"] == "from_cwd"
    monkeypatch.delenv("MACRO_TEST_VAR")
