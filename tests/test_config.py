import os

import pytest

from macro.config import load_env


@pytest.mark.parametrize("use_parent", [False, True])
def test_load_env_searches_working_directory_and_parents(tmp_path, monkeypatch, use_parent):
    monkeypatch.setattr(os, "environ", os.environ.copy())
    monkeypatch.delenv("MACRO_ENV_TEST_VALUE", raising=False)
    (tmp_path / ".env").write_text("MACRO_ENV_TEST_VALUE=from_file\n")
    workdir = tmp_path / "analysis" if use_parent else tmp_path
    workdir.mkdir(exist_ok=True)
    monkeypatch.chdir(workdir)

    assert load_env()
    assert os.environ["MACRO_ENV_TEST_VALUE"] == "from_file"


def test_load_env_preserves_existing_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("MACRO_ENV_TEST_VALUE", "from_environment")
    (tmp_path / ".env").write_text("MACRO_ENV_TEST_VALUE=from_file\n")
    monkeypatch.chdir(tmp_path)

    assert load_env()
    assert os.environ["MACRO_ENV_TEST_VALUE"] == "from_environment"


def test_load_env_returns_false_without_a_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not load_env()
