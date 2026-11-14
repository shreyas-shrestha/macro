"""WRDS extract loaders: schema validation, typing and the holdout guard (synthetic CSVs)."""

import pandas as pd
import pytest

from macro.data.wrds import (
    ExtractError,
    load_compustat_ratings,
    load_trace_master,
    load_trace_trades,
)


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_trace_loader_keeps_identifiers_as_text_and_types_fields(tmp_path):
    path = _write(
        tmp_path,
        "trace.csv",
        "trd_exctn_dt,cusip_id,rptd_pr,yld_pt,trc_st,ascii_rptd_vol_tx\n"
        "2010-01-04,037833100,101.5,5.25,T,1MM+\n"
        "2010-01-05,00206rab8,99,,T,5000\n",
    )
    out = load_trace_trades(path)
    assert out["cusip_id"].tolist() == ["037833100", "00206RAB8"]
    assert out["trd_exctn_dt"].dtype.kind == "M"
    assert out["rptd_pr"].tolist() == [101.5, 99.0]
    assert pd.isna(out.loc[1, "yld_pt"])
    assert out["ascii_rptd_vol_tx"].tolist() == ["1MM+", "5000"]  # raw text, not coerced
    assert "msg_seq_nb" in out.attrs["missing_optional"]


def test_missing_required_columns_are_named(tmp_path):
    path = _write(tmp_path, "trace.csv", "trd_exctn_dt,cusip_id\n2010-01-04,037833100\n")
    with pytest.raises(ExtractError, match=r"\['rptd_pr', 'yld_pt'\]"):
        load_trace_trades(path)


def test_unparseable_values_raise_instead_of_becoming_missing(tmp_path):
    path = _write(
        tmp_path,
        "trace.csv",
        "trd_exctn_dt,cusip_id,rptd_pr,yld_pt\n2010-01-04,037833100,abc,5\n",
    )
    with pytest.raises(ExtractError, match="rptd_pr"):
        load_trace_trades(path)


def test_holdout_rows_are_rejected(tmp_path):
    path = _write(
        tmp_path,
        "trace.csv",
        "trd_exctn_dt,cusip_id,rptd_pr,yld_pt\n2018-01-02,037833100,100,5\n",
    )
    with pytest.raises(ExtractError, match="holdout"):
        load_trace_trades(path)


def test_ratings_loader_preserves_gvkey_zeros_and_rejects_holdout(tmp_path):
    path = _write(
        tmp_path,
        "ratings.csv",
        "gvkey,datadate,splticrm,sic\n001004,2009-12-31,BB+,0100\n",
    )
    out = load_compustat_ratings(path)
    assert out.loc[0, "gvkey"] == "001004"
    assert out.loc[0, "sic"] == "0100"
    late = _write(tmp_path, "late.csv", "gvkey,datadate,splticrm\n001004,2018-01-31,BB+\n")
    with pytest.raises(ExtractError, match="holdout"):
        load_compustat_ratings(late)


def test_master_loader_allows_future_maturities(tmp_path):
    path = _write(
        tmp_path,
        "master.csv",
        "cusip_id,mtrty_dt,cpn_rt,grade\n037833100,2030-05-15,4.5,H\n037833AB6,,,\n",
    )
    out = load_trace_master(path)
    assert out.loc[0, "mtrty_dt"] == pd.Timestamp("2030-05-15")
    assert pd.isna(out.loc[1, "mtrty_dt"])
    assert out.loc[0, "grade"] == "H"
