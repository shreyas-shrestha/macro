"""Point-in-time sector assignment tests (#24 residual extension)."""

import pandas as pd
import pytest

from macro.data.sector import asof_interval, gics_sector_asof


def _hgic():
    return pd.DataFrame(
        {
            "gvkey": ["001", "001", "002"],
            "indfrom": pd.to_datetime(["2005-01-01", "2010-06-01", "2008-01-01"]),
            "indthru": pd.to_datetime(["2010-05-31", None, "2012-12-31"]),
            "gsector": ["40", "60", "10"],
        }
    )


def _obs(rows):
    return pd.DataFrame(rows, columns=["gvkey", "trd_exctn_dt"]).assign(
        trd_exctn_dt=lambda f: pd.to_datetime(f["trd_exctn_dt"])
    )


def test_later_reclassification_does_not_reach_back():
    obs = _obs([("001", "2010-05-31"), ("001", "2010-06-01"), ("001", "2016-01-04")])
    assert gics_sector_asof(obs, _hgic()).tolist() == ["40", "60", "60"]


def test_dates_before_first_spell_or_after_last_stay_missing():
    obs = _obs(
        [("002", "2007-12-31"), ("002", "2013-01-02"), ("003", "2010-01-04"), (None, "2010-01-04")]
    )
    assert gics_sector_asof(obs, _hgic()).isna().all()


def test_adding_future_spells_does_not_change_past_assignments():
    obs = _obs([(g, d) for g in ("001", "002") for d in ("2006-03-01", "2009-03-02", "2011-03-01")])
    known = _hgic()
    full = pd.concat(
        [
            known.assign(indthru=known["indthru"].fillna(pd.Timestamp("2013-12-31"))),
            pd.DataFrame(
                {
                    "gvkey": ["001", "002"],
                    "indfrom": pd.to_datetime(["2014-01-01", "2013-01-01"]),
                    "indthru": [pd.NaT, pd.NaT],
                    "gsector": ["55", "15"],
                }
            ),
        ]
    )
    early = obs["trd_exctn_dt"] < "2013-01-01"
    pd.testing.assert_series_equal(
        gics_sector_asof(obs, full).loc[early], gics_sector_asof(obs, known).loc[early]
    )


def test_overlapping_spells_raise():
    bad = pd.concat(
        [
            _hgic(),
            pd.DataFrame(
                {
                    "gvkey": ["002"],
                    "indfrom": pd.to_datetime(["2012-01-01"]),
                    "indthru": [pd.NaT],
                    "gsector": ["15"],
                }
            ),
        ]
    )
    with pytest.raises(ValueError, match="overlap"):
        gics_sector_asof(_obs([("002", "2010-01-04")]), bad)


def test_rows_and_index_are_preserved():
    obs = _obs([("002", "2009-01-02"), ("001", "2006-01-03")])
    obs.index = [9, 4]
    out = asof_interval(
        obs,
        _hgic(),
        key="gvkey",
        date_col="trd_exctn_dt",
        start_col="indfrom",
        end_col="indthru",
        value_cols=["gsector"],
    )
    assert out.index.tolist() == [9, 4]
    assert out["gsector"].tolist() == ["10", "40"]
    assert out["spell_start"].iloc[0] == pd.Timestamp("2008-01-01")
