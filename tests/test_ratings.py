"""Synthetic tests for point-in-time S&P rating handling."""

import numpy as np
import pandas as pd
import pytest

from macro.data.ratings import (
    asof_ratings,
    normalize_sp_rating,
    rating_class,
    unrecognized_ratings,
)


def test_ig_hy_boundary_and_missing():
    raw = pd.Series(["AAA", "bbb-", " BB+ ", "D", None, "NR", "SD"])
    assert list(rating_class(raw).astype(object)) == ["IG", "IG", "HY", "HY", pd.NA, pd.NA, pd.NA]


def test_unrated_is_never_hy():
    out = rating_class(pd.Series([None, "", "NR"], dtype="object"))
    assert out.isna().all()


def test_unrecognized_ratings_are_surfaced():
    counts = unrecognized_ratings(pd.Series(["NR", "SD", "NR", "BB", None, ""]))
    assert counts.to_dict() == {"NR": 2, "SD": 1}
    assert normalize_sp_rating(pd.Series(["SD"])).isna().all()


def _ratings():
    return pd.DataFrame(
        {
            "issuer": ["X", "X", "X", "Y"],
            "datadate": pd.to_datetime(["2009-11-30", "2009-12-31", "2010-01-31", "2009-12-31"]),
            "splticrm": ["BBB-", "BBB-", "BB+", "A"],
        }
    )


def _obs(dates, issuer="X"):
    return pd.DataFrame({"issuer": issuer, "date": pd.to_datetime(dates)})


def test_downgrade_does_not_affect_earlier_observations():
    obs = _obs(["2010-01-15", "2010-01-30", "2010-01-31", "2010-02-15"])
    out = asof_ratings(obs, _ratings(), issuer_key="issuer", date_col="date")
    assert out["rating"].tolist() == ["BBB-", "BBB-", "BB+", "BB+"]
    assert out["rating_class"].tolist() == ["IG", "IG", "HY", "HY"]
    assert out["rating_date"].iloc[0] == pd.Timestamp("2009-12-31")


def test_no_future_rating_is_backfilled():
    obs = _obs(["2009-11-01", "2009-11-30"])
    out = asof_ratings(obs, _ratings(), issuer_key="issuer", date_col="date")
    assert pd.isna(out["rating"].iloc[0])
    assert out["rating"].iloc[1] == "BBB-"


def test_ratings_do_not_leak_across_issuers():
    obs = pd.DataFrame({"issuer": ["Y", "Z", None], "date": pd.to_datetime(["2010-01-31"] * 3)})
    out = asof_ratings(obs, _ratings(), issuer_key="issuer", date_col="date")
    assert out["rating"].iloc[0] == "A"
    assert out["rating"].iloc[1:].isna().all()


def test_latest_missing_rating_means_unrated_now():
    ratings = pd.DataFrame(
        {
            "issuer": ["X", "X"],
            "datadate": pd.to_datetime(["2009-12-31", "2010-01-31"]),
            "splticrm": ["BB", None],
        }
    )
    out = asof_ratings(_obs(["2010-02-10"]), ratings, issuer_key="issuer", date_col="date")
    assert pd.isna(out["rating"].iloc[0])
    assert pd.isna(out["rating_class"].iloc[0])


def test_tolerance_expires_stale_ratings():
    out = asof_ratings(
        _obs(["2010-02-15", "2010-09-30"]),
        _ratings(),
        issuer_key="issuer",
        date_col="date",
        tolerance=pd.Timedelta(days=93),
    )
    assert out["rating"].iloc[0] == "BB+"
    assert pd.isna(out["rating"].iloc[1])


def test_duplicate_rating_records_raise():
    ratings = pd.concat([_ratings(), _ratings().iloc[[0]]])
    with pytest.raises(ValueError, match="share"):
        asof_ratings(_obs(["2010-01-15"]), ratings, issuer_key="issuer", date_col="date")


def test_rows_and_index_order_are_preserved():
    obs = _obs(["2010-02-15", "2009-12-15", "2010-01-15"])
    obs.index = [7, 3, 5]
    out = asof_ratings(obs, _ratings(), issuer_key="issuer", date_col="date")
    assert out.index.tolist() == [7, 3, 5]
    assert out["rating"].tolist() == ["BB+", "BBB-", "BBB-"]


def test_adding_later_rating_records_never_changes_earlier_rows():
    rng = np.random.default_rng(0)
    months = pd.date_range("2005-01-31", periods=48, freq="ME")
    grades = list(pd.Series(["AA", "A-", "BBB", "BBB-", "BB+", "B", "CCC"]))
    ratings = pd.DataFrame(
        {
            "issuer": np.repeat(["X", "Y"], len(months)),
            "datadate": np.tile(months, 2),
            "splticrm": rng.choice(grades, size=2 * len(months)),
        }
    )
    obs = pd.DataFrame(
        {
            "issuer": rng.choice(["X", "Y"], size=300),
            "date": pd.Timestamp("2005-01-01") + pd.to_timedelta(rng.integers(0, 1460, 300), "D"),
        }
    )
    full = asof_ratings(obs, ratings, issuer_key="issuer", date_col="date")
    for cutoff in pd.to_datetime(["2006-06-30", "2007-03-15", "2008-01-01"]):
        known = ratings.loc[ratings["datadate"] <= cutoff]
        partial = asof_ratings(obs, known, issuer_key="issuer", date_col="date")
        early = obs["date"] <= cutoff
        pd.testing.assert_series_equal(partial.loc[early, "rating"], full.loc[early, "rating"])


def test_asof_join_accepts_mixed_datetime_resolutions():
    obs = _obs(["2010-01-15"])
    obs["date"] = obs["date"].astype("datetime64[s]")  # as read back from Parquet date32
    ratings = _ratings()
    ratings["datadate"] = ratings["datadate"].astype("datetime64[us]")
    out = asof_ratings(obs, ratings, issuer_key="issuer", date_col="date")
    assert out["rating"].tolist() == ["BBB-"]
