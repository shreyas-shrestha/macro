"""Treasury curve interpolation and timing (synthetic data, no network)."""

import numpy as np
import pandas as pd
import pytest

from macro.data.treasury import (
    CMT_SERIES,
    fetch_cmt_curve,
    interpolate_curve,
    load_cmt_curve,
    treasury_spread_bp,
)


def _curve():
    return pd.DataFrame(
        {1.0: [1.0, 2.0], 2.0: [2.0, np.nan], 10.0: [4.0, 5.0], 30.0: [5.0, np.nan]},
        index=pd.to_datetime(["2010-01-04", "2010-01-05"]),
    )


def test_linear_interpolation_between_nodes():
    out = interpolate_curve(
        _curve(), pd.Series(pd.to_datetime(["2010-01-04"] * 3)), [1.5, 6.0, 20.0]
    )
    assert out.tolist() == pytest.approx([1.5, 3.0, 4.5])


def test_flat_extrapolation_beyond_available_nodes():
    dates = pd.Series(pd.to_datetime(["2010-01-04", "2010-01-04", "2010-01-05"]))
    out = interpolate_curve(_curve(), dates, [0.25, 40.0, 25.0])
    # 2010-01-05 has no 30-year node, so 25 years is flat at the 10-year value.
    assert out.tolist() == pytest.approx([1.0, 5.0, 5.0])


def test_uses_only_the_same_dates_curve():
    # 2010-01-05 is missing the 2-year node; the 2010-01-04 value must not be borrowed.
    out = interpolate_curve(_curve(), pd.Series(pd.to_datetime(["2010-01-05"])), [2.0])
    assert out.iloc[0] == pytest.approx(2.0 + (5.0 - 2.0) * (1.0 / 9.0))


def test_missing_date_or_maturity_gives_missing():
    dates = pd.Series(pd.to_datetime(["2010-01-06", "2010-01-04"]))
    out = interpolate_curve(_curve(), dates, [5.0, np.nan])
    assert out.isna().all()


def test_too_few_nodes_gives_missing():
    curve = pd.DataFrame({1.0: [np.nan], 10.0: [3.0]}, index=pd.to_datetime(["2010-01-04"]))
    out = interpolate_curve(curve, pd.Series(pd.to_datetime(["2010-01-04"])), [5.0])
    assert out.isna().all()


def test_spread_in_basis_points_and_index_preserved():
    frame = pd.DataFrame(
        {
            "trd_exctn_dt": pd.to_datetime(["2010-01-04", "2010-01-04"]),
            "remaining_maturity_y": [6.0, 1.0],
            "yield_pct": [8.0, 1.25],
        },
        index=[7, 3],
    )
    out = treasury_spread_bp(frame, _curve())
    assert out.index.tolist() == [7, 3]
    assert out["spread_bp"].tolist() == pytest.approx([500.0, 25.0])


def _write_fred(directory, rows_by_series):
    directory.mkdir(parents=True, exist_ok=True)
    for series in CMT_SERIES:
        rows = rows_by_series.get(series, [("2010-01-04", "1.0")])
        text = "observation_date," + series + "\n" + "\n".join(f"{d},{v}" for d, v in rows)
        (directory / f"{series}.csv").write_text(text + "\n")


def test_load_curve_treats_blank_as_missing_and_masks_dgs30_gap(tmp_path):
    _write_fred(
        tmp_path,
        {
            "DGS10": [("2006-02-08", "4.5"), ("2006-02-09", "")],
            "DGS30": [("2006-02-08", "4.6"), ("2006-02-09", "4.51")],
        },
    )
    curve = load_cmt_curve(tmp_path)
    assert np.isnan(curve.loc["2006-02-08", 30.0])  # inside the discontinuation gap
    assert curve.loc["2006-02-09", 30.0] == 4.51
    assert np.isnan(curve.loc["2006-02-09", 10.0])


def test_load_curve_rejects_holdout_dates(tmp_path):
    _write_fred(tmp_path, {"DGS10": [("2018-01-02", "2.4")]})
    with pytest.raises(ValueError, match="holdout"):
        load_cmt_curve(tmp_path)


def test_fetch_refuses_holdout_end(tmp_path):
    with pytest.raises(ValueError, match="holdout"):
        fetch_cmt_curve(tmp_path, end="2018-01-02")
