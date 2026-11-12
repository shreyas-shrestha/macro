import numpy as np
import pandas as pd
import pytest

from macro.data.calendar import nyse_sessions
from macro.features.move import Move
from macro.features.testing import assert_no_lookahead


def test_move_uses_only_available_closes_and_prior_zscore_window():
    dates = nyse_sessions(Move.start_date, "2005-01-03")
    raw = pd.Series(np.arange(len(dates), dtype=float) + 50, index=dates)
    raw.iloc[300:302] = np.nan
    signal = Move(loader=lambda start, end: raw.loc[start:end])

    assert_no_lookahead(signal, [dates[260], dates[301], dates[302]], end_date=dates[-1])
    features = signal.run(dates[-1])

    assert np.isnan(features.iloc[0]["move_level"])
    assert features.iloc[1]["move_level"] == raw.iloc[0]
    assert features.iloc[21]["move_change_20"] == 20
    history = raw.iloc[:252]
    assert features["move_z_252"].iloc[:252].isna().all()
    assert features.iloc[253]["move_z_252"] == pytest.approx(
        (raw.iloc[252] - history.mean()) / history.std()
    )
    assert features.iloc[301]["move_level"] == raw.iloc[299]
    assert np.isnan(features.iloc[302]["move_level"])
    assert features.iloc[303]["move_level"] == raw.iloc[302]


def test_move_zscore_tolerates_at_most_seven_missing_levels():
    dates = nyse_sessions(Move.start_date, "2005-01-03")
    raw = pd.Series(np.arange(len(dates), dtype=float) + 50, index=dates)
    raw.iloc[100:108] = np.nan
    signal = Move(loader=lambda start, end: raw.loc[start:end])
    features = signal.run(dates[-1])
    assert pd.notna(features.iloc[253]["move_z_252"])

    raw.iloc[108] = np.nan
    features = signal.run(dates[-1])
    assert pd.isna(features.iloc[253]["move_z_252"])
