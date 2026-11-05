import numpy as np
import pandas as pd
import pytest

from macro.features.base import INDEX_NAME, Signal, SignalError
from macro.features.testing import assert_no_lookahead

START = pd.Timestamp("2000-01-03")
END = pd.Timestamp("2000-06-30")
TRADING_DAYS = pd.bdate_range(START, END)
# Raw series indexed by observation date s.
RAW = pd.Series(np.arange(len(TRADING_DAYS), dtype=float), index=TRADING_DAYS)


def frame(index, **cols):
    index = pd.DatetimeIndex(index, name=INDEX_NAME)
    return pd.DataFrame(cols or {"x": np.zeros(len(index))}, index=index)


class Toy(Signal):
    name = "toy"
    source = "test"
    start_date = START
    release_lag = 0

    def build(self, end_date):
        days = TRADING_DAYS[TRADING_DAYS <= end_date]
        return frame(days, x=np.arange(len(days), dtype=float))


class Lagged(Signal):
    """Value observed at s enters at s + 1 trading day."""

    name = "lagged"
    source = "test"
    start_date = TRADING_DAYS[1]
    release_lag = 1

    def build(self, end_date):
        shifted = RAW.shift(self.release_lag).iloc[self.release_lag :]
        shifted = shifted[shifted.index <= end_date]
        return frame(shifted.index, x=shifted.to_numpy())


class FullSampleZ(Toy):
    """Leaks: z-score uses the mean and std of the whole sample."""

    def build(self, end_date):
        out = super().build(end_date)
        out["x"] = (out["x"] - out["x"].mean()) / out["x"].std()
        return out


def test_toy_signal_runs():
    out = Toy().run(END)
    assert len(out) == len(TRADING_DAYS)
    assert out.index.name == INDEX_NAME


def test_cannot_instantiate_without_build():
    class NoBuild(Signal):
        name = "x"
        source = "x"
        start_date = START
        release_lag = 0

    with pytest.raises(TypeError):
        NoBuild()


@pytest.mark.parametrize(
    "attr, value",
    [
        ("name", None),
        ("name", ""),
        ("source", 1),
        ("start_date", "2000-01-03"),
        ("release_lag", 1.0),
        ("release_lag", True),
        ("release_lag", -1),
    ],
)
def test_bad_attributes_raise(attr, value):
    bad = type("Bad", (Toy,), {attr: value})
    with pytest.raises(SignalError):
        bad().run(END)


def test_missing_attribute_raises():
    class Missing(Signal):
        name = "missing"
        source = "test"
        start_date = START

        def build(self, end_date):
            return frame([START])

    with pytest.raises(SignalError, match="release_lag"):
        Missing().run(END)


def bad_output(output):
    return type("BadOutput", (Toy,), {"build": lambda self, end_date: output})()


@pytest.mark.parametrize(
    "output, match",
    [
        ([1, 2], "DataFrame"),
        (pd.DataFrame({"x": [1.0]}, index=[0]), "DatetimeIndex"),
        (pd.DataFrame({"x": [1.0]}, index=pd.DatetimeIndex([START], name="date")), "named"),
        (frame([START, START]), "duplicate"),
        (frame([TRADING_DAYS[1], TRADING_DAYS[0]]), "sorted"),
        (frame([START, END + pd.Timedelta(days=1)]), "after end_date"),
        (frame([START - pd.Timedelta(days=1), START]), "before start_date"),
        (frame([START], x=["a"]), "non-numeric"),
    ],
)
def test_bad_output_raises(output, match):
    with pytest.raises(SignalError, match=match):
        bad_output(output).run(END)


def test_lagged_value_appears_one_day_after_observation():
    out = Lagged().run(END)
    s = TRADING_DAYS[10]
    next_day = TRADING_DAYS[11]
    assert out.loc[next_day, "x"] == RAW[s]
    assert out.loc[s, "x"] != RAW[s]


def test_lagged_signal_has_no_lookahead():
    assert_no_lookahead(Lagged(), cutoffs=TRADING_DAYS[[5, 50, 100]], end_date=END)


def test_full_sample_zscore_is_caught():
    with pytest.raises(AssertionError, match="lookahead"):
        assert_no_lookahead(FullSampleZ(), cutoffs=[TRADING_DAYS[50]], end_date=END)


def test_cutoff_after_end_date_rejected():
    with pytest.raises(ValueError):
        assert_no_lookahead(Toy(), cutoffs=[END], end_date=END)
