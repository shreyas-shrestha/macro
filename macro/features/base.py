"""Base class for signals.

Interface version 0. Provisional until #8 and #10 merge; changes go through a PR that
updates every existing subclass in the same change. Signal modules depend only on
``build``'s signature and the four class attributes below, never on other base-class
internals.

Timing convention
-----------------
- Decision date t is an NYSE trading day. The row indexed t may use only information
  published by 16:00 ET (market close) on t.
- A series observed on day s with release lag L trading days enters the row at s + L,
  never earlier.
- The forecast made at t targets realized variance over trading days t+1 ... t+h. Day
  t's return is never part of the target.

Examples: BAA10Y for day s is published by FRED the next morning, so L = 1 and it is
first usable at decision date s + 1. The VIX close on t has L = 0 and is usable at t.

Writing a signal
----------------
Subclass ``Signal``, set ``name``, ``source``, ``start_date`` and ``release_lag``, and
implement ``build(end_date)``. The harness calls ``run``, which calls ``build`` and then
``validate``. Each signal's tests call ``macro.features.testing.assert_no_lookahead``.

``release_lag`` is metadata: the worst-case lag in trading days across the signal's
inputs, used for documentation. The base class does not shift data by it; per-series
lags and vintages are applied by the loaders in ``macro.data``.
"""

from abc import ABC, abstractmethod

import pandas as pd

from macro.data.calendar import nyse_sessions

INDEX_NAME = "decision_date"

_REQUIRED_ATTRIBUTES = {
    "name": str,
    "source": str,
    "start_date": pd.Timestamp,
    "release_lag": int,
}


class SignalError(ValueError):
    """A signal's attributes or output break the interface contract."""


class Signal(ABC):
    """A feature set indexed by decision date.

    Class attributes:
        name: short identifier, e.g. ``"credit"``.
        source: where the inputs come from, e.g. ``"FRED"``.
        start_date: first decision date the signal can produce.
        release_lag: worst-case input lag in trading days (metadata only).
    """

    name: str
    source: str
    start_date: pd.Timestamp
    release_lag: int

    @abstractmethod
    def build(self, end_date: pd.Timestamp) -> pd.DataFrame:
        """Return features for decision dates from ``start_date`` through ``end_date``.

        Must use only information available at each row's decision date (see module
        docstring). Index: sorted, unique, timezone-naive NYSE session dates in a
        ``DatetimeIndex`` named ``decision_date`` (midnight date labels, not timestamps).
        Columns: numeric.
        """

    def run(self, end_date) -> pd.DataFrame:
        """Check attributes, call ``build(end_date)``, validate and return its output."""
        end_date = pd.Timestamp(end_date)
        self.check_attributes()
        frame = self.build(end_date)
        self.validate(frame, end_date)
        return frame

    def check_attributes(self) -> None:
        """Raise ``SignalError`` if a required class attribute is missing or mistyped."""
        cls = type(self).__name__
        for attr, kind in _REQUIRED_ATTRIBUTES.items():
            if not hasattr(self, attr):
                raise SignalError(f"{cls} is missing class attribute {attr!r}")
            value = getattr(self, attr)
            if not isinstance(value, kind) or isinstance(value, bool):
                raise SignalError(f"{cls}.{attr} must be {kind.__name__}, got {value!r}")
        if not self.name:
            raise SignalError(f"{cls}.name must be non-empty")
        if self.release_lag < 0:
            raise SignalError(f"{cls}.release_lag must be >= 0, got {self.release_lag}")

    def validate(self, frame: pd.DataFrame, end_date: pd.Timestamp) -> None:
        """Raise ``SignalError`` if ``frame`` breaks the output contract.

        Checks: DataFrame with a ``DatetimeIndex`` named ``decision_date``; index sorted
        and unique; timezone-naive NYSE session dates; no rows after ``end_date`` or
        before ``start_date``; numeric columns.
        Does not detect lookahead inside values; use ``assert_no_lookahead`` for that.
        """
        cls = type(self).__name__
        if not isinstance(frame, pd.DataFrame):
            raise SignalError(f"{cls}.build must return a DataFrame, got {type(frame)}")
        index = frame.index
        if not isinstance(index, pd.DatetimeIndex):
            raise SignalError(f"{cls} index must be a DatetimeIndex, got {type(index)}")
        if index.name != INDEX_NAME:
            raise SignalError(f"{cls} index must be named {INDEX_NAME!r}, got {index.name!r}")
        if index.tz is not None:
            raise SignalError(f"{cls} index must use timezone-naive decision dates")
        if not index.is_unique:
            raise SignalError(f"{cls} index has duplicate decision dates")
        if not index.is_monotonic_increasing:
            raise SignalError(f"{cls} index is not sorted")
        if len(index) and index[-1] > end_date:
            raise SignalError(f"{cls} has rows after end_date {end_date.date()}")
        if len(index) and index[0] < self.start_date:
            raise SignalError(f"{cls} has rows before start_date {self.start_date.date()}")
        if len(index):
            sessions = nyse_sessions(index[0].normalize(), index[-1].normalize())
            bad_dates = index[~index.isin(sessions)]
            if len(bad_dates):
                raise SignalError(f"{cls} has non-NYSE session dates: {bad_dates.tolist()}")
        bad = [c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])]
        if bad:
            raise SignalError(f"{cls} has non-numeric columns: {bad}")
