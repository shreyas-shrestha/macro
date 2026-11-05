"""Base class for signals.

Interface version 0, provisional until #8 and #10 merge. A change here updates every
existing subclass in the same PR.

Timing convention (the single source; CONTRIBUTING.md has the worked example):
decision date t is an NYSE trading day, and the row indexed t uses only information
published by the 16:00 ET close on t. A series observed on day s with release lag L
enters at s + L. The forecast made at t targets realized variance over t+1 ... t+h.
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
        release_lag: worst-case input lag in trading days. Documentation only; the
            loaders in ``macro.data`` apply the actual lags.
    """

    name: str
    source: str
    start_date: pd.Timestamp
    release_lag: int

    @abstractmethod
    def build(self, end_date: pd.Timestamp) -> pd.DataFrame:
        """Return features for decision dates from ``start_date`` through ``end_date``.

        Index: ``DatetimeIndex`` named ``decision_date`` of NYSE session dates (see
        ``macro.data.calendar.nyse_sessions``). Columns: numeric. ``validate`` lists the
        exact checks.
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
