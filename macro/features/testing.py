"""Test helpers for signals."""

from collections.abc import Iterable

import pandas as pd

from macro.config import DEV_END
from macro.features.base import Signal


def assert_no_lookahead(signal: Signal, cutoffs: Iterable, end_date=DEV_END) -> None:
    """Assert that building at earlier cutoffs reproduces the full build's rows.

    Runs ``signal.run(end_date)`` and ``signal.run(cutoff)`` for each cutoff, and asserts
    the cutoff build equals the full build truncated at the cutoff. A row that changes
    when later data is added (e.g. a full-sample z-score) fails. Each cutoff must be
    before ``end_date`` (default ``DEV_END``) and leave at least one row to compare.
    """
    end_date = pd.Timestamp(end_date)
    full = signal.run(end_date)
    for cutoff in cutoffs:
        cutoff = pd.Timestamp(cutoff)
        if cutoff >= end_date:
            raise ValueError(f"cutoff {cutoff.date()} must be before end_date {end_date.date()}")
        expected = full.loc[:cutoff]
        if expected.empty:
            raise ValueError(f"cutoff {cutoff.date()} leaves no rows to compare")
        partial = signal.run(cutoff)
        try:
            pd.testing.assert_frame_equal(partial, expected)
        except AssertionError as err:
            raise AssertionError(
                f"{type(signal).__name__} output up to {cutoff.date()} changes when built "
                f"through {end_date.date()}: lookahead\n{err}"
            ) from None
