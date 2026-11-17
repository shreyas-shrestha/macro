import pandas as pd
import pytest

from macro.data.calendar import nyse_sessions
from macro.models.schedule import refit_schedule, training_end

SESSIONS = nyse_sessions("2000-01-03", "2002-12-31")


def test_each_year_refits_at_the_last_session_of_the_previous_year():
    dates = pd.DatetimeIndex(["2001-01-31", "2001-06-29", "2002-03-28"])
    schedule = list(refit_schedule(SESSIONS, dates))
    assert [refit for refit, _ in schedule] == [
        pd.Timestamp("2000-12-29"),
        pd.Timestamp("2001-12-31"),
    ]
    assert schedule[0][1].equals(dates[:2])


def test_training_targets_end_by_the_refit_session():
    # Five sessions back from 2001-12-31 skips Christmas Day.
    assert training_end(SESSIONS, pd.Timestamp("2001-12-31"), 5) == pd.Timestamp("2001-12-21")


def test_training_end_refuses_a_refit_too_early_in_the_sessions():
    with pytest.raises(ValueError, match="fewer than 5 sessions"):
        training_end(SESSIONS, SESSIONS[2], 5)


def test_refit_schedule_needs_a_session_before_the_first_year():
    with pytest.raises(ValueError, match="no session before 2000"):
        list(refit_schedule(SESSIONS, pd.DatetimeIndex(["2000-06-30"])))
