import pandas as pd

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
    refit = pd.Timestamp("2001-12-31")
    end = training_end(SESSIONS, refit, 5)
    assert SESSIONS[SESSIONS.get_loc(end) + 5] == refit
