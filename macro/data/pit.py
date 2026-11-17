"""Point-in-time alignment of raw series onto NYSE decision dates."""

import pandas as pd


def align(raw: pd.Series, sessions: pd.DatetimeIndex, lag: int, max_carry: int) -> pd.Series:
    """Re-index ``raw`` onto ``sessions`` by the session each value becomes usable.

    ``raw`` is indexed by the date a value is stamped: its observation date, or for
    vintage data the date it was first published. A value stamped s is usable ``lag``
    sessions after the first session on or after s, so a value stamped on a day the NYSE
    is closed (Good Friday, a weekend) is never usable earlier than that session's own
    value. When several values become usable on one session, the latest stamp wins, and
    among equal stamps the last row in ``raw``. Sessions with no new value carry the last
    one forward for at most ``max_carry`` sessions; longer gaps stay NaN.
    """
    raw = raw.dropna().sort_index(kind="stable")
    usable = sessions.searchsorted(raw.index, side="left") + lag
    keep = usable < len(sessions)
    stamped = pd.Series(raw.to_numpy()[keep], index=sessions[usable[keep]])
    stamped = stamped[~stamped.index.duplicated(keep="last")]
    aligned = stamped.reindex(sessions)
    return aligned.ffill(limit=max_carry) if max_carry else aligned
