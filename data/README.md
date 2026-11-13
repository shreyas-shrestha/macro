# Data

Every series is read with `macro.data.series.load(name, end_date)`. It returns values on
NYSE sessions (index `decision_date`), where the value at *t* is the latest one usable by
the 16:00 ET close on *t*. Raw downloads are cached in `data/cache/` (git-ignored) and
always run through 2017-12-31; `load` refuses later dates unless `holdout=True`, and
holdout reads are never cached.

## Rules

- **Lag.** A value stamped on day *s* is usable *L* sessions after the first NYSE
  session on or after *s*. The stamp is the observation date, or for vintage data the
  date the value was first published.
- **Calendar.** Values stamped on days the NYSE is closed (Good Friday, bond-market-only
  days) are never usable earlier than the next session's own value. When several values
  become usable on one session, the latest stamp wins.
- **Carry.** A session with no new value carries the last one forward for at most the
  series' carry limit. Longer gaps stay missing. Nothing is back-filled.

## Series

| Name | Source | ID | Start | Lag | Carry | Notes |
|---|---|---|---|---|---|---|
| SPX | Yahoo | `^GSPC` | 1985-01-02 | 0 | 0 | Price index close; known at the close |
| VIX | Yahoo | `^VIX` | 1990-01-02 | 0 | 0 | Known at the close |
| VIX3M | Yahoo | `^VIX3M` | 2006-07-17 | 0 | 0 | Formerly VXV; Yahoo serves the full history under `^VIX3M` |
| MOVE | Yahoo | `^MOVE` | 2002-11-12 | 1 | 3 | ICE publishes after the equity close; Yahoo has gaps |
