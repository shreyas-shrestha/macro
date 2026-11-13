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
- **Calendar.** Values stamped on days the NYSE is closed but another source still prints
  (weekends, Good Friday) are never usable earlier than the next session's own value.
  When several values become usable on one session, the latest stamp wins.
- **Carry.** A session with no new value carries the last one forward for at most the
  series' carry limit. Longer gaps stay missing. Nothing is back-filled.

## Series

| Name | Source | ID | Start | Lag | Carry | Revised | Notes |
|---|---|---|---|---|---|---|---|
| SPX | Yahoo | `^GSPC` | 1985-01-02 | 0 | 0 | No | Price index, 16:00 close |
| VIX | Yahoo | `^VIX` | 1990-01-02 | 0 | 0 | No | 16:15 close; see below |
| VIX3M | Yahoo | `^VIX3M` | 2006-07-17 | 0 | 0 | No | 16:15 close; formerly VXV, full history under `^VIX3M` |
| MOVE | Yahoo | `^MOVE` | 2002-11-12 | 1 | 3 | No | Published after the equity close; 81 missing sessions 2002–2017, one gap of 5 |

**VIX timing exception.** CBOE's VIX and VIX3M closes are struck at 16:15 ET, 15 minutes
after the cutoff in CONTRIBUTING.md. They are used at lag 0 anyway: they are the
benchmark every signal must beat, and a one-day lag would handicap the benchmark and
flatter the signals. VIX3M is also an input to the term-structure signal (#16), where
the extra 15 minutes help the signal instead; #16 reports a lag-1 run as a robustness
check.
