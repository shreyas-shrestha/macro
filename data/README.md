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

*Start* is the first date requested; a series may begin later. *Lag* and *Carry* are in
NYSE sessions.

| Name | Source | ID | Start | Lag | Carry | Revised | Notes |
|---|---|---|---|---|---|---|---|
| SPX | Yahoo | `^GSPC` | 1985-01-02 | 0 | 0 | No | Price index, 16:00 close |
| VIX | Yahoo | `^VIX` | 1990-01-02 | 0 | 0 | No | 16:15 close; see below |
| VIX3M | Yahoo | `^VIX3M` | 2006-07-17 | 0 | 0 | No | 16:15 close; formerly VXV, full history under `^VIX3M` |
| MOVE | Yahoo | `^MOVE` | 2002-11-12 | 1 | 3 | No | Published after the equity close; 81 missing sessions 2002–2017, one gap of 5 |
| BAA10Y | FRED | `BAA10Y` | 1990-01-02 | 2 | 3 | No | Baa minus 10-year Treasury; FRED rows below share the lag and carry rules |
| DBAA | FRED | `DBAA` | 1990-01-02 | 2 | 3 | No | Moody's seasoned Baa yield |
| DAAA | FRED | `DAAA` | 1990-01-02 | 2 | 3 | No | Moody's seasoned Aaa yield |
| DGS10 | FRED | `DGS10` | 1990-01-02 | 2 | 3 | No | 10-year Treasury constant maturity |
| T10Y2Y | FRED | `T10Y2Y` | 1990-01-02 | 2 | 3 | No | 10-year minus 2-year |
| T10Y3M | FRED | `T10Y3M` | 1990-01-02 | 2 | 3 | No | 10-year minus 3-month |
| CPFF | FRED | `CPFF` | 1997-01-02 | 2 | 3 | No | 3-month commercial paper minus fed funds |
| NFCI | ALFRED | `NFCI` | 1990-01-02 | 1 | 10 | Yes | Weekly; first releases only, so nothing before 2011 (see below) |
| ICSA | ALFRED | `ICSA` | 1990-01-02 | 1 | 10 | Yes | Weekly initial claims; first releases only |
| EMBIG | BCRP | `PD04707XD` | 1998-01-02 | 2 | 3 | Not checked | Global; BCRP has nothing after 2017-01-09 |
| EMBIG_LATAM | BCRP | `PD04708XD` | 1998-01-02 | 2 | 3 | Not checked | Latin America |
| EMBIG_PERU | BCRP | `PD04709XD` | 1998-01-02 | 2 | 3 | Not checked | |
| EMBIG_ARGENTINA | BCRP | `PD04710XD` | 1998-01-02 | 2 | 3 | Not checked | |
| EMBIG_BRAZIL | BCRP | `PD04711XD` | 1998-01-02 | 2 | 3 | Not checked | |
| EMBIG_ECUADOR | BCRP | `PD04712XD` | 1998-01-02 | 2 | 3 | Not checked | |
| EMBIG_MEXICO | BCRP | `PD04713XD` | 1998-01-02 | 2 | 3 | Not checked | |
| EMBIG_VENEZUELA | BCRP | `PD04714XD` | 1998-01-02 | 2 | 3 | Not checked | |
| EMBIG_COLOMBIA | BCRP | `PD04715XD` | 1998-01-02 | 2 | 3 | Not checked | |

**VIX timing exception.** CBOE's VIX and VIX3M closes are struck at 16:15 ET, 15 minutes
after the cutoff in CONTRIBUTING.md. They are used at lag 0 anyway: they are the
benchmark every signal must beat, and a one-day lag would handicap the benchmark and
flatter the signals. VIX3M is also an input to the term-structure signal (#16), where
the extra 15 minutes help the signal instead; #16 reports a lag-1 run as a robustness
check.

**FRED lag.** FRED posts these daily series the next business day, but not always
before the 16:00 ET close (#25 found BAA10Y for 2026-11-09 posted at 17:04 ET on
2026-11-10). Vintage dates carry no time of day, so *L* = 2 is the first lag that is
safe on every day.

**ALFRED first releases.** NFCI and ICSA are revised, so they are read as first
releases, stamped by the date each value was first published, plus one session because
that date has no time of day. ALFRED began tracking NFCI in 2011; its whole earlier
history shares that first publication date, so it enters at once on that date and
nothing is usable before it. Treat NFCI results before mid-2011 as unavailable, not
estimated (#14). The same holds for ICSA before its first ALFRED vintage.

**EMBIG.** J.P. Morgan spreads redistributed by BCRP, dated on Peruvian business days.
BCRP does not publish posting times, so the FRED lag rule is used.
