# Data

Every series is read with `macro.data.series.load(name, end_date)`. It returns values on
NYSE sessions (index `decision_date`), where the value at *t* is the latest one usable by
the 16:00 ET close on *t*. Raw downloads are cached in `data/cache/` (git-ignored) and
always run through 2017-12-31; `load` refuses later dates unless `holdout=True`, and
holdout reads are never cached.

## Rules

- **Lag.** A value stamped on day *s* is usable *L* sessions after the first NYSE
  session on or after *s*. The stamp is the observation date for Yahoo and BCRP, the
  posting day for FRED (the release day for weekly claims), and the first-publication date
  for ALFRED.
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
| BAA10Y | FRED | `BAA10Y` | 1990-01-02 | 1 | 3 | No | Baa minus 10-year Treasury |
| DBAA | FRED | `DBAA` | 1990-01-02 | 1 | 3 | No | Moody's seasoned Baa yield |
| DAAA | FRED | `DAAA` | 1990-01-02 | 1 | 3 | No | Moody's seasoned Aaa yield |
| DGS10 | FRED | `DGS10` | 1990-01-02 | 1 | 3 | No | 10-year Treasury constant maturity |
| T10Y2Y | FRED | `T10Y2Y` | 1990-01-02 | 1 | 3 | No | 10-year minus 2-year |
| T10Y3M | FRED | `T10Y3M` | 1990-01-02 | 1 | 3 | No | 10-year minus 3-month |
| CPFF | FRED | `CPFF` | 1997-01-02 | 1 | 3 | No | 3-month commercial paper minus fed funds; 16 missing sessions in Nov 2008–Jan 2009, one gap of 9 |
| NFCI | ALFRED | `NFCI` | 1990-01-02 | 1 | 10 | Yes | Weekly; first ALFRED vintage 2011-06-02 |
| ANFCI | ALFRED | `ANFCI` | 1990-01-02 | 1 | 10 | Yes | NFCI adjusted for economic conditions; first ALFRED vintage 2011-06-02 |
| ANFCI_REVISED | FRED | `ANFCI` | 1990-01-02 | 1 | 10 | Yes | Revised history, **not point in time**; exploratory runs only, see below |
| ICSA | ALFRED | `ICSA` | 1990-01-02 | 1 | 10 | Yes | Weekly initial claims; first ALFRED vintage 2009-06-04 |
| ICNSA | FRED, ALFRED | `ICNSA` | 1988-01-02 | 1 | 10 | Yes | Weekly initial claims, not seasonally adjusted; first releases from 2009-05-30; stamped by release day, see below |
| EMBIG | BCRP | `PD04707XD` | 1998-01-02 | 2 | 3 | Not checked | Global; BCRP has nothing after 2017-01-09, so #17 uses the regional series |
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

**FRED posting day.** FRED posts a day's value on the next Federal Reserve business
day, not always before the 16:00 ET close (#25 found BAA10Y for 2026-11-09 posted at
17:04 ET on 2026-11-10). So FRED values are stamped with that posting day and used one
session later. Counting on the Fed calendar matters around Columbus Day and Veterans
Day, when the NYSE is open but FRED does not post: Friday 2017-10-06's value is posted
Tuesday 10-10 and first used Wednesday 10-11.

Checked against ALFRED's first-release dates for July 2014–2017: FRED posted 97.5% of
DGS10 values on the stamped posting day and the rest 1–3 business days later. Those
delays are taken to be FRED's, since the Fed's H.15 release
(federalreserve.gov/releases/h15) posts each business day's rates on the next one; this
was not checked against the H.15 archive. *L* = 1 stays. Moody's Baa and Aaa yields
(BAA10Y, DBAA, DAAA) for 2016-10-07 to 2017-03-15 first appeared on FRED on 2017-03-22,
after the Fed dropped them from H.15. Moody's subscribers had them daily, so they are
kept as market data at *L* = 1, the same reasoning as EMBIG below. Nothing confirms the
backfill matches what subscribers saw, so #11 should report whether its results change
when that window is excluded.

**ALFRED first releases.** NFCI, ANFCI and ICSA are revised, so they are read as first
releases, stamped by the date each value was first published, plus one session because
that date has no time of day. The Chicago Fed releases NFCI and ANFCI at 08:30 ET on
Wednesday (Thursday in holiday weeks) for the week through the previous Friday, and
ALFRED's vintage dates fall on those days (295 Wednesdays and 49 Thursdays, 2011–2017).
*L* = 0 would therefore be accurate; *L* = 1 is kept as a one-session margin. ALFRED's
first vintages (NFCI and ANFCI 2011-05-25, ICSA 2009-05-28) carry the whole earlier
history, but its initial-release query (`output_type=4`) omits every observation in a
series' first vintage. So `alfred.fetch` starts with the second vintage (the dates in the
table), nothing is usable before it, and none of the three covers 2008. A feature with a
52-week window starts a year after that date. One ANFCI week, ending 2012-10-26,
was released late, together with the next week on 2012-11-07.

**Revised ANFCI (ANFCI_REVISED).** For #14's exploratory run on 2000–2017 only: ANFCI
has no first releases before 2011, so this reads the whole history from the vintage
published on 2017-12-31. Every value includes revisions made after its release, so
results from it are labelled "revised history, not real time" and cannot carry a label.
Each week ending Friday *f* is stamped *f* + 6 (the Thursday release of holiday weeks;
normal weeks come out on Wednesday) with *L* = 1. That timing is the 2011–2017 schedule
applied to earlier years, when the index was not yet published.

**Weekly claims (ICNSA).** The Department of Labor releases initial claims for the week
ending Saturday *s* at 08:30 ET on Thursday *s* + 5, or on Wednesday when that Thursday
is a holiday (Thanksgiving every year; also Veterans Day 2010, July 4 2013, Christmas 2014
and New Year 2015). FRED dates the week by *s*, so the posting-day rule would stamp it
Monday *s* + 2, three days before release. ICNSA has its own source, `claims`: each week
is stamped *s* + 5 with *L* = 1. *L* = 0 would be accurate for a Thursday release; *L* = 1
is a one-session margin, so a normal week is first used on Friday and a week released on
Wednesday is used two sessions late (the Monday after Thanksgiving, for example).

Weeks from 2009-05-30 on take their ALFRED first release, so they are point in time.
All 448 first releases from 2009-06-04 to 2017-12-28 came on or before the stamp (435 on
*s* + 5, 13 on *s* + 4), and the fetcher refuses any that did not. Earlier weeks have no
first release (ALFRED's first ICNSA vintage, 2009-05-28, is dropped as described above)
and take the vintage published on 2017-12-31, which includes revisions made after each
week's release. This is a known deviation: first releases differ from that vintage by a
median 0.5% (90th percentile 1.4%, maximum 3.8% of the first release) over 2010–2017,
and features with a year-long window read some pre-2009 weeks until mid-2010. First
releases are lower than the vintage in 83% of weeks, so over 2009-06 to 2010-06
`claims_yoy` is biased down by about 0.006 (12% of its s.d.). ICNSA is
used rather than the seasonally adjusted ICSA because ICSA is also revised whenever its
seasonal factors are re-estimated.

**EMBIG.** J.P. Morgan spreads redistributed by BCRP. They are market data, known to
J.P. Morgan subscribers after the US close on *s*, so *L* = 1 is the true lag and the
*L* = 2 used here is conservative. BCRP itself posts weeks late (in November 2026 its
latest value was 2026-09-03), dates values on every weekday including US and Peruvian
holidays, and sometimes repeats a stale value. Holiday-dated values only matter when the
next session has none. Whether BCRP revises past values is not checked; #17 should
compare a later refresh against its cached download.
