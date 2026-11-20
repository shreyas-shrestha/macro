# Data

Every series is read with `macro.data.series.load(name, end_date)`. It returns values on
NYSE sessions (index `decision_date`), where the value at *t* is the latest one usable by
the 16:00 ET close on *t*. Raw downloads are cached in `data/cache/` (git-ignored) and
always run through 2017-12-31; `load` refuses later dates unless `holdout=True`, and
holdout reads are never cached.

## Rules

- **Lag.** A value stamped on day *s* is usable *L* sessions after the first NYSE
  session on or after *s*. The stamp is the observation date for Yahoo and BCRP, the
  posting day for FRED, and the first-publication date for ALFRED.
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
| ICSA | ALFRED | `ICSA` | 1990-01-02 | 1 | 10 | Yes | Weekly initial claims; first ALFRED vintage 2009-06-04 |
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

**ALFRED first releases.** NFCI and ICSA are revised, so they are read as first
releases, stamped by the date each value was first published, plus one session because
that date has no time of day. ALFRED began tracking NFCI in 2011; its whole earlier
history shares that first publication date, so it enters at once on that date and
nothing is usable before it. Treat NFCI results before mid-2011 as unavailable, not
estimated (#14). The same holds for ICSA before its first ALFRED vintage. Neither series covers
2008, so #14 cannot compare them with other signals on that episode unless it
deliberately uses a later vintage for the early history. `load` keeps
one value per session, so that back history collapses to its latest observation; a
feature with a 52-week window starts a year after the first vintage unless the signal
reads the first releases with `alfred.fetch` and rebuilds the history itself.

**EMBIG.** J.P. Morgan spreads redistributed by BCRP. They are market data, known to
J.P. Morgan subscribers after the US close on *s*, so *L* = 1 is the true lag and the
*L* = 2 used here is conservative. BCRP itself posts weeks late (in November 2026 its
latest value was 2026-09-03), dates values on every weekday including US and Peruvian
holidays, and sometimes repeats a stale value. Holiday-dated values only matter when the
next session has none. Whether BCRP revises past values is not checked; #17 should
compare a later refresh against its cached download.

## TRACE high-yield series (#24)

Apart from the Treasury curve, these come from WRDS (licensed; the repo is public, so they
cannot be redistributed) and are not in the `series.load` registry. Raw pulls and the
built series live in the git-ignored `data/cache/`; only code and summary diagnostics are
committed. Each signal module applies its own lag and is covered by an
`assert_no_lookahead` test.

| Series | Source | Start | Release lag (trading days) | Notes |
|---|---|---|---|---|
| HY / IG issuer-median TRACE spreads (`hy_spread_bp`, `ig_spread_bp`, `hy_ig_gap_bp`) | WRDS `trace_enhanced.trace_enhanced`, `trace_enhanced.camasterfile`, `comp.adsprate`, `comp.security`, `wrdsapps_link_crsp_bond.bondcrsp_link`, `crsp.stocknames` | 2006-01-09 (bond-days); first decision date 2006-01-10 | 1 | Pulled and built by `scripts/hy_credit/run_hy_once.py` (build step: `python -m macro.data.hy_runs build-dev`) into `data/cache/hy_credit/` (licensed, never committed). The last Legacy Compustat rating is 2017-02-28, so the rated panel ends 2017-06-01. See `research/hy_credit.md`. |
| Treasury constant-maturity curve (`DGS1MO` ... `DGS30`) | FRED / Federal Reserve H.15 | 2000-01-03 | 1 | Public, not WRDS. Read by `macro.data.treasury` only to build same-day maturity-matched spreads (not a signal on its own, so not in the registry); cached as CSV in `data/cache/fred/` through 2017-12-29; `DGS30` treated as missing 2002-02-18 to 2006-02-08 (discontinued). |
| Residual (within-rating) TRACE spread stress (`residual_p90`, `residual_iqr`, `residual_positive_share`) | Derived from the HY panel above | 2006-01-09 (positive share from 2006-04-05) | 1 | Built by `scripts/hy_credit/residual_research.py` into `data/cache/hy_residual/` (licensed issuer-level rows, never committed); aggregates in `results/hy_credit/residual/`. Development-only, descriptive. Ends 2017-06-01 with the ratings. |
| Point-in-time GICS sector per issuer (`gsector`) | WRDS `comp.co_hgic` (historical GICS spells) | spells from 1999-06-30; capped 2017-12-31 | n/a (effective-dated) | Pulled once by `scripts/hy_credit/wrds_pull_sector.py` into `data/cache/hy_residual/sector/` (licensed, never committed); assigned by `macro.data.sector.gics_sector_asof`. 98.3% of rated issuer-days classified. |
