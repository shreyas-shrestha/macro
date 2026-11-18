# Slow macro: jobless claims and financial conditions (#14)

## Hypothesis

Weekly labor and financial-conditions data forecast S&P 500 realized variance beyond
VIX and recent realized variance, and any gain grows with the horizon (Engle, Ghysels
and Sohn 2013; Paye 2012).

## Data and timing

Details and checks are in data/README.md.

- **Claims: ICNSA** (initial claims, not seasonally adjusted), 1988–2017. Each week is
  stamped on its release Thursday, *s* + 5, with *L* = 1. All 448 ALFRED first releases
  from 2009-06-04 to 2017-12-28 came on or before that stamp.
- **What is point in time.** Weeks from 2009-05-30 on use their ALFRED first release.
  Earlier weeks use the vintage published on 2017-12-31, which includes revisions made
  after each week's release; ALFRED has nothing earlier. So the claims features are
  point in time from 2010-06-24, when their year-long windows no longer reach back
  before 2009-05-30. Before that they use revised data: a measured deviation (below).
- **Financial conditions: ANFCI** first releases from ALFRED, usable from 2011-06-03.
  ANFCI still contains market volatility measures; it removes the economic-conditions
  component, not the VIX overlap.

## Features (fixed before any scoring)

Computed on the weekly values indexed by release date, then put on NYSE sessions with
the input's lag and carry. Each weekly value uses only its own release and earlier
ones, so a session sees only weeks already released.

| Feature | Definition |
|---|---|
| `claims_yoy` | log(4-week average / the same average 52 weeks earlier) |
| `claims_vs_low` | log(4-week average / its minimum over the prior 52 weeks) |
| `anfci_level` | ANFCI as first released |
| `anfci_change_13w` | first release minus the first release 13 weeks earlier |

The year-ago week is matched by date (364 days back), not position. No winsorizing:
the 4-week average damps one-week outliers (Katrina 2005, Sandy 2012).

`anfci_change_13w` subtracts values from two vintages, because ANFCI's whole history is
re-estimated weekly. Both were public at the later release, so there is no lookahead,
but it is noisier than the change within the later release's vintage. Over all 331
releases where both exist (weeks ending 2011-08-26 to 2017-12-22), the two correlate
0.843, with a median absolute difference of 0.100 and a maximum of 0.55 (s.d. of the
within-vintage change 0.348). Rebuilding the within-vintage change in the signal needs
every vintage, so it is left out.

## Revision check (claims)

How large is the pre-2009 deviation? Features built from ALFRED first releases (what
the signal uses from mid-2010) and from the 2017-12-31 vintage (what it uses before),
weekly, 2010-06-24 to 2017-12-28 (393 weeks):

| Feature | Correlation | Max abs difference | Median abs difference | Feature s.d. |
|---|---|---|---|---|
| `claims_yoy` | 0.9958 | 0.0147 | 0.0029 | 0.050 |
| `claims_vs_low` | 0.9997 | 0.0156 | 0.0018 | 0.156 |

Raw values over 2010–2017: the vintage differs from the first release by a median 0.5%
of the first release (90th percentile 1.4%, maximum 3.8%; 3.65% as a share of the
vintage value). If pre-2009 revisions were similar, the deviation is small next to the
features' variation.

## Models

Provisional runner (`experiments/run_slow_macro.py`, `configs/slow_macro.yaml`) until
#10 lands, then rerun under #10. Ridge on log realized variance; baseline log trailing
RV and log VIX; yearly expanding refit, embargo *h*; month-end grid; baseline and signal
models on identical rows. Blocks: claims (2000–2017), ANFCI and claims + ANFCI
(2013–2017, 47 to 59 month-ends by horizon; first refit at the end of 2012). Horizons 5,
21, 63, 126, 252. Clark–West uses Newey–West with a lag equal to the number of later
month-ends whose target windows overlap (0, 1, 3, 6, 12 for those horizons): #10's "lag
*h*" is in sessions and the grid is monthly.

## Pre-registered threshold (to be posted on #14 before any run)

- **Primary:** claims block at *h* = 21: at least 5% lower QLIKE than the baseline with
  Clark–West *p* < 0.10.
- **Secondary, declared in advance:** the same test at *h* = 63.
- ANFCI blocks are exploratory: 47 to 59 forecasts, and the first fit has 82 to 329
  training sessions, too few to carry the label.

## Scorecard

Pending: not run until the threshold is posted.

## Not done yet

- The indicative 2000–2017 ANFCI run on revised history ("revised history, not real
  time").
- The breakdown by stress episode (2008, 2010, 2011, 2015).
- The R²_OOS-versus-horizon chart.
