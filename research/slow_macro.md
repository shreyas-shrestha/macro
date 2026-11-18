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

## Pre-registered threshold (posted on #14 before any run)

- **Primary:** claims block at *h* = 21: at least 5% lower QLIKE than the baseline with
  Clark–West *p* < 0.10.
- **Secondary, declared in advance:** the same test at *h* = 63.
- ANFCI blocks are exploratory: 47 to 59 forecasts, and the first fit has 82 to 329
  training sessions, too few to carry the label.

## Scorecard

Development period only. Gain is the QLIKE reduction against the baseline on the same
rows (positive is better); *p* is one-sided Clark–West.

| Block | *h* | *n* | QLIKE base | QLIKE signal | Gain | R²_OOS | CW *p* |
|---|---|---|---|---|---|---|---|
| claims | 5 | 215 | 0.2623 | 0.2646 | −0.9% | −0.009 | 0.97 |
| claims | **21** | 214 | 0.2465 | 0.2467 | **−0.1%** | 0.003 | **0.16** |
| claims | **63** | 213 | 0.2893 | 0.2871 | **+0.8%** | −0.006 | **0.29** |
| claims | 126 | 210 | 0.3399 | 0.3259 | +4.1% | 0.008 | 0.21 |
| claims | 252 | 203 | 0.3595 | 0.3477 | +3.3% | 0.005 | 0.18 |
| anfci | 5 | 59 | 0.2768 | 0.2921 | −5.6% | −0.087 | 0.72 |
| anfci | 21 | 58 | 0.2967 | 0.4556 | −53.6% | −0.255 | 0.78 |
| anfci | 63 | 57 | 0.1927 | 0.2483 | −28.8% | −0.180 | 0.90 |
| anfci | 126 | 54 | 0.1713 | 0.1950 | −13.8% | −0.123 | 0.90 |
| anfci | 252 | 47 | 0.2408 | 0.1999 | +17.0% | 0.204 | 0.03 |
| claims + anfci | 5 | 59 | 0.2768 | 0.4018 | −45.2% | −0.306 | 0.91 |
| claims + anfci | 21 | 58 | 0.2967 | 0.5050 | −70.2% | −0.428 | 0.81 |
| claims + anfci | 63 | 57 | 0.1927 | 0.2704 | −40.3% | −0.300 | 0.80 |
| claims + anfci | 126 | 54 | 0.1713 | 0.1955 | −14.1% | −0.134 | 0.80 |
| claims + anfci | 252 | 47 | 0.2408 | 0.1404 | +41.7% | 0.327 | 0.04 |

**Label: `not-advancing`.** Both pre-registered tests fail: the primary misses the 5% bar
with a gain of −0.1%, and the secondary reaches only +0.8% with *p* = 0.29.

## Reading

- **The claims block's gains all come from 2008.** Summed QLIKE improvement by year:
  2008 contributes 0.52, 2.24, 3.74 and 2.99 at *h* = 21, 63, 126 and 252. Without 2008
  the block is worse than the baseline at every horizon (−0.56, −1.77, −0.81, −0.59).
  The +3–4% at 126 and 252 days is one episode, not a horizon effect, and the
  pre-2009 claims values are the latest vintage, not first releases.
- **The long-horizon ANFCI result should not be believed.** 47 month-ends with target
  windows a year long cover about four non-overlapping years (2013–2017), Newey–West
  with 12 lags on 47 points is unreliable, and the same block loses 14–54% at every
  shorter horizon. The short-horizon losses come from fits on 82 to 329 training
  sessions, mostly 2011–2012.
- What remains for the claims block is a weak hint that labor data adds information
  in recessions only. The development period holds one recession with real-time claims
  timing, so this cannot be tested here.

## Not done yet

- The indicative 2000–2017 ANFCI run on revised history ("revised history, not real
  time").
- The breakdown by stress episode (2008, 2010, 2011, 2015).
- The R²_OOS-versus-horizon chart.
