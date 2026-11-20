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

Scored with the shared harness (#10): `experiments/run_signal.py` with
`configs/claims.yaml`, `configs/anfci.yaml` and `configs/claims_anfci.yaml`. Ridge on
log realized variance; baseline log trailing RV and log VIX; yearly expanding refit,
embargo *h*; month-end grid; every model on identical rows; refits with fewer than 504
training rows are skipped. Horizons 5, 21, 63, 126, 252. Claims + ANFCI is also scored
against baseline + claims, which shows what ANFCI adds beyond claims.

An earlier stand-in runner gave identical claims results at every horizon. ANFCI
results differ from it: the stand-in fit from the end of 2012 on as few as 82 rows,
while the harness's 504-row minimum moves the first ANFCI forecast to 2014 (2015 at
*h* = 126 and 252).

## Pre-registered threshold (posted on #14 before any run)

- **Primary:** claims block at *h* = 21: at least 5% lower QLIKE than the baseline with
  Clark–West *p* < 0.10.
- **Secondary, declared in advance:** the same test at *h* = 63.
- ANFCI blocks are exploratory and cannot carry the label: real-time ANFCI starts in
  2011, leaving 23 to 47 forecasts.

## Scorecard

Development period only. Gain is the QLIKE reduction against the baseline on the same
rows (positive is better); *p* is one-sided Clark–West.

| Block | Against | *h* | *n* | QLIKE ref | QLIKE signal | Gain | R²_OOS | CW *p* |
|---|---|---|---|---|---|---|---|---|
| claims | baseline | 5 | 215 | 0.2623 | 0.2646 | −0.9% | −0.009 | 0.97 |
| claims | baseline | **21** | 214 | 0.2465 | 0.2467 | **−0.1%** | 0.003 | **0.16** |
| claims | baseline | **63** | 213 | 0.2893 | 0.2871 | **+0.8%** | −0.006 | **0.29** |
| claims | baseline | 126 | 210 | 0.3399 | 0.3259 | +4.1% | 0.008 | 0.21 |
| claims | baseline | 252 | 203 | 0.3595 | 0.3477 | +3.3% | 0.005 | 0.18 |
| anfci | baseline | 5 | 47 | 0.3158 | 0.3359 | −6.4% | −0.100 | 0.79 |
| anfci | baseline | 21 | 46 | 0.3302 | 0.5407 | −63.8% | −0.329 | 0.87 |
| anfci | baseline | 63 | 45 | 0.2301 | 0.3015 | −31.0% | −0.198 | 0.92 |
| anfci | baseline | 126 | 30 | 0.2811 | 0.3213 | −14.3% | −0.127 | 0.93 |
| anfci | baseline | 252 | 23 | 0.2146 | 0.1799 | +16.2% | 0.253 | 0.09 |
| claims + anfci | baseline | 5 | 47 | 0.3158 | 0.4384 | −38.8% | −0.301 | 0.95 |
| claims + anfci | baseline | 21 | 46 | 0.3302 | 0.5351 | −62.1% | −0.372 | 0.86 |
| claims + anfci | baseline | 63 | 45 | 0.2301 | 0.2863 | −24.4% | −0.136 | 0.67 |
| claims + anfci | baseline | 126 | 30 | 0.2811 | 0.3127 | −11.2% | −0.093 | 0.79 |
| claims + anfci | baseline | 252 | 23 | 0.2146 | 0.1667 | +22.3% | 0.312 | 0.07 |
| claims + anfci | + claims | 5 | 47 | 0.3823 | 0.4384 | −14.7% | −0.111 | 0.84 |
| claims + anfci | + claims | 21 | 46 | 0.3536 | 0.5351 | −51.4% | −0.306 | 0.88 |
| claims + anfci | + claims | 63 | 45 | 0.2311 | 0.2863 | −23.9% | −0.154 | 0.87 |
| claims + anfci | + claims | 126 | 30 | 0.2672 | 0.3127 | −17.0% | −0.151 | 0.94 |
| claims + anfci | + claims | 252 | 23 | 0.2120 | 0.1667 | +21.4% | 0.290 | 0.06 |

R²_OOS is on squared errors of log volatility. Forecasts are in
`results/forecasts/{claims,anfci,claims_anfci}_h*.csv`.

**Label: `not-advancing`.** Both pre-registered tests fail: the primary misses the 5% bar
with a gain of −0.1%, and the secondary reaches only +0.8% with *p* = 0.29.

## Reading

- **The claims block's gains all come from one stretch of 2008.** Summed QLIKE
  improvement in 2008 is 0.52, 2.24, 3.74 and 2.99 at *h* = 21, 63, 126 and 252; over
  every other year it is −0.56, −1.77, −0.81 and −0.59. Within 2008 the gain comes from
  forecasts made in April to August, before Lehman: rising claims raised the forecast
  while VIX was still moderate. Over the crisis itself (September 2008 to March 2009) the
  block gains nothing at *h* = 63 and loses at 126 and 252, then keeps losing into mid
  2009, when claims were still high and volatility was falling. The +3–4% at 126 and 252
  days is one early warning, not a horizon effect.
- **The 2008 claims values are not point in time.** Weeks before mid-2009 use the
  2017-12-31 vintage (see `data/README.md`), so the one episode that carries the long
  horizons is also the one scored on revised data.
- **The ANFCI results should not be believed in either direction.** At *h* = 252 there
  are 23 month-ends from 2015 to 2016, about two non-overlapping years, and Newey–West
  with 12 lags on 23 points is unreliable; at every shorter horizon the same features
  lose 6–64%. Against baseline + claims, ANFCI adds nothing up to *h* = 126.
- What remains for the claims block is a weak hint that a rise in claims can warn of a
  volatility regime before VIX does. The development period has one such episode, scored
  on revised claims, so it cannot be tested here.

## Not done yet

- The indicative 2000–2017 ANFCI run on revised history ("revised history, not real
  time").
- The R²_OOS-versus-horizon chart.
