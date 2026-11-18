# Slow macro: jobless claims and financial conditions (#14)

## Hypothesis

Weekly labor and financial-conditions data forecast S&P 500 realized variance beyond
VIX and recent realized variance, and any gain grows with the horizon (Engle, Ghysels
and Sohn 2013; Paye 2012).

## Data and timing

- **Claims: ICNSA** (initial claims, not seasonally adjusted), FRED, 1988–2017. The
  seasonally adjusted ICSA is not used because its seasonal factors are re-estimated
  with later data.
- **Release day.** The week ending Saturday *s* is released Thursday *s* + 5 at 08:30 ET
  (Wednesday when Thursday is a holiday). FRED's posting-day rule would stamp it
  Monday *s* + 2, three days early, so ICNSA has its own rule: stamp *s* + 5, *L* = 1.
  All 448 ALFRED first releases from 2009-06-04 to 2017-12-28 came on or before the
  stamp: 435 on *s* + 5, 13 on *s* + 4 (Thanksgiving, Veterans Day 2010, July 4 2013,
  Christmas 2014, New Year 2015). ALFRED has no ICNSA vintages before 2009-05-28, so
  the rule is unchecked before then.
- **Vintage.** ICNSA is read as published on 2017-12-31, not the latest vintage, so no
  revision made after the development sample is used. Before 2009 no first releases
  exist, so revised values are used; the revision check below bounds the effect.
- **Financial conditions: ANFCI** first releases from ALFRED, real time from 2011-06-02.
  ALFRED's initial-release query drops the first vintage (2011-05-25), which held the
  back history, so nothing is usable earlier. ANFCI still contains market volatility
  measures; it removes the economic-conditions component, not the VIX overlap.
- No missing weeks in either series. One ANFCI week (2012-10-26) was released late,
  together with the next one on 2012-11-07.

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
but it is noisier than the change within one vintage: on 41 releases sampled 2011–2017
the two correlate 0.86, with a median absolute difference of 0.10 (feature s.d. 0.36).
Rebuilding the within-vintage change needs every vintage, so it is left out.

## Revision check (claims)

Features built from ALFRED first releases and from the 2017-12-31 vintage, weekly,
2010-06-24 to 2017-12-28 (393 weeks; first releases need a year of history):

| Feature | Correlation | Max abs difference | Median abs difference | Feature s.d. |
|---|---|---|---|---|
| `claims_yoy` | 0.9958 | 0.0147 | 0.0029 | 0.050 |
| `claims_vs_low` | 0.9997 | 0.0156 | 0.0018 | 0.156 |

Raw values: first releases differ from the vintage by a median 0.5% (90th percentile
1.4%). Using revised values before 2009 should therefore leak next to nothing.

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
