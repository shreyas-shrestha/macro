# Funding and liquidity: commercial paper minus fed funds (#15)

## Hypothesis

When short-term funding gets scarce, the rate on 3-month commercial paper rises relative
to the overnight fed funds rate, and that stress forecasts S&P 500 realized variance
beyond VIX and recent realized variance. Funding and market liquidity reinforce each
other, so a funding squeeze can raise volatility (Brunnermeier and Pedersen 2009). The
commercial paper market is where the 2007–2009 funding run showed first: asset-backed
paper ran in August 2007 (Covitz, Liang and Suarez 2013; Kacperczyk and Schnabl 2010),
and after Lehman the Fed set up the Commercial Paper Funding Facility to backstop the
market and cap term paper rates (Adrian, Kimbrough and Marchioni 2011). Credit spreads
more broadly carry information about future activity beyond default risk (Gilchrist and
Zakrajšek 2012); this signal asks the narrower question for funding spreads and
volatility.

## Data and timing

Details and checks are in data/README.md.

- **CPFF** (FRED): 3-month AA financial commercial paper rate (RIFSPPFAAD90NB) minus
  the effective fed funds rate, percentage points, daily from 1997. Each value is
  stamped on FRED's posting day (the next Federal Reserve business day) and used one
  session later, *L* = 1, carried at most 3 sessions. Not revised.
- The first usable session is 1997-01-06. 16 sessions in Nov 2008–Jan 2009 have no
  value within the carry limit (one gap of 9, 2008-12-12 to 12-24); they stay NaN.
  No month-end decision date in 2000–2017 loses a feature to them.
- CPFF is still posted on FRED (latest observation 2026-11-18), so the holdout run needs
  no substitute series.

**Caveats, from the levels on sessions** (2000–2017; nothing here is measured against
volatility):

- Mean 0.12 pp, s.d. 0.28, maximum 2.91 (October 2008). From 2007-08-01 to 2009-06-30
  the mean is 0.53; outside that window it is 0.08 (s.d. 0.16).
- Negative through most of 2001 (91% of sessions; minimum −0.86 in September). The
  3-month paper rate prices the rate cuts expected over its term, while fed funds is
  overnight, so CPFF also moves with the expected path of policy, not only with stress.
- Elevated in 2016: mean 0.25; 0.31 on average in the second half (quarterly means 0.30
  and 0.32), around the money-market fund reform that took effect in October 2016, a
  regulatory shift in demand for commercial paper rather than market stress.
- The effective fed funds rate changed methodology on 2016-03-01 (the New York Fed's
  volume-weighted median), a break in the spread's second leg.

## Features (fixed before any scoring)

Computed on CPFF as `macro.data.series.load` returns it, on NYSE sessions from 1997, and
kept from 2000-01-03, so every feature is complete from the first row. Windows count
sessions, missing ones included, and are not extended to make up for them.

| Feature | Definition |
|---|---|
| `cpff_level` | CPFF, percentage points |
| `cpff_change_20d` | CPFF at *t* minus CPFF 20 sessions earlier; NaN if either is missing |
| `cpff_z_252` | (CPFF at *t* − mean) / sample s.d. (ddof = 1) of the 252 sessions ending at *t*, with at least 200 non-missing |

The z-score's window includes *t*; nothing after *t* enters.

## Models

Scored with the shared harness (#10): `experiments/run_signal.py` with
`configs/funding.yaml`. Ridge on log realized variance; baseline log trailing RV and log
VIX; yearly expanding refit, embargo *h*; month-end grid; every model on identical rows;
refits with fewer than 504 training rows are skipped. Horizons 5, 21, 63. The refit at
the end of 2001 has 495, 479 and 437 training rows at *h* = 5, 21 and 63, so the first
forecast is 2003-01-31 at every horizon, as for other signals starting in 2000. The
concentration check rescores the *h* = 21 forecasts with
`experiments/score_excluding.py --forecasts results/forecasts/funding_h21.csv --exclude
2007-08-01 2009-06-30`.

## Pre-registered threshold (posted on #15 before any run)

- **Primary (carries the label):** at h = 21, at least 5% lower QLIKE than the log RV +
  log VIX baseline, with Clark–West p < 0.10.
- h = 5 and h = 63 are reported and cannot carry the label.
- **Concentration check, declared in advance:** recompute the QLIKE reduction and
  Clark–West at h = 21 without decision dates from 2007-08-01 to 2009-06-30 (from the
  commercial-paper freeze to the end of the recession). If the signal passes only with
  that window included, it's labeled `not-advancing`.

## Scorecard

**To be filled after the run.** Development period only.

| Rows | *h* | *n* | QLIKE baseline | QLIKE signal | Gain | R²_OOS | CW *p* |
|---|---|---|---|---|---|---|---|
| all | 5 | | | | | | |
| all | **21** | | | | | | |
| all | 63 | | | | | | |
| without 2007-08 to 2009-06 | **21** | | | | | | |
| 2007-08 to 2009-06 only | 21 | | | | | | |

**Label:** to be filled after the run.

## Reading

**To be filled after the run.**

## References

- Adrian, T., K. Kimbrough and D. Marchioni (2011). The Federal Reserve's Commercial
  Paper Funding Facility. *FRBNY Economic Policy Review* 17(1).
- Brunnermeier, M. and L. Pedersen (2009). Market Liquidity and Funding Liquidity.
  *Review of Financial Studies* 22(6).
- Covitz, D., N. Liang and G. Suarez (2013). The Evolution of a Financial Crisis:
  Collapse of the Asset-Backed Commercial Paper Market. *Journal of Finance* 68(3).
- Gilchrist, S. and E. Zakrajšek (2012). Credit Spreads and Business Cycle Fluctuations.
  *American Economic Review* 102(4).
- Kacperczyk, M. and P. Schnabl (2010). When Safe Proved Risky: Commercial Paper during
  the Financial Crisis of 2007–2009. *Journal of Economic Perspectives* 24(1).
