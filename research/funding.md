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

Development period only. Gain is the QLIKE reduction against the baseline on the same
rows (positive is better); *p* is one-sided Clark–West; R²_OOS is on squared errors of
log volatility. The last two rows come from `experiments/score_excluding.py`.

| Rows | *h* | *n* | QLIKE baseline | QLIKE signal | Gain | R²_OOS | CW *p* |
|---|---|---|---|---|---|---|---|
| all | 5 | 179 | 0.2461 | 0.2462 | −0.04% | 0.002 | 0.09 |
| all | **21** | 178 | 0.2581 | 0.2527 | **+2.1%** | −0.008 | **0.19** |
| all | 63 | 177 | 0.3233 | 0.4234 | −31.0% | −0.226 | 0.97 |
| without 2007-08 to 2009-06 | **21** | 155 | 0.2190 | 0.2252 | **−2.8%** | −0.032 | **0.77** |
| 2007-08 to 2009-06 only | 21 | 23 | 0.5221 | 0.4386 | +16.0% | 0.097 | 0.05 |

Forecasts are in `results/forecasts/funding_h{5,21,63}.csv`.

**Label: `not-advancing`.** The primary test fails, with a gain of +2.1% against the 5%
bar and *p* = 0.19. Without the crisis window the gain is −2.8% (*p* = 0.77).

## Reading

- **The *h* = 21 gain is three forecasts.** Summed over the 178 month-ends, the QLIKE
  improvement is 0.96. The forecasts of 2008-08-29, 09-30 and 10-31 contribute 2.34;
  every other month together loses 1.38, including −0.82 in 2009 and −0.45 in 2006.
  Outside 2007–2009 the largest single gain is September 2005 (0.42).
- **The gain is a large spread times a small coefficient.** Each horizon is a separate
  fit. In the *h* = 21 refit used for 2008, the standardized ridge coefficients are +0.06
  on `cpff_level` and −0.08 on `cpff_z_252`, next to +0.75 on log VIX. In September 2008
  the level was 2.32, above anything in training (maximum 1.72, a one-day spike at the
  2000 turn of year; 2007 peaked at 0.78) and 11 training s.d. above its mean, so its
  small slope added about 0.6 in logs; the z-score (4.6, inside the range reached in
  August 2007) took back half. Net, CPFF raised the forecast by a third: the level
  extrapolated far beyond anything the 2007 episode reached, not a relationship the
  sample can estimate.
- **Then it overshoots.** The 2009 refit, now including 2008, raises the level coefficient
  to +0.20. CPFF spiked again in January 2009 (1.95) while volatility was already well
  below its October peak: the forecast was 2.3 times the baseline at *h* = 21 and 4.4
  times at *h* = 63, and the losses run through 2009.
- **At *h* = 63 the sign was wrong going into the crisis.** The refit used for 2008 has a
  negative slope on the level (−0.07), so as CPFF rose from May to October 2008 the model
  cut its forecast to 0.4–0.85 times the baseline just before the largest variance in the
  sample. QLIKE punishes that heavily: −13.8 in 2008 alone. A plausible cause, not
  tested here, is 2001–2002, when CPFF was negative because of expected rate cuts while
  volatility was high; it is in every training window before 2008.
- **The *h* = 5 *p* = 0.09 is not a gain.** Clark–West tests the log forecasts, while QLIKE
  is flat (−0.04%); the large gains and losses in 2008 and 2009 cancel.
- What remains is that CPFF was elevated in August 2008 (0.89, z-score 1.5) and spiked
  after Lehman, while VIX had not yet caught up. The development period has one such
  crisis (2007–2009), and the same features lose in its aftermath and in quiet years, so
  that lead cannot be tested here.

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
