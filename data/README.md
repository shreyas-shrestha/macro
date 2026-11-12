Datasets.

## Credit spreads (#11)

| Series | Source | Start | Frequency | Revised | Release lag *L* |
|---|---|---|---|---|---|
| BAA10Y (Baa minus 10-year CMT) | FRED, Interest Rate Spreads | 1986-01-02 | Daily, business days | No* | 2 NYSE sessions |
| DBAA (Moody's seasoned Baa yield) | FRED / Moody's | 1986-01-02 | Daily | No* | 2 NYSE sessions |
| DAAA (Moody's seasoned Aaa yield) | FRED / Moody's | 1983-01-03 | Daily | No* | 2 NYSE sessions |
| BAMLH0A0HYM2 (ICE BofA HY OAS) | FRED / ICE | Rolling 3 years only | Daily | Rarely | Not used |

\* To be confirmed from ALFRED vintages by `experiments/credit_diagnostics.py`.

**Release lag.** BAA10Y for day *s* is the Moody's Baa yield minus Treasury's 10-year
constant maturity yield for *s*. FRED posts it on the next business day, but not
necessarily before the 16:00 ET cutoff. The observation for Monday 2026-11-09 was
posted at 16:04 CST (17:04 ET) on Tuesday 2026-11-10, an hour after the close. ALFRED
vintage dates have no time of day, so a one-day vintage date does not show that the
value was usable at *s* + 1. The signal therefore uses *L* = 2: a value observed on *s*
first enters the row two NYSE sessions after *s*. This is stricter than the *L* = 1
example in `CONTRIBUTING.md`. If the team decides to assume a pre-close posting time,
change `Credit.release_lag`; the features and tests follow it.

**Calendar.** FRED dates these series on US business days. An observation on a day the
NYSE is closed but the bond market is open (e.g. Good Friday) is stamped from the next
NYSE session, so it never arrives earlier than that session's own value. When several
observations become usable on one session, the latest wins.

**Fill rule.** Bond-market holidays that are NYSE sessions (Columbus Day, Veterans Day)
leave one-session FRED gaps. The last usable value is carried forward for at most three
sessions; longer gaps stay missing. Nothing is back-filled.

**High-yield OAS.** Since April 2026 FRED serves only a rolling three-year window for
all ICE BofA series, including BAMLH0A0HYM2, so it has no development-sample (2000–2017)
history and is not used. Full history needs ICE directly; TRACE-based HY spreads are
tracked in #24.
