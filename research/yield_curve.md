# Signal: rates and yield curve

Issue #12 · Owner: Michael Zuo · Status: **not advancing** (stand-in evaluation; rerun under #10)

## Result in one line

Adding yield-curve features to the RV + VIX benchmark made S&P 500 volatility forecasts
slightly worse at 5, 21 and 63 days, so the signal does not meet its threshold.

| Horizon (sessions) | Forecasts | QLIKE, benchmark | QLIKE, with yield curve | Change | Clark–West p |
|---|---|---|---|---|---|
| 5 | 167 | 0.2580 | 0.2646 | 2.6% worse | 0.16 |
| 21 | 166 | 0.2615 | 0.2815 | 7.7% worse | 0.25 |
| 63 | 165 | 0.3220 | 0.3346 | 3.9% worse | 0.09 |

QLIKE is a forecast error, so lower is better. Forecasts are monthly, 2004 through 2017.

## Question

Does the shape of the Treasury yield curve say anything about future S&P 500 volatility
that the VIX and recent realized volatility do not already say?

The idea: a flat or fast-moving curve reflects a changing growth or policy outlook, and
turbulence in rates can spill into equities. The competing view is that equity option
prices react to the same news on the same day, so the VIX already contains it.

## Features (`macro/features/yield_curve.py`)

| Column | Definition | What it measures |
|---|---|---|
| `t10y2y_level` | 10-year minus 2-year Treasury yield | Slope of the curve today |
| `t10y2y_change_20` | Slope minus its value 20 sessions earlier | How fast the curve is moving |
| `t10y2y_z_252` | Slope standardized by the previous 252 sessions (excludes *t*) | How unusual today's slope is |
| `dgs10_vol_20` | Standard deviation of daily 10-year yield changes, 20 sessions | How jumpy rates have been |

Inputs are T10Y2Y and DGS10 from FRED, read with `macro.data.series.load` (one-session
lag, carry of at most three sessions). Windows match the credit and MOVE signals.

T10Y3M is listed on the issue but not used. It tracks T10Y2Y closely, and with the
3-month rate near zero from late 2008 to late 2015 it was mostly the 10-year yield.

Candidates considered and dropped: an inversion flag (only two inversions in 2000–2017),
the short-rate level (near zero for about seven years), and the signed change in the
10-year yield (falling yields in 2008 and rising yields in 2013 both came with equity
volatility, so one coefficient cannot fit both).

## Pre-registered threshold

Posted on #12 before any run: advancing if the augmented model improves QLIKE by at least
2% over the RV + VIX benchmark at one or more horizons, with Clark–West p < 0.05. Any
effect was expected at h = 63, not h = 5.

No horizon meets either condition. The result would also fail the 5% at h = 21 bar used
by the credit and MOVE signals.

## How it was tested (`experiments/run_yield_curve.py`)

- **Benchmark.** A regression of log realized variance over the next *h* sessions on the
  HAR terms (log realized variance over the last 1, 5 and 22 sessions) and log VIX squared.
- **Augmented.** The same regression plus the four features.
- **Walk-forward.** Both models refit once a year on all data up to that point, then
  forecast the following year's month-end dates. No forecast uses later data.
- **Same rows.** Both models train on identical rows, so the only difference is the four
  extra columns.
- **Scores.** QLIKE on the variance forecasts, and a one-sided Clark–West test on the
  log-variance errors with Newey–West standard errors.

Training starts in 2000, where the features start, so the first forecasts are in 2004.

## Reading the result

**The forecasts got worse, not better.** The model must estimate four extra coefficients.
When the features carry little real information, those coefficients mostly fit noise, and
the noise costs accuracy on new data.

**There may be a faint signal at 63 days.** Clark–West asks whether the features hold any
information after allowing for the noise of estimating them. At h = 63 the statistic is
positive with p = 0.09: suggestive, not significant, and not enough to improve the
forecasts. This is the horizon where an effect was expected.

**This agrees with earlier work.** A quick literature check (papers not read in full)
found the term spread reported as useful mainly at 6 to 12 months, well beyond the
horizons tested here, with little out-of-sample gain at one month.

## Caveats

- **Stand-in evaluation.** The shared harness (#10) is not merged. This script follows
  the README protocol, but the benchmark specification is my reading of it. Rerun under
  #10 before the label is final.
- **Benchmark trained from 2000.** At h = 63 it scores 0.322, against 0.302 for the HAR
  baseline in `results/baselines.csv` over the same years, which trains from 1990. Both
  models share this handicap, so the comparison is fair, but the levels are not directly
  comparable with the baseline table.
- **Overlap with MOVE.** `dgs10_vol_20` is realized rates volatility and MOVE (#13) is
  implied rates volatility. If either advances, check them together.
- **Holiday effect.** On sessions where the previous yield is carried forward (Columbus
  Day, Veterans Day), the daily change is zero and the next is a two-day change. This is
  what was knowable on those sessions and is left as is.

## Checks

- `tests/test_yield_curve.py`: `assert_no_lookahead` at three cutoffs, hand-computed
  values for each feature, and a test that today's value is outside its own z-score
  window. Both a full-sample z-score and a window that includes today make the tests fail.
- `tests/test_run_yield_curve.py`: on simulated data, random features show no gain and a
  feature that leaks the target shows a large one; forecasts are unchanged when later data
  is removed.
- No holdout data (2018 onward) was loaded.

## Reproduce

```bash
uv run pytest
uv run python experiments/run_yield_curve.py   # prints the table, writes results/yield_curve.csv
```

## Possible follow-ups (exploratory, not part of the pre-registered test)

- Rerun under #10 when it merges.
- Longer horizons (126 and 252 sessions), where the published evidence is stronger.
- A single feature (the slope level alone) to cut the estimation noise.
