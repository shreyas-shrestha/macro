# Signal: credit spreads (BAA10Y)

Issue #11 · Owner: Sahil Ravani · Status: **features and diagnostics done, awaiting harness (#31)**

## Hypothesis

Corporate credit spreads price default risk and the risk premium on it. Bond investors
often reprice credit risk before equity option sellers do. If so, a high or rising
Baa–Treasury spread should predict higher S&P 500 realized variance over the next month,
beyond what is already in the VIX and trailing realized volatility. The competing view
is that credit and equity volatility respond to the same shocks on the same day. In that
case the VIX already contains the spread's information and the increment is about zero.

## Data

BAA10Y, DBAA and DAAA from FRED through `macro.data.series.load`, which stamps each value
by FRED's posting day and uses it one session later. A value observed on *s* normally
enters at *s* + 2, because FRED can post after the 16:00 ET close. ALFRED shows no
revisions (0 of 943 BAA10Y and DBAA observations, 2014–2017). Gaps are single
bond-market holidays only (73 sessions since 1986, none longer than one day).
Moody's yields for 2016-10-07 to 2017-03-15 reached FRED late, all on 2017-03-22. The
robustness run blanks them (`Credit(exclude_backfill=True)`).
FRED now serves only three years of the ICE BofA HY OAS (BAMLH0A0HYM2), so it is not
usable for 2000–2017. #24 covers HY spreads from TRACE.

## Features (`macro/features/credit.py`)

| Column | Definition |
|---|---|
| `baa10y_level` | BAA10Y, percentage points |
| `baa10y_change_20` | BAA10Y minus its value 20 sessions earlier |
| `baa10y_z_252` | Level standardized by mean and std of the previous 252 sessions (excludes *t*) |
| `baa_aaa_level` | DBAA − DAAA: default-risk spread that does not move with Treasury yields |

Four features, all point-in-time. `tests/test_credit.py` serves synthetic data through
the real loader. It runs `assert_no_lookahead` at three cutoffs (and on the backfill
variant), and checks the two-session lag and hand-computed change and z-score values.

## Pre-registered threshold

Posted on #11 before any run: at h = 21, QLIKE at least 5% lower than the VIX + RV
no-signal model, with Clark–West p < 0.10 (Newey–West lag 21). h = 5 and h = 63 are
reported but do not decide the label. The "credit + MOVE" run is reported alongside and
does not change the label.

## Scorecard

*Pending #31* (the harness). Runs at h ∈ {5, 21, 63}, development sample 2000–2017
only: (1) VIX + RV baseline vs. + credit; (2) the same without the 2016–17 backfill.
Still to arrange with #31/#13: (3) + MOVE, (4) + credit + MOVE.

| Model | h | QLIKE | Δ vs baseline | R²_OOS | CW p |
|---|---|---|---|---|---|
| — | — | — | — | — | — |

## Reading

*To be written after the runs.* Points to address: whether any gain comes from 2008
alone (check the by-episode breakdown), whether level or change carries it, and how
much is left once MOVE is included. #13 suggests credit and rates volatility are one
factor.
