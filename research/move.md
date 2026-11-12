## MOVE and S&P 500 variance
Treasury option-implied volatility may capture rate uncertainty that carries
over to equities after accounting for VIX, realized variance and BAA10Y credit
spread features.

Use Yahoo daily MOVE close with a one-session lag, its 20-session point
change and a z-score estimated from the preceding 252 NYSE sessions with at least 245 available levels. See
[data timing and gaps](../data/README.md) for alignment and filling.

At each horizon (5, 21 and 63 sessions), compare VIX + RV with and without
MOVE. Repeat with credit features in both models. For each nested pair,
score both forecasts on identical decision dates. Report QLIKE and the
one-sided Clark–West p-value for each pair.

The proposed advancement threshold is at least 5% lower QLIKE and a
Clark–West p-value below 0.10 at 21 sessions for the primary VIX + RV
comparison. The credit comparison tests whether any gain survives adding
credit. Post the threshold to issue #13 before running forecasts.

The Yahoo pull returned 3,695 closes from January 2, 2003 through December
29, 2017, with 81 missing closes across 3,776 NYSE sessions. The longest
gap was five sessions, March 18–22, 2013. The [diagnostic plot](move_diagnostics.png)
shows raw closes, 20-session changes and missing sessions before filling.

The 81 missing sessions form 64 one-session gaps, six two-session gaps and
one five-session gap. Requiring 252 complete prior levels after one-session
forward-fill left 2,385 complete feature rows and removed all of 2013 and
2015–2017. Allowing up to seven unavailable levels in the rolling window
retained 3,504 complete rows, including at least 242 per year in 2013–2017.
With the same rolling rule, no fill retained 2,552 rows and a two-session
fill retained 3,518. The one-session rule keeps nearly all of the coverage
of two-session fill while carrying fewer stale closes. These checks use
feature availability only.

For the scorecard, train and score each nested pair on matching eligible
rows. Use a common evaluation sample across both pairs for the headline
comparison if credit coverage differs. Keep 2018 onward locked.

Scorecard and threshold status are pending the shared harness and credit
features. There is no empirical result yet on whether MOVE adds information.
