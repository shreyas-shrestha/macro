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


## MOVE data and diagnostic plot

Source: [Yahoo Finance ^MOVE historical prices](https://finance.yahoo.com/quote/%5EMOVE/history/). The plot uses unadjusted daily closes from 2003 through 2017.

[View the diagnostic plot](move_diagnostics.png)

The three panels show the daily close, the change in index points from 20 NYSE sessions earlier, and the NYSE sessions with no Yahoo close. The plot uses raw closes before applying the signal's release lag or forward-fill.

### Reproduce the plot

Run this from the repository root, where `pyproject.toml` is located, with `macro/features/move.py` present on your local branch.

```bash
uv run --with matplotlib python - <<'PY'
import matplotlib.pyplot as plt
import pandas as pd

from macro.data.calendar import nyse_sessions
from macro.features.move import load_move

start = pd.Timestamp("2003-01-01")
end = pd.Timestamp("2017-12-31")
close = load_move(start, end).reindex(nyse_sessions(start, end))
change = close.diff(20)
missing_dates = close.index[close.isna()]

fig, axes = plt.subplots(
    3, 1, sharex=True, figsize=(10, 6),
    layout="constrained", height_ratios=[2, 2, 1],
)

axes[0].set_title(
    "Yahoo Finance MOVE | Development sample, 2003–2017",
    loc="left", weight="bold",
)
axes[0].plot(close.index, close, color="#24537b", linewidth=0.85)
axes[0].set_ylabel("Daily close")

axes[1].plot(change.index, change, color="#197971", linewidth=0.75)
axes[1].axhline(0, color="#748291", linewidth=0.6)
axes[1].set_ylabel("20-session change")

axes[2].eventplot(
    missing_dates, lineoffsets=0.5, linelengths=0.7, colors="#bd4a2d",
)
axes[2].set_ylim(0, 1)
axes[2].set_yticks([])
axes[2].set_ylabel("Missing close")

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)

fig.savefig("research/move_diagnostics.png", dpi=150)
```
