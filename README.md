# Macro Volatility Research

Trading Club @ GT, Macro team. Fall 2026.

## What we're trying to find out

Everyone knows volatility clusters. If the market was wild last month it will probably be wild next month, and VIX already tells you what options traders expect. Our question is whether the macro picture adds anything on top of that.

Specifically: do credit spreads, the yield curve, Treasury volatility, funding stress, slow-moving economic data and emerging-market spreads tell you something about S&P 500 volatility over the next one to three months that VIX and a GARCH model don't already know?

The intuition is that stress often shows up in credit and rates markets before it fully shows up in equity options. If that's true, it should be measurable. If it isn't, we want to be able to say so with evidence.

## What the end result looks like

By the end of the semester we want three things:

1. **A scorecard.** Every signal tested the same way, on the same data, against the same baselines, with a clear yes or no on whether it added information. We expect some to fail. A clean negative result is still a result.
2. **A combined forecast.** The signals that survive are combined into one volatility forecast, and we show whether that beats the best single signal and the standard benchmarks.
3. **A strategy.** That forecast drives how much S&P 500 exposure to hold, scaled down when predicted volatility is high. We show whether the better forecast actually produces a better risk-adjusted return after costs, compared with doing the same thing off VIX or GARCH alone.

All of this is tested once, at the end, on data from 2018 onward that nobody touches during development. That's the number we'd show someone who works at a real shop.

## How it fits together

- **One target.** Realized S&P 500 variance over the next 5, 21 and 63 trading days.
- **One set of baselines.** Trailing realized vol, HAR-RV, GARCH(1,1), VIX. Every signal has to beat these.
- **One signal per person.** Each is a small module that produces daily features using only information that was actually available on that day.
- **One evaluation.** Walk-forward, out of sample, same metrics for everyone. Development on 2000–2017, holdout 2018 to present.

## The signals

| Signal | What it measures | Data |
|---|---|---|
| [Credit spreads](https://github.com/shreyas-shrestha/macro/issues/11) | How much extra yield risky companies pay | BAA10Y, FRED |
| [Rates and yield curve](https://github.com/shreyas-shrestha/macro/issues/12) | Level, slope and instability of Treasury yields | DGS10, T10Y2Y, T10Y3M, FRED |
| [Rates volatility](https://github.com/shreyas-shrestha/macro/issues/13) | Implied volatility in the Treasury market | MOVE |
| [Slow macro](https://github.com/shreyas-shrestha/macro/issues/14) | Financial conditions and jobless claims | NFCI, ICSA, FRED |
| [Funding and liquidity](https://github.com/shreyas-shrestha/macro/issues/15) | Cost of short-term borrowing vs the Fed rate | CPFF, FRED |
| [VIX term structure](https://github.com/shreyas-shrestha/macro/issues/16) | Near-term vs three-month implied vol | VIX, VIX3M, CBOE |
| [EM sovereign spreads](https://github.com/shreyas-shrestha/macro/issues/17) | Emerging-market sovereign risk | EMBIG, BCRP |

The supporting work (data layer, target and baselines, backtest harness, combination, strategy and the final holdout run) is tracked in the [issues](https://github.com/shreyas-shrestha/macro/issues).

## Getting started

```bash
git clone https://github.com/shreyas-shrestha/macro.git
cd macro
cp .env.example .env          # add a free FRED API key from fred.stlouisfed.org
python data/data_check.py     # confirms the data pulls work
```

Project setup and conventions are in `CONTRIBUTING.md` once the package scaffold lands. Research proposals and write-ups live in `research/`.
