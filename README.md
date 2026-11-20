# Macro Team, Trading Club @ GT

The Macro Team is the global macro group within the Trading Club at Georgia Tech. The team covers rates, credit, currencies, sovereign risk and monetary policy, builds systematic signals from macro data, and tests them to the standards used in professional quantitative research. This repository holds the team's code, data definitions, research write-ups and results.

## Current work: macro signals for equity volatility

Fall 2026. We are testing whether macroeconomic and cross-asset stress indicators carry information about future S&P 500 volatility beyond what is already in the VIX and standard time-series models. The hypothesis is that stress appears in credit, rates and funding markets before it is fully priced in equity options.

Each member owns one signal. Every signal is evaluated under the same protocol: an increment to a benchmark that already includes realized volatility and the VIX, walk-forward and out of sample, scored by QLIKE with Clark-West tests for significance, with point-in-time data and a lookahead test required before merge. Development uses 2000 through 2017. The period from 2018 onward is held out and evaluated once at the end of the semester.

The signals that pass their pre-stated thresholds are combined into a single forecast, which drives a volatility-targeting strategy on the S&P 500 compared against the same rule on GARCH and VIX forecasts.

| Signal | Indicators | Source | Issue |
|---|---|---|---|
| Credit spreads | Baa minus 10-year Treasury, Baa minus Aaa | FRED | [#11](https://github.com/shreyas-shrestha/macro/issues/11) |
| Rates and yield curve | 10-year yield, 10y–2y and 10y–3m slopes | FRED | [#12](https://github.com/shreyas-shrestha/macro/issues/12) |
| Rates volatility | MOVE index | ICE via Yahoo | [#13](https://github.com/shreyas-shrestha/macro/issues/13) |
| Slow macro | NFCI, initial jobless claims | ALFRED vintages | [#14](https://github.com/shreyas-shrestha/macro/issues/14) |
| Funding and liquidity | 3-month commercial paper minus fed funds | FRED | [#15](https://github.com/shreyas-shrestha/macro/issues/15) |
| VIX term structure | VIX relative to VIX3M | CBOE | [#16](https://github.com/shreyas-shrestha/macro/issues/16) |
| EM sovereign spreads | EMBIG regional spreads | BCRP | [#17](https://github.com/shreyas-shrestha/macro/issues/17) |
| High-yield credit | TRACE HY and IG issuer-median spreads, HY minus IG | WRDS TRACE Enhanced, Compustat ratings, FRED H.15 | [#24](https://github.com/shreyas-shrestha/macro/issues/24) |

Infrastructure, combination, strategy and the final held-out evaluation are tracked on the [issues page](https://github.com/shreyas-shrestha/macro/issues). Proposals and write-ups are in `research/`.

## Getting started

```bash
git clone https://github.com/shreyas-shrestha/macro.git
cd macro
cp .env.example .env          # requires a FRED API key (free at fred.stlouisfed.org)
python data/data_check.py     # verifies data access
```

Environment setup and contribution guidelines are in `CONTRIBUTING.md` once the package scaffold is merged.
