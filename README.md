# Macro Team, Trading Club @ GT

Research repository of the Macro Team at the Trading Club at Georgia Tech.

## About the team

The Macro Team studies how macroeconomic conditions, monetary policy and cross-asset stress transmit into asset prices, and whether that transmission can be measured and forecast with the discipline expected of professional quantitative research. Work is organized as research programs. Each program states a hypothesis, fixes an evaluation protocol before results are produced, and ends in a reproducible set of results and a written report. Members own individual signals or infrastructure components within a program, and all code, data definitions and results live in this repository.

## Current program: macro signals for equity volatility (Fall 2026)

This program evaluates whether macroeconomic and cross-asset stress indicators contain information about future S&P 500 volatility beyond what is already embedded in implied volatility and standard time-series models. The work is organized as a set of independent signals tested under a common protocol, followed by a combined forecast and a volatility-managed strategy evaluated on a held-out period.

### Research question

Equity volatility is persistent, and the VIX provides a forward-looking market estimate of it. Both are strong benchmarks. The hypothesis under test is that financial stress often becomes visible in credit, rates and funding markets before it is fully reflected in equity option prices, and that indicators from those markets therefore improve volatility forecasts at horizons of one to three months. Each signal in the project addresses one component of this hypothesis. A negative result, where a signal adds no information conditional on the benchmarks, is treated as a valid and reportable finding.

### Methodology

The forecast target is the realized variance of S&P 500 daily log returns over the subsequent 5, 21 and 63 trading days, annualized. Models are estimated in log variance and forecasts are converted back with a training-sample bias correction.

Benchmarks are trailing 21-day realized volatility, the HAR-RV model of Corsi (2009), a GARCH(1,1) model re-estimated on an expanding window, and the squared VIX scaled to the forecast horizon. Every signal is evaluated as an increment to a benchmark specification that already includes recent realized volatility and the VIX.

Each signal is implemented as a module that produces a daily feature set from a declared data source with a declared release lag, such that no feature value depends on information unavailable at the close of the decision date. Point-in-time alignment is handled centrally in the data layer, and a lookahead test is required for every signal before it is merged.

Evaluation is walk-forward and out of sample, with model re-estimation at annual intervals and an embargo equal to the forecast horizon between the training and forecast dates. The primary loss is QLIKE on variance, which is robust to measurement error in the realized variance proxy (Patton 2011). Incremental predictive content is assessed with the Clark and West (2007) test for nested models using Newey-West standard errors, and results are reported by calendar year and by stress episode.

Model development and selection use data from 2000 through 2017. The period from 2018 to the present is a held-out sample that is not accessed during development. All signals, the combination and the strategy are evaluated on it once, at the end of the semester.

### Signals

| Signal | Indicators | Source | Issue |
|---|---|---|---|
| Credit spreads | Baa minus 10-year Treasury, Baa minus Aaa | FRED | [#11](https://github.com/shreyas-shrestha/macro/issues/11) |
| Rates and yield curve | 10-year yield, 10y–2y and 10y–3m slopes | FRED | [#12](https://github.com/shreyas-shrestha/macro/issues/12) |
| Rates volatility | MOVE index | ICE via Yahoo | [#13](https://github.com/shreyas-shrestha/macro/issues/13) |
| Slow macro | NFCI, initial jobless claims | ALFRED vintages | [#14](https://github.com/shreyas-shrestha/macro/issues/14) |
| Funding and liquidity | 3-month commercial paper minus fed funds | FRED | [#15](https://github.com/shreyas-shrestha/macro/issues/15) |
| VIX term structure | VIX relative to VIX3M | CBOE | [#16](https://github.com/shreyas-shrestha/macro/issues/16) |
| EM sovereign spreads | EMBIG regional spreads | BCRP | [#17](https://github.com/shreyas-shrestha/macro/issues/17) |

### Deliverables

The program produces three results. The first is a scorecard in which every signal is evaluated under the protocol above and classified as advancing or not advancing against a threshold stated by its owner before any results were produced. The second is a combined forecast built from the advancing signals, using equal weighting and a constrained linear stack, scored against the best individual signal and the benchmarks. The third is a volatility-targeting strategy on the S&P 500 driven by the combined forecast, with a leverage cap and transaction costs, compared with the same rule driven by the GARCH and VIX forecasts. The headline figures are the held-out QLIKE improvement of the combination over GARCH, its Clark-West p-value, and the difference in Sharpe ratio between the strategies.

Supporting work, including the data layer, benchmarks, backtest harness, combination and the final held-out evaluation, is tracked on the [issues page](https://github.com/shreyas-shrestha/macro/issues).

## Getting started

```bash
git clone https://github.com/shreyas-shrestha/macro.git
cd macro
cp .env.example .env          # requires a FRED API key (free at fred.stlouisfed.org)
python data/data_check.py     # verifies data access
```

Environment setup, repository conventions and the definition of done for a signal are documented in `CONTRIBUTING.md` once the package scaffold is merged. Research proposals and signal write-ups are in `research/`.

## References

Clark, T. and West, K. (2007). Approximately normal tests for equal predictive accuracy in nested models. *Journal of Econometrics*.
Corsi, F. (2009). A simple approximate long-memory model of realized volatility. *Journal of Financial Econometrics*.
Moreira, A. and Muir, T. (2017). Volatility-managed portfolios. *Journal of Finance*.
Patton, A. (2011). Volatility forecast comparison using imperfect volatility proxies. *Journal of Econometrics*.
Paye, B. (2012). Déjà vol: Predictive regressions for aggregate stock market volatility using macroeconomic variables. *Journal of Financial Economics*.
