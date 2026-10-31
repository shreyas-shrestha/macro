# Macro Volatility Research — Trading Club @ GT

Can macro stress signals forecast S&P 500 volatility beyond what VIX and GARCH already know, and does a better forecast change how much risk you should hold?

This repo is the Macro team's semester project. It is built the way a research pod would build it: one target, one evaluation harness, one locked holdout, and many competing signals that each plug into the same pipeline.

## The question

Volatility clusters and mean-reverts, so recent realized vol and VIX already forecast it well. Our hypothesis is that stress shows up in credit, rates, funding and macro data before it is fully priced in equity options. Each signal tests one slice of that hypothesis. Several will fail. That is a finding.

## Approach

**Target.** Realized variance of S&P 500 log returns over the next 5, 21 and 63 trading days, annualized. Models work in log variance. Defined once in `macro/targets.py`.

**Baselines.** Trailing 21-day realized vol, HAR-RV, GARCH(1,1) refit on an expanding window, and VIX². Every signal has to beat these.

**Signals.** One owner each. A signal is a module that returns a daily feature frame (level, change, rolling z-score) with a declared source, start date and release lag. Nothing in a feature may use information unavailable at that day's close, and a test enforces it.

**Evaluation.** Every signal is scored the same way: baseline features plus signal features versus baseline alone, expanding window, annual refit, embargo equal to the horizon. Primary loss is QLIKE on variance. Significance by Clark–West with Newey–West errors. Results broken down by year and by stress episode.

**Development vs holdout.** All development uses 2000–2017. **2018 to present is locked.** It is run once, at the end of the semester, for every signal and the combination together. The harness refuses to run past 2017 without an explicit `--holdout` flag.

**Combination.** Signals that meet their pre-registered threshold are combined by equal-weight averaging and by a constrained stack, then scored like any other signal.

**Strategy.** The combined forecast drives a vol-targeting position in SPY with a leverage cap and transaction costs, compared against the same rule on GARCH and VIX forecasts. Vol selling is studied only through a variance-risk-premium proxy, since we have no options data.

## Signals

| Signal | Series | Source | Issue |
|---|---|---|---|
| Credit spreads | BAA10Y, Baa−Aaa | FRED | [#11](https://github.com/shreyas-shrestha/macro/issues/11) |
| Rates and yield curve | DGS10, T10Y2Y, T10Y3M | FRED | [#12](https://github.com/shreyas-shrestha/macro/issues/12) |
| Rates volatility | MOVE | Yahoo | [#13](https://github.com/shreyas-shrestha/macro/issues/13) |
| Slow macro | NFCI, ICSA | ALFRED vintages | [#14](https://github.com/shreyas-shrestha/macro/issues/14) |
| Funding and liquidity | CPFF | FRED | [#15](https://github.com/shreyas-shrestha/macro/issues/15) |
| VIX term structure | VIX, VIX3M | CBOE | [#16](https://github.com/shreyas-shrestha/macro/issues/16) |
| EM sovereign spreads | EMBIG | BCRP | [#17](https://github.com/shreyas-shrestha/macro/issues/17) |

Infrastructure: [#7](https://github.com/shreyas-shrestha/macro/issues/7) scaffold, [#8](https://github.com/shreyas-shrestha/macro/issues/8) data layer, [#9](https://github.com/shreyas-shrestha/macro/issues/9) target and baselines, [#10](https://github.com/shreyas-shrestha/macro/issues/10) harness. Then [#18](https://github.com/shreyas-shrestha/macro/issues/18) combination, [#19](https://github.com/shreyas-shrestha/macro/issues/19) strategy, [#20](https://github.com/shreyas-shrestha/macro/issues/20) holdout and final report.

## Repo layout

```
macro/            installable package
  data/           loaders (FRED, ALFRED, Yahoo, CBOE, BCRP), parquet cache, point-in-time alignment
  features/       one module per signal, all implementing features/base.py
  targets.py      realized variance definition and horizons
  models/         baselines, linear, HMM, quantile GBM, combination
  backtest/       walk-forward splits, scoring, standard report
  strategy/       vol targeting, costs, VRP proxy
configs/          one YAML per experiment
experiments/      one script per experiment; writes to results/
results/          tracked CSV summaries and figures (no raw data)
tests/            lookahead, target and scoring tests
research/         proposals and one-page write-ups per signal
data/             raw and cached data, gitignored
```

The package skeleton lands with [#7](https://github.com/shreyas-shrestha/macro/issues/7). Until then only `research/` and `data/data_check.py` exist.

## Getting started

```bash
git clone https://github.com/shreyas-shrestha/macro.git
cd macro
uv sync                      # once pyproject.toml exists
cp .env.example .env         # add your FRED_API_KEY (free at fred.stlouisfed.org)
python data/data_check.py    # confirms FRED and Yahoo pulls work
```

## How we work

- One issue, one owner, one branch, one pull request, one review. `main` is protected.
- Claim an issue by assigning yourself and posting in the team channel.
- Before running anything, post your pre-registered success threshold as a comment on your issue. Default: at least 5% lower QLIKE than the no-signal model at h = 21 with Clark–West p < 0.10.
- **Definition of done for a signal:** feature module with lookahead test, experiment config, forecast CSVs for all three horizons, scorecard row, one-page write-up in `research/`. Then label the issue `advancing` or `not-advancing`.
- Do not touch data after 2017-12-31 for anything except the final holdout run.
- Notebooks are for exploration only and are stripped of output before commit. Anything that produces a result lives in the package.

## Results

Scorecard to be filled in as signals complete. Headline holdout numbers (combination QLIKE improvement over GARCH, Clark–West p-value, vol-targeting Sharpe difference) will appear here after [#20](https://github.com/shreyas-shrestha/macro/issues/20).

## References

Two Sigma (2021), *A Machine Learning Approach to Regime Modeling* · Corsi (2009), HAR-RV · Patton (2011), robust loss functions for volatility forecasts · Clark & West (2007), nested model tests · Moreira & Muir (2017), *Volatility-Managed Portfolios* · Paye (2012), macro variables and volatility · Gilchrist & Zakrajšek (2012), excess bond premium
