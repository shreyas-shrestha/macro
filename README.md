# Macro Volatility Project

Trading Club @ GT, Macro team, Fall 2026.

## The goal

We want to predict how volatile the S&P 500 will be over the next month or so, using macro data.

Predicting volatility is not new. If the market has been jumpy recently, it will probably stay jumpy, and VIX already tells you what the options market expects. So the real question is:

**Does macro data (credit spreads, interest rates, funding markets, jobless claims, etc.) tell us anything about future volatility that VIX and standard models don't already know?**

Our bet is that stress tends to show up in bond and credit markets before it shows up in stocks. If that's true, we should be able to measure it. If it's not true, we want to be able to show that too.

## What we're building

1. **Signals.** Each person owns one macro signal and tests whether it helps predict volatility. Everyone's signal gets tested the exact same way so the results are comparable.
2. **A combined forecast.** We take the signals that actually work and combine them into one forecast.
3. **A strategy.** We use that forecast to decide how much S&P 500 to hold: less when we expect high volatility, more when we expect calm. Then we check whether it beats doing the same thing with just VIX or a standard model.

Everything gets built and tested on data from 2000 to 2017. At the end of the semester we run it once on 2018 to today, which nobody is allowed to look at before then. That final result is what we show people.

## The signals

| Signal | What it is |
|---|---|
| [Credit spreads](https://github.com/shreyas-shrestha/macro/issues/11) | How much more companies pay to borrow than the government |
| [Rates and yield curve](https://github.com/shreyas-shrestha/macro/issues/12) | Where Treasury yields are and how the curve is shaped |
| [Rates volatility](https://github.com/shreyas-shrestha/macro/issues/13) | How jumpy the Treasury market is (MOVE index) |
| [Slow macro](https://github.com/shreyas-shrestha/macro/issues/14) | Financial conditions index and jobless claims |
| [Funding and liquidity](https://github.com/shreyas-shrestha/macro/issues/15) | How expensive short-term borrowing is for banks and companies |
| [VIX term structure](https://github.com/shreyas-shrestha/macro/issues/16) | 1-month VIX vs 3-month VIX |
| [EM sovereign spreads](https://github.com/shreyas-shrestha/macro/issues/17) | Risk on emerging-market government bonds |

The rest of the work (data pipeline, baselines, backtesting, combining, strategy) is on the [issues page](https://github.com/shreyas-shrestha/macro/issues).

## Getting started

```bash
git clone https://github.com/shreyas-shrestha/macro.git
cd macro
cp .env.example .env          # add a free FRED API key from fred.stlouisfed.org
python data/data_check.py     # checks the data pulls work
```

Setup details and how to contribute will be in `CONTRIBUTING.md` once the code scaffold is in. Research write-ups are in `research/`.
