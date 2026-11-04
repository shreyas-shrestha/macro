# Macro volatility project

Trading Club at GT, macro team, fall 2026.

We are trying to predict how volatile the S&P 500 is going to be over the next month or so, using macro data instead of just the stock market itself.

Volatility is already fairly predictable. When the market has been jumpy it usually stays jumpy for a while, and the VIX tells you what options traders expect. Anything we build has to do better than that or there is no point. The idea we are testing is that trouble tends to show up in credit and bond markets a bit before it shows up in stocks. Credit spreads widen, the yield curve inverts, it gets more expensive for banks to borrow, and a few weeks later the stock market gets rough. Whether that lead really exists in the data, and whether it is big enough to matter once you already have the VIX, is what the semester is about.

Each person on the team owns one signal. You take a piece of macro data, turn it into a daily series, and test whether adding it to a standard volatility model makes the forecasts better. Everyone runs the same test on the same dates so we can line the results up at the end. Some of the signals will not work. We will write those up too.

Once we know which signals help, we combine them into one forecast and use it to decide how much S&P 500 to hold. Less when we expect a rough month, more when we expect a quiet one. Then we check whether that did better than making the same decision off the VIX alone.

There is one rule we are strict about. All of the development happens on data from 2000 through 2017. Data from 2018 onward is off limits until the end of the semester, when we run everything on it once. That final run is the result.

## The signals

- Credit spreads ([#11](https://github.com/shreyas-shrestha/macro/issues/11)). The gap between what companies pay to borrow and what the government pays.
- Rates and the yield curve ([#12](https://github.com/shreyas-shrestha/macro/issues/12)). The level of Treasury yields and the shape of the curve.
- Treasury volatility ([#13](https://github.com/shreyas-shrestha/macro/issues/13)). The MOVE index, roughly the VIX for bonds.
- Slow macro ([#14](https://github.com/shreyas-shrestha/macro/issues/14)). The Chicago Fed financial conditions index and weekly jobless claims.
- Funding ([#15](https://github.com/shreyas-shrestha/macro/issues/15)). Commercial paper rates relative to the fed funds rate, which spike when banks stop lending to each other.
- VIX term structure ([#16](https://github.com/shreyas-shrestha/macro/issues/16)). One-month VIX against three-month VIX.
- Emerging market spreads ([#17](https://github.com/shreyas-shrestha/macro/issues/17)). What emerging market governments pay to borrow in dollars.

The data pipeline, baselines, backtester and strategy work are on the [issues page](https://github.com/shreyas-shrestha/macro/issues) as well.

## Getting started

```bash
git clone https://github.com/shreyas-shrestha/macro.git
cd macro
cp .env.example .env          # add a free FRED API key from fred.stlouisfed.org
python data/data_check.py     # checks the data pulls work
```

Setup and contribution notes will go in `CONTRIBUTING.md` once the code scaffold is merged. Research proposals and write-ups are in the `research` folder.
