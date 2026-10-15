# Can Macro Regimes Tell Us When to Stop Selling Volatility?
Alex Epstein

## Idea
Options are usually priced for more volatility than actually happens. That gap is called the variance risk premium, and it's why selling options makes money most of the time. The problem is that every few years (2008, 2018, 2020) the market moves way more than expected and option sellers lose a lot at once.

My question is whether macro conditions can tell us when we're in a "stressed" regime where that premium disappears, so we know when to stop selling.

## Signal
- **Credit spread** (`BAA10Y`, FRED): extra yield risky companies pay over Treasuries. Goes up when people worry about defaults.
- **Yield curve** (`T10Y3M`, FRED): long-term minus short-term rates. Goes negative before recessions.
- **Funding spread** (`CPFF`, FRED): short-term borrowing costs vs. the Fed rate. Spikes when lending dries up.
- **MOVE** (yfinance): the VIX for Treasury bonds.
- **VIX and S&P 500** (yfinance): to measure the premium itself.

All of these are daily market data, so no issues with data getting revised later. The data starts in 2003 because that's when MOVE starts on Yahoo. I tried the high-yield spread too, but FRED only gives 3 years of it now.

## Model
A hidden Markov model (HMM) with 2 or 3 states, trained only on the macro data. It outputs the probability that we're in a stressed regime each day. I'm leaving VIX out of the HMM on purpose, so that if it works, we know it's the macro data helping and not just VIX.

## Test
1. Is the premium (VIX vs. the volatility that actually happens over the next month) smaller or negative in the stressed regime, even after accounting for VIX?
2. Backtest a simple strategy that sells volatility each month and sizes down when the stress probability is high. Compare it to:
   - always selling
   - stopping when VIX is high
   - stopping when a GARCH forecast says vol is coming

Success means smaller drawdowns and smaller worst-case months than the other strategies, without a worse Sharpe ratio. The model is trained on 2003–2009 and tested on later years it hasn't seen.

## Why It Might Fail
- There are only about 6 real volatility blowups since 2003, so the model could easily overfit.
- Some crashes (Feb 2018, Mar 2020) happened fast, and macro data only reacted after. The model might flag stress too late.
- VIX might already include everything the macro data knows.

## Sources
- Two Sigma, "A Machine Learning Approach to Regime Modeling"
- Carr & Wu, "Variance Risk Premiums" (2009)
- FRED: [BAA10Y](https://fred.stlouisfed.org/series/BAA10Y), [T10Y3M](https://fred.stlouisfed.org/series/T10Y3M), [CPFF](https://fred.stlouisfed.org/series/CPFF)
