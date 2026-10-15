# Can Macro Data Improve Equity Volatility Forecasts?
Sahil Ravani

## Signal
Use daily S&P 500 prices (^GSPC) and VIX (^VIX) from Yahoo Finance through yfinance, starting in 2000. Add two FRED indicators through fredapi: the 10-year minus 2-year Treasury yield spread (T10Y2Y) and the Baa corporate yield minus 10-year Treasury yield spread (BAA10Y). Use each macro indicator's level and five-session change.

The hypothesis is that credit spreads and interest-rate conditions provide information about future volatility beyond what recent equity returns and VIX already capture. Widening credit spreads may signal increasing financial stress; the yield curve may reflect changing economic expectations.

## Model
Build a ridge regression to forecast the log of next-week S&P 500 variance. Define the target as the average squared daily log return over the next five trading sessions, multiplied by 252 to annualize it.

Inputs will include trailing average squared returns over 1, 5, and 22 sessions, VIX, and the macro features. Ridge regression provides an interpretable starting point and limits overfitting. Fit feature scaling on training data only. Convert predictions back to variance using a training-only correction for log-transformation bias.

Generate forecasts after the market close. Initially lag macro observations by two trading sessions, verify historical publication timing, and increase the lag if needed. Use only past information when filling missing observations.

## Test
Use 2000–2009 for initial training, 2010–2014 for chronological validation, and 2015–2025 for the final test. Freeze model choices before evaluating the test period, then refit monthly using only labels whose full five-session outcomes are already observed.

Compare against:
- A trailing 22-session historical variance forecast.
- GARCH(1,1), with five daily variance forecasts aggregated to the same target.
- The identical ridge model without macro features, retaining VIX.

Use QLIKE as the primary variance-forecast metric and volatility RMSE as a secondary metric. A proposed success threshold is at least 5% lower test QLIKE than both GARCH and the no-macro model. Assess uncertainty with a paired block bootstrap that accounts for overlapping five-session targets, and report performance across years and stress periods.

If forecasting improves, test a capped SPY/cash volatility-targeting strategy that reduces exposure when predicted volatility rises. Execute after the signal is available and compare net Sharpe ratio, maximum drawdown, and turnover against the same strategy using baseline forecasts.

## Why It Might Fail
Macro indicators may describe stress after it has started rather than predict it. Recent returns and VIX may already contain the useful information. Few independent crises, changing market relationships, and publication-time leakage could also create misleading results. Stronger forecasting accuracy may not translate into better strategy performance after trading costs.

## Sources
- Two Sigma, "A Machine Learning Approach to Regime Modeling":
  https://www.twosigma.com/wp-content/uploads/2021/10/Machine-Learning-Approach-to-Regime-Modeling_.pdf
  Motivation for studying market conditions; this proposal uses regression rather than the paper's Gaussian mixture model.
- FRED yield curve: https://fred.stlouisfed.org/series/T10Y2Y
- FRED credit spread: https://fred.stlouisfed.org/series/BAA10Y
