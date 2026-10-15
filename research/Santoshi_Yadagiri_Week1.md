# Forecasting typical and high-end volatility from interest-rate conditions
**Santoshi Yadagiri | Trading @ GT**

## Signal

I’d like to test whether interest-rate conditions help forecast both typical equity volatility and how high volatility could reasonably get over the next month. The hypothesis is that rate instability may be especially useful for forecasting the upper end of future volatility, even if it adds little to the typical forecast.

I would use the 10-year Treasury yield from FRED (**DGS10**) and the 10-year minus 2-year Treasury spread (**T10Y2Y**). Features would include the 21-day change in the Treasury yield, the standard deviation of daily yield changes over that window, and the yield-curve level and 21-day change. These describe the direction and instability of rates without using MOVE or credit spreads. I would also include VIX (**VIXCLS**) and trailing 5-, 21-, and 63-day realized volatility. FRED data can be downloaded with `fredapi`, and equity prices for a team-selected index or ETF with `yfinance`.

## Model

I would use gradient-boosted trees with quantile loss to predict the median and 90th percentile of realized volatility over the next 21 trading days. The median estimates a typical outcome, while the 90th percentile provides an upper risk estimate. It is not a maximum or a guaranteed bound.

Unlike a regime model, the trees would learn nonlinear relationships directly from the features. For example, they could test whether rate instability matters more when VIX is already elevated or when the yield curve is inverted. I would limit tree depth and tune model settings on chronological validation data to reduce overfitting. Realized volatility would be the square root of 252 times the average squared daily log return over the relevant window.

## Test

I would use an expanding-window backtest, fitting on earlier observations and forecasting later ones. Training targets must end before each forecast date, and FRED inputs must be aligned to their actual availability. A final test period would be kept separate from model selection.

The main comparison would be identical quantile models using only VIX and recent equity volatility. I would also compare with linear quantile regression using the same inputs. GARCH(1,1) would provide a central volatility benchmark, with mean absolute error used to compare its forecast against the predicted median.

The main metric would be quantile, or pinball, loss at both forecast levels. For the 90th-percentile forecast, I would check whether realized volatility falls below it approximately 90% of the time, including during stressed periods, without making the estimate unnecessarily high. Success means lower held-out quantile loss than the market-only models, particularly at the upper quantile. Overlapping targets would be accounted for when assessing uncertainty.

If the upper forecast is useful, we could test using it to set conservative exposure limits. I would compare this with sizing from the median forecast and recent volatility, including trading costs, drawdowns, cash returns, and missed upside.

## Why It Might Fail

VIX and recent volatility may already capture the useful rate information. Trees can overfit historical episodes, and the limited number of extreme events makes upper-quantile forecasts difficult to validate. The two forecasts could also cross, so we would check that the upper estimate stays above the median. Finally, an overly cautious forecast could reduce exposure too often and hurt returns despite looking safer.

## References

FRED: [DGS10](https://fred.stlouisfed.org/series/DGS10), [T10Y2Y](https://fred.stlouisfed.org/series/T10Y2Y), and [VIXCLS](https://fred.stlouisfed.org/series/VIXCLS).

[Scikit-learn: quantile regression with gradient boosting](https://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_quantile.html).
