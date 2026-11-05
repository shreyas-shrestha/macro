"""Signal combination and vol-targeting strategy.

Contract: combines signals that passed their pre-registered thresholds into one
forecast and sizes an S&P 500 position to a volatility target, compared against the
same rule on GARCH and VIX forecasts. Filled by #18 and #19.
"""
