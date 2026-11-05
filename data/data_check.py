# Quick test that we can pull data. Needs FRED_API_KEY in .env (see .env.example).
# Run with: uv run python data/data_check.py
# Requests stop at DEV_END so the 2018+ holdout is never loaded.
import os

from macro.config import DEV_END, DEV_START, HOLDOUT_START, load_env

# Load .env before importing the HTTP clients so SSL_CERT_FILE reaches OpenSSL.
load_env()

import yfinance as yf  # noqa: E402
from fredapi import Fred  # noqa: E402

fred = Fred(api_key=os.getenv("FRED_API_KEY"))

# yfinance's end date is exclusive; FRED's observation_end is inclusive.
spx = yf.download("^GSPC", start=DEV_START, end=HOLDOUT_START)["Close"]
vix = yf.download("^VIX", start=DEV_START, end=HOLDOUT_START)["Close"]
credit = fred.get_series("BAA10Y", observation_start=DEV_START, observation_end=DEV_END)

# Slice again in case a provider returns rows outside the requested range.
for series in (spx, vix, credit):
    print(series.loc[DEV_START:DEV_END].tail())
