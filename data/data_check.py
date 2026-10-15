# Quick test that we can pull data. Needs FRED_API_KEY in a .env file.
import os
import ssl

import certifi
import yfinance as yf
from dotenv import load_dotenv
from fredapi import Fred

# fixes SSL errors on Mac
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

load_dotenv()
fred = Fred(api_key=os.getenv("FRED_API_KEY"))

spx = yf.download("^GSPC", start="2000-01-01")["Close"]
vix = yf.download("^VIX", start="2000-01-01")["Close"]
credit = fred.get_series("BAA10Y", observation_start="2000-01-01")

print(spx.tail())
print(vix.tail())
print(credit.tail())
