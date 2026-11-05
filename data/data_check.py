# Quick test that we can pull data. Needs FRED_API_KEY in .env (see .env.example).
# Run with: uv run python data/data_check.py
import os

from dotenv import load_dotenv

# Load .env before importing the HTTP clients so SSL_CERT_FILE reaches OpenSSL.
load_dotenv()

import yfinance as yf  # noqa: E402
from fredapi import Fred  # noqa: E402

fred = Fred(api_key=os.getenv("FRED_API_KEY"))

spx = yf.download("^GSPC", start="2000-01-01")["Close"]
vix = yf.download("^VIX", start="2000-01-01")["Close"]
credit = fred.get_series("BAA10Y", observation_start="2000-01-01")

print(spx.tail())
print(vix.tail())
print(credit.tail())
