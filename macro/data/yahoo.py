"""Daily closes from Yahoo Finance."""

import pandas as pd
import yfinance as yf


def fetch(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Closes for ``ticker`` dated ``start`` through ``end`` inclusive."""
    # Yahoo's end date is exclusive.
    frame = yf.download(
        ticker,
        start=start,
        end=end + pd.Timedelta(days=1),
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )
    if frame.empty:
        raise RuntimeError(f"Yahoo returned no data for {ticker}")
    return frame["Close"].astype(float).loc[start:end]
