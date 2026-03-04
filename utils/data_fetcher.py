"""
utils/data_fetcher.py
---------------------
Purpose:
    Provides functions to fetch historical stock price data from Yahoo Finance
    for both Indian (NSE) and US stock markets. Used by app.py and the test suite.

Dependencies:
    - yfinance: The Yahoo Finance Python library used to pull OHLCV data.

Functions:
    - fetch_indian_stock(ticker, period): Fetches NSE-listed stock data.
    - fetch_us_stock(ticker, period): Fetches US-listed stock data.
"""

import yfinance as yf
import pandas as pd


def fetch_indian_stock(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for an Indian (NSE) stock.

    Args:
        ticker (str): NSE stock symbol with .NS suffix (e.g., 'RELIANCE.NS').
        period (str): Download period recognized by yfinance (default: '2y').

    Returns:
        pd.DataFrame: DataFrame with columns [Open, High, Low, Close, Volume]
                      indexed by Date, or None if fetching fails.
    """
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if data.empty:
            return None
        # Flatten multi-level columns if present (yfinance >=0.2 behaviour)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        print(f"[data_fetcher] Error fetching {ticker}: {e}")
        return None


def fetch_us_stock(ticker: str, period: str = "2y") -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a US stock.

    Args:
        ticker (str): US stock symbol (e.g., 'AAPL').
        period (str): Download period recognized by yfinance (default: '2y').

    Returns:
        pd.DataFrame: DataFrame with columns [Open, High, Low, Close, Volume]
                      indexed by Date, or None if fetching fails.
    """
    try:
        data = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data
    except Exception as e:
        print(f"[data_fetcher] Error fetching {ticker}: {e}")
        return None
