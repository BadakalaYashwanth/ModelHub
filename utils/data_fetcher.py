"""
utils/data_fetcher.py
---------------------
Purpose:
    Provides robust functions to fetch historical stock price data from Yahoo Finance
    for both Indian (NSE) and US stock markets.

    Handles yfinance >=0.2.x MultiIndex column changes and network-level issues
    that are common on cloud-hosted environments (e.g., Streamlit Cloud).
"""

import time
import pandas as pd
import yfinance as yf


def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten MultiIndex columns returned by yfinance >=0.2.x for single tickers.

    Args:
        data (pd.DataFrame): Raw DataFrame from yf.download().

    Returns:
        pd.DataFrame: Cleaned DataFrame with single-level column names.
    """
    if isinstance(data.columns, pd.MultiIndex):
        # For a single ticker yfinance returns ('Close', 'TICKER') etc.
        # Drop the ticker level and keep only the price-type level.
        data.columns = data.columns.get_level_values(0)

    # Standardise: rename 'Adj Close' -> 'Close' if 'Close' is missing
    if "Adj Close" in data.columns and "Close" not in data.columns:
        data = data.rename(columns={"Adj Close": "Close"})

    # Keep only the standard OHLCV columns that exist
    standard_cols = ["Open", "High", "Low", "Close", "Volume"]
    data = data[[c for c in standard_cols if c in data.columns]]

    data.dropna(subset=["Close"], inplace=True)
    return data


def _download_with_retry(ticker: str, period: str, retries: int = 3) -> pd.DataFrame | None:
    """
    Download data with retry logic to handle transient network errors on cloud.

    Args:
        ticker  (str): Yahoo Finance ticker symbol.
        period  (str): Data period string (e.g. '2y').
        retries (int): Number of attempts before giving up.

    Returns:
        pd.DataFrame | None: Cleaned DataFrame or None on failure.
    """
    for attempt in range(1, retries + 1):
        try:
            data = yf.download(
                ticker,
                period=period,
                progress=False,
                auto_adjust=True,
                threads=False,   # More stable on cloud environments
            )
            if data is not None and not data.empty:
                return _clean_dataframe(data)

            # Fallback: use Ticker.history() which is more reliable
            tk = yf.Ticker(ticker)
            data = tk.history(period=period, auto_adjust=True)
            if data is not None and not data.empty:
                return _clean_dataframe(data)

        except Exception as e:
            print(f"[data_fetcher] Attempt {attempt} failed for {ticker}: {e}")
            if attempt < retries:
                time.sleep(2 * attempt)  # Back-off: 2s, 4s

    return None


def fetch_indian_stock(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    """
    Fetch historical OHLCV data for an Indian (NSE) stock.

    Args:
        ticker (str): NSE ticker with .NS suffix (e.g., 'RELIANCE.NS').
        period (str): yfinance period string (default: '2y').

    Returns:
        pd.DataFrame | None: OHLCV DataFrame indexed by Date, or None.
    """
    return _download_with_retry(ticker, period)


def fetch_us_stock(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    """
    Fetch historical OHLCV data for a US-listed stock.

    Args:
        ticker (str): US ticker symbol (e.g., 'AAPL').
        period (str): yfinance period string (default: '2y').

    Returns:
        pd.DataFrame | None: OHLCV DataFrame indexed by Date, or None.
    """
    return _download_with_retry(ticker, period)
