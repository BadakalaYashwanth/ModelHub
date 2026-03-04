"""
utils/data_fetcher.py
---------------------
Purpose:
    Fetches historical OHLCV stock data from Yahoo Finance for Indian (NSE)
    and US markets.

    On cloud platforms (Streamlit Cloud, Render, Railway) Yahoo Finance can
    rate-limit plain requests.  We work around this by:
        1. Attaching a browser-like User-Agent via a custom requests.Session
        2. Using Ticker.history() which is more robust than yf.download()
        3. Retrying up to 3 times with exponential back-off
        4. Falling back to yf.download() if Ticker.history() fails
"""

import time
import random
import requests
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------------
# Browser-like session — avoids Yahoo Finance rate-limiting on cloud servers
# ---------------------------------------------------------------------------
def _make_session() -> requests.Session:
    """Return a requests.Session that mimics a real browser."""
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }
    )
    return session


# ---------------------------------------------------------------------------
# Column normalisation
# ---------------------------------------------------------------------------
def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame | None:
    """
    Normalise a raw yfinance DataFrame:
        - Flatten MultiIndex columns (yfinance >= 0.2.x single-ticker quirk)
        - Ensure a 'Close' column exists
        - Drop rows where Close is NaN
    """
    if data is None or data.empty:
        return None

    # Flatten multi-level columns: ('Close', 'AAPL') -> 'Close'
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Remove duplicate column names that can appear after flattening
    data = data.loc[:, ~data.columns.duplicated()]

    # 'Adj Close' -> 'Close' fallback
    if "Adj Close" in data.columns and "Close" not in data.columns:
        data = data.rename(columns={"Adj Close": "Close"})

    if "Close" not in data.columns:
        return None

    # Keep only standard OHLCV columns
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data = data[keep].copy()
    data.dropna(subset=["Close"], inplace=True)

    return data if not data.empty else None


# ---------------------------------------------------------------------------
# Core download with retry
# ---------------------------------------------------------------------------
def _fetch(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    """
    Try to download data using Ticker.history() with a browser session,
    fall back to yf.download() on failure, retry up to 3 times.

    Args:
        ticker (str): Yahoo Finance ticker symbol.
        period (str): Data period (e.g., '2y', '1y', '6mo').

    Returns:
        pd.DataFrame | None: Cleaned OHLCV DataFrame or None.
    """
    session = _make_session()

    for attempt in range(1, 4):
        try:
            # ── Primary: Ticker.history() with browser session ──────────────
            tk = yf.Ticker(ticker, session=session)
            data = tk.history(period=period, auto_adjust=True)
            cleaned = _clean_dataframe(data)
            if cleaned is not None:
                return cleaned

        except Exception as e:
            print(f"[data_fetcher] Ticker.history attempt {attempt} failed for {ticker}: {e}")

        try:
            # ── Fallback: yf.download() with browser session ─────────────────
            data = yf.download(
                ticker,
                period=period,
                progress=False,
                auto_adjust=True,
                threads=False,
                session=session,
            )
            cleaned = _clean_dataframe(data)
            if cleaned is not None:
                return cleaned

        except Exception as e:
            print(f"[data_fetcher] yf.download attempt {attempt} failed for {ticker}: {e}")

        # Exponential back-off with a small random jitter
        wait = (2 ** attempt) + random.uniform(0, 1)
        print(f"[data_fetcher] Waiting {wait:.1f}s before retry {attempt + 1}...")
        time.sleep(wait)

    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_indian_stock(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    """
    Fetch historical OHLCV data for an Indian (NSE) stock.

    Args:
        ticker (str): NSE ticker with .NS suffix (e.g., 'RELIANCE.NS').
        period (str): yfinance period string — '1y', '2y', '5y', etc.

    Returns:
        pd.DataFrame | None: OHLCV DataFrame indexed by Date, or None on failure.
    """
    return _fetch(ticker, period)


def fetch_us_stock(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    """
    Fetch historical OHLCV data for a US-listed stock.

    Args:
        ticker (str): US ticker symbol (e.g., 'AAPL').
        period (str): yfinance period string.

    Returns:
        pd.DataFrame | None: OHLCV DataFrame indexed by Date, or None on failure.
    """
    return _fetch(ticker, period)
