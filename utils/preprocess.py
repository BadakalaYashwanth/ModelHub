"""
utils/preprocess.py
-------------------
Purpose:
    Contains functions to clean raw stock data, compute technical indicators,
    and reshape the data into LSTM-ready sequences. Called by app.py and the
    test pipeline.

Functions:
    - add_technical_indicators(df): Adds RSI, SMA, EMA, Bollinger Bands to OHLCV data.
    - preprocess_for_lstm(df, sequence_length): Scales 'Close' prices and builds
      sliding window sequences (X, y) for LSTM training/inference.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a raw OHLCV DataFrame with common technical indicators.

    Indicators added:
        - RSI  (14-period Relative Strength Index)
        - SMA20 / SMA50 (Simple Moving Averages)
        - EMA20          (Exponential Moving Average)
        - BB_upper / BB_lower (Bollinger Bands, 20-period ±2 std)

    Args:
        df (pd.DataFrame): DataFrame with at least a 'Close' column.

    Returns:
        pd.DataFrame: Original DataFrame with new indicator columns appended.
                      Rows with NaN values (warm-up period) are dropped.
    """
    df = df.copy()

    # --- RSI ---
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    # --- Moving Averages ---
    df["SMA20"] = df["Close"].rolling(window=20).mean()
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # --- Bollinger Bands ---
    rolling_std = df["Close"].rolling(window=20).std()
    df["BB_upper"] = df["SMA20"] + (2 * rolling_std)
    df["BB_lower"] = df["SMA20"] - (2 * rolling_std)

    # Drop warm-up NaN rows
    df.dropna(inplace=True)
    return df


def preprocess_for_lstm(
    df: pd.DataFrame, sequence_length: int = 60
):
    """
    Scale 'Close' prices with MinMaxScaler and build sliding-window sequences.

    Args:
        df (pd.DataFrame): DataFrame with a 'Close' column (post indicator enrichment).
        sequence_length (int): Number of past days in each input sequence (default: 60).

    Returns:
        tuple:
            X (np.ndarray): Shape (n_samples, sequence_length, 1) — input sequences.
            y (np.ndarray): Shape (n_samples,) — target next-day close price (scaled).
            scaler (MinMaxScaler): Fitted scaler for inverse-transforming predictions.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[["Close"]])

    X, y = [], []
    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i - sequence_length : i, 0])
        y.append(scaled[i, 0])

    X = np.array(X).reshape(-1, sequence_length, 1)
    y = np.array(y)
    return X, y, scaler
