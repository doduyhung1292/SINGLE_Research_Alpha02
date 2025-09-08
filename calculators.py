import math
import numpy as np
import pandas as pd


def calculate_bollinger_bands(prices, window=500, num_std=2):
    """
    Calculate Bollinger Bands for a series of prices.

    IMPORTANT: This function assumes price data is properly sorted from oldest to newest.
    For market data with timestamps, be sure to use ensure_data_sort_oldest_to_newest()
    from utils.py before passing to this function.

    Implementation follows TradingView's ta.sma() and ta.stdev() behavior:
    - Uses ddof=1 for standard deviation (divides by n-1)
    - Requires at least 'window' data points before calculating

    Args:
        prices (array-like): Series of price data
        window (int): Window size for the moving average
        num_std (float): Number of standard deviations for the bands

    Returns:
        tuple: (middle_band, upper_band, lower_band)
    """
    # Convert to pandas Series if not already
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices)

    # Calculate middle band (SMA) with min_periods to match TradingView behavior
    middle_band = prices.rolling(window=window, min_periods=window).mean()

    # Calculate standard deviation with ddof=1 to match TradingView behavior
    std_dev = prices.rolling(window=window, min_periods=window).std(ddof=1)

    # Calculate upper and lower bands
    upper_band = middle_band + (std_dev * num_std)
    lower_band = middle_band - (std_dev * num_std)

    return middle_band, upper_band, lower_band



def delay(series, period):
    """
    Trả về series bị trễ đi 'period' bước. Nếu không đủ dữ liệu, trả về NaN ở đầu.
    """
    series = np.array(series)
    delayed = np.roll(series, period)
    delayed[:period] = np.nan
    return delayed


def calculate_new_spread(Close, Open):
    """
    Tính spread theo công thức mới:
    Close - Open
    """

    Close = np.array(Close, dtype=float)
    Open = np.array(Open, dtype=float)

    alpha = Close - Open
    return alpha.tolist()
