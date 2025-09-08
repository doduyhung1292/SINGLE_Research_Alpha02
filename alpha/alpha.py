import numpy as np
import pandas as pd
from typing import Tuple, Dict, Callable, Any, Union

def alpha1(close_a: pd.Series, close_b: pd.Series, **kwargs) -> pd.Series:
    """
    Alpha 1: LOG(CloseA) - LOG(CloseB)
    
    Args:
        close_a: Close price series for symbol A
        close_b: Close price series for symbol B
        
    Returns:
        Alpha values series
    """
    # Ensure both series have the same index
    common_index = close_a.index.intersection(close_b.index)
    close_a = close_a.loc[common_index]
    close_b = close_b.loc[common_index]
    
    # Calculate log of close prices
    log_a = np.log(close_a)
    log_b = np.log(close_b)
    
    # Return difference
    return log_a - log_b

def alpha2(close_a: pd.Series, close_b: pd.Series, window: int = 50, **kwargs) -> pd.Series:
    """
    Alpha 2: (CloseA-SMA(CloseA,20))/SMA(CloseA,20) - (CloseB-SMA(CloseB,20))/SMA(CloseB,20)
    
    Args:
        close_a: Close price series for symbol A
        close_b: Close price series for symbol B
        window: Window size for SMA calculation (default: 20)
        
    Returns:
        Alpha values series
    """
    # Ensure both series have the same index
    common_index = close_a.index.intersection(close_b.index)
    close_a = close_a.loc[common_index]
    close_b = close_b.loc[common_index]
    
    # Calculate SMA for both symbols
    sma_a = close_a.rolling(window=window).mean()
    sma_b = close_b.rolling(window=window).mean()
    
    # Calculate normalized deviation from SMA
    dev_a = (close_a - sma_a) / sma_a
    dev_b = (close_b - sma_b) / sma_b
    
    # Return difference
    alpha = dev_a - dev_b
    
    return alpha.dropna()

def alpha3(close_a: pd.Series, close_b: pd.Series, delay: int = 50, **kwargs) -> pd.Series:
    """
    Alpha 3: (CloseA-DELAY(CloseA,50))/DELAY(CloseA,50) - (CloseB-DELAY(CloseB,50))/DELAY(CloseB,50)
    
    Args:
        close_a: Close price series for symbol A
        close_b: Close price series for symbol B
        delay: Delay periods (default: 20)
        
    Returns:
        Alpha values series
    """
    # Ensure both series have the same index
    common_index = close_a.index.intersection(close_b.index)
    close_a = close_a.loc[common_index]
    close_b = close_b.loc[common_index]
    
    # Calculate delayed values
    delay_a = close_a.shift(50)
    delay_b = close_b.shift(50)
    
    # Calculate normalized changes
    change_a = (close_a - delay_a) / delay_a
    change_b = (close_b - delay_b) / delay_b
    
    # Return difference
    alpha = change_a - change_b
    
    return alpha.dropna()

def alpha4(open_a: pd.Series, close_a: pd.Series, open_b: pd.Series, close_b: pd.Series, **kwargs) -> pd.Series:
    """
    Alpha 4: OpenA/CloseA - OpenB/CloseB
    
    Args:
        open_a: Open price series for symbol A
        close_a: Close price series for symbol A
        open_b: Open price series for symbol B
        close_b: Close price series for symbol B
        
    Returns:
        Alpha values series
    """
    # Find common index for all series
    common_index = open_a.index.intersection(close_a.index).intersection(open_b.index).intersection(close_b.index)
    
    # Filter series to common index
    open_a = open_a.loc[common_index]
    close_a = close_a.loc[common_index]
    open_b = open_b.loc[common_index]
    close_b = close_b.loc[common_index]
    
    # Calculate ratios
    ratio_a = open_a / close_a
    ratio_b = open_b / close_b
    
    # Return difference
    return ratio_a - ratio_b

def alpha5(high_a: pd.Series, low_a: pd.Series, close_a: pd.Series, 
           high_b: pd.Series, low_b: pd.Series, close_b: pd.Series, **kwargs) -> pd.Series:
    """
    Alpha 5: ((HighA + LowA)/2 - CloseA)/CloseA - ((HighB + LowB)/2 - CloseB)/CloseB
    
    Args:
        high_a: High price series for symbol A
        low_a: Low price series for symbol A
        close_a: Close price series for symbol A
        high_b: High price series for symbol B
        low_b: Low price series for symbol B
        close_b: Close price series for symbol B
        
    Returns:
        Alpha values series
    """
    # Find common index for all series
    common_index = high_a.index.intersection(low_a.index).intersection(close_a.index).intersection(
        high_b.index).intersection(low_b.index).intersection(close_b.index)
    
    # Filter series to common index
    high_a = high_a.loc[common_index]
    low_a = low_a.loc[common_index]
    close_a = close_a.loc[common_index]
    high_b = high_b.loc[common_index]
    low_b = low_b.loc[common_index]
    close_b = close_b.loc[common_index]
    
    # Calculate mid prices
    mid_a = (high_a + low_a) / 2
    mid_b = (high_b + low_b) / 2
    
    # Calculate normalized differences
    norm_diff_a = (mid_a - close_a) / close_a
    norm_diff_b = (mid_b - close_b) / close_b
    
    # Return difference
    return norm_diff_a - norm_diff_b

def alpha6(high_a: pd.Series, low_a: pd.Series, volume_a: pd.Series,
           high_b: pd.Series, low_b: pd.Series, volume_b: pd.Series, window: int = 10, **kwargs) -> pd.Series:
    """
    Alpha 6: CORRELATION(HighA/LowA, VolumeA, 10) - CORRELATION(HighB/LowB, VolumeB, 10)
    
    Args:
        high_a: High price series for symbol A
        low_a: Low price series for symbol A
        volume_a: Volume series for symbol A
        high_b: High price series for symbol B
        low_b: Low price series for symbol B
        volume_b: Volume series for symbol B
        window: Window size for correlation calculation (default: 10)
        
    Returns:
        Alpha values series
    """
    # Find common index for all series
    common_index = high_a.index.intersection(low_a.index).intersection(volume_a.index).intersection(
        high_b.index).intersection(low_b.index).intersection(volume_b.index)
    
    # Filter series to common index
    high_a = high_a.loc[common_index]
    low_a = low_a.loc[common_index]
    volume_a = volume_a.loc[common_index]
    high_b = high_b.loc[common_index]
    low_b = low_b.loc[common_index]
    volume_b = volume_b.loc[common_index]
    
    # Calculate high/low ratios
    ratio_a = high_a / low_a
    ratio_b = high_b / low_b
    
    # Replace infinite values with NaN
    ratio_a = ratio_a.replace([np.inf, -np.inf], np.nan)
    ratio_b = ratio_b.replace([np.inf, -np.inf], np.nan)
    
    # Calculate rolling correlations with min_periods to handle NaNs
    corr_a = ratio_a.rolling(window=window, min_periods=4).corr(volume_a)
    corr_b = ratio_b.rolling(window=window, min_periods=4).corr(volume_b)
    
    # Replace NaN with 0 in correlations
    corr_a = corr_a.fillna(0)
    corr_b = corr_b.fillna(0)
    
    # Return difference
    alpha = corr_a - corr_b
    
    return alpha

def alpha7(vwap_a: pd.Series, volume_a: pd.Series, 
           vwap_b: pd.Series, volume_b: pd.Series, window: int = 10, **kwargs) -> pd.Series:
    """
    Alpha 7: CORRELATION(VwapA, VolumeA, 10) - CORRELATION(VwapB, VolumeB, 10)
    
    Args:
        vwap_a: VWAP series for symbol A
        volume_a: Volume series for symbol A
        vwap_b: VWAP series for symbol B
        volume_b: Volume series for symbol B
        window: Window size for correlation calculation (default: 10)
        
    Returns:
        Alpha values series
    """
    # Find common index for all series
    common_index = vwap_a.index.intersection(volume_a.index).intersection(
        vwap_b.index).intersection(volume_b.index)
    
    # Filter series to common index
    vwap_a = vwap_a.loc[common_index]
    volume_a = volume_a.loc[common_index]
    vwap_b = vwap_b.loc[common_index]
    volume_b = volume_b.loc[common_index]
    
    # Handle NaN and Infinity values more thoroughly
    # Replace infinite values with NaN first
    vwap_a = vwap_a.replace([np.inf, -np.inf], np.nan)
    vwap_b = vwap_b.replace([np.inf, -np.inf], np.nan)
    
    # Check if we have too many NaN values
    nan_count_a = vwap_a.isna().sum()
    nan_count_b = vwap_b.isna().sum()
    
    print(f"nan_count_a: {nan_count_a}, nan_count_b: {nan_count_b}")
    # Calculate rolling correlations with min_periods to handle NaNs
    try:
        corr_a = vwap_a.rolling(window=window, min_periods=4).corr(volume_a)
        corr_b = vwap_b.rolling(window=window, min_periods=4).corr(volume_b)
        
        # Replace NaN and infinite values with 0 in correlations
        corr_a = corr_a.replace([np.inf, -np.inf], np.nan).fillna(0)
        corr_b = corr_b.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Return difference
        alpha = corr_a - corr_b
        
        # Final check for any remaining NaN or infinite values
        alpha = alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return alpha
    except Exception as e:
        print(f"Error calculating alpha7 correlation: {e}")
        return pd.Series(0, index=common_index)

def calculate_vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate Volume Weighted Average Price (VWAP)
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        volume: Volume series
        
    Returns:
        VWAP series
    """
    try:   
        # Typical price = (high + low + close) / 3
        typical_price = (high + low + close) / 3
        
        # Check for invalid values
        if typical_price.isna().any():
            print(f"Warning: NaN values in typical price calculation")
            typical_price = typical_price.fillna(method='ffill').fillna(method='bfill')
        
        # VWAP = sum(typical_price * volume) / sum(volume)
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        
        # Handle any remaining NaN or infinite values
        vwap = vwap.replace([np.inf, -np.inf], np.nan)
        vwap = vwap.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        return vwap
    except Exception as e:
        print(f"Error calculating VWAP: {e}")
        return pd.Series(close)  # Fallback to using close price if VWAP calculation fails

def calculate_all_alphas(df_a: pd.DataFrame, df_b: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate all alpha values for a pair of symbols
    
    Args:
        df_a: DataFrame containing OHLCV data for symbol A
        df_b: DataFrame containing OHLCV data for symbol B
        
    Returns:
        Dictionary of alpha series with keys 'alpha1', 'alpha2', etc.
    """
    # DEBUG PRINT
    print("Starting calculate_all_alphas...")
    print(f"df_a shape: {df_a.shape}, df_b shape: {df_b.shape}")
    
    # Calculate VWAP for both symbols
    vwap_a = calculate_vwap(df_a['high'], df_a['low'], df_a['close'], df_a['volume'])
    vwap_b = calculate_vwap(df_b['high'], df_b['low'], df_b['close'], df_b['volume'])
    
    # Calculate all alphas - return as dictionary
    alphas = {
        'alpha1': alpha1(df_a['close'], df_b['close']),
        'alpha2': alpha2(df_a['close'], df_b['close']),
        'alpha3': alpha3(df_a['close'], df_b['close']),
        'alpha4': alpha4(df_a['open'], df_a['close'], df_b['open'], df_b['close']),
        'alpha5': alpha5(df_a['high'], df_a['low'], df_a['close'], df_b['high'], df_b['low'], df_b['close']),
        'alpha6': alpha6(df_a['high'], df_a['low'], df_a['volume'], df_b['high'], df_b['low'], df_b['volume']),
        'alpha7': alpha7(vwap_a, df_a['volume'], vwap_b, df_b['volume'])
    }
    
    # DEBUG PRINT
    print(f"Returning a dictionary with {len(alphas)} alphas")
    for alpha_name, alpha_series in alphas.items():
        print(f"{alpha_name} - Type: {type(alpha_series)}, Length: {len(alpha_series) if isinstance(alpha_series, pd.Series) else 'Not a Series'}")
    
    return alphas

# Dictionary mapping alpha names to their respective functions
ALPHA_FUNCTIONS = {
    'alpha1': alpha1,
    'alpha2': alpha2,
    'alpha3': alpha3,
    'alpha4': alpha4,
    'alpha5': alpha5,
    'alpha6': alpha6,
    'alpha7': alpha7
}

# Dictionary with alpha descriptions
ALPHA_DESCRIPTIONS = {
    'alpha1': "LOG(CloseA) - LOG(CloseB)",
    'alpha2': "(CloseA-SMA(CloseA,20))/SMA(CloseA,20) - (CloseB-SMA(CloseB,20))/SMA(CloseB,20)",
    'alpha3': "(CloseA-DELAY(CloseA,20))/DELAY(CloseA,20) - (CloseB-DELAY(CloseB,20))/DELAY(CloseB,20)",
    'alpha4': "OpenA/CloseA - OpenB/CloseB",
    'alpha5': "((HighA + LowA)/2 - CloseA)/CloseA - ((HighB + LowB)/2 - CloseB)/CloseB",
    'alpha6': "CORRELATION(HighA/LowA, VolumeA, 10) - CORRELATION(HighB/LowB, VolumeB, 10)",
    'alpha7': "CORRELATION(VwapA, VolumeA, 10) - CORRELATION(VwapB, VolumeB, 10)"
}

def get_alpha_function(alpha_name: str) -> Callable:
    """
    Get the function for a specific alpha by name
    
    Args:
        alpha_name: Name of the alpha (e.g., 'alpha1', 'alpha2', etc.)
        
    Returns:
        The alpha function
    """
    if alpha_name not in ALPHA_FUNCTIONS:
        raise ValueError(f"Unknown alpha: {alpha_name}. Available alphas: {list(ALPHA_FUNCTIONS.keys())}")
    return ALPHA_FUNCTIONS[alpha_name]

def calculate_spread(df_a: pd.DataFrame, df_b: pd.DataFrame, alpha_name: str, **kwargs) -> pd.Series:
    """
    Calculate the spread between two assets using the specified alpha
    
    Args:
        df_a: DataFrame containing OHLCV data for symbol A
        df_b: DataFrame containing OHLCV data for symbol B
        alpha_name: Name of the alpha to use
        **kwargs: Additional arguments to pass to the alpha function
        
    Returns:
        Calculated spread series
    """
    alpha_func = get_alpha_function(alpha_name)
    
    # For alpha1, alpha2, alpha3, we only need close prices
    if alpha_name in ['alpha1', 'alpha2', 'alpha3']:
        return alpha_func(df_a['close'], df_b['close'], **kwargs)
    
    # For alpha4
    elif alpha_name == 'alpha4':
        return alpha_func(df_a['open'], df_a['close'], df_b['open'], df_b['close'], **kwargs)
    
    # For alpha5
    elif alpha_name == 'alpha5':
        return alpha_func(df_a['high'], df_a['low'], df_a['close'], 
                         df_b['high'], df_b['low'], df_b['close'], **kwargs)
    
    # For alpha6
    elif alpha_name == 'alpha6':
        return alpha_func(df_a['high'], df_a['low'], df_a['volume'], 
                         df_b['high'], df_b['low'], df_b['volume'], **kwargs)
    
    # For alpha7
    elif alpha_name == 'alpha7':
        vwap_a = calculate_vwap(df_a['high'], df_a['low'], df_a['close'], df_a['volume'])
        vwap_b = calculate_vwap(df_b['high'], df_b['low'], df_b['close'], df_b['volume'])
        return alpha_func(vwap_a, df_a['volume'], vwap_b, df_b['volume'], **kwargs)
    
    else:
        raise ValueError(f"Unknown alpha: {alpha_name}")

def list_available_alphas() -> None:
    """
    Print a list of available alpha functions with their descriptions
    """
    print("Available Alpha Functions:")
    print("-------------------------")
    for name, desc in ALPHA_DESCRIPTIONS.items():
        print(f"{name}: {desc}") 