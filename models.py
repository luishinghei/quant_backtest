import numpy as np
import pandas as pd


# def ma_diff(data: pd.Series, ma_short: int = 10, ma_long: int = 20) -> pd.Series:
#     """Calculate the moving average difference of a time series."""
#     ma_short = int(ma_short)
#     ma_long = int(ma_long)
#     return data.rolling(ma_short).mean() - data.rolling(ma_long).mean()

def ma_pct_diff(data: pd.Series, ma: int = 10) -> pd.Series:
    """Calculate the moving average percentage difference of a time series."""
    ma = int(ma)
    return data / data.rolling(ma).mean() - 1

def ma_crossover(data: pd.Series, ma_short: int = 10, ma_long: int = 20) -> pd.Series:
    """Calculate the moving average crossover of a time series."""
    ma_short = int(ma_short)
    ma_long = int(ma_long)
    return np.where(data.rolling(ma_short).mean() > data.rolling(ma_long).mean(), 1, 0)

def z_score(data: pd.Series, window: int = 20) -> pd.Series:
    """Calculate the rolling Z-score of a time series."""
    window = int(window)
    # data = pd.to_numeric(data, errors='coerce')  # Convert to numeric
    data = data.fillna(0)
    rolling_mean = data.rolling(window).mean()
    rolling_std = data.rolling(window).std()
    return (data - rolling_mean) / rolling_std

def min_max_scaler(data: pd.Series, window: int = 20) -> pd.Series:
    """Calculate the rolling Min-Max scaling of a time series."""
    window = int(window)
    return (data - data.rolling(window).min()) / (data.rolling(window).max() - data.rolling(window).min())

def precentile_rank(data: pd.Series, window: int = 20) -> pd.Series:
    """Calculate the rolling percentile rank of a time series."""
    window = int(window)
    return data.rolling(window).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])

def robust_scaling(data: pd.Series, window_size: int = 20) -> pd.Series:
    """Calculate the rolling Robust scaling of a time series."""
    window_size = int(window_size)
    
    rolling_median = data.rolling(window_size).median()
    rolling_iqr = data.rolling(window_size).quantile(0.75) - data.rolling(window_size).quantile(0.25)
    rolling_iqr = rolling_iqr.replace(0, np.nan)
    
    return (data - rolling_median) / rolling_iqr

def rsi(data: pd.Series, window: int = 14) -> pd.Series:
    """Calculate the Relative Strength Index (RSI) of a time series."""
    window = int(window)
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))