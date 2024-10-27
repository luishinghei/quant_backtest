import pandas as pd

import os

from .timeframe import TimeFrame


def fetch_price(start: str,
                end: str | None = None,
                asset: str = 'btcusdt',
                interval: str = '1h',
                data_source: str = 'binance',
                ) -> pd.DataFrame:
    start_date = pd.to_datetime(start)
    if end is None:
        end_date = pd.to_datetime('today')
    else:
        end_date = pd.to_datetime(end)
    
    time_frame_str = interval.lower()
    
    package_dir = os.path.dirname(__file__)
    if data_source == 'binance':
        dir_path = os.path.join(package_dir, 'price_data', 'klines')
    else:
        raise ValueError(f"Unsupported data source: {data_source}")
    
    if time_frame_str != '1m':
        file_path = os.path.join(dir_path, f'{asset}_{time_frame_str}.csv')
        price_df = pd.read_csv(file_path, usecols=['timestamp', 'close'], index_col=0, parse_dates=True)
    else:
        folder_path = os.path.join(dir_path, f'{asset}_{time_frame_str}')
        price_df_list = []
        
        start_month = start_date.strftime('%Y-%m')
        end_month = end_date.strftime('%Y-%m')
        
        for month in pd.date_range(start=start_month, end=end_month, freq='MS'):
            file_name = f'{asset}_{time_frame_str}_{month.strftime("%Y-%m")}.csv'
            file_path = os.path.join(folder_path, file_name)
            
            temp_df = pd.read_csv(file_path, usecols=['timestamp', 'close'], index_col=0, parse_dates=True)
            price_df_list.append(temp_df)
            
        price_df = pd.concat(price_df_list, axis=0)
        
    return price_df[(price_df.index >= start_date) & (price_df.index <= end_date)]

def concat_price(df: pd.DataFrame, asset: str = 'btcusdt', data_source: str = 'binance') -> pd.DataFrame:
    time_delta = df.index[1] - df.index[0]
    time_frame_str = None
    
    for timeframe in TimeFrame:
        if time_delta == timeframe.value.time_delta:
            time_frame_str = timeframe.value.time_frame_str
            break
    
    if time_frame_str is None:
        raise ValueError(f"Unsupported time frame with time delta: {time_delta}")
    
    package_dir = os.path.dirname(__file__)
    if data_source == 'binance':
        dir_path = os.path.join(package_dir, 'price_data', 'klines')
    else:
        raise ValueError(f"Unsupported data source: {data_source}")
    
    if time_frame_str != '1m':
        file_path = os.path.join(dir_path, f'{asset}_{time_frame_str}.csv')
        price_df = pd.read_csv(file_path, usecols=['timestamp', 'close'], index_col=0, parse_dates=True)
    else:
        folder_path = os.path.join(dir_path, f'{asset}_{time_frame_str}')
        price_df_list = []
        
        start_month = df.index[0].strftime('%Y-%m')
        end_month = df.index[-1].strftime('%Y-%m')
        
        for month in pd.date_range(start=start_month, end=end_month, freq='MS'):
            file_name = f'{asset}_{time_frame_str}_{month.strftime("%Y-%m")}.csv'
            file_path = os.path.join(folder_path, file_name)
            
            temp_df = pd.read_csv(file_path, usecols=['timestamp', 'close'], index_col=0, parse_dates=True)
            price_df_list.append(temp_df)
            
        price_df = pd.concat(price_df_list, axis=0)
    
    return pd.concat([df, price_df], axis=1, join='inner')