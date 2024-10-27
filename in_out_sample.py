import pandas as pd

from typing import Callable, Tuple
from .backtest_engine import BacktestEngine
from .fee import TransactionCost


def split_and_backtest(data: pd.DataFrame, 
                       strategy_function: Callable, 
                       alpha: pd.DataFrame | pd.Series,
                       split_ratio: float = 0.7,
                       transaction_cost: float | TransactionCost = 0,
                       **stratergy_params) -> Tuple[BacktestEngine, BacktestEngine]:
    split_index = int(len(data) * split_ratio)
    in_sample_data = data.iloc[:split_index].copy()
    out_of_sample_data = data.iloc[split_index:].copy()
    
    in_smaple_alpha = alpha.iloc[:split_index].copy()
    out_of_sample_alpha = alpha.iloc[split_index:].copy()

    # Create the backtest engine instance and run the backtest
    in_smaple_engine = BacktestEngine(in_sample_data, strategy_function, in_smaple_alpha, transaction_cost, **stratergy_params)
    in_smaple_engine.run()
    
    out_of_sample_engine = BacktestEngine(out_of_sample_data, strategy_function, out_of_sample_alpha, transaction_cost, **stratergy_params)
    out_of_sample_engine.run()
    
    return in_smaple_engine, out_of_sample_engine