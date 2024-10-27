import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Callable

from .fee import TransactionCost
from .timeframe import TimeFrame


class BacktestEngine:
    def __init__(self, 
                 data: pd.DataFrame, 
                 strategy_function: Callable,
                 alpha: pd.DataFrame | pd.Series,
                 transaction_cost: float | TransactionCost = 0,
                 **stratergy_params):
        self.alpha = alpha
        self.data = data.copy()
        self.strategy_function = strategy_function
        self.transaction_cost = transaction_cost
        self.stratergy_params = stratergy_params
        
        self.timeframe_str = self._get_timeframe()  # 'M15'
        self.annualized_factor = self._get_annualized_factor()  # 365 * 24 * 4
        
        self.strategy_name = self._format_strategy_name(self.strategy_function.__name__)  # 'M15 Ma Diff'
        self.params_str = self._format_strategy_params(self.stratergy_params)  # 'ma=10 | diff=1'
    
    def _get_timeframe(self) -> str:
        time_delta = self.data.index[1] - self.data.index[0]
        
        for timeframe in TimeFrame:
            if time_delta == timeframe.value.time_delta:
                return timeframe.name
        
        raise ValueError(f'Unsupported timeframe: {time_delta}')
    
    def _get_annualized_factor(self) -> int:
        timeframe_str = self.timeframe_str
        return TimeFrame[timeframe_str].value.annualized_factor
    
    def _format_strategy_name(self, strategy_name: str) -> str:
        formatted_strategy_name = strategy_name.replace('_', ' ').title()
        return f'{self.timeframe_str} {formatted_strategy_name}'

    def _format_strategy_params(self, strategy_params: dict) -> str:
        if not strategy_params:
            return ""
        return ' | '.join([f'{key}={value}' for key, value in strategy_params.items()])
    
    def _fetch_price_data(self, path: str, alpha) -> pd.DataFrame:
        price_df = pd.read_csv(path, index_col=0, parse_dates=True)
        
        alpha_start_date = alpha.index[0]
        alpha_end_date = alpha.index[-1]
        
        if price_df.index[0] > alpha_start_date or price_df.index[-1] < alpha_end_date:
            raise ValueError('Price data does not overlap with alpha data.')
        
        return price_df[(price_df.index >= alpha_start_date) & (price_df.index <= alpha_end_date)]
        
    def run(self) -> None:
        self.data['price_ret'] = self.data['close'].pct_change()
        self.data['signal'] = self.strategy_function(self.alpha, **self.stratergy_params)
        self.data['positions'] = self.data['signal'].shift(1).fillna(0)
        self.data['transaction_cost'] = self.data['positions'].diff().abs() * self.transaction_cost
        self.data['pnl'] = self.data['price_ret'] * self.data['positions'] - self.data['transaction_cost']
        self.data['cum_pnl'] = self.data['pnl'].cumsum()
        self.data['drawdown'] = self.data['cum_pnl'] - self.data['cum_pnl'].cummax()
    
    def run_with_params(self, **stratergy_params) -> None:
        # Update the strategy parameters
        if stratergy_params:
            self.stratergy_params = stratergy_params
            self.params_str = self._format_strategy_params(self.stratergy_params)
        self.run()
    
    
    @property
    def annual_return(self) -> float:
        """Calculate and cache the annualized return."""
        return self.data['pnl'].mean() * self.annualized_factor
    
    @property
    def max_drawdown(self) -> float:
        """Calculate and cache the maximum drawdown."""
        return self.data['drawdown'].min()
    
    @property
    def sharpe(self) -> float:
        """Calculate and cache the Sharpe ratio."""
        if self.data['pnl'].std() == 0:
            return np.nan
        return self.data['pnl'].mean() / self.data['pnl'].std() * np.sqrt(self.annualized_factor)
    
    @property
    def calmar(self) -> float:
        """Calculate and cache the Calmar ratio."""
        if self.max_drawdown == 0.0:
            return np.nan
        return self.annual_return / abs(self.max_drawdown)
    
    @property
    def exposure(self) -> float:
        """Calculate the exposure."""
        return self.data['positions'].abs().mean()
    
    @property
    def long_short_ratio(self) -> float | str:
        """Calculate the long and short ratio."""
        long_period = self.data['positions'][self.data['positions'] > 0].count()
        short_period = self.data['positions'][self.data['positions'] < 0].count()
        
        if short_period == 0:
            return 'Long Only'
        elif long_period == 0:
            return 'Short Only'
        
        return round(long_period / short_period, 2)
    
    @property
    def no_of_trades(self) -> int:
        """Calculate the number of trades."""
        return int(self.data['positions'].fillna(0).diff().abs().sum() // 2)
    
    @property
    def dd_duration(self) -> float:
        """Calculate the duration of the maximum drawdown."""
        mdd_date = self.data['drawdown'].idxmin()
        mdd_start_date = self.data['cum_pnl'][:mdd_date].idxmax()
    
        post_mdd_data = self.data['cum_pnl'][mdd_date:]
        new_high_data = post_mdd_data[post_mdd_data > self.data['cum_pnl'][mdd_start_date]]
    
        if new_high_data.empty:
            new_high_date = self.data.index[-1]
        else:
            new_high_date = new_high_data.index[0]
        
        return (new_high_date - mdd_start_date).days
    
    def _get_rolling_sharpe(self, days=60) -> pd.Series:
        """Calculate the rolling Sharpe ratio."""
        if self.data['pnl'].std() == 0:
            return np.nan
        
        window = int(days * self.annualized_factor / 365)
        return self.data['pnl'].rolling(window=window).mean() / self.data['pnl'].rolling(window=window).std() * np.sqrt(self.annualized_factor)
    
    def stats(self):
        """Display all key statistics."""
        align = 20  # Set the alignment width as a variable

        print(f'{self.strategy_name} | {self.params_str}')
        print(f'-' * 42)
        print(f'Sharpe                {self.sharpe:>{align}.2f}')
        print(f'Calmar                {self.calmar:>{align}.2f}')
        print(f'-' * 42)
        print(f'Exposure              {self.exposure:>{align}.2f}')
        print(f'No of Trades          {self.no_of_trades:>{align}}')
        print(f'No of Data Points     {len(self.data):>{align}}')
        print(f'Start Date            {str(self.data.index[0]):>{align}}')
        print(f'End Date              {str(self.data.index[-1]):>{align}}')
        print(f'-' * 42)
        print(f'Annu Ret [%]          {self.annual_return * 100:>{align}.2f}')
        print(f'Max DD [%]            {self.max_drawdown * 100:>{align}.2f}')
        print(f'Max DD Dur [days]     {self.dd_duration:>{align}}')
        print(f'Long/Short Ratio      {self.long_short_ratio:>{align}}')
    
    def report(self):
        self.stats()
        self.plot_pnl()
    
    def plot_pnl(self, slice_range: tuple[int, int] = None, benchmark: bool = False):
        start, end = slice_range if slice_range else (None, None)
        pnl = self.data['pnl'].iloc[start:end]
        cum_pnl = pnl.cumsum()
        label = f'{self.params_str} | sr:{self.sharpe:.2f} | cr:{self.calmar:.2f}'
        
        if benchmark:
            plt.plot(self.data['price_ret'].cumsum(), lw=1, zorder=0, color='grey', label='Buy and Hold')
            plt.legend()
        
        plt.plot(cum_pnl, label=label, lw=1.5)
        plt.title(self.strategy_name)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()
    
    def plot_dd(self):
        plt.plot(self.data['drawdown'])
        plt.title('Drawdown')
        plt.grid(alpha=0.3)
        plt.xticks(rotation=30)
        plt.show()
    
    def plot_positions(self):
        plt.plot(self.data['positions'])
        plt.title('Positions')
        plt.grid(alpha=0.3)
        plt.xticks(rotation=30)
        plt.show()
    
    def plot_rolling_sharpe(self, days=60, ma=60, sharpe=2):
        rolling_sharpe = self._get_rolling_sharpe(days).dropna()
        rolling_sharpe_ma = rolling_sharpe.rolling(window=ma).mean()
        # plt.figure(figsize=(12, 3))
        plt.plot(rolling_sharpe)
        plt.plot(rolling_sharpe_ma, label=f'{ma}-MA')
        plt.hlines(sharpe, self.data.index[0], self.data.index[-1],colors='red', linestyles='dashed')
        plt.title(f'Rolling Sharpe ({days}-Days)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xticks(rotation=30)
        plt.show()
    
    def plot(self):
        # create subplot of pnl, drawdown, rolling sharpes
        gridspec_kw = {'height_ratios': [4, 1, 1]}
        fig, ax = plt.subplots(3, 1, figsize=(12, 12), sharex=True, gridspec_kw=gridspec_kw)
        
        # set the figure size
        
        label = f'{self.params_str} | sr:{self.sharpe:.2f} | cr:{self.calmar:.2f}'
        
        ax[0].plot(self.data['cum_pnl'], label=label, lw=1.5)
        ax[0].set_title(f'{self.strategy_name} PnL')
        ax[0].grid(alpha=0.3)
        
        # Plot Drawdown
        ax[1].plot(self.data['drawdown'], lw=1.5)
        ax[1].set_title('Drawdown')
        ax[1].grid(alpha=0.3)
        
        # Plot Rolling Sharpe
        ax[2].plot(self.metrics.get_rolling_sharpe(), lw=1.5)
        ax[2].set_title('Rolling Sharpe')
        ax[2].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()