import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .backtest_engine import BacktestEngine
from .fee import TransactionCost
from typing import Callable



class Optimizer:
    def __init__(self, 
                 data: pd.DataFrame | pd.Series, 
                 strategy_function: Callable,
                 alpha: pd.DataFrame | pd.Series,
                 transaction_cost: float | TransactionCost = 0,
                 **strategy_params: np.ndarray):
        self.data = data
        self.strategy_function = strategy_function
        self.alpha = alpha
        self.transaction_cost = transaction_cost
        self.strategy_params = strategy_params
        self.bt_results = {}
        self.pnls_df = pd.DataFrame()
    
    def run(self):
        param_values = [v for v in self.strategy_params.values()]  # [array([10, 12, 14, 16, 18]), array([1. , 1.5])]
        param_names = [k for k in self.strategy_params.keys()]  # ['ma', 'diff']
        param_grid = np.array(np.meshgrid(*param_values)).T.reshape(-1, len(param_values))  # [[10.  1.], [12.  1.], [14.  1.], [16.  1.]]
        
        for combination in param_grid:
            param_dict = {param_names[i]: combination[i] for i in range(len(param_names))}  # {'ma': 10.0, 'diff': 1.0}
            
            engine = BacktestEngine(self.data, self.strategy_function, self.alpha, self.transaction_cost, **param_dict)
            engine.run()
            # concat engine.data['cum_pnl'] to the pnls_df
            self.pnls_df = pd.concat([self.pnls_df, engine.data['cum_pnl']], axis=1)
            # renmae the column name to the combination of parameters
            # self.pnls_df.rename({'cum_pnl': f'{engine.params_str}'}, inplace=True)
            self.bt_results[tuple(combination)] = (engine.sharpe, engine.calmar)  # {(10.0, 1.0): (0.885685874816949, 0.11145790279401147),
            
        self.results_df = pd.DataFrame.from_dict(self.bt_results, orient='index', columns=['Sharpe', 'Calmar'])
        self.results_df.index = pd.MultiIndex.from_tuples(self.results_df.index, names=param_names)
        
        return self.results_df
    
    def plot_heatmap(self, annot=True, center=None):
        """Plot heatmaps for Sharpe and Calmar ratios."""
        # Convert MultiIndex DataFrame to pivot tables for heatmap
        sharpe_pivot = self.results_df['Sharpe'].unstack(level=-1)
        calmar_pivot = self.results_df['Calmar'].unstack(level=-1)
        
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        
        # Round x-tick and y-tick labels to avoid floating point errors
        x_ticks = [f'{x:.2f}' for x in sharpe_pivot.columns]
        y_ticks = [f'{y:.2f}' for y in sharpe_pivot.index]
        
        # Plot Sharpe Ratio Heatmap
        sns.heatmap(sharpe_pivot, annot=annot, center=center, cmap='PiYG', fmt='.2f', xticklabels=x_ticks, yticklabels=y_ticks, ax=ax[0])
        ax[0].set_title('Sharpe')
        ax[0].set_xlabel(list(self.strategy_params.keys())[1])
        ax[0].set_ylabel(list(self.strategy_params.keys())[0])

        # Plot Calmar Ratio Heatmap
        sns.heatmap(calmar_pivot, annot=annot, center=center, cmap='PiYG', fmt='.2f', xticklabels=x_ticks, yticklabels=y_ticks, ax=ax[1])
        ax[1].set_title('Calmar')
        ax[1].set_xlabel(list(self.strategy_params.keys())[1])
        ax[1].set_ylabel(list(self.strategy_params.keys())[0])
        
        # # set the xticks rotation to 45 degrees
        plt.setp(ax[0].get_xticklabels(), rotation=30)
        plt.setp(ax[1].get_xticklabels(), rotation=30)
        
        plt.tight_layout()
        plt.show()
    
    def plot_pnl(self):
        """Plot all cumulative PnL curves for each parameter combination."""
        plt.plot(self.pnls_df, lw=1)
        plt.title('PnL')
        # plt.title(engine.strategy_name)
        plt.grid(alpha=0.3)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()
    
    @property
    def params(self):
        return self.strategy_params