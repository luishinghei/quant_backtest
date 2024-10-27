# Quant Backtest Engine

Welcome to the Quant Backtest for Crypto package. This package is designed to help you perform backtesting on cryptocurrency trading strategies.


## Usage

To get started with using this package, you can refer to the `example.ipynb` notebook included in this directory. This notebook demonstrates how to use the package to perform backtesting on a sample trading strategy.

## Features

- Easy setup and installation
- Comprehensive backtesting capabilities
- Support for various cryptocurrency trading strategies
- Detailed performance metrics and analysis

## Backtest Engine

The `BacktestEngine` class in [`backtest_engine.py`] is the core of the backtesting framework. It handles the execution of trading strategies and the calculation of performance metrics.

### Example

```python
import backtest as bt

# Define your strategy function
def double_rsi_momentum(df, rsi_short=14, rsi_long=25):
    df['rsi_short'] = bt.rsi(df['close'], rsi_short)
    df['rsi_long'] = bt.rsi(df['close'], rsi_long)
    df['spread'] = df['rsi_long'] - df['rsi_short']
    df['signal'] = np.where(df['spread'] < -5, 1, 0)
    return df['signal']

# Fetch price data
btc = bt.fetch_price('BTCUSDT', '1d')

# Define transaction cost
fee = bt.TransactionCost.bybit_taker

# Initialize and run the backtest
bt1 = bt.BacktestEngine(btc, double_rsi_momentum, btc, fee, rsi_short=17, rsi_long=65)
bt1.run()
bt1.report()
```
![image](https://github.com/user-attachments/assets/218b228d-0c59-40dc-b56a-95b087c4212e)

## Optimizer

The `Optimizer` class in [`optimizer.py`] helps in optimizing the parameters of your trading strategy.

### Example

```python
rsi_short = np.arange(5, 60, 5)
rsi_long = np.arange(30, 100, 5)

opt1 = bt.Optimizer(btc, double_rsi_momentum, btc, fee, rsi_short=rsi_short, rsi_long=rsi_long)
opt1.run()
opt1.plot_heatmap()
```

## Visualization

The package includes several methods for visualizing the performance of your trading strategies, such as `plot_pnl`, `plot_rolling_sharpe`, and `plot`.

### Example

```python
bt1.plot_pnl()
bt1.plot_rolling_sharpe()
bt1.plot()
```

For more detailed examples and usage, please refer to the [`example.ipynb`] notebook.
