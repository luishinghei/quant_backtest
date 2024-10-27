from dataclasses import dataclass


@dataclass(frozen=True)
class TransactionCost:
    bybit_taker: float = 0.00055
    bybit_maker: float = 0.0002