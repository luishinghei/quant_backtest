import pandas as pd

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class TimeframeInfo:
    time_frame_str: str
    time_delta: pd.Timedelta
    annualized_factor: int


class TimeFrame(Enum):
    D1 = TimeframeInfo('1d', pd.Timedelta(days=1), 365)
    H8 = TimeframeInfo('8h', pd.Timedelta(hours=8), 365 * 3)
    H4 = TimeframeInfo('4h', pd.Timedelta(hours=4), 365 * 6)
    H1 = TimeframeInfo('1h', pd.Timedelta(hours=1), 365 * 24)
    M15 = TimeframeInfo('15m', pd.Timedelta(minutes=15), 365 * 24 * 4)
    M10 = TimeframeInfo('10m', pd.Timedelta(minutes=10), 365 * 24 * 6)
    M5 = TimeframeInfo('5m', pd.Timedelta(minutes=5), 365 * 24 * 12)
    M3 = TimeframeInfo('3m', pd.Timedelta(minutes=3), 365 * 24 * 20)
    M1 = TimeframeInfo('1m', pd.Timedelta(minutes=1), 365 * 24 * 60)
    W1 = TimeframeInfo('1w', pd.DateOffset(weeks=1), 52)
    MN1 = TimeframeInfo('1mn', pd.DateOffset(months=1), 12)