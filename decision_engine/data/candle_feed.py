"""
decision_engine/data/candle_feed.py

Two separate concerns, deliberately kept apart:

1. load_historical_candles() — fetches real historical OHLCV data (via
   yfinance). This has NOT been run against the real network from this
   environment — no path to Yahoo Finance from this sandbox. Verify it
   yourself before relying on it.
2. CandleFeed — replays an already-loaded candle series at a controllable
   rate, maintains the rolling window, and drives a SignalModel. This is
   pure logic with no network dependency, so it's fully testable with a
   synthetic DataFrame — see tests/test_candle_feed.py.

Keeping these separate means the replay/windowing logic (the part that's
actually easy to get subtly wrong — off-by-one windows, stale buffers) is
tested independently of whether yfinance is reachable today.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Callable

import numpy as np
import pandas as pd

from decision_engine.events import EventBus
from decision_engine.signals.base import SignalModel

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def load_historical_candles(ticker: str, period: str = "60d", interval: str = "5m") -> pd.DataFrame:
    """
    Fetch historical OHLCV via yfinance. Not verified from this sandbox —
    no network path to Yahoo Finance here. Confirm column names/behaviour
    yourself; yfinance's return shape has changed across versions before.
    """
    import yfinance as yf  # imported lazily so the rest of this module has no hard dependency on it

    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if data.empty:
        raise ValueError(f"yfinance returned no data for {ticker!r} (period={period!r}, interval={interval!r})")
    return data[OHLCV_COLUMNS]


class CandleFeed:
    """
    Replays a candle DataFrame through a SignalModel at a controllable
    rate, publishing a SentimentEvent to the bus for every candle once
    the rolling window is full.
    """

    def __init__(self, model: SignalModel, bus: EventBus):
        self._model = model
        self._bus = bus
        self._window: deque[np.ndarray] = deque(maxlen=model.window_size())

    def run(
        self,
        candles: pd.DataFrame,
        ticker: str,
        rate_hz: float = 60.0,
        on_event: Callable[[object], None] | None = None,
    ) -> None:
        """
        Iterates `candles` (expects columns OHLCV_COLUMNS, oldest row
        first) at `rate_hz` rows/second. `on_event` is an optional extra
        callback (e.g. for logging/testing) called with every published
        event, in addition to whatever's subscribed on the bus.

        rate_hz=60 is a target, not a guarantee — actual achievable rate
        depends on model inference latency plus Python loop overhead.
        Benchmark before trusting it hits 60 for real; nothing here
        corrects for drift if a single iteration runs long.
        """
        interval_seconds = 1.0 / rate_hz if rate_hz > 0 else 0.0
        missing = [c for c in OHLCV_COLUMNS if c not in candles.columns]
        if missing:
            raise ValueError(f"candles is missing required columns: {missing}")

        for _, row in candles.iterrows():
            self._window.append(row[OHLCV_COLUMNS].to_numpy(dtype=np.float32))

            if len(self._window) == self._model.window_size():
                window_array = np.stack(self._window)
                event = self._model.predict(ticker, window_array)
                self._bus.publish(event)
                if on_event is not None:
                    on_event(event)

            if interval_seconds > 0:
                time.sleep(interval_seconds)