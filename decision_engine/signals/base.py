"""
decision_engine/signals/base.py

Every signal model — momentum, mean-reversion, volatility, whatever comes
next — implements this same interface. It's what lets the feeder and the
UI stay ignorant of which specific model they're driving: swap one in or
out without touching anything else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from decision_engine.events import SentimentEvent

# A candle window: shape (window_size, 5) — columns are
# [open, high, low, close, volume], oldest row first.
CandleWindow = np.ndarray


class SignalModel(ABC):
    name: str

    @abstractmethod
    def window_size(self) -> int:
        """How many candles this model needs to make a prediction."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, ticker: str, window: CandleWindow) -> SentimentEvent:
        """
        window has shape (self.window_size(), 5). Raises ValueError if the
        shape doesn't match — fail loudly here rather than let a shape bug
        silently produce a garbage sentiment score.
        """
        raise NotImplementedError

    def _check_window(self, window: CandleWindow) -> None:
        expected = (self.window_size(), 5)
        if window.shape != expected:
            raise ValueError(f"{self.name}: expected window shape {expected}, got {window.shape}")