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


def normalise_window(window: CandleWindow) -> np.ndarray:
    """
    Shared input normalisation for every window-based model (originally
    MomentumSignalModel-only; promoted here once FlagSignalModel needed the
    identical transform). Prices relative to the window's first close (so
    a model sees shape, not absolute price level — the same network has to
    work for a $30 stock and a $300 one). Volume log-scaled and divided by
    its own max in the window, with a small epsilon to avoid divide-by-zero
    on a flat/zero-volume window.
    """
    first_close = window[0, 3]
    price_features = (window[:, :4] / first_close) - 1.0
    log_volume = np.log1p(window[:, 4])
    max_log_volume = log_volume.max()
    volume_feature = log_volume / (max_log_volume + 1e-8)
    return np.column_stack([price_features, volume_feature]).astype(np.float32)


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