"""
decision_engine/signals/candlestick_signal.py

The short-horizon price-action indicator: does the window's last candle
look like a rejection of the move that preceded it? Structurally the
closest relative of FlagSignalModel (context + a reaction to it), just on
a single final candle rather than a multi-candle consolidation. Same
tiny network shape as MomentumSignalModel; see
decision_engine/training/candlestick_data_set.py for exactly how the
label is computed, and train_candlestick.py for training.

Shortest window in this package (18) — candlestick patterns are
inherently short-horizon; this only needs enough prior candles to judge
whether there was a real move to reject.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from decision_engine.events import SentimentEvent
from decision_engine.signals.base import CandleWindow, SignalModel, normalise_window


class CandlestickSignalModel(SignalModel):
    name = "candlestick_v0"

    def __init__(self, window_size: int = 18, seed: int = 42, weights_path: str | Path | None = None):
        self._window_size = window_size
        tf.random.set_seed(seed)

        inputs = tf.keras.Input(shape=(window_size, 5), name="candles")
        x = tf.keras.layers.Flatten()(inputs)
        x = tf.keras.layers.Dense(32, activation="relu")(x)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
        sentiment = tf.keras.layers.Dense(1, activation="tanh", name="sentiment")(x)
        confidence = tf.keras.layers.Dense(1, activation="sigmoid", name="confidence")(x)
        self._model = tf.keras.Model(inputs=inputs, outputs=[sentiment, confidence])

        # Warm up now (pays the first-call graph-tracing cost here, not
        # mid-loop) — matters if this is ever driven at a tight frame rate.
        self._model(np.zeros((1, window_size, 5), dtype=np.float32), training=False)

        if weights_path is not None:
            self.load_weights(weights_path)

    def load_weights(self, path: str | Path) -> None:
        self._model.load_weights(str(path))

    def save_weights(self, path: str | Path) -> None:
        self._model.save_weights(str(path))

    def window_size(self) -> int:
        return self._window_size

    @property
    def keras_model(self) -> tf.keras.Model:
        """The underlying Keras model, for training. Signal model consumers
        (CandleFeed, etc.) should use predict() instead — this is for
        train_candlestick.py."""
        return self._model

    def predict(self, ticker: str, window: CandleWindow) -> SentimentEvent:
        self._check_window(window)
        normalised = normalise_window(window)
        sentiment_out, confidence_out = self._model(normalised[None, ...], training=False)
        return SentimentEvent(
            event_type="sentiment",
            source=self.name,
            ticker=ticker,
            sentiment=float(sentiment_out[0, 0]),
            confidence=float(confidence_out[0, 0]),
        )
