"""
decision_engine/signals/support_resistance_signal.py

The market-structure indicator: is price testing a level it established
earlier in the window (the extremes of a "formation" segment), and is it
breaking through or being rejected? Unlike momentum/flag/RSI/squeeze/
divergence (all readings of a trend or pattern shape over the whole
window), this is about a specific price level. Same tiny network shape
as MomentumSignalModel; see
decision_engine/training/support_resistance_data_set.py for exactly how
the label is computed, and train_support_resistance.py for training.

Window 40, split 80% formation / 20% test (see
support_resistance_data_set.py) — needs a decent formation period for a
level to be meaningful, per standard S/R theory (a level tested more
often during formation is treated as more real).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from decision_engine.events import SentimentEvent
from decision_engine.signals.base import CandleWindow, SignalModel, normalise_window


class SupportResistanceSignalModel(SignalModel):
    name = "support_resistance_v0"

    def __init__(self, window_size: int = 40, seed: int = 42, weights_path: str | Path | None = None):
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
        train_support_resistance.py."""
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
