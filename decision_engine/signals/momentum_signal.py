"""
decision_engine/signals/momentum_signal.py

The first signal model — deliberately small and simple, since the goal
right now is proving the plumbing (event bus -> UI) works end-to-end,
not squeezing out predictive power.

Weights are randomly initialised UNLESS a weights_path is given (or
load_weights() is called after construction). See
decision_engine/training/train_momentum.py for how to actually train
this — training predicts the next candle's return from the window, the
simplest form of a momentum label.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from decision_engine.events import SentimentEvent
from decision_engine.signals.base import CandleWindow, SignalModel


class MomentumSignalModel(SignalModel):
    name = "momentum_v0"

    def __init__(self, window_size: int = 30, seed: int = 42, weights_path: str | Path | None = None):
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
        train_momentum.py."""
        return self._model

    @staticmethod
    def _normalise(window: CandleWindow) -> np.ndarray:
        """
        Prices relative to the window's first close (so the model sees
        shape, not absolute price level — same network has to work for a
        $30 stock and a $300 one). Volume log-scaled and divided by its
        own max in the window, with a small epsilon to avoid divide-by-zero
        on a flat/zero-volume window.
        """
        first_close = window[0, 3]
        price_features = (window[:, :4] / first_close) - 1.0
        log_volume = np.log1p(window[:, 4])
        max_log_volume = log_volume.max()
        volume_feature = log_volume / (max_log_volume + 1e-8)
        return np.column_stack([price_features, volume_feature]).astype(np.float32)

    def predict(self, ticker: str, window: CandleWindow) -> SentimentEvent:
        self._check_window(window)
        normalised = self._normalise(window)
        sentiment_out, confidence_out = self._model(normalised[None, ...], training=False)
        return SentimentEvent(
            event_type="sentiment",
            source=self.name,
            ticker=ticker,
            sentiment=float(sentiment_out[0, 0]),
            confidence=float(confidence_out[0, 0]),
        )