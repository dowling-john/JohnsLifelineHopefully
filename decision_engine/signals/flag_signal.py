"""
decision_engine/signals/flag_signal.py

The second signal model — the "flag" pattern: a sharp prior move (the
pole) followed by a tight consolidation (the flag) that's expected to
break on in the pole's direction. Structurally identical to
MomentumSignalModel (same tiny network, same SentimentEvent shape) —
what differs is the label it's trained on. See
decision_engine/training/flag_data_set.py for exactly how that label is
computed from a window, and train_flag.py for training.

Uses a longer window than momentum (45 vs 30 candles) — a pole-then-flag
shape needs room for both phases, not just a single trend read.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import tensorflow as tf

from decision_engine.events import SentimentEvent
from decision_engine.signals.base import CandleWindow, SignalModel, normalise_window


class FlagSignalModel(SignalModel):
    name = "flag_v0"

    def __init__(self, window_size: int = 45, seed: int = 42, weights_path: str | Path | None = None):
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
        train_flag.py."""
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
