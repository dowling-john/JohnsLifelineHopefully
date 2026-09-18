"""
decision_engine/training/data_set.py

The question this label answers: "does THIS window currently show upward
or downward momentum, and how confident/clean is that trend?" — not "what
will the next candle do?". The earlier next-candle-return version was
trying to forecast unknowable future noise from a short window, which is
close to unlearnable and is why it wouldn't train well. This version's
label is a deterministic function of the window itself: fit a straight
line through the window's closes.

  sentiment_label  = the fitted line's slope, expressed as a relative
                      price move over the window, clipped to [-1, 1].
                      Positive = uptrend, negative = downtrend.
  confidence_label = R² of that linear fit, clipped to [0, 1]. High when
                      the window's prices closely follow a straight line
                      (clean trend, up or down); low when they're choppy/
                      directionless. Deliberately independent of
                      sentiment's sign — a flat, genuinely range-bound
                      window and a strong trend can both be described
                      confidently, just as "confidently neutral" or
                      "confidently trending" (this matches the intent
                      already stated in SentimentEvent's docstring).

No future candle is used anywhere here, so — unlike the old version —
every window in the series gets a label; there's no need to hold back the
last row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from decision_engine.data.candle_feed import OHLCV_COLUMNS
from decision_engine.signals.base import normalise_window
from decision_engine.training.trend_fit import fit_trend

DEFAULT_TREND_SCALE = 0.02  # relative move over the window that saturates sentiment to +-1


def build_training_dataset(
    candles: pd.DataFrame,
    window_size: int,
    trend_scale: float = DEFAULT_TREND_SCALE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X, y_sentiment, y_confidence):
      X shape:            (n_examples, window_size, 5) — normalised the
                           same way MomentumSignalModel uses at inference
                           (see signals/base.py::normalise_window)
      y_sentiment shape:  (n_examples,)
      y_confidence shape: (n_examples,)

    Every window of `window_size` consecutive candles produces one
    example, labelled from that window alone.
    """
    missing = [c for c in OHLCV_COLUMNS if c not in candles.columns]
    if missing:
        raise ValueError(f"candles is missing required columns: {missing}")

    n_examples = len(candles) - window_size + 1
    if n_examples <= 0:
        raise ValueError(
            f"Not enough candles ({len(candles)}) for window_size={window_size} "
            f"— need at least window_size rows."
        )

    X = np.empty((n_examples, window_size, 5), dtype=np.float32)
    y_sentiment = np.empty(n_examples, dtype=np.float32)
    y_confidence = np.empty(n_examples, dtype=np.float32)

    for i in range(n_examples):
        window = candles.iloc[i : i + window_size][OHLCV_COLUMNS].to_numpy(dtype=np.float32)
        X[i] = normalise_window(window)

        closes = window[:, 3].astype(np.float64)
        relative_trend, r_squared = fit_trend(closes)

        y_sentiment[i] = float(np.clip(relative_trend / trend_scale, -1.0, 1.0))
        y_confidence[i] = r_squared

    return X, y_sentiment, y_confidence


def chronological_split(
    X: np.ndarray, y_sentiment: np.ndarray, y_confidence: np.ndarray, train_fraction: float = 0.8
) -> tuple[tuple, tuple]:
    """
    Time-based split, NOT a random shuffle — shuffling before splitting
    would mix later windows into training and earlier ones into
    validation, giving an optimistic, misleading picture of how well this
    generalises to genuinely unseen (later) data. First `train_fraction`
    of examples (in time order) is train, the rest val.
    """
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    split_at = int(len(X) * train_fraction)
    train = (X[:split_at], y_sentiment[:split_at], y_confidence[:split_at])
    val = (X[split_at:], y_sentiment[split_at:], y_confidence[split_at:])
    return train, val