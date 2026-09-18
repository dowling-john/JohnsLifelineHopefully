"""
decision_engine/training/candlestick_data_set.py

The question this label answers: "does the window's LAST candle look
like a rejection of the move that came before it, and how much did that
prior move actually matter?" A short-horizon price-action signal — one
general "wick-rejection" measure rather than a hand-coded zoo of named
patterns (hammer/engulfing/doji/...), same "deterministic function of the
window" philosophy as the rest of this package. Structurally the closest
relative of flag's label: context (~pole) + a reaction to it, just on a
single final candle instead of a multi-candle consolidation.

  prior_trend, prior_r2 = fit_trend(closes[:-1]) — everything except the
                           last candle is "the move to potentially reverse"
  prior_quality = trend_quality(prior_trend, prior_r2, scale) — same
                  "clean AND big-enough" composite flag's pole_quality
                  uses; a reversal candle only means something after a
                  real prior move

  For the last candle: close_position = (close-low)/(high-low) — 0 at
  the candle's low, 1 at its high.
    after a downtrend (prior_trend < 0): a bullish rejection has the
      close near the top AND a long lower wick
        reversal_strength = close_position * lower_wick_ratio
        direction = +1
    after an uptrend (prior_trend > 0): the mirror image
        reversal_strength = (1 - close_position) * upper_wick_ratio
        direction = -1

  y_confidence = prior_quality * reversal_strength
  y_sentiment  = clip(direction * y_confidence, -1, 1)

No future candle is used — every window gets a label, same as the other
indicators in this package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from decision_engine.data.candle_feed import OHLCV_COLUMNS
from decision_engine.signals.base import normalise_window
from decision_engine.training.trend_fit import fit_trend, trend_quality

DEFAULT_PRIOR_TREND_SCALE = 0.02  # prior segment's relative move that saturates prior_quality's magnitude term
_EPSILON = 1e-8

_OPEN, _HIGH, _LOW, _CLOSE = 0, 1, 2, 3


def compute_candlestick_label(
    window: np.ndarray,
    prior_trend_scale: float = DEFAULT_PRIOR_TREND_SCALE,
) -> tuple[float, float]:
    """
    window: shape (window_size, 5), OHLCV columns, oldest row first.
    Returns (y_sentiment, y_confidence) — see module docstring.
    """
    prior_closes = window[:-1, _CLOSE].astype(np.float64)
    prior_trend, prior_r2 = fit_trend(prior_closes)
    prior_quality = trend_quality(prior_trend, prior_r2, prior_trend_scale)

    last_open, last_high, last_low, last_close = window[-1, _OPEN:_CLOSE + 1].astype(np.float64)
    candle_range = (last_high - last_low) + _EPSILON
    close_position = (last_close - last_low) / candle_range

    if prior_trend < 0:
        lower_wick = min(last_open, last_close) - last_low
        reversal_strength = close_position * (lower_wick / candle_range)
        direction = 1.0
    elif prior_trend > 0:
        upper_wick = last_high - max(last_open, last_close)
        reversal_strength = (1.0 - close_position) * (upper_wick / candle_range)
        direction = -1.0
    else:
        reversal_strength = 0.0
        direction = 0.0

    y_confidence = float(np.clip(prior_quality * reversal_strength, 0.0, 1.0))
    y_sentiment = float(np.clip(direction * y_confidence, -1.0, 1.0))
    return y_sentiment, y_confidence


def build_candlestick_training_dataset(
    candles: pd.DataFrame,
    window_size: int,
    prior_trend_scale: float = DEFAULT_PRIOR_TREND_SCALE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same shapes/semantics as data_set.py::build_training_dataset,
    labelled for a candlestick reversal instead of raw momentum — see
    compute_candlestick_label."""
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
        y_sentiment[i], y_confidence[i] = compute_candlestick_label(window, prior_trend_scale)

    return X, y_sentiment, y_confidence
