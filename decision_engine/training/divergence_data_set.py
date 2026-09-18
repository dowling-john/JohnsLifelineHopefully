"""
decision_engine/training/divergence_data_set.py

The question this label answers: "does volume confirm or contradict the
price trend in this window, and if it contradicts, which way does that
suggest price will turn?" The first indicator in this package where
volume drives the signal rather than lightly confirming it (flag's
volume-contraction factor is a minor input; here it's the whole point).

fit_trend (same generic helper momentum/flag use) is applied to BOTH the
window's closes and its volume:

  price_trend, price_r2   = fit_trend(closes)
  volume_trend, volume_r2 = fit_trend(volume)

Classic divergence reading:
  price up + volume down   -> a rally on fading participation (weakening)
                               -> expect DOWN
  price down + volume up   -> selling into rising volume (capitulation /
                               accumulation) -> expect UP
  price/volume same sign   -> confirmation, not divergence -> no signal
                               here (that's momentum's territory, not this
                               indicator's)

  divergence_gate = clip(-sign(price_trend) * sign(volume_trend), 0, 1)
                     (1 when price/volume trends are opposite, 0 when
                     they agree)
  y_confidence = price_r2 * volume_r2 * divergence_gate — both trends
                 need to be clean AND opposite for this to mean anything
  y_sentiment  = clip(-sign(price_trend) * y_confidence, -1, 1) — bet
                 against the (unconfirmed-by-volume) price trend

No future candle is used — every window gets a label, same as the other
indicators in this package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from decision_engine.data.candle_feed import OHLCV_COLUMNS
from decision_engine.signals.base import normalise_window
from decision_engine.training.trend_fit import fit_trend


def compute_divergence_label(window: np.ndarray) -> tuple[float, float]:
    """
    window: shape (window_size, 5), OHLCV columns, oldest row first.
    Returns (y_sentiment, y_confidence) — see module docstring.
    """
    closes = window[:, 3].astype(np.float64)
    volume = window[:, 4].astype(np.float64)

    price_trend, price_r2 = fit_trend(closes)
    volume_trend, volume_r2 = fit_trend(volume)

    divergence_gate = float(np.clip(-np.sign(price_trend) * np.sign(volume_trend), 0.0, 1.0))
    y_confidence = float(np.clip(price_r2 * volume_r2 * divergence_gate, 0.0, 1.0))
    y_sentiment = float(np.clip(-np.sign(price_trend) * y_confidence, -1.0, 1.0))
    return y_sentiment, y_confidence


def build_divergence_training_dataset(
    candles: pd.DataFrame,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same shapes/semantics as data_set.py::build_training_dataset,
    labelled for volume/price divergence instead of raw momentum — see
    compute_divergence_label."""
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
        y_sentiment[i], y_confidence[i] = compute_divergence_label(window)

    return X, y_sentiment, y_confidence
