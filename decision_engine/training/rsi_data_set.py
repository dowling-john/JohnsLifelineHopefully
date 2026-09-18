"""
decision_engine/training/rsi_data_set.py

The question this label answers: "is this window overbought/oversold
enough that reversion is likely, and how much should a strong prevailing
trend make us doubt that reversion call?" Classic RSI (average gain vs
average loss over the window) rather than a curve fit — a genuinely
different lens on the same closes than momentum/flag use, and the
deliberate counterpoint to them: they bet on continuation, this bets on
reversion.

  RSI = 100 - 100 / (1 + avg_gain / avg_loss), over all diffs in the
        window. 50 = balanced, >70 classically "overbought", <30
        "oversold" — here we use continuous extremity rather than hard
        thresholds.

  y_sentiment  = clip(-(RSI - 50) / RSI_SCALE, -1, 1) — negative
                 (overbought → expect down) when RSI is high, positive
                 (oversold → expect up) when RSI is low.
  y_confidence = extremity * (1 - REVERSION_TREND_DISCOUNT * trend_r2)
               where extremity = clip(|RSI-50| / RSI_SCALE, 0, 1) and
               trend_r2 is fit_trend's cleanliness score for the whole
               window's closes. Discounted — not zeroed — when the window
               is itself a clean trend: naive mean-reversion calls are
               exactly what gets run over by a real trend, so a strongly
               trending window should reduce this indicator's confidence
               even if RSI is technically extreme.

No future candle is used — every window gets a label, same as
momentum/flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from decision_engine.data.candle_feed import OHLCV_COLUMNS
from decision_engine.signals.base import normalise_window
from decision_engine.training.trend_fit import fit_trend

DEFAULT_RSI_SCALE = 25.0  # |RSI-50| that saturates extremity/sentiment magnitude to 1
DEFAULT_REVERSION_TREND_DISCOUNT = 0.5  # how much a clean whole-window trend discounts confidence
_EPSILON = 1e-8


def compute_rsi(closes: np.ndarray) -> float:
    """Classic RSI (simple-average variant, not Wilder's smoothing —
    deterministic and simple enough to match this package's other labels)
    over every diff in `closes`. Returns a value in [0, 100]."""
    diffs = np.diff(closes)
    gains = np.clip(diffs, 0.0, None)
    losses = np.clip(-diffs, 0.0, None)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    rs = avg_gain / (avg_loss + _EPSILON)
    return float(100.0 - 100.0 / (1.0 + rs))


def compute_rsi_label(
    window: np.ndarray,
    rsi_scale: float = DEFAULT_RSI_SCALE,
    reversion_trend_discount: float = DEFAULT_REVERSION_TREND_DISCOUNT,
) -> tuple[float, float]:
    """
    window: shape (window_size, 5), OHLCV columns, oldest row first.
    Returns (y_sentiment, y_confidence) — see module docstring.
    """
    closes = window[:, 3].astype(np.float64)
    rsi = compute_rsi(closes)
    _, trend_r2 = fit_trend(closes)

    y_sentiment = float(np.clip(-(rsi - 50.0) / rsi_scale, -1.0, 1.0))
    extremity = float(np.clip(abs(rsi - 50.0) / rsi_scale, 0.0, 1.0))
    y_confidence = float(np.clip(extremity * (1.0 - reversion_trend_discount * trend_r2), 0.0, 1.0))
    return y_sentiment, y_confidence


def build_rsi_training_dataset(
    candles: pd.DataFrame,
    window_size: int,
    rsi_scale: float = DEFAULT_RSI_SCALE,
    reversion_trend_discount: float = DEFAULT_REVERSION_TREND_DISCOUNT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same shapes/semantics as data_set.py::build_training_dataset,
    labelled for mean-reversion instead of raw momentum — see
    compute_rsi_label."""
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
        y_sentiment[i], y_confidence[i] = compute_rsi_label(window, rsi_scale, reversion_trend_discount)

    return X, y_sentiment, y_confidence
