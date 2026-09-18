"""
decision_engine/training/squeeze_data_set.py

The question this label answers: "is volatility unusually compressed
right now relative to its own recent history — the kind of squeeze that
classically precedes a breakout — and if so, which way does the little
drift inside the squeeze itself lean?" A genuinely different dimension
from momentum/flag/RSI: this is about the *regime* (how much price is
moving around), not its direction or shape.

The window splits into a small "recent" tail (SQUEEZE_RECENT_FRACTION,
default 25%) and the "history" baseline (the rest):

  band_width(segment) = std(closes) / mean(closes) — a Bollinger-style
                         normalised volatility measure

  squeeze_ratio   = band_width(recent) / band_width(history)
  squeeze_quality = clip(1 - squeeze_ratio, 0, 1) — high when recent
                    volatility is much lower than the window's own
                    historical volatility; ~0 when recent volatility is
                    the same or higher (no squeeze at all)

  y_confidence = squeeze_quality — deliberately direction-independent:
                 "is a breakout setting up", regardless of which way.
  y_sentiment  = a *weak* directional lean, not a strong call — a squeeze's
                 breakout direction genuinely isn't knowable in advance,
                 so this only nudges toward whatever slight drift the
                 recent segment itself shows (trend_quality on the recent
                 segment), damped by WEAK_DIRECTION_FACTOR (default 0.5)
                 and gated by squeeze_quality (a lean inside a non-squeeze
                 means nothing).

No future candle is used — every window gets a label, same as the other
indicators in this package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from decision_engine.data.candle_feed import OHLCV_COLUMNS
from decision_engine.signals.base import normalise_window
from decision_engine.training.trend_fit import fit_trend, trend_quality

DEFAULT_RECENT_FRACTION = 0.25  # fraction of the window treated as "recent"; rest is the history baseline
DEFAULT_RECENT_TREND_SCALE = 0.01  # recent segment's relative move that saturates the directional lean
WEAK_DIRECTION_FACTOR = 0.5  # a squeeze's breakout direction is a hint, not a call — damp it
_EPSILON = 1e-8


def _recent_history_split(window_size: int, recent_fraction: float) -> int:
    """Returns the index where "history" ends and "recent" begins —
    recent is the LAST `recent_fraction` of the window (opposite emphasis
    from flag's pole/flag split, where the interesting small segment comes
    first)."""
    recent_len = int(round(window_size * recent_fraction))
    recent_len = min(max(recent_len, 2), window_size - 2)  # keep at least 2 candles on each side
    return window_size - recent_len


def _band_width(closes: np.ndarray) -> float:
    return float(closes.std() / (closes.mean() + _EPSILON))


def compute_squeeze_label(
    window: np.ndarray,
    recent_fraction: float = DEFAULT_RECENT_FRACTION,
    recent_trend_scale: float = DEFAULT_RECENT_TREND_SCALE,
) -> tuple[float, float]:
    """
    window: shape (window_size, 5), OHLCV columns, oldest row first.
    Returns (y_sentiment, y_confidence) — see module docstring.
    """
    split = _recent_history_split(len(window), recent_fraction)
    history_closes = window[:split, 3].astype(np.float64)
    recent_closes = window[split:, 3].astype(np.float64)

    squeeze_ratio = _band_width(recent_closes) / (_band_width(history_closes) + _EPSILON)
    squeeze_quality = float(np.clip(1.0 - squeeze_ratio, 0.0, 1.0))

    recent_trend, recent_r2 = fit_trend(recent_closes)
    directional_lean = trend_quality(recent_trend, recent_r2, recent_trend_scale)

    y_sentiment = float(
        np.clip(np.sign(recent_trend) * directional_lean * squeeze_quality * WEAK_DIRECTION_FACTOR, -1.0, 1.0)
    )
    y_confidence = squeeze_quality
    return y_sentiment, y_confidence


def build_squeeze_training_dataset(
    candles: pd.DataFrame,
    window_size: int,
    recent_fraction: float = DEFAULT_RECENT_FRACTION,
    recent_trend_scale: float = DEFAULT_RECENT_TREND_SCALE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same shapes/semantics as data_set.py::build_training_dataset,
    labelled for a volatility squeeze instead of raw momentum — see
    compute_squeeze_label."""
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
        y_sentiment[i], y_confidence[i] = compute_squeeze_label(window, recent_fraction, recent_trend_scale)

    return X, y_sentiment, y_confidence
