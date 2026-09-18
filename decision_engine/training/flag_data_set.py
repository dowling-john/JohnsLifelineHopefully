"""
decision_engine/training/flag_data_set.py

The question this label answers: "does THIS window currently show a flag
pattern building — a sharp prior move (the pole) followed by a tight
consolidation (the flag) — and if so, which direction is it likely to
break, and how strong/clean is the pattern?" Like momentum's label (see
data_set.py), this is a deterministic function of the window itself: no
future candle is used, so every window gets a label.

The window is split at `pole_fraction` of its length:
  pole = window[:split]   — the prior move
  flag = window[split:]   — the consolidation

Both segments get a straight-line fit (fit_trend, shared with momentum's
label — see trend_fit.py):

  pole_quality = clean AND big-enough pole
               = pole_r2 * clip(|pole_trend| / POLE_TREND_SCALE, 0, 1)

  flag_quality = tight AND not-still-trending consolidation, further
                 discounted if volume didn't contract during it (a flag's
                 volume classically dries up during the consolidation —
                 confirming it's a pause, not just noise)
               = tightness * stillness * volume_factor
    tightness  = 1 - clip(flag's own fit residual std / |pole price move|, 0, 1)
                 — how little the flag wobbles around its own drift,
                 relative to how big a move it's consolidating after
    stillness  = 1 - clip(|flag_trend| / (|pole_trend| + eps), 0, 1)
                 — a flag drifts a little; a flag that's still trending as
                 hard as the pole is just the pole continuing, not a flag
    volume_factor = 1.0 if mean(flag volume) <= mean(pole volume), else a
                    linear falloff to VOLUME_FLOOR as flag volume exceeds
                    pole volume by up to 2x

  y_sentiment  = clip(sign(pole_trend) * pole_quality * flag_quality, -1, 1)
               — direction the pattern favours; magnitude = how strong the
                 read is (a weak/absent pattern reads as ~0, same as
                 momentum's "confidently neutral" case)
  y_confidence = clip(pole_quality * flag_quality, 0, 1)
               — direction-independent pattern strength: literally "how
                 strong is the flag pattern"

All constants below are named starting points (same spirit as momentum's
DEFAULT_TREND_SCALE) — watch directional_accuracy on real data and adjust.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from decision_engine.data.candle_feed import OHLCV_COLUMNS
from decision_engine.signals.base import normalise_window
from decision_engine.training.trend_fit import fit_residual_std, fit_trend, trend_quality

DEFAULT_POLE_FRACTION = 0.4  # fraction of the window treated as the pole; the rest is the flag
DEFAULT_POLE_TREND_SCALE = 0.03  # pole's relative move that saturates pole_quality's magnitude term
VOLUME_FLOOR = 0.3  # volume_factor never drops below this even if flag volume balloons
_EPSILON = 1e-8


def _pole_flag_split(window_size: int, pole_fraction: float) -> int:
    split = int(round(window_size * pole_fraction))
    return min(max(split, 2), window_size - 2)  # keep at least 2 candles on each side


def compute_flag_label(
    window: np.ndarray,
    pole_fraction: float = DEFAULT_POLE_FRACTION,
    pole_trend_scale: float = DEFAULT_POLE_TREND_SCALE,
) -> tuple[float, float]:
    """
    window: shape (window_size, 5), OHLCV columns, oldest row first.
    Returns (y_sentiment, y_confidence) — see module docstring.
    """
    split = _pole_flag_split(len(window), pole_fraction)
    pole_closes = window[:split, 3].astype(np.float64)
    flag_closes = window[split:, 3].astype(np.float64)
    pole_volume = window[:split, 4].astype(np.float64)
    flag_volume = window[split:, 4].astype(np.float64)

    pole_trend, pole_r2 = fit_trend(pole_closes)
    flag_trend, _ = fit_trend(flag_closes)
    flag_residual_std = fit_residual_std(flag_closes)
    pole_price_move = abs(pole_closes[-1] - pole_closes[0])

    pole_quality = trend_quality(pole_trend, pole_r2, pole_trend_scale)

    tightness = 1.0 - float(np.clip(flag_residual_std / (pole_price_move + _EPSILON), 0.0, 1.0))
    stillness = 1.0 - float(np.clip(abs(flag_trend) / (abs(pole_trend) + _EPSILON), 0.0, 1.0))

    volume_ratio = float(flag_volume.mean() / (pole_volume.mean() + _EPSILON))
    volume_factor = 1.0 if volume_ratio <= 1.0 else max(VOLUME_FLOOR, 1.0 - (volume_ratio - 1.0))

    flag_quality = tightness * stillness * volume_factor

    y_sentiment = float(np.clip(np.sign(pole_trend) * pole_quality * flag_quality, -1.0, 1.0))
    y_confidence = float(np.clip(pole_quality * flag_quality, 0.0, 1.0))
    return y_sentiment, y_confidence


def build_flag_training_dataset(
    candles: pd.DataFrame,
    window_size: int,
    pole_fraction: float = DEFAULT_POLE_FRACTION,
    pole_trend_scale: float = DEFAULT_POLE_TREND_SCALE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (X, y_sentiment, y_confidence) — same shapes/semantics as
    data_set.py::build_training_dataset, just labelled for the flag
    pattern instead of raw momentum. Every window of `window_size`
    consecutive candles produces one example, labelled from that window
    alone (see compute_flag_label).
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
        y_sentiment[i], y_confidence[i] = compute_flag_label(window, pole_fraction, pole_trend_scale)

    return X, y_sentiment, y_confidence
