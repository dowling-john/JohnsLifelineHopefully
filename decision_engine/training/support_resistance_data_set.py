"""
decision_engine/training/support_resistance_data_set.py

The question this label answers: "is price testing a level it
established earlier in this window, and is it breaking through or being
rejected?" A market-structure signal rather than a shape-of-the-trend
one — the first indicator in this package about a specific PRICE LEVEL
rather than a trend/pattern over the whole window.

The window splits into "formation" (FORMATION_FRACTION, default 80%,
whose High/Low extremes ARE the support/resistance levels) and "test"
(the last 20%, whose behaviour near those levels we read):

  vol_scale = std(formation closes) — the yardstick every distance below
              is measured against, so this generalises across price
              levels/tickers the same way momentum's trend_scale does

  proximity_high/low = how close the test segment's High/Low got to
                        reaching the formation's High/Low — 1.0 once it
                        reaches OR clears the level (a strong breakout
                        overshooting the level is still a full test of
                        it, not a miss), falling off only if it fell
                        short and never got there
  -> whichever is closer is "the level being tested"

  level_significance = how many formation candles came within half a
                        vol_scale of that level (more touches = a more
                        real level, standard S/R theory), scaled by
                        LEVEL_SIGNIFICANCE_SCALE (default 3 touches -> 1.0)

  decisiveness = how far the window's LAST close ended up on the far
                 side (breakout) or near side (rejection) of the level,
                 in vol_scale units

  direction: last close beyond the level = breakout, continuing that way;
             last close still on the near side after testing = rejection,
             bouncing the other way

  quality        = proximity * level_significance * decisiveness
  y_sentiment    = clip(direction * quality, -1, 1)
  y_confidence   = clip(quality, 0, 1)

This is more a small deterministic algorithm than a single formula (as
support/resistance reads generally are) — still fully computable from the
window alone, no future leakage, same as every other label in this
package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from decision_engine.data.candle_feed import OHLCV_COLUMNS
from decision_engine.signals.base import normalise_window

DEFAULT_FORMATION_FRACTION = 0.8  # fraction of the window whose extremes form the levels; rest is the test
DEFAULT_TOUCH_TOLERANCE_FACTOR = 0.5  # vol_scale multiples within which a formation candle counts as a "touch"
DEFAULT_LEVEL_SIGNIFICANCE_SCALE = 3.0  # touches that saturate level_significance to 1.0
_EPSILON = 1e-8

_HIGH, _LOW, _CLOSE = 1, 2, 3


def _formation_test_split(window_size: int, formation_fraction: float) -> int:
    split = int(round(window_size * formation_fraction))
    return min(max(split, 2), window_size - 2)  # keep at least 2 candles on each side


def compute_support_resistance_label(
    window: np.ndarray,
    formation_fraction: float = DEFAULT_FORMATION_FRACTION,
    touch_tolerance_factor: float = DEFAULT_TOUCH_TOLERANCE_FACTOR,
    level_significance_scale: float = DEFAULT_LEVEL_SIGNIFICANCE_SCALE,
) -> tuple[float, float]:
    """
    window: shape (window_size, 5), OHLCV columns, oldest row first.
    Returns (y_sentiment, y_confidence) — see module docstring.
    """
    split = _formation_test_split(len(window), formation_fraction)
    formation = window[:split].astype(np.float64)
    test = window[split:].astype(np.float64)

    formation_closes = formation[:, _CLOSE]
    vol_scale = float(formation_closes.std()) + _EPSILON

    formation_high = float(formation[:, _HIGH].max())
    formation_low = float(formation[:, _LOW].min())
    test_high = float(test[:, _HIGH].max())
    test_low = float(test[:, _LOW].min())
    last_close = float(window[-1, _CLOSE])

    shortfall_high = max(formation_high - test_high, 0.0)  # 0 once the test segment reaches/clears the level
    shortfall_low = max(test_low - formation_low, 0.0)
    proximity_high = float(np.clip(1.0 - shortfall_high / vol_scale, 0.0, 1.0))
    proximity_low = float(np.clip(1.0 - shortfall_low / vol_scale, 0.0, 1.0))

    tolerance = touch_tolerance_factor * vol_scale
    touches_high = int(np.sum(formation[:, _HIGH] >= formation_high - tolerance))
    touches_low = int(np.sum(formation[:, _LOW] <= formation_low + tolerance))

    if proximity_high >= proximity_low:
        proximity = proximity_high
        level = formation_high
        significance = float(np.clip(touches_high / level_significance_scale, 0.0, 1.0))
        if last_close > level:
            direction, decisiveness = 1.0, (last_close - level) / vol_scale  # breakout up
        else:
            direction, decisiveness = -1.0, (level - last_close) / vol_scale  # rejected back down
    else:
        proximity = proximity_low
        level = formation_low
        significance = float(np.clip(touches_low / level_significance_scale, 0.0, 1.0))
        if last_close < level:
            direction, decisiveness = -1.0, (level - last_close) / vol_scale  # breakdown
        else:
            direction, decisiveness = 1.0, (last_close - level) / vol_scale  # rejected back up

    decisiveness = float(np.clip(decisiveness, 0.0, 1.0))
    quality = proximity * significance * decisiveness

    y_sentiment = float(np.clip(direction * quality, -1.0, 1.0))
    y_confidence = float(np.clip(quality, 0.0, 1.0))
    return y_sentiment, y_confidence


def build_support_resistance_training_dataset(
    candles: pd.DataFrame,
    window_size: int,
    formation_fraction: float = DEFAULT_FORMATION_FRACTION,
    touch_tolerance_factor: float = DEFAULT_TOUCH_TOLERANCE_FACTOR,
    level_significance_scale: float = DEFAULT_LEVEL_SIGNIFICANCE_SCALE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same shapes/semantics as data_set.py::build_training_dataset,
    labelled for support/resistance rejection-vs-breakout instead of raw
    momentum — see compute_support_resistance_label."""
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
        y_sentiment[i], y_confidence[i] = compute_support_resistance_label(
            window, formation_fraction, touch_tolerance_factor, level_significance_scale
        )

    return X, y_sentiment, y_confidence
