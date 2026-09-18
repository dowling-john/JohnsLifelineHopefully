"""
decision_engine/training/trend_fit.py

The one piece of math every deterministic-label dataset in this package is
built on: fit a straight line through a run of closes and report both its
direction/size and how clean the fit is. Originally private to
data_set.py's momentum label; promoted here once flag_data_set.py needed
the exact same fit on sub-segments of a window (the "pole" and the "flag").
"""

from __future__ import annotations

import numpy as np

_R2_EPSILON = 1e-8  # guards ss_tot == 0 on a perfectly flat (zero-variance) segment


def fit_trend(closes: np.ndarray) -> tuple[float, float]:
    """
    Returns (relative_trend, r_squared) for a run of closes.
    relative_trend = fitted slope * (n-1) / first_close — the line's total
    implied move over the segment, as a fraction of its starting price.
    r_squared clipped to [0, 1] (a fit worse than the flat mean is treated
    as "no linear trend detected", not a negative score).
    """
    n = len(closes)
    x = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(x, closes, 1)
    fitted = slope * x + intercept

    ss_res = np.sum((closes - fitted) ** 2)
    ss_tot = np.sum((closes - closes.mean()) ** 2)
    r_squared = 1.0 - ss_res / (ss_tot + _R2_EPSILON)
    r_squared = float(np.clip(r_squared, 0.0, 1.0))

    relative_trend = float(slope * (n - 1) / closes[0])
    return relative_trend, r_squared


def trend_quality(trend: float, r_squared: float, scale: float) -> float:
    """
    "Clean AND big-enough move" composite shared by every label below that
    needs to ask that question about some segment of a window (flag's
    pole, candlestick's prior-move context, ...): r_squared rewards a
    clean fit, the clipped magnitude term rewards a move that's actually
    large enough to matter, relative to `scale`.
    """
    return r_squared * float(np.clip(abs(trend) / scale, 0.0, 1.0))


def fit_residual_std(closes: np.ndarray) -> float:
    """
    Standard deviation of closes around their own linear fit (not around
    their mean) — how much a segment wobbles around its own trendline,
    regardless of whether that trendline is flat or sloped. Used by the
    flag label to measure how "tight" a consolidation channel is: a flag
    can legitimately drift a little, what matters is how little it strays
    from that drift.
    """
    n = len(closes)
    x = np.arange(n, dtype=np.float64)
    slope, intercept = np.polyfit(x, closes, 1)
    fitted = slope * x + intercept
    return float(np.std(closes - fitted))
