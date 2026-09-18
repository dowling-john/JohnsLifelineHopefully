"""
decision_engine/training/dataset.py

Builds a training dataset where the target represents the mathematical
momentum of the price curve contained within each candle window.

The model is NOT trained to predict the next candle.

Instead, each window is treated as a price curve. We fit a linear regression
to log(close) across the window and use:

    momentum = slope * R²

The slope represents the direction and rate of price movement.

R² represents how consistently the prices follow that direction.

Therefore:

    positive momentum -> upward trend
    negative momentum -> downward trend
    near-zero momentum -> flat / noisy market

The result is normalised into approximately [-1, +1].

The confidence target represents the strength of the underlying trend,
primarily based on R².

This means the model is learning:

    recent candles -> mathematical momentum of those candles

rather than:

    recent candles -> next candle return
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from decision_engine.data.candle_feed import OHLCV_COLUMNS
from decision_engine.signals.momentum_signal import MomentumSignalModel


def calculate_momentum(
    closes: np.ndarray,
    momentum_scale: float = 0.5,
) -> tuple[float, float]:

    closes = np.asarray(closes, dtype=np.float64).reshape(-1)

    if closes.size < 2:
        raise ValueError("At least two closes are required")

    if np.any(closes <= 0):
        raise ValueError("Close prices must be greater than zero")

    if momentum_scale <= 0:
        raise ValueError(
            f"momentum_scale must be > 0, got {momentum_scale}"
        )

    log_prices = np.log(closes)

    x = np.arange(
        log_prices.size,
        dtype=np.float64,
    )

    slope, intercept = np.polyfit(
        x,
        log_prices,
        1,
    )

    slope = float(np.asarray(slope).reshape(()))
    intercept = float(np.asarray(intercept).reshape(()))

    predicted = slope * x + intercept

    ss_total = float(
        np.sum(
            (log_prices - np.mean(log_prices)) ** 2
        )
    )

    if ss_total <= 1e-12:
        r_squared = 0.0
    else:
        ss_residual = float(
            np.sum(
                (log_prices - predicted) ** 2
            )
        )

        r_squared = 1.0 - (
            ss_residual / ss_total
        )

        r_squared = float(
            np.clip(r_squared, 0.0, 1.0)
        )

    raw_momentum = slope * r_squared

    momentum = float(
        np.clip(
            raw_momentum / momentum_scale,
            -1.0,
            1.0,
        )
    )

    confidence = float(r_squared)

    return momentum, confidence


def build_training_dataset(
    candles: pd.DataFrame,
    window_size: int,
    momentum_scale: float = 1.00,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the training dataset.

    Returns:

        X:
            Shape (n_examples, window_size, 5)

        y_momentum:
            Shape (n_examples,)

            Mathematical momentum of each candle window.

        y_confidence:
            Shape (n_examples,)

            R² of the price trend within each window.

    IMPORTANT:

        The target does NOT use the next candle.

        Each example is:

            candles[i : i + window_size]
                    |
                    v
            mathematical curve
                    |
                    v
               momentum

    This means the model learns to recognise momentum from the candle
    sequence itself.
    """

    missing = [
        column
        for column in OHLCV_COLUMNS
        if column not in candles.columns
    ]

    if missing:
        raise ValueError(
            f"candles is missing required columns: {missing}"
        )

    if window_size < 2:
        raise ValueError(
            f"window_size must be >= 2, got {window_size}"
        )

    n_examples = len(candles) - window_size + 1

    if n_examples <= 0:
        raise ValueError(
            f"Not enough candles ({len(candles)}) for "
            f"window_size={window_size}"
        )

    closes = candles["Close"].to_numpy(dtype=np.float64)

    X = np.empty(
        (n_examples, window_size, 5),
        dtype=np.float32,
    )

    y_momentum = np.empty(
        n_examples,
        dtype=np.float32,
    )

    y_confidence = np.empty(
        n_examples,
        dtype=np.float32,
    )

    for i in range(n_examples):

        # ---------------------------------------------------------------
        # Input
        # ---------------------------------------------------------------

        window = candles.iloc[
            i : i + window_size
        ][OHLCV_COLUMNS].to_numpy(dtype=np.float32)

        X[i] = MomentumSignalModel._normalise(window)

        # ---------------------------------------------------------------
        # Target
        # ---------------------------------------------------------------

        window_closes = closes[
            i : i + window_size
        ]

        momentum, confidence = calculate_momentum(
            window_closes,
            momentum_scale=momentum_scale,
        )

        y_momentum[i] = momentum
        y_confidence[i] = confidence

    return (
        X,
        y_momentum,
        y_confidence,
    )


def chronological_split(
    X: np.ndarray,
    y_momentum: np.ndarray,
    y_confidence: np.ndarray,
    train_fraction: float = 0.8,
) -> tuple[tuple, tuple]:
    """
    Split the dataset chronologically.

    No random shuffling is performed.

    Returns:

        (
            (X_train, ym_train, yc_train),
            (X_val, ym_val, yc_val),
        )
    """

    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            f"train_fraction must be in (0, 1), "
            f"got {train_fraction}"
        )

    split_at = int(len(X) * train_fraction)

    if split_at <= 0 or split_at >= len(X):
        raise ValueError(
            "train_fraction produces an empty train or validation set"
        )

    train = (
        X[:split_at],
        y_momentum[:split_at],
        y_confidence[:split_at],
    )

    validation = (
        X[split_at:],
        y_momentum[split_at:],
        y_confidence[split_at:],
    )

    return train, validation
