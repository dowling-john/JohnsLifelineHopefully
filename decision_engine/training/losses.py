"""
decision_engine/training/losses.py

Plain MSE has a real failure mode for this task: return labels are small,
noisy, and roughly zero-mean, so the loss-minimising prediction under MSE
alone is close to zero for everything — technically low loss, but useless
as a "which direction, how strongly" signal, which is the whole point of
a momentum model.

directional_mse fixes this by penalising a wrong-sign prediction more
heavily than a same-sign one of equal magnitude error, on top of the
usual squared-error magnitude term — so getting the direction right
matters more to the optimiser than it does under raw MSE, while the
squared-error term still carries magnitude information (how big the
momentum is), which a pure classification/direction-only loss would
throw away entirely.

directional_accuracy is a plain, non-differentiable (metric-only, never
used as a loss) way to actually see whether that's working: fraction of
predictions with the correct sign.
"""

from __future__ import annotations

import tensorflow as tf


def make_directional_mse(direction_penalty: float = 2.0):
    """
    direction_penalty controls how much more a wrong-sign prediction costs
    than a same-sign one of equal magnitude error. 0.0 reduces to plain
    MSE; higher values push the optimiser harder toward getting the sign
    right, at some cost to magnitude precision. 2.0 is a reasonable
    starting point, not a tuned value — watch directional_accuracy on
    real data and adjust.
    """
    def directional_mse(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        squared_error = tf.square(y_true - y_pred)
        wrong_direction = tf.cast(tf.sign(y_true) != tf.sign(y_pred), tf.float32)
        weight = 1.0 + direction_penalty * wrong_direction
        return tf.reduce_mean(squared_error * weight)

    return directional_mse


def directional_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Fraction of examples where the predicted sign matches the actual
    sign. Metric only — tf.sign has essentially no useful gradient, so
    this must never be used as the loss itself."""
    correct = tf.cast(tf.sign(y_true) == tf.sign(y_pred), tf.float32)
    return tf.reduce_mean(correct)