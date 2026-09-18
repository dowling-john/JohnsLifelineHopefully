"""
decision_engine/training/train_rsi.py

Trains RsiSignalModel to answer "is this window overbought/oversold
enough that reversion is likely, discounted by how strongly it's
trending" — classic RSI over the window itself (see rsi_data_set.py for
exactly how), the same deterministic-label approach as train_momentum.py
and train_flag.py, just a different label. The dataset building,
splitting, and training loop share the same mechanics already exercised
for momentum/flag — that proves the mechanics work, not that the
resulting model is useful on real prices. You'll need to judge that
yourself once you run it.

Usage:
    python -m decision_engine.training.train_rsi --ticker AAPL --epochs 20
"""

from __future__ import annotations

import argparse

import numpy as np

from decision_engine.data.candle_feed import load_historical_candles
from decision_engine.signals.rsi_signal import RsiSignalModel
from decision_engine.training.data_set import chronological_split
from decision_engine.training.losses import directional_accuracy, make_directional_mse
from decision_engine.training.rsi_data_set import (
    DEFAULT_REVERSION_TREND_DISCOUNT,
    DEFAULT_RSI_SCALE,
    build_rsi_training_dataset,
)

DEFAULT_WEIGHTS_PATH = "decision_engine/signals/weights/rsi_v0.weights.h5"
DIRECTIONAL_ACCURACY_THRESHOLD = 0.05  # see the val-set diagnostic below


def train(
    ticker: str,
    period: str = "60d",
    interval: str = "5m",
    window_size: int = 20,
    rsi_scale: float = DEFAULT_RSI_SCALE,
    reversion_trend_discount: float = DEFAULT_REVERSION_TREND_DISCOUNT,
    direction_penalty: float = 2.0,
    epochs: int = 20,
    weights_out: str = DEFAULT_WEIGHTS_PATH,
) -> RsiSignalModel:
    candles = load_historical_candles(ticker, period=period, interval=interval)
    X, y_sentiment, y_confidence = build_rsi_training_dataset(
        candles, window_size, rsi_scale, reversion_trend_discount
    )
    (X_train, ys_train, yc_train), (X_val, ys_val, yc_val) = chronological_split(X, y_sentiment, y_confidence)

    model = RsiSignalModel(window_size=window_size)
    model.keras_model.compile(
        optimizer="adam",
        loss={"sentiment": make_directional_mse(direction_penalty), "confidence": "mse"},
        metrics={"sentiment": [directional_accuracy]},
    )
    history = model.keras_model.fit(
        X_train,
        {"sentiment": ys_train, "confidence": yc_train},
        validation_data=(X_val, {"sentiment": ys_val, "confidence": yc_val}),
        epochs=epochs,
        verbose=2,
    )

    final_train_loss = history.history["loss"][-1]
    final_val_loss = history.history["val_loss"][-1]
    # Keras names this key from the metric function's name — directional_accuracy.
    final_val_dir_acc = history.history.get("val_sentiment_directional_accuracy", [None])[-1]

    print(f"\nFinal train loss: {final_train_loss:.4f}  |  val loss: {final_val_loss:.4f}")
    if final_val_dir_acc is not None:
        print(f"Val directional accuracy (all examples): {final_val_dir_acc:.2%}  "
              f"(50% = coin flip, no directional skill)")

    # Unlike momentum, RSI's label is rarely exactly zero (it's a
    # continuous extremity measure), so this split matters less here than
    # for flag/candlestick/etc — kept anyway for consistency across every
    # indicator in this package, and because a threshold-restricted number
    # is still the more meaningful one: "did we get the direction right
    # when the call actually mattered" rather than diluted by near-50 RSI
    # examples that are closer to noise than to a real signal.
    meaningful = np.abs(ys_val) > DIRECTIONAL_ACCURACY_THRESHOLD
    if meaningful.any():
        pred_sentiment, _ = model.keras_model.predict(X_val, verbose=0)
        pred_sentiment = pred_sentiment.flatten()
        acc_on_meaningful = float(
            (np.sign(pred_sentiment[meaningful]) == np.sign(ys_val[meaningful])).mean()
        )
        print(f"Val directional accuracy (only where |label| > {DIRECTIONAL_ACCURACY_THRESHOLD}, "
              f"{meaningful.sum()}/{len(ys_val)} examples): {acc_on_meaningful:.2%}")

    if final_val_loss > final_train_loss * 1.5:
        print("Warning: val loss notably higher than train loss — likely overfitting. "
              "Consider more data, a smaller model, or fewer epochs.")

    model.save_weights(weights_out)
    print(f"Saved weights to {weights_out}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--period", default="60d")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--window-size", type=int, default=20)
    parser.add_argument("--rsi-scale", type=float, default=DEFAULT_RSI_SCALE,
                         help="|RSI-50| that saturates sentiment magnitude to 1 (default: 25)")
    parser.add_argument("--reversion-trend-discount", type=float, default=DEFAULT_REVERSION_TREND_DISCOUNT,
                         help="How much a clean whole-window trend discounts confidence (default: 0.5)")
    parser.add_argument("--direction-penalty", type=float, default=2.0,
                         help="Extra loss weight for a wrong-sign prediction, on top of the squared-error "
                              "magnitude term (default: 2.0; 0.0 = plain MSE)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--weights-out", default=DEFAULT_WEIGHTS_PATH)
    args = parser.parse_args()

    train(
        ticker=args.ticker, period=args.period, interval=args.interval,
        window_size=args.window_size, rsi_scale=args.rsi_scale,
        reversion_trend_discount=args.reversion_trend_discount, direction_penalty=args.direction_penalty,
        epochs=args.epochs, weights_out=args.weights_out,
    )


if __name__ == "__main__":
    main()
