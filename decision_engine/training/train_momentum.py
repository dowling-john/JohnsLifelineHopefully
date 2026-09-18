"""
decision_engine/training/train_momentum.py

Trains MomentumSignalModel on the simplest possible momentum label: given
a window, predict the next candle's return (see dataset.py for exactly
how). Not run against real yfinance data from this environment — no
network path to Yahoo Finance here (see candle_feed.py). The dataset
building, splitting, and training loop are tested against synthetic data
instead (tests/test_dataset.py, tests/test_train_momentum.py) — that
proves the mechanics work, not that the resulting model is any good on
real prices. You'll need to judge that yourself once you run it.

Usage:
    python -m decision_engine.training.train_momentum --ticker AAPL --epochs 20
"""

from __future__ import annotations

import argparse

from decision_engine.data.candle_feed import load_historical_candles
from decision_engine.signals.momentum_signal import MomentumSignalModel
from decision_engine.training.data_set import build_training_dataset, chronological_split
from decision_engine.training.losses import directional_accuracy, make_directional_mse, make_momentum_loss

DEFAULT_WEIGHTS_PATH = "decision_engine/signals/weights/momentum_v0.weights.h5"


def train(
    ticker: str,
    period: str = "60d",
    interval: str = "5m",
    window_size: int = 30,
    return_scale: float = 0.5,
    direction_penalty: float = 2.0,
    epochs: int = 20,
    weights_out: str = DEFAULT_WEIGHTS_PATH,
) -> MomentumSignalModel:
    candles = load_historical_candles(ticker, period=period, interval=interval)
    X, y_sentiment, y_confidence = build_training_dataset(candles, window_size, return_scale)
    (X_train, ys_train, yc_train), (X_val, ys_val, yc_val) = chronological_split(X, y_sentiment, y_confidence)

    model = MomentumSignalModel(window_size=window_size)
    model.keras_model.compile(
        optimizer="adam",
        loss={"sentiment": make_momentum_loss(), "confidence": "mse"},
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
        print(f"Val directional accuracy: {final_val_dir_acc:.2%}  "
              f"(50% = coin flip, no directional skill)")
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
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--return-scale", type=float, default=0.5,
                         help="Return magnitude that saturates sentiment to +-1 (default: 0.5%%)")
    parser.add_argument("--direction-penalty", type=float, default=2.0,
                         help="Extra loss weight for a wrong-sign prediction, on top of the squared-error "
                              "magnitude term (default: 2.0; 0.0 = plain MSE)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--weights-out", default=DEFAULT_WEIGHTS_PATH)
    args = parser.parse_args()

    train(
        ticker=args.ticker, period=args.period, interval=args.interval,
        window_size=args.window_size, return_scale=args.return_scale,
        direction_penalty=args.direction_penalty,
        epochs=args.epochs, weights_out=args.weights_out,
    )


if __name__ == "__main__":
    main()