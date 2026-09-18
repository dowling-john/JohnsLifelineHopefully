"""
decision_engine/training/train_flag.py

Trains FlagSignalModel to answer "is a flag pattern building in this
window, which direction is it likely to break, and how strong/clean is
the pattern" — a pole+consolidation fit over the window itself (see
flag_data_set.py for exactly how), the same deterministic-label approach
as train_momentum.py, just a different label. The dataset building,
splitting, and training loop share the same mechanics already exercised
for momentum — that proves the mechanics work, not that the resulting
model is useful on real prices. You'll need to judge that yourself once
you run it.

Usage:
    python -m decision_engine.training.train_flag --ticker AAPL --epochs 20
"""

from __future__ import annotations

import argparse

import numpy as np

from decision_engine.data.candle_feed import load_historical_candles
from decision_engine.signals.flag_signal import FlagSignalModel
from decision_engine.training.data_set import chronological_split
from decision_engine.training.flag_data_set import (
    DEFAULT_POLE_FRACTION,
    DEFAULT_POLE_TREND_SCALE,
    build_flag_training_dataset,
)
from decision_engine.training.losses import directional_accuracy, make_directional_mse

DEFAULT_WEIGHTS_PATH = "decision_engine/signals/weights/flag_v0.weights.h5"


def train(
    ticker: str,
    period: str = "60d",
    interval: str = "5m",
    window_size: int = 45,
    pole_fraction: float = DEFAULT_POLE_FRACTION,
    pole_trend_scale: float = DEFAULT_POLE_TREND_SCALE,
    direction_penalty: float = 2.0,
    epochs: int = 20,
    weights_out: str = DEFAULT_WEIGHTS_PATH,
) -> FlagSignalModel:
    candles = load_historical_candles(ticker, period=period, interval=interval)
    X, y_sentiment, y_confidence = build_flag_training_dataset(candles, window_size, pole_fraction, pole_trend_scale)
    (X_train, ys_train, yc_train), (X_val, ys_val, yc_val) = chronological_split(X, y_sentiment, y_confidence)

    model = FlagSignalModel(window_size=window_size)
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

    # The raw metric above is misleading for this label specifically: most
    # windows show no real flag at all, so y_sentiment is EXACTLY 0 for a
    # large majority of examples — and tf.sign(0) never equals tf.sign of a
    # real-valued prediction, so directional_accuracy counts every one of
    # those "correctly predicted near-zero" examples as a directional miss.
    # Restricting to examples where a flag label is actually present is the
    # number that means "did we get the breakout direction right when it
    # mattered" — see this session's verification for why 0.05 was chosen
    # (it cleanly separates "no pattern" noise from real pattern examples
    # on AAPL 60d/5m data).
    flag_present = np.abs(ys_val) > 0.05
    if flag_present.any():
        pred_sentiment, _ = model.keras_model.predict(X_val, verbose=0)
        pred_sentiment = pred_sentiment.flatten()
        acc_on_flags = float(
            (np.sign(pred_sentiment[flag_present]) == np.sign(ys_val[flag_present])).mean()
        )
        print(f"Val directional accuracy (only where a flag label is present, "
              f"{flag_present.sum()}/{len(ys_val)} examples): {acc_on_flags:.2%}")

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
    parser.add_argument("--window-size", type=int, default=45)
    parser.add_argument("--pole-fraction", type=float, default=DEFAULT_POLE_FRACTION,
                         help="Fraction of the window treated as the pole; the rest is the flag (default: 0.4)")
    parser.add_argument("--pole-trend-scale", type=float, default=DEFAULT_POLE_TREND_SCALE,
                         help="Pole's relative move that saturates the pole-quality magnitude term (default: 3%%)")
    parser.add_argument("--direction-penalty", type=float, default=2.0,
                         help="Extra loss weight for a wrong-sign prediction, on top of the squared-error "
                              "magnitude term (default: 2.0; 0.0 = plain MSE)")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--weights-out", default=DEFAULT_WEIGHTS_PATH)
    args = parser.parse_args()

    train(
        ticker=args.ticker, period=args.period, interval=args.interval,
        window_size=args.window_size, pole_fraction=args.pole_fraction,
        pole_trend_scale=args.pole_trend_scale, direction_penalty=args.direction_penalty,
        epochs=args.epochs, weights_out=args.weights_out,
    )


if __name__ == "__main__":
    main()
