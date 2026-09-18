"""
decision_engine_main.py

Wires everything up and launches the live sentiment UI:
  historical candles -> CandleFeed -> MomentumSignalModel -> EventBus
  -> WebSocket -> browser

This is the decision-engine counterpart to the repo's existing main.py
(which is the broker smoke-test harness) — kept as a separate file
rather than overloading main.py, since they start two different things
for two different purposes.

Usage:
    python decision_engine_main.py --ticker AAPL
    python decision_engine_main.py --ticker TSLA_US_EQ --rate-hz 30 --port 8001

Then open http://localhost:8000 (or whatever --port you gave).

Not run against a real browser or real yfinance data from this
environment — see decision_engine/ui_server.py's docstring. Verify
yourself: `python decision_engine_main.py`, then open the URL it prints
and confirm the sentiment value is actually updating.
"""

from __future__ import annotations

import argparse

import uvicorn

from decision_engine import ui_server


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticker", default="AAPL", help="Ticker to feed through the signal model (default: AAPL)")
    parser.add_argument("--period", default="60d", help="yfinance history period, e.g. 60d, 1y (default: 60d)")
    parser.add_argument("--interval", default="5m", help="yfinance candle interval, e.g. 1m, 5m, 1h (default: 5m)")
    parser.add_argument(
        "--rate-hz", type=float, default=60.0,
        help="Replay rate in candles/second — a target, not a guarantee, see CandleFeed.run() (default: 60)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    # configure() records the intent; the actual fetch+feed starts inside
    # the server's own startup (see ui_server._lifespan) so there's no
    # race between the feed starting and the server's event loop existing.
    ui_server.configure(ticker=args.ticker, period=args.period, interval=args.interval, rate_hz=args.rate_hz)

    print(f"Starting UI for {args.ticker} at http://{args.host}:{args.port} (rate_hz={args.rate_hz})")
    uvicorn.run(ui_server.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
