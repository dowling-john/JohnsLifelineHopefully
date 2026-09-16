"""
main.py

Manual smoke-test harness — run this by hand against the DEMO (paper)
environment to sanity-check the broker against the real API, after the
automated tests in tests/ are passing.

This is NOT an automated test and is NOT run in CI. It makes real network
calls to demo.trading212.com using real (demo-account) credentials, and
was never run by Claude — there's no network path to trading212.com from
that sandbox. Run it yourself before trusting this against real money.

Usage:
    export T212_API_KEY=...
    export T212_API_SECRET=...
    python main.py                       # read-only checks only
    python main.py --place-test-order    # also places and cancels one
                                          # tiny market order, DEMO ONLY
"""

from __future__ import annotations

import argparse
import os
import sys

from trading_implementations.trading212_api_broker import Environment, Trading212APIBroker
from trading_interface import Order, OrderType, Side


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"Missing required environment variable: {name}", file=sys.stderr)
        sys.exit(1)
    return value


def run_read_only_checks(broker: Trading212APIBroker) -> None:
    print("== get_account() ==")
    account = broker.get_account()
    print(f"  currency: {account.currency}")
    print(f"  cash (available to trade): {account.cash}")
    print(f"  invested: {account.invested}")
    print(f"  total value: {account.total_value}")

    print("\n== get_positions() ==")
    positions = broker.get_positions()
    if not positions:
        print("  (no open positions)")
    for p in positions:
        print(f"  {p.ticker}: qty={p.quantity} avg_paid={p.average_price_paid} "
              f"current={p.current_price} unrealised_pnl={p.unrealised_pnl}")

    print("\n== get_quote() for each held position ==")
    for p in positions:
        quote = broker.get_quote(p.ticker)
        print(f"  {p.ticker}: price={quote.price} bid={quote.bid} ask={quote.ask}")
    if not positions:
        print("  (skipped — no positions to price; get_quote() can't price anything else, see spec doc)")


def run_test_order(broker: Trading212APIBroker, ticker: str) -> None:
    print(f"\n== placing a tiny test MARKET buy order for {ticker} (DEMO account) ==")
    confirm = input(
        f"About to place a real order (quantity=1) for {ticker} against "
        f"{broker._environment.name}. Type 'yes' to continue: "
    )
    if confirm.strip().lower() != "yes":
        print("Aborted.")
        return

    order = Order(ticker=ticker, side=Side.BUY, quantity=1, order_type=OrderType.MARKET)
    result = broker.place_order(order)
    print(f"  placed: id={result.broker_order_id} status={result.status}")

    input("Press Enter to cancel this order again (or Ctrl+C to leave it as-is)...")
    broker.cancel_order(result.broker_order_id)
    print("  cancel request sent.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--place-test-order", action="store_true",
        help="Also place (and offer to cancel) one tiny market order. DEMO only — refuses on LIVE.",
    )
    parser.add_argument(
        "--ticker", default="AAPL_US_EQ",
        help="Ticker to use for --place-test-order (default: AAPL_US_EQ)",
    )
    args = parser.parse_args()

    api_key = _require_env("T212_API_KEY")
    api_secret = _require_env("T212_API_SECRET")

    # Deliberately hardcoded to DEMO — this script is for manual sanity
    # checks, not for touching a live account. Change this only once
    # you've decided you're ready to, and understand what you're doing.
    broker = Trading212APIBroker(api_key=api_key, api_secret=api_secret, environment=Environment.DEMO)

    run_read_only_checks(broker)

    if args.place_test_order:
        run_test_order(broker, args.ticker)


if __name__ == "__main__":
    main()