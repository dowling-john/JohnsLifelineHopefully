"""
tests/test_trading_interface.py

Tests for the core interface types themselves — no broker, no network.
These are the cheapest possible tests to run and catch the most basic
class of bug (validation logic, derived-property math), so they run
first.
"""

from datetime import datetime, timezone

import pytest

from trading_interface import Order, OrderType, Quote, Side


def _now():
    return datetime.now(timezone.utc)


class TestQuote:
    def test_spread_none_when_bid_ask_missing(self):
        q = Quote(ticker="AAPL_US_EQ", price=100.0, bid=None, ask=None, currency="USD", as_of=_now())
        assert q.spread is None
        assert q.spread_pct is None

    def test_spread_computed_when_bid_ask_present(self):
        q = Quote(ticker="AAPL_US_EQ", price=100.0, bid=99.9, ask=100.1, currency="USD", as_of=_now())
        assert q.spread == pytest.approx(0.2)
        assert q.spread_pct == pytest.approx(0.002)

    def test_spread_pct_none_when_price_zero(self):
        q = Quote(ticker="X", price=0.0, bid=0.0, ask=0.0, currency="USD", as_of=_now())
        # spread is 0.0 (not None) here, but spread_pct must not divide by zero
        assert q.spread == 0.0
        assert q.spread_pct is None


class TestOrder:
    def test_market_order_is_valid(self):
        order = Order(ticker="AAPL_US_EQ", side=Side.BUY, quantity=1.0, order_type=OrderType.MARKET)
        assert order.side == Side.BUY

    def test_zero_or_negative_quantity_rejected(self):
        with pytest.raises(ValueError):
            Order(ticker="AAPL_US_EQ", side=Side.BUY, quantity=0, order_type=OrderType.MARKET)
        with pytest.raises(ValueError):
            Order(ticker="AAPL_US_EQ", side=Side.SELL, quantity=-5, order_type=OrderType.MARKET)

    def test_limit_order_requires_limit_price(self):
        with pytest.raises(ValueError):
            Order(ticker="AAPL_US_EQ", side=Side.BUY, quantity=1.0, order_type=OrderType.LIMIT)
        # doesn't raise once limit_price is supplied
        Order(ticker="AAPL_US_EQ", side=Side.BUY, quantity=1.0, order_type=OrderType.LIMIT, limit_price=100.0)

    def test_stop_order_requires_stop_price(self):
        with pytest.raises(ValueError):
            Order(ticker="AAPL_US_EQ", side=Side.SELL, quantity=1.0, order_type=OrderType.STOP)
        Order(ticker="AAPL_US_EQ", side=Side.SELL, quantity=1.0, order_type=OrderType.STOP, stop_price=90.0)

    def test_stop_limit_requires_both(self):
        with pytest.raises(ValueError):
            Order(
                ticker="AAPL_US_EQ", side=Side.BUY, quantity=1.0,
                order_type=OrderType.STOP_LIMIT, stop_price=90.0,
            )
        Order(
            ticker="AAPL_US_EQ", side=Side.BUY, quantity=1.0,
            order_type=OrderType.STOP_LIMIT, stop_price=90.0, limit_price=91.0,
        )