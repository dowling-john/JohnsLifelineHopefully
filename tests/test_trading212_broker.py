"""
tests/test_trading212_broker.py

Confirms the Trading212UIBroker + Trading212APIBroker composition
resolves methods to the intended parent — the whole point of the
mixin design, and easy to silently get wrong if the class structure
changes later.
"""

from unittest.mock import patch

from trading_implementations.trading212_api_broker import Environment, Trading212APIBroker
from trading_implementations.trading212_broker import Trading212Broker
from trading_implementations.trading212_ui_broker import Trading212UIBroker


class TestMethodResolution:
    def test_get_quote_resolves_to_ui_broker(self):
        assert Trading212Broker.get_quote is Trading212UIBroker.get_quote

    def test_get_account_resolves_to_api_broker(self):
        assert Trading212Broker.get_account is Trading212APIBroker.get_account

    def test_place_order_resolves_to_api_broker(self):
        assert Trading212Broker.place_order is Trading212APIBroker.place_order

    def test_get_positions_resolves_to_api_broker(self):
        assert Trading212Broker.get_positions is Trading212APIBroker.get_positions

    def test_cancel_order_resolves_to_api_broker(self):
        assert Trading212Broker.cancel_order is Trading212APIBroker.cancel_order

    def test_get_order_resolves_to_api_broker(self):
        assert Trading212Broker.get_order is Trading212APIBroker.get_order


class TestInit:
    def test_both_parents_initialised(self):
        broker = Trading212Broker(
            api_key="key",
            api_secret="secret",
            environment=Environment.DEMO,
            customer_session="cust",
            trading212_session_live="live",
        )
        # API-broker state present
        assert broker._environment == Environment.DEMO
        assert "Authorization" in broker._session.headers
        # UI-broker state present
        assert broker._customer_session == "cust"
        assert broker._trading212_session_live == "live"
        assert broker._latest_quotes == {}


class TestStartStop:
    def test_start_calls_connect(self):
        broker = Trading212Broker(
            api_key="key", api_secret="secret", environment=Environment.DEMO,
            customer_session="cust", trading212_session_live="live",
        )
        with patch.object(broker, "connect") as connect:
            broker.start()
        connect.assert_called_once()

    def test_stop_calls_disconnect(self):
        broker = Trading212Broker(
            api_key="key", api_secret="secret", environment=Environment.DEMO,
            customer_session="cust", trading212_session_live="live",
        )
        with patch.object(broker, "disconnect") as disconnect:
            broker.stop()
        disconnect.assert_called_once()