"""
tests/test_trading212_ui_broker.py

No real WebSocket connection is ever opened — websocket.WebSocketApp and
threading.Thread are both patched. These tests validate the message
parsing, the cookie header construction, and get_quote()'s behaviour,
not real connectivity to Trading212.
"""

from unittest.mock import MagicMock, patch

import pytest

from trading_implementations.trading212_ui_broker import QuoteUnavailableError, Trading212UIBroker


@pytest.fixture
def ui_broker():
    return Trading212UIBroker(customer_session="cust-session-abc", trading212_session_live="live-session-xyz")


class TestGetQuote:
    def test_raises_when_nothing_received_yet(self, ui_broker):
        with pytest.raises(QuoteUnavailableError):
            ui_broker.get_quote("AAPL_US_EQ")

    def test_returns_quote_after_message_received(self, ui_broker):
        message = '{"type": "price", "ticker": "AAPL_US_EQ", "bid": 190.0, "ask": 190.4, "currency": "USD"}'
        ui_broker._on_message(None, message)

        quote = ui_broker.get_quote("AAPL_US_EQ")
        assert quote.bid == 190.0
        assert quote.ask == 190.4
        assert quote.price == pytest.approx(190.2)
        assert quote.spread == pytest.approx(0.4)

    def test_ignores_non_price_messages(self, ui_broker):
        message = '{"type": "account", "cash": 100.0}'
        ui_broker._on_message(None, message)
        with pytest.raises(QuoteUnavailableError):
            ui_broker.get_quote("AAPL_US_EQ")

    def test_ignores_malformed_json(self, ui_broker):
        # must not raise — a malformed frame shouldn't crash the listener thread
        ui_broker._on_message(None, "not valid json{{{")

    def test_quote_for_one_ticker_does_not_affect_another(self, ui_broker):
        ui_broker._on_message(
            None, '{"type": "price", "ticker": "AAPL_US_EQ", "bid": 190.0, "ask": 190.4, "currency": "USD"}'
        )
        with pytest.raises(QuoteUnavailableError):
            ui_broker.get_quote("MSFT_US_EQ")


class TestConnect:
    def test_builds_cookie_header_from_session_values(self, ui_broker):
        with patch("trading_implementations.trading212_ui_broker.websocket.WebSocketApp") as mock_ws_app, \
             patch("trading_implementations.trading212_ui_broker.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            ui_broker.connect()

        _, kwargs = mock_ws_app.call_args
        headers = kwargs["header"]
        assert any("CUSTOMER_SESSION=cust-session-abc" in h for h in headers)
        assert any("TRADING212_SESSION_LIVE=live-session-xyz" in h for h in headers)
        mock_thread.return_value.start.assert_called_once()


class TestSubscribe:
    def test_raises_if_not_connected(self, ui_broker):
        with pytest.raises(RuntimeError):
            ui_broker.subscribe(["AAPL_US_EQ"])

    def test_sends_subscribe_message_once_connected(self, ui_broker):
        with patch("trading_implementations.trading212_ui_broker.websocket.WebSocketApp") as mock_ws_app, \
             patch("trading_implementations.trading212_ui_broker.threading.Thread") as mock_thread:
            mock_thread.return_value = MagicMock()
            ui_broker.connect()
            ui_broker.subscribe(["AAPL_US_EQ", "MSFT_US_EQ"])

        mock_ws_app.return_value.send.assert_called_once()