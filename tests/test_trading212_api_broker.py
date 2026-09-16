"""
tests/test_trading212_api_broker.py

All HTTP calls are mocked (patched on requests.Session) — these tests
never touch the real Trading212 API, demo or live. They exist to catch
schema mismatches and logic bugs before real credentials are ever used;
see the get_account() field-name bug this caught during development.
"""

from unittest.mock import MagicMock, patch

import pytest

from trading_implementations.trading212_api_broker import Environment, QuoteUnavailableError, Trading212APIBroker
from trading_interface import Order, OrderStatus, OrderType, Side


def _mock_response(json_data, status_ok=True):
    response = MagicMock()
    response.json.return_value = json_data
    if status_ok:
        response.raise_for_status.return_value = None
    else:
        response.raise_for_status.side_effect = Exception("HTTP error")
    return response


@pytest.fixture
def broker():
    b = Trading212APIBroker(api_key="key", api_secret="secret", environment=Environment.DEMO)
    # Rate limiter would otherwise sleep between repeated calls in the same
    # test — irrelevant here and just slows the suite down.
    b._rate_limiter.wait = lambda key: None
    return b


class TestGetAccount:
    def test_maps_real_schema_correctly(self, broker):
        """
        Encodes the actual /equity/account/summary response shape (cash is
        nested; invested/total live under investments/totalValue) — this
        is exactly the shape the original implementation got wrong.
        """
        payload = {
            "currency": "GBP",
            "id": 12345,
            "cash": {"availableToTrade": 150.0, "inPies": 10.0, "reservedForOrders": 5.0},
            "investments": {
                "currentValue": 900.0,
                "totalCost": 850.0,
                "realizedProfitLoss": 20.0,
                "unrealizedProfitLoss": 50.0,
            },
            "totalValue": 1050.0,
        }
        with patch.object(broker._session, "get", return_value=_mock_response(payload)):
            account = broker.get_account()

        assert account.currency == "GBP"
        assert account.cash == 150.0
        assert account.invested == 850.0
        assert account.total_value == 1050.0


class TestGetPositions:
    def test_maps_positions_correctly(self, broker):
        payload = [
            {
                "instrument": {"ticker": "AAPL_US_EQ", "currency": "USD", "isin": "x", "name": "Apple"},
                "quantity": 2.0,
                "averagePricePaid": 180.0,
                "currentPrice": 190.0,
                "quantityAvailableForTrading": 2.0,
                "quantityInPies": 0.0,
                "walletImpact": {
                    "currency": "USD",
                    "currentValue": 380.0,
                    "totalCost": 360.0,
                    "unrealizedProfitLoss": 20.0,
                    "fxImpact": 0.0,
                },
            }
        ]
        with patch.object(broker._session, "get", return_value=_mock_response(payload)):
            positions = broker.get_positions()

        assert len(positions) == 1
        p = positions[0]
        assert p.ticker == "AAPL_US_EQ"
        assert p.quantity == 2.0
        assert p.average_price_paid == 180.0
        assert p.current_price == 190.0
        assert p.unrealised_pnl == 20.0
        assert p.currency == "USD"


class TestGetQuote:
    def test_returns_price_for_held_ticker(self, broker):
        payload = [
            {
                "instrument": {"ticker": "AAPL_US_EQ", "currency": "USD", "isin": "x", "name": "Apple"},
                "quantity": 2.0,
                "averagePricePaid": 180.0,
                "currentPrice": 190.0,
                "quantityAvailableForTrading": 2.0,
                "quantityInPies": 0.0,
                "walletImpact": {
                    "currency": "USD", "currentValue": 380.0, "totalCost": 360.0,
                    "unrealizedProfitLoss": 20.0, "fxImpact": 0.0,
                },
            }
        ]
        with patch.object(broker._session, "get", return_value=_mock_response(payload)):
            quote = broker.get_quote("AAPL_US_EQ")

        assert quote.price == 190.0
        assert quote.bid is None
        assert quote.ask is None

    def test_raises_for_non_held_ticker(self, broker):
        with patch.object(broker._session, "get", return_value=_mock_response([])):
            with pytest.raises(QuoteUnavailableError):
                broker.get_quote("MSFT_US_EQ")


class TestPlaceOrder:
    def _order_payload(self, order_id=1, status="NEW", filled_qty=0.0):
        return {
            "id": order_id,
            "status": status,
            "filledQuantity": filled_qty,
            "ticker": "AAPL_US_EQ",
            "type": "MARKET",
        }

    def test_market_buy_sends_positive_quantity(self, broker):
        with patch.object(broker._session, "post", return_value=_mock_response(self._order_payload())) as post:
            order = Order(ticker="AAPL_US_EQ", side=Side.BUY, quantity=3.0, order_type=OrderType.MARKET)
            result = broker.place_order(order)

        sent_payload = post.call_args.kwargs["json"]
        assert sent_payload["quantity"] == 3.0
        assert result.broker_order_id == "1"
        assert result.status == OrderStatus.PENDING

    def test_market_sell_sends_negative_quantity(self, broker):
        with patch.object(broker._session, "post", return_value=_mock_response(self._order_payload())) as post:
            order = Order(ticker="AAPL_US_EQ", side=Side.SELL, quantity=3.0, order_type=OrderType.MARKET)
            broker.place_order(order)

        sent_payload = post.call_args.kwargs["json"]
        assert sent_payload["quantity"] == -3.0

    def test_non_market_order_not_implemented(self, broker):
        order = Order(
            ticker="AAPL_US_EQ", side=Side.BUY, quantity=1.0,
            order_type=OrderType.LIMIT, limit_price=100.0,
        )
        with pytest.raises(NotImplementedError):
            broker.place_order(order)

    def test_idempotency_guard_prevents_duplicate_http_call(self, broker):
        order = Order(
            ticker="AAPL_US_EQ", side=Side.BUY, quantity=1.0,
            order_type=OrderType.MARKET, client_order_id="abc-123",
        )
        with patch.object(broker._session, "post", return_value=_mock_response(self._order_payload())) as post:
            first = broker.place_order(order)
            second = broker.place_order(order)

        assert post.call_count == 1, "second call with the same client_order_id must not hit the API again"
        assert first == second

    def test_different_client_order_ids_both_send(self, broker):
        order1 = Order(
            ticker="AAPL_US_EQ", side=Side.BUY, quantity=1.0,
            order_type=OrderType.MARKET, client_order_id="abc-1",
        )
        order2 = Order(
            ticker="AAPL_US_EQ", side=Side.BUY, quantity=1.0,
            order_type=OrderType.MARKET, client_order_id="abc-2",
        )
        with patch.object(
            broker._session, "post",
            side_effect=[_mock_response(self._order_payload(order_id=1)), _mock_response(self._order_payload(order_id=2))],
        ) as post:
            broker.place_order(order1)
            broker.place_order(order2)

        assert post.call_count == 2


class TestCancelAndGetOrder:
    def test_cancel_order_calls_correct_url(self, broker):
        with patch.object(broker._session, "delete", return_value=_mock_response({})) as delete:
            broker.cancel_order("42")
        called_url = delete.call_args.args[0]
        assert called_url.endswith("/equity/orders/42")

    def test_get_order_maps_result(self, broker):
        payload = {"id": 42, "status": "FILLED", "filledQuantity": 3.0, "ticker": "AAPL_US_EQ", "type": "MARKET"}
        with patch.object(broker._session, "get", return_value=_mock_response(payload)):
            result = broker.get_order("42")
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == 3.0


@pytest.mark.parametrize(
    "t212_status,expected",
    [
        ("LOCAL", OrderStatus.PENDING),
        ("CONFIRMED", OrderStatus.PENDING),
        ("PARTIALLY_FILLED", OrderStatus.PARTIALLY_FILLED),
        ("FILLED", OrderStatus.FILLED),
        ("CANCELLED", OrderStatus.CANCELLED),
        ("REJECTED", OrderStatus.REJECTED),
    ],
)
def test_status_mapping(broker, t212_status, expected):
    payload = {"id": 1, "status": t212_status, "filledQuantity": 0.0, "ticker": "AAPL_US_EQ", "type": "MARKET"}
    with patch.object(broker._session, "get", return_value=_mock_response(payload)):
        result = broker.get_order("1")
    assert result.status == expected