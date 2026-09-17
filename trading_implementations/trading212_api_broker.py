"""
trading_implementations/trading212_api_broker.py

Concrete BrokerInterface implementation against Trading212's official
Public API (v0, beta). Works for both the demo (paper) and live
environments — same API shape, different base URL.

Known gaps in the underlying API (see the interface spec doc for detail):

1. No live quote endpoint. `get_quote()` can only return a price for a
   ticker you currently hold (via /equity/positions). For anything else,
   it raises QuoteUnavailableError — there is no official way to price an
   instrument you don't already own. The decision engine needs a separate
   market-data source for candidate instruments; this broker cannot
   provide one.
2. No bid/ask anywhere. Quote.bid and Quote.ask are always None here.
3. Order endpoints are not idempotent in this beta — place_order() guards
   against accidental duplicate submission using client_order_id, but
   only within this process's lifetime (in-memory dedupe, not persisted).
4. get_quote() calls get_positions() under the hood, so calling it in a
   loop (once per held ticker) multiplies /equity/positions calls. A short
   TTL cache on get_positions() absorbs this — see positions_cache_ttl.
   Confirmed against the real demo API: without this, a two-position
   portfolio's read-only checks alone triggered a 429.

Requires the `requests` package (not yet in pyproject.toml — add it
before this module is used: `poetry add requests`).
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import requests

from trading_interface import (
    AccountSnapshot,
    BrokerInterface,
    Order,
    OrderResult,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    Side,
)


class QuoteUnavailableError(RuntimeError):
    """Raised when a price was requested for a ticker the API can't price."""


class Environment(Enum):
    DEMO = "https://demo.trading212.com/api/v0"
    LIVE = "https://live.trading212.com/api/v0"


# Rate limits taken from the API reference (docs.trading212.com/api), as of
# the beta docs read for this project. Re-check before relying on these —
# the API is explicitly still evolving.
_MIN_INTERVAL_SECONDS: dict[str, float] = {
    "GET /equity/account/summary": 5.0,
    "GET /equity/positions": 1.0,
    "GET /equity/orders": 5.0,
    "GET /equity/orders/{id}": 1.0,
    "POST /equity/orders/market": 1.2,       # 50 req / 60s
    "DELETE /equity/orders/{id}": 1.2,       # 50 req / 60s
}

# Trading212 order status -> our OrderStatus
_STATUS_MAP: dict[str, OrderStatus] = {
    "LOCAL": OrderStatus.PENDING,
    "UNCONFIRMED": OrderStatus.PENDING,
    "CONFIRMED": OrderStatus.PENDING,
    "NEW": OrderStatus.PENDING,
    "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
    "FILLED": OrderStatus.FILLED,
    "CANCELLING": OrderStatus.PENDING,
    "CANCELLED": OrderStatus.CANCELLED,
    "REJECTED": OrderStatus.REJECTED,
    "REPLACING": OrderStatus.PENDING,
    "REPLACED": OrderStatus.PENDING,
}


_MAX_RETRIES_ON_429 = 3
_DEFAULT_BACKOFF_SECONDS = 2.0  # used when the response has no Retry-After header


@dataclass
class _RateLimiter:
    """Minimal per-endpoint-key rate limiter: sleeps just enough to respect
    the documented minimum interval. Not a full token bucket — good enough
    for a single-process trading loop, not for concurrent callers.

    This is a best-effort guard, not a guarantee: the real demo environment
    has rate-limited us even while respecting these documented intervals
    (see module docstring, point 4) — callers still need 429 handling on
    top of this, not instead of it.
    """

    _last_call: dict[str, float] = None

    def __post_init__(self) -> None:
        self._last_call = {}

    def wait(self, key: str) -> None:
        min_interval = _MIN_INTERVAL_SECONDS.get(key, 0.0)
        last = self._last_call.get(key)
        if last is not None:
            elapsed = time.monotonic() - last
            remaining = min_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)
        self._last_call[key] = time.monotonic()


class Trading212APIBroker(BrokerInterface):
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        environment: Environment,
        positions_cache_ttl: float = 2.0,
    ):
        self._environment = environment
        self._session = requests.Session()
        credentials = base64.b64encode(f"{api_key}:{api_secret}".encode("utf-8")).decode("utf-8")
        self._session.headers.update({"Authorization": f"Basic {credentials}"})
        self._rate_limiter = _RateLimiter()
        # In-memory idempotency guard: client_order_id -> broker OrderResult.
        # Does not survive a process restart — acceptable for now, but a
        # real deployment should persist this (e.g. to disk or a DB) before
        # going live, so a crash-and-restart mid-order can't double-submit.
        self._submitted_orders: dict[str, OrderResult] = {}
        # Short-TTL cache for get_positions(), since get_quote() calls it
        # internally — without this, pricing N held tickers means N+1 calls
        # to /equity/positions in quick succession. See module docstring,
        # point 4 — this was observed triggering a real 429 in practice.
        self._positions_cache_ttl = positions_cache_ttl
        self._positions_cache: tuple[float, list[Position]] | None = None

    # -- internal helpers ----------------------------------------------

    def _request(self, method: str, path: str, rate_key: str, **kwargs: Any) -> requests.Response:
        url = f"{self._environment.value}{path}"
        for attempt in range(_MAX_RETRIES_ON_429 + 1):
            self._rate_limiter.wait(rate_key)
            response = self._session.request(method, url, **kwargs)
            if response.status_code != 429:
                response.raise_for_status()
                return response
            if attempt == _MAX_RETRIES_ON_429:
                response.raise_for_status()  # give up — raises HTTPError
            retry_after = response.headers.get("Retry-After")
            delay = float(retry_after) if retry_after else _DEFAULT_BACKOFF_SECONDS * (attempt + 1)
            time.sleep(delay)
        raise AssertionError("unreachable")  # loop always returns or raises

    def _get(self, path: str, rate_key: str, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", path, rate_key, params=params).json()

    def _post(self, path: str, rate_key: str, payload: dict[str, Any]) -> Any:
        return self._request("POST", path, rate_key, json=payload).json()

    def _delete(self, path: str, rate_key: str) -> None:
        self._request("DELETE", path, rate_key)

    @staticmethod
    def _to_order_result(data: dict[str, Any]) -> OrderResult:
        return OrderResult(
            broker_order_id=str(data["id"]),
            status=_STATUS_MAP.get(data["status"], OrderStatus.PENDING),
            filled_quantity=data.get("filledQuantity") or 0.0,
            filled_price=None,  # not returned directly here; see get_order/history for fills
        )

    # -- BrokerInterface --------------------------------------------------

    def get_account(self) -> AccountSnapshot:
        data = self._get("/equity/account/summary", "GET /equity/account/summary")
        return AccountSnapshot(
            currency=data["currency"],
            cash=data["cash"]["availableToTrade"],
            invested=data["investments"]["totalCost"],
            total_value=data["totalValue"],
            as_of=datetime.now(timezone.utc),
        )

    def get_positions(self, force_refresh: bool = False) -> list[Position]:
        """
        Cached for `positions_cache_ttl` seconds (default 2s) — repeated
        calls within that window (e.g. from get_quote() in a loop) reuse
        the cached list instead of hitting the API again. Pass
        force_refresh=True when you specifically need the latest state
        (e.g. right after placing an order).
        """
        if not force_refresh and self._positions_cache is not None:
            cached_at, cached_positions = self._positions_cache
            if time.monotonic() - cached_at < self._positions_cache_ttl:
                return cached_positions

        data = self._get("/equity/positions", "GET /equity/positions")
        positions = [
            Position(
                ticker=item["instrument"]["ticker"],
                quantity=item["quantity"],
                average_price_paid=item["averagePricePaid"],
                current_price=item["currentPrice"],
                unrealised_pnl=item["walletImpact"]["unrealizedProfitLoss"],
                currency=item["walletImpact"]["currency"],
            )
            for item in data
        ]
        self._positions_cache = (time.monotonic(), positions)
        return positions

    def get_quote(self, ticker: str) -> Quote:
        """
        Only works for a ticker you currently hold — the API has no way to
        price anything else. Raises QuoteUnavailableError otherwise.
        """
        for position in self.get_positions():
            if position.ticker == ticker:
                return Quote(
                    ticker=ticker,
                    price=position.current_price,
                    bid=None,
                    ask=None,
                    currency=position.currency,
                    as_of=datetime.now(timezone.utc),
                )
        raise QuoteUnavailableError(
            f"No price available for {ticker!r}: Trading212's API only exposes "
            f"prices for held positions. A candidate instrument needs an "
            f"external market-data source."
        )

    def place_order(self, order: Order) -> OrderResult:
        if order.client_order_id and order.client_order_id in self._submitted_orders:
            # Already sent this exact order in this process — return the
            # original result rather than risk a duplicate submission
            # against a non-idempotent endpoint.
            return self._submitted_orders[order.client_order_id]

        if order.order_type != OrderType.MARKET:
            raise NotImplementedError(
                f"{order.order_type} not yet implemented — market orders only for now."
            )

        signed_quantity = order.quantity if order.side == Side.BUY else -order.quantity
        payload = {
            "ticker": order.ticker,
            "quantity": signed_quantity,
            "extendedHours": order.extended_hours,
        }
        data = self._post("/equity/orders/market", "POST /equity/orders/market", payload)
        result = self._to_order_result(data)

        if order.client_order_id:
            self._submitted_orders[order.client_order_id] = result
        return result

    def cancel_order(self, broker_order_id: str) -> None:
        self._delete(f"/equity/orders/{broker_order_id}", "DELETE /equity/orders/{id}")

    def get_order(self, broker_order_id: str) -> OrderResult:
        data = self._get(f"/equity/orders/{broker_order_id}", "GET /equity/orders/{id}")
        return self._to_order_result(data)