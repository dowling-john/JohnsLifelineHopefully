"""
trading_implementations/trading212_ui_broker.py

Trading212UIBroker — a QUOTE-ONLY mixin, not a full BrokerInterface.

This deliberately implements only get_quote(). It does not implement
place_order, get_positions, etc. — those stay on Trading212APIBroker.
Keeping it single-purpose is what makes composing it via inheritance
safe: there's no overlap with Trading212APIBroker's methods for the
MRO to have to arbitrate.

⚠️ IMPORTANT — read before using this class

This talks to Trading212's *internal* web-platform API, not the official
Public API. It is not sanctioned or documented by Trading212, has no
official API key auth, and works by reusing session cookies lifted out
of a logged-in browser session (see community references below). That
means:

  - It can change or break without any notice — Trading212 owes it
    nothing, unlike the versioned, documented public API.
  - Session cookies expire and need refreshing periodically; there's no
    long-lived credential like the API key/secret pair.
  - This is materially higher risk than the official API in terms of
    both reliability and standing with Trading212 — go in eyes open,
    and don't rely on it for anything beyond read-only quote-watching
    until you've satisfied yourself it's acceptable for your use.

Community reference implementations (unofficial, unmaintained, use at
your own risk — ported here for the general shape, not verified endpoint
by endpoint): https://github.com/KO-9/node-trading212

_WS_URL below is a placeholder. Fill it in yourself after inspecting your
own browser's network traffic while logged into live.trading212.com — I'm
not going to assert an unverified private endpoint as fact.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

import websocket  # pip install websocket-client (not yet in pyproject.toml)

from trading_interface import Quote


class QuoteUnavailableError(RuntimeError):
    """Raised when no quote has been received yet for a requested ticker."""


# TODO: fill in after inspecting your own browser's WS traffic at
# live.trading212.com — this is not a verified endpoint.
_WS_URL_PLACEHOLDER = "wss://REPLACE_ME.trading212.com/REPLACE_ME"


class Trading212UIBroker:
    """
    Mixin providing get_quote() with real bid/ask, via Trading212's
    internal (unofficial) WebSocket feed.

    Not a BrokerInterface on its own — combine with an executing broker,
    e.g.:

        class Trading212Broker(Trading212UIBroker, Trading212APIBroker):
            ...
    """

    def __init__(self, customer_session: str, trading212_session_live: str):
        self._customer_session = customer_session
        self._trading212_session_live = trading212_session_live
        self._latest_quotes: dict[str, Quote] = {}
        self._lock = threading.Lock()
        self._ws: websocket.WebSocketApp | None = None
        self._ws_thread: threading.Thread | None = None

    # -- connection lifecycle --------------------------------------------

    def connect(self) -> None:
        """Open the WebSocket and start listening in a background thread."""
        cookie_header = (
            f"CUSTOMER_SESSION={self._customer_session}; "
            f"TRADING212_SESSION_LIVE={self._trading212_session_live}"
        )
        self._ws = websocket.WebSocketApp(
            _WS_URL_PLACEHOLDER,
            header=[f"Cookie: {cookie_header}"],
            on_message=self._on_message,
            on_error=self._on_error,
        )
        self._ws_thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._ws_thread.start()

    def disconnect(self) -> None:
        if self._ws is not None:
            self._ws.close()

    def subscribe(self, tickers: list[str]) -> None:
        """
        Subscribe to live price updates for the given tickers. Quotes only
        start arriving (and get_quote() only starts working) for a ticker
        after it's been subscribed here — mirrors the bulkSubscribe pattern
        in the community reference implementations.
        """
        if self._ws is None:
            raise RuntimeError("call connect() before subscribe()")
        # TODO: exact subscribe message shape is unverified — inspect your
        # own browser's WS traffic and adjust this payload to match.
        self._ws.send(json.dumps({"action": "subscribe", "tickers": tickers}))

    # -- BrokerInterface-compatible method --------------------------------

    def get_quote(self, ticker: str) -> Quote:
        with self._lock:
            quote = self._latest_quotes.get(ticker)
        if quote is None:
            raise QuoteUnavailableError(
                f"No quote received yet for {ticker!r} — make sure subscribe() "
                f"was called for it and the WebSocket connection is live."
            )
        return quote

    # -- internal ----------------------------------------------------------

    def _on_message(self, _ws, message: str) -> None:
        # TODO: exact message shape is unverified — adjust field names below
        # once you've inspected the real payloads from your own session.
        try:
            data = json.loads(message)
        except json.JSONDecodeError:
            return
        if data.get("type") != "price":
            return
        quote = Quote(
            ticker=data["ticker"],
            price=(data["bid"] + data["ask"]) / 2,
            bid=data["bid"],
            ask=data["ask"],
            currency=data.get("currency", "GBP"),
            as_of=datetime.now(timezone.utc),
        )
        with self._lock:
            self._latest_quotes[quote.ticker] = quote

    def _on_error(self, _ws, error) -> None:
        # Deliberately not swallowed silently — surface it, since a dead
        # feed means get_quote() will start raising QuoteUnavailableError
        # for tickers that were previously working.
        print(f"Trading212UIBroker WebSocket error: {error}")