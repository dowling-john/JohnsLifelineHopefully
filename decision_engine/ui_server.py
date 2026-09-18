"""
decision_engine/ui_server.py

Local web server: runs the CandleFeed against a signal model in a
background thread, and broadcasts every SentimentEvent to connected
browsers over a WebSocket. Serves a single static page (static/index.html)
that renders the live sentiment stream.

This has NOT been run against a real browser or real yfinance data from
this environment — no network path to Yahoo Finance here, and no browser
to click through. The WebSocket wiring itself is tested (see
tests/test_ui_server.py, using FastAPI's TestClient — real WebSocket
protocol, no network), but "does it look right in an actual browser" is
for you to check.

Run it via the repo's decision-engine entry point (see decision_engine_main.py)
rather than `uvicorn` directly, so the feed starts automatically:
    python decision_engine_main.py --ticker AAPL
Then open http://localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from decision_engine.data.candle_feed import CandleFeed, load_historical_candles
from decision_engine.events import Event, EventBus
from decision_engine.signals.momentum_signal import MomentumSignalModel

_bus = EventBus()
_connections: list[WebSocket] = []
_connections_lock = threading.Lock()
_main_loop: asyncio.AbstractEventLoop | None = None
_startup_config: dict | None = None


def configure(ticker: str, period: str = "60d", interval: str = "5m", rate_hz: float = 60.0) -> None:
    """
    Call before uvicorn.run() to have the feed start automatically once
    the server's event loop is up (see main.py). This is deliberately
    separate from start_feed() itself: start_feed() fetches data
    synchronously and would block server startup on a network call if
    invoked directly before uvicorn.run(); configure() just records the
    intent, and the actual fetch+feed happens inside _lifespan, after the
    loop exists — so there's no race where a sentiment event arrives
    before _main_loop is set and gets silently dropped.
    """
    global _startup_config
    _startup_config = {"ticker": ticker, "period": period, "interval": interval, "rate_hz": rate_hz}


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    global _main_loop
    _main_loop = asyncio.get_event_loop()
    if _startup_config is not None:
        start_feed(**_startup_config)
    yield


app = FastAPI(lifespan=_lifespan)


def _broadcast(event: Event) -> None:
    """
    Called from the feeder's background thread — hops onto the asyncio
    event loop to actually send, since WebSocket sends aren't thread-safe
    to call directly from another thread.
    """
    if _main_loop is None:
        return
    payload = json.dumps(
        {
            "event_type": event.event_type,
            "source": event.source,
            "ticker": getattr(event, "ticker", None),
            "sentiment": getattr(event, "sentiment", None),
            "confidence": getattr(event, "confidence", None),
            "as_of": event.as_of.isoformat(),
        }
    )
    with _connections_lock:
        targets = list(_connections)
    for ws in targets:
        asyncio.run_coroutine_threadsafe(_safe_send(ws, payload), _main_loop)


async def _safe_send(ws: WebSocket, payload: str) -> None:
    try:
        await ws.send_text(payload)
    except Exception:
        with _connections_lock:
            if ws in _connections:
                _connections.remove(ws)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    with _connections_lock:
        _connections.append(websocket)
    try:
        while True:
            await websocket.receive_text()  # just keeps the connection alive; ignores client messages
    except WebSocketDisconnect:
        with _connections_lock:
            if websocket in _connections:
                _connections.remove(websocket)


def start_feed(ticker: str = "AAPL", period: str = "60d", interval: str = "5m", rate_hz: float = 60.0) -> None:
    """
    Starts a background thread that fetches real historical data (this
    fetch happens ON the background thread, not here — so calling this
    never blocks the caller on a network round-trip) and replays it
    through the model. Safe to call directly once the server's event
    loop is already running (e.g. from a request handler); to have it
    start automatically at server startup, use configure() instead — see
    its docstring for why that indirection exists.
    """
    _bus.subscribe("sentiment", _broadcast)
    model = MomentumSignalModel(weights_path="/Users/johndowling/Documents/Projects/JohnsLifelineHopefully/decision_engine/signals/weights/momentum_v0.weights.h5")

    def _run() -> None:
        candles = load_historical_candles(ticker, period=period, interval=interval)
        CandleFeed(model, _bus).run(candles, ticker=ticker, rate_hz=rate_hz)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()


def start_feed_from_dataframe(candles: pd.DataFrame, ticker: str, rate_hz: float = 60.0) -> None:
    """Same as start_feed, but takes an already-loaded DataFrame — useful
    for testing or for a data source other than yfinance."""
    _bus.subscribe("sentiment", _broadcast)
    model = MomentumSignalModel()

    def _run() -> None:
        CandleFeed(model, _bus).run(candles, ticker=ticker, rate_hz=rate_hz)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()