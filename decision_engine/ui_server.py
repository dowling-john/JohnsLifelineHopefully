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
import logging
import threading
from collections import deque
from pathlib import Path

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from decision_engine.data.candle_feed import CandleFeed, LiveCandleFeed, load_historical_candles
from decision_engine.events import Event, EventBus
from decision_engine.signals.base import SignalModel
from decision_engine.signals.candlestick_signal import CandlestickSignalModel
from decision_engine.signals.divergence_signal import DivergenceSignalModel
from decision_engine.signals.flag_signal import FlagSignalModel
from decision_engine.signals.momentum_signal import MomentumSignalModel
from decision_engine.signals.rsi_signal import RsiSignalModel
from decision_engine.signals.squeeze_signal import SqueezeSignalModel
from decision_engine.signals.support_resistance_signal import SupportResistanceSignalModel
from decision_engine.training.train_candlestick import DEFAULT_WEIGHTS_PATH as CANDLESTICK_DEFAULT_WEIGHTS_PATH
from decision_engine.training.train_divergence import DEFAULT_WEIGHTS_PATH as DIVERGENCE_DEFAULT_WEIGHTS_PATH
from decision_engine.training.train_flag import DEFAULT_WEIGHTS_PATH as FLAG_DEFAULT_WEIGHTS_PATH
from decision_engine.training.train_momentum import DEFAULT_WEIGHTS_PATH as MOMENTUM_DEFAULT_WEIGHTS_PATH
from decision_engine.training.train_rsi import DEFAULT_WEIGHTS_PATH as RSI_DEFAULT_WEIGHTS_PATH
from decision_engine.training.train_squeeze import DEFAULT_WEIGHTS_PATH as SQUEEZE_DEFAULT_WEIGHTS_PATH
from decision_engine.training.train_support_resistance import (
    DEFAULT_WEIGHTS_PATH as SUPPORT_RESISTANCE_DEFAULT_WEIGHTS_PATH,
)

# Every active indicator, declaratively — add a new (ModelClass,
# default_weights_path) pair here and _load_models(), CandleFeed/
# LiveCandleFeed, /api/status, and the frontend's indicator grid all pick
# it up automatically; nothing else in this file needs to change.
ACTIVE_MODELS: list[tuple[type[SignalModel], str]] = [
    (MomentumSignalModel, MOMENTUM_DEFAULT_WEIGHTS_PATH),
    (FlagSignalModel, FLAG_DEFAULT_WEIGHTS_PATH),
    (RsiSignalModel, RSI_DEFAULT_WEIGHTS_PATH),
    (SqueezeSignalModel, SQUEEZE_DEFAULT_WEIGHTS_PATH),
    (DivergenceSignalModel, DIVERGENCE_DEFAULT_WEIGHTS_PATH),
    (SupportResistanceSignalModel, SUPPORT_RESISTANCE_DEFAULT_WEIGHTS_PATH),
    (CandlestickSignalModel, CANDLESTICK_DEFAULT_WEIGHTS_PATH),
]

logger = logging.getLogger(__name__)

_bus = EventBus()
_connections: list[WebSocket] = []
_connections_lock = threading.Lock()
_main_loop: asyncio.AbstractEventLoop | None = None
_startup_config: dict | None = None
_broadcast_subscribed = False
_feed_thread: threading.Thread | None = None
_feed_stop_event: threading.Event | None = None
# Which feed is currently driving events — read by _broadcast so the UI is
# told honestly whether a sentiment value is a live decision or a replayed
# historical one. Only one feed runs per process, so a module-level flag is
# enough (set by start_feed/start_feed_from_dataframe vs start_live_feed).
_is_replay = True
_model_status: list[dict] = []  # [{"name": ..., "window_size": ...}, ...], set from the running models, exposed via /api/status
# Which feed is actually running right now, for /api/status — distinct from
# _startup_config, which only reflects configure()'s *intended* startup feed
# and goes stale if start_feed/start_live_feed is later called directly.
_current_ticker: str | None = None
_current_interval: str | None = None
_current_poll_seconds: float | None = None

# Rolling buffer of recently-seen candles (as the same dicts _broadcast sends
# over the socket) — lets /api/candles hand a fresh connection instant price
# history instead of it sitting empty until enough "candle" WS messages
# arrive on their own.
_CANDLE_HISTORY_MAXLEN = 500
_candle_history: deque[dict] = deque(maxlen=_CANDLE_HISTORY_MAXLEN)
_history_lock = threading.Lock()

_FRONTEND_DIST = Path(__file__).parent.parent / "frontend" / "dist"


def _load_models() -> list[SignalModel]:
    """
    Builds every model in ACTIVE_MODELS, each independently trying its own
    default trained weights file and falling back to (and warning about)
    a randomly-initialised network if it isn't there yet — e.g. a
    not-yet-trained indicator doesn't block the others from working.
    """
    models = []
    for model_cls, default_weights_path in ACTIVE_MODELS:
        path = Path(default_weights_path)
        if path.exists():
            models.append(model_cls(weights_path=path))
        else:
            logger.warning("No trained weights found at %s — running %s with randomly initialised weights.",
                            path, model_cls.__name__)
            models.append(model_cls())
    return models


def _ensure_broadcast_subscribed() -> None:
    """
    Subscribing _broadcast on every start_feed() call would duplicate the
    subscription — and therefore duplicate every message sent — if
    start_feed is ever called more than once in a process (e.g. the feed
    is restarted without restarting the server). Guard against that here
    rather than relying on every call site to remember not to.
    """
    global _broadcast_subscribed
    if not _broadcast_subscribed:
        _bus.subscribe("candle", _broadcast)
        _bus.subscribe("candle", _record_candle_history)
        _bus.subscribe("sentiment", _broadcast)
        _broadcast_subscribed = True


def configure(
    ticker: str,
    period: str = "60d",
    interval: str = "5m",
    rate_hz: float = 60.0,
    live: bool = True,
    poll_seconds: float = 15.0,
    lookback_period: str = "5d",
) -> None:
    """
    Call before uvicorn.run() to have the feed start automatically once
    the server's event loop is up (see main.py). This is deliberately
    separate from start_feed()/start_live_feed() themselves: they fetch
    data synchronously and would block server startup on a network call
    if invoked directly before uvicorn.run(); configure() just records
    the intent, and the actual fetch+feed happens inside _lifespan, after
    the loop exists — so there's no race where a sentiment event arrives
    before _main_loop is set and gets silently dropped.

    live=True (the default) drives real-time decisions off the current
    market via start_live_feed — this is what "is the model looking at
    now" means for actual trading use. live=False falls back to the old
    start_feed historical replay, which is only useful for a fast demo /
    sanity check that the plumbing works, not for live decisions.
    """
    global _startup_config
    _startup_config = {
        "ticker": ticker,
        "period": period,
        "interval": interval,
        "rate_hz": rate_hz,
        "live": live,
        "poll_seconds": poll_seconds,
        "lookback_period": lookback_period,
    }


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    global _main_loop
    _main_loop = asyncio.get_event_loop()
    if _startup_config is not None:
        config = dict(_startup_config)
        if config.pop("live"):
            start_live_feed(
                ticker=config["ticker"],
                interval=config["interval"],
                poll_seconds=config["poll_seconds"],
                lookback_period=config["lookback_period"],
            )
        else:
            start_feed(ticker=config["ticker"], period=config["period"], interval=config["interval"],
                       rate_hz=config["rate_hz"])
    yield


app = FastAPI(lifespan=_lifespan)

# Serves the built React app's JS/CSS (frontend/dist/assets/*, Vite's
# default output layout) at the same absolute paths its built index.html
# references. Guarded on existence: mounting a missing directory raises at
# import time, and frontend/dist won't exist until `npm run build` has been
# run at least once — see index()'s fallback to the legacy static page for
# the same reasoning applied to "/".
if (_FRONTEND_DIST / "assets").exists():
    app.mount("/assets", StaticFiles(directory=_FRONTEND_DIST / "assets"), name="frontend-assets")


def _serialize_event(event: Event) -> dict:
    """
    Shared shape for an event whether it goes out over the WebSocket (see
    _broadcast) or is handed back from GET /api/candles (see
    _record_candle_history) — one schema for the frontend to parse either
    way. is_replay reflects whichever feed (start_feed/
    start_feed_from_dataframe vs start_live_feed) is actually running —
    see the module-level docstring on `_is_replay` — so the UI can say
    honestly whether a value is a live decision or a replayed historical
    one (see index.html / the frontend).
    """
    return {
        "event_type": event.event_type,
        "source": event.source,
        "ticker": getattr(event, "ticker", None),
        "sentiment": getattr(event, "sentiment", None),
        "confidence": getattr(event, "confidence", None),
        "open": getattr(event, "open", None),
        "high": getattr(event, "high", None),
        "low": getattr(event, "low", None),
        "close": getattr(event, "close", None),
        "volume": getattr(event, "volume", None),
        "candle_timestamp": event.timestamp.isoformat() if getattr(event, "timestamp", None) else None,
        "as_of": event.as_of.isoformat(),
        "is_replay": _is_replay,
    }


def _broadcast(event: Event) -> None:
    """
    Called from the feeder's background thread — hops onto the asyncio
    event loop to actually send, since WebSocket sends aren't thread-safe
    to call directly from another thread.

    Sends both CandleEvent (price data, every row) and SentimentEvent
    (model output, once the window fills) over the same connection — the
    client tells them apart by event_type.
    """
    if _main_loop is None:
        return
    payload = json.dumps(_serialize_event(event))
    with _connections_lock:
        targets = list(_connections)
    for ws in targets:
        asyncio.run_coroutine_threadsafe(_safe_send(ws, payload), _main_loop)


def _record_candle_history(event: Event) -> None:
    """Keeps the rolling buffer GET /api/candles serves in sync with what's
    being broadcast — called directly from the feeder thread, so just needs
    the lock, no event-loop hop like _broadcast."""
    with _history_lock:
        _candle_history.append(_serialize_event(event))


async def _safe_send(ws: WebSocket, payload: str) -> None:
    try:
        await ws.send_text(payload)
    except Exception:
        with _connections_lock:
            if ws in _connections:
                _connections.remove(ws)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    """
    Serves the built React app (frontend/dist/index.html) once it exists;
    falls back to the legacy static page otherwise, so the server still
    works standalone before anyone's run `npm run build` in frontend/.
    """
    frontend_index = _FRONTEND_DIST / "index.html"
    if frontend_index.exists():
        return frontend_index.read_text()
    html_path = Path(__file__).parent / "static" / "index.html"
    return html_path.read_text()


@app.get("/favicon.svg")
async def favicon() -> Response:
    """The one file Vite's build (public/favicon.svg) emits at the dist
    root rather than under /assets, so the /assets mount alone won't serve
    it."""
    path = _FRONTEND_DIST / "favicon.svg"
    if not path.exists():
        raise HTTPException(status_code=404)
    return Response(content=path.read_bytes(), media_type="image/svg+xml")


@app.get("/api/status")
async def api_status() -> dict:
    """Lets the frontend know what it's actually looking at — ticker,
    candle interval, live vs replay, and each active model's name/window
    size (how many candles must arrive before that model's sentiment
    starts appearing — different models can need different amounts)."""
    return {
        "ticker": _current_ticker,
        "interval": _current_interval,
        "mode": "replay" if _is_replay else "live",
        "poll_seconds": _current_poll_seconds,
        "models": _model_status,
    }


@app.get("/api/candles")
async def api_candles() -> list[dict]:
    """Recent candle history for the frontend to seed its chart with on
    load/reconnect — see _record_candle_history for how this fills up."""
    with _history_lock:
        return list(_candle_history)


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


def _stop_existing_feed() -> None:
    """Signals any previously-started LiveCandleFeed to stop polling, and
    clears the candle history buffer. Only relevant when
    start_feed/start_live_feed is called more than once in a process
    (e.g. switching ticker without restarting the server) — CandleFeed's
    replay has no equivalent stop hook since it just runs to completion
    on its own, and the history buffer needs clearing either way so a
    ticker switch doesn't leave stale candles from the old ticker in
    /api/candles."""
    global _feed_stop_event
    if _feed_stop_event is not None:
        _feed_stop_event.set()
    _feed_stop_event = None
    with _history_lock:
        _candle_history.clear()


def start_feed(
    ticker: str = "AAPL",
    period: str = "60d",
    interval: str = "5m",
    rate_hz: float = 60.0,
) -> threading.Thread:
    """
    Starts a background thread that fetches real historical data (this
    fetch happens ON the background thread, not here — so calling this
    never blocks the caller on a network round-trip) and fast-replays it
    through every active model (see _load_models). This is NOT the
    live/real-time path — it plays back a fixed historical batch at an
    artificial rate_hz, useful for a quick demo or sanity check that the
    plumbing works. For actual real-time decisions, use start_live_feed
    instead. Safe to call directly once the server's event loop is
    already running (e.g. from a request handler); to have it start
    automatically at server startup, use configure() instead — see its
    docstring for why that indirection exists.

    Returns the Thread. The live UI use case should ignore this (fire and
    forget) — it's exposed so callers that need determinism (tests, a
    one-shot script) can thread.join() it and know the feed has fully
    finished, rather than the process just happening to still be running
    when they check.
    """
    global _feed_thread, _is_replay, _model_status, _current_ticker, _current_interval, _current_poll_seconds
    _stop_existing_feed()
    _ensure_broadcast_subscribed()
    _is_replay = True
    models = _load_models()
    _model_status = [{"name": m.name, "window_size": m.window_size()} for m in models]
    _current_ticker, _current_interval, _current_poll_seconds = ticker, interval, None

    def _run() -> None:
        candles = load_historical_candles(ticker, period=period, interval=interval)
        CandleFeed(models, _bus).run(candles, ticker=ticker, rate_hz=rate_hz)

    thread = threading.Thread(target=_run, daemon=True)
    _feed_thread = thread
    thread.start()
    return thread


def start_feed_from_dataframe(candles: pd.DataFrame, ticker: str, rate_hz: float = 60.0) -> threading.Thread:
    """Same as start_feed, but takes an already-loaded DataFrame — useful
    for testing or for a data source other than yfinance. See start_feed's
    docstring re: the returned Thread."""
    global _feed_thread, _is_replay, _model_status, _current_ticker, _current_interval, _current_poll_seconds
    _stop_existing_feed()
    _ensure_broadcast_subscribed()
    _is_replay = True
    models = _load_models()
    _model_status = [{"name": m.name, "window_size": m.window_size()} for m in models]
    _current_ticker, _current_interval, _current_poll_seconds = ticker, None, None

    def _run() -> None:
        CandleFeed(models, _bus).run(candles, ticker=ticker, rate_hz=rate_hz)

    thread = threading.Thread(target=_run, daemon=True)
    _feed_thread = thread
    thread.start()
    return thread


def start_live_feed(
    ticker: str = "AAPL",
    interval: str = "5m",
    poll_seconds: float = 15.0,
    lookback_period: str = "5d",
) -> threading.Thread:
    """
    The real-time path: seeds every active model's window from recent
    history, then polls for the current market's latest candle on
    `poll_seconds` and publishes a fresh SentimentEvent per model each
    time a new candle closes — see LiveCandleFeed. This is what should be
    running for actual trading decisions; start_feed's fast historical
    replay is for demos only. Same fire-and-forget/thread.join() contract
    as start_feed.
    """
    global _feed_thread, _feed_stop_event, _is_replay, _model_status
    global _current_ticker, _current_interval, _current_poll_seconds
    _stop_existing_feed()
    _ensure_broadcast_subscribed()
    _is_replay = False
    models = _load_models()
    _model_status = [{"name": m.name, "window_size": m.window_size()} for m in models]
    _current_ticker, _current_interval, _current_poll_seconds = ticker, interval, poll_seconds
    stop_event = threading.Event()
    _feed_stop_event = stop_event

    def _run() -> None:
        LiveCandleFeed(models, _bus).run(
            ticker=ticker,
            interval=interval,
            poll_seconds=poll_seconds,
            lookback_period=lookback_period,
            stop_event=stop_event,
        )

    thread = threading.Thread(target=_run, daemon=True)
    _feed_thread = thread
    thread.start()
    return thread