"""
decision_engine/data/candle_feed.py

Three separate concerns, deliberately kept apart:

1. load_historical_candles() — fetches real historical OHLCV data (via
   yfinance), and flattens the MultiIndex columns yfinance now returns
   (('Open', 'AAPL'), ('High', 'AAPL'), ... even for a single ticker) down
   to plain 'Open'/'High'/etc. — every consumer below expects flat OHLCV
   column names and breaks confusingly otherwise (a MultiIndex Series
   looked up by a single label returns a length-1 Series, not a scalar).
2. CandleFeed — replays an already-loaded candle series at a controllable
   rate, maintains the rolling window, and drives a SignalModel. Use this
   for backtesting / demoing against a fixed historical batch, NOT for
   live decision-making — it has no concept of "now", it just fast-plays
   whatever DataFrame it's handed. This is pure logic with no network
   dependency, so it's fully testable with a synthetic DataFrame — see
   tests/test_candle_feed.py.
3. LiveCandleFeed — the actual real-time path. Seeds its window from
   recent history, then polls for the current/most-recently-completed
   candle on a timer and feeds each new one through the model as it
   arrives, so a prediction reflects the current market rather than a
   historical batch replayed at an artificial rate. The network fetch is
   injectable (see `fetch`), so the polling/windowing logic is testable
   the same way CandleFeed is, without hitting yfinance or real wall-clock
   time.

Keeping these separate means the replay/windowing logic (the part that's
actually easy to get subtly wrong — off-by-one windows, stale buffers) is
tested independently of whether yfinance is reachable today.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from decision_engine.events import CandleEvent, EventBus
from decision_engine.signals.base import SignalModel

logger = logging.getLogger(__name__)

OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _predict_ready_models(models: Sequence[SignalModel], window: deque[np.ndarray], ticker: str) -> list:
    """
    Given the shared candle window (sized to the largest model's
    window_size — see CandleFeed/LiveCandleFeed), predicts with every
    model that already has enough history, each against its own most
    recent window_size() candles sliced off the same shared window. A
    model needing fewer candles than another starts publishing sooner —
    e.g. a 30-candle momentum model and a 45-candle flag model sharing one
    45-candle window.
    """
    events = []
    window_list: list[np.ndarray] | None = None
    for model in models:
        size = model.window_size()
        if len(window) >= size:
            if window_list is None:
                window_list = list(window)
            window_array = np.stack(window_list[-size:])
            events.append(model.predict(ticker, window_array))
    return events


def _candle_event(ticker: str, timestamp: pd.Timestamp, row: pd.Series) -> CandleEvent:
    """Builds the CandleEvent for one OHLCV row — shared by CandleFeed and
    LiveCandleFeed so the two publish identically-shaped price data."""
    return CandleEvent(
        event_type="candle",
        source="candle_feed",
        ticker=ticker,
        open=float(row["Open"]),
        high=float(row["High"]),
        low=float(row["Low"]),
        close=float(row["Close"]),
        volume=float(row["Volume"]),
        timestamp=timestamp.to_pydatetime() if hasattr(timestamp, "to_pydatetime") else timestamp,
    )


def load_historical_candles(ticker: str, period: str = "60d", interval: str = "5m") -> pd.DataFrame:
    """
    Fetch historical OHLCV via yfinance. Confirmed against the real network
    (2026-09) with the yfinance version pinned in pyproject.toml: for a
    single ticker it currently returns MultiIndex columns shaped
    (field, ticker) — e.g. ('Open', 'AAPL') — rather than plain 'Open',
    which is flattened away below. yfinance's return shape has changed
    across versions before, so if this breaks again on an upgrade, check
    `yf.download(...).columns` first.
    """
    import yfinance as yf  # imported lazily so the rest of this module has no hard dependency on it

    data = yf.download(ticker, period=period, interval=interval, progress=False)
    if data.empty:
        raise ValueError(f"yfinance returned no data for {ticker!r} (period={period!r}, interval={interval!r})")
    if isinstance(data.columns, pd.MultiIndex):
        data = data.copy()
        data.columns = data.columns.get_level_values(0)
    return data[OHLCV_COLUMNS]


class CandleFeed:
    """
    Replays a candle DataFrame through one or more SignalModels at a
    controllable rate, off one shared rolling window (sized to the
    largest model's window_size — see _predict_ready_models). Publishes a
    CandleEvent for every candle (price data, so a UI has something to
    show immediately) and, once each model's own window is full, that
    model's SentimentEvent too — so models with different window_size()s
    (e.g. momentum's 30 vs flag's 45) start publishing at different
    points without fetching/replaying the data twice.
    """

    def __init__(self, models: Sequence[SignalModel], bus: EventBus):
        if not models:
            raise ValueError("CandleFeed requires at least one model")
        self._models = list(models)
        self._bus = bus
        self._window: deque[np.ndarray] = deque(maxlen=max(m.window_size() for m in self._models))

    def run(
        self,
        candles: pd.DataFrame,
        ticker: str,
        rate_hz: float = 60.0,
        on_event: Callable[[object], None] | None = None,
    ) -> None:
        """
        Iterates `candles` (expects columns OHLCV_COLUMNS, oldest row
        first) at `rate_hz` rows/second. `on_event` is an optional extra
        callback (e.g. for logging/testing) called with every published
        event, in addition to whatever's subscribed on the bus.

        rate_hz=60 is a target, not a guarantee — actual achievable rate
        depends on model inference latency plus Python loop overhead.
        Benchmark before trusting it hits 60 for real; nothing here
        corrects for drift if a single iteration runs long.
        """
        interval_seconds = 1.0 / rate_hz if rate_hz > 0 else 0.0
        missing = [c for c in OHLCV_COLUMNS if c not in candles.columns]
        if missing:
            raise ValueError(f"candles is missing required columns: {missing}")

        for timestamp, row in candles.iterrows():
            self._window.append(row[OHLCV_COLUMNS].to_numpy(dtype=np.float32))

            candle_event = _candle_event(ticker, timestamp, row)
            self._bus.publish(candle_event)
            if on_event is not None:
                on_event(candle_event)

            for sentiment_event in _predict_ready_models(self._models, self._window, ticker):
                self._bus.publish(sentiment_event)
                if on_event is not None:
                    on_event(sentiment_event)

            if interval_seconds > 0:
                time.sleep(interval_seconds)


class LiveCandleFeed:
    """
    Drives one or more SignalModels (see CandleFeed's docstring re: sharing
    one window sized to the largest model) off the actual current market,
    not a historical
    batch. Seeds its rolling window from recent history (so the very
    first poll already has a full window and can publish immediately),
    then repeatedly re-fetches and appends only the candles newer than
    the last one it saw — i.e. it reacts once per completed candle, on
    the same cadence the market produces them, rather than compressing
    history into a fast artificial replay.

    Note this only reacts to a *closed* candle boundary: yfinance's most
    recent intraday row is often the still-forming candle and can update
    in place before it closes — this feed won't publish again until that
    row's timestamp is superseded by the next one. Good enough for
    "decide once per candle"; sub-candle updates would be a further step.
    """

    def __init__(self, models: Sequence[SignalModel], bus: EventBus):
        if not models:
            raise ValueError("LiveCandleFeed requires at least one model")
        self._models = list(models)
        self._bus = bus
        self._window: deque[np.ndarray] = deque(maxlen=max(m.window_size() for m in self._models))
        self._last_timestamp: pd.Timestamp | None = None

    def _ingest(self, candles: pd.DataFrame, ticker: str, publish_sentiment: bool) -> None:
        """
        `publish_sentiment` gates only the SentimentEvent — CandleEvents are
        always published for every new row (seed rows included), since raw
        price data is useful to a UI the moment it's seen, unlike a
        prediction, which needs a full window to mean anything.
        """
        missing = [c for c in OHLCV_COLUMNS if c not in candles.columns]
        if missing:
            raise ValueError(f"candles is missing required columns: {missing}")

        new_rows = candles if self._last_timestamp is None else candles[candles.index > self._last_timestamp]
        for timestamp, row in new_rows.iterrows():
            self._window.append(row[OHLCV_COLUMNS].to_numpy(dtype=np.float32))
            self._last_timestamp = timestamp

            self._bus.publish(_candle_event(ticker, timestamp, row))

            if publish_sentiment:
                for event in _predict_ready_models(self._models, self._window, ticker):
                    self._bus.publish(event)

    def run(
        self,
        ticker: str,
        interval: str = "5m",
        lookback_period: str = "5d",
        poll_seconds: float = 15.0,
        fetch: Callable[[], pd.DataFrame] | None = None,
        max_polls: int | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        """
        Blocks, polling forever (like CandleFeed.run, meant to be driven
        from a background thread) unless `max_polls` or `stop_event` cuts
        it short — both exist for tests/callers that need this to
        terminate deterministically rather than trusting the process to
        still be alive when they check.

        `fetch` defaults to re-fetching `lookback_period` of real
        `interval` candles via load_historical_candles(ticker, ...) —
        overridable so the polling/windowing logic can be tested without
        the network or real time.sleep (pass a stub `fetch` and
        poll_seconds=0).
        """
        fetch = fetch or (lambda: load_historical_candles(ticker, period=lookback_period, interval=interval))

        # Seed the window from history without a sentiment publish — a
        # prediction made from stale seed candles isn't a live signal, so
        # the first sentiment publish should only happen once the window
        # is built from an explicit poll below. CandleEvents for the seed
        # rows still publish (see _ingest), so a UI has price history
        # immediately.
        self._ingest(fetch(), ticker, publish_sentiment=False)

        polls = 0
        while stop_event is None or not stop_event.is_set():
            if max_polls is not None and polls >= max_polls:
                return
            if poll_seconds > 0:
                time.sleep(poll_seconds)
            polls += 1
            try:
                candles = fetch()
            except Exception:
                logger.exception("LiveCandleFeed: poll failed for %s, will retry next interval", ticker)
                continue
            self._ingest(candles, ticker, publish_sentiment=True)