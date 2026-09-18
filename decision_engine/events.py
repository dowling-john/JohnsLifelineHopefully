"""
decision_engine/events.py

The event bus: how completely separate, independently-trained networks
talk to each other. A signal model never calls another model directly —
it publishes an event; anything downstream (the UI, the future engine
model) subscribes and reacts. This is the same decoupling principle as
an event-bus architecture in any other language — Go included — just
applied to model outputs instead of device/scheduler events.

Deliberately synchronous and in-process for now: one Python process,
subscribers called directly on publish, no threading/async complexity.
That's the right amount of machinery for "one model, prove it works" —
upgrading to threaded/async delivery (needed once the UI server and a
real-time feed run concurrently) is a contained, later change, not a
redesign, because publish()/subscribe() stay the same shape.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable


@dataclass(frozen=True)
class Event:
    """Base event — every event on the bus carries at least this."""
    event_type: str
    source: str  # which model/component published this
    as_of: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class SentimentEvent(Event):
    """
    Published by a signal model after looking at a candle window.

    sentiment is a continuous score in [-1, 1]: -1 strongly bearish,
    0 neutral, +1 strongly bullish. confidence in [0, 1] is how sure the
    model is — separate from sentiment's sign/magnitude, since a model
    can be confidently neutral or unconfidently bullish.
    """
    ticker: str = ""
    sentiment: float = 0.0
    confidence: float = 0.0

    def __post_init__(self) -> None:
        if not -1.0 <= self.sentiment <= 1.0:
            raise ValueError(f"sentiment must be in [-1, 1], got {self.sentiment}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


Subscriber = Callable[[Event], None]


class EventBus:
    """Simple synchronous pub/sub, keyed by event_type."""

    def __init__(self) -> None:
        self._subscribers: dict[str, list[Subscriber]] = defaultdict(list)

    def subscribe(self, event_type: str, callback: Subscriber) -> None:
        self._subscribers[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: Subscriber) -> None:
        self._subscribers[event_type].remove(callback)

    def publish(self, event: Event) -> None:
        for callback in self._subscribers.get(event.event_type, []):
            callback(event)