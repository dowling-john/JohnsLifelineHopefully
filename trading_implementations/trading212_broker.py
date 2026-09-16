"""
trading_implementations/trading212_broker.py

Trading212Broker — the broker the decision engine actually uses. Composes:

  - Trading212UIBroker  -> provides get_quote() with real bid/ask, via the
                            unofficial WebSocket feed (see that module's
                            docstring for the caveats — read them).
  - Trading212APIBroker -> provides everything else (account, positions,
                            orders), via the official, documented API.

Because Trading212UIBroker only implements get_quote() and nothing else
overlaps between the two parents, MRO resolution is unambiguous: get_quote
resolves to the UI mixin, every other BrokerInterface method resolves to
Trading212APIBroker.

When Trading212 eventually ships an official quote endpoint, don't reshuffle
the base classes to "fix" this — override get_quote() explicitly here
instead, so the change is visible in one obvious place:

    def get_quote(self, ticker: str) -> Quote:
        return Trading212APIBroker.get_quote(self, ticker)

That keeps the swap a one-line, explicit, reviewable change rather than a
side-effect of reordering base classes.
"""

from __future__ import annotations

from trading_implementations.trading212_api_broker import Environment, Trading212APIBroker
from trading_implementations.trading212_ui_broker import Trading212UIBroker


class Trading212Broker(Trading212UIBroker, Trading212APIBroker):
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        environment: Environment,
        customer_session: str,
        trading212_session_live: str,
    ):
        # Explicit calls rather than a cooperative super() chain — with two
        # unrelated __init__ signatures, being explicit here is clearer
        # than trying to make MRO-cooperative multiple inheritance work.
        Trading212APIBroker.__init__(self, api_key, api_secret, environment)
        Trading212UIBroker.__init__(self, customer_session, trading212_session_live)

    def start(self) -> None:
        """Open the UI WebSocket connection. Call once before trading."""
        self.connect()

    def stop(self) -> None:
        self.disconnect()