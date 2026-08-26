"""In-memory bounded event bus for active runs (Studio S1.5 reconnect).

Each active run owns an ``EventBroker`` that:
- assigns a strictly increasing, unique ``event_seq`` to every published event
  (no event may reuse a seq);
- keeps a bounded ring buffer of the most recent events so a reconnecting
  client can be replayed via ``after_seq``;
- fans events out to SSE subscribers. Subscribers are plain queues; removing
  a subscriber never affects the run controller.

This is a short-lived in-memory structure only. Full event persistence is
left to S2 (SQLite); complete logs keep going to ``training.log``.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from typing import Optional

DEFAULT_MAX_EVENTS = 2000
DEFAULT_SUBSCRIBER_QUEUE = 500


class EventBroker:
    def __init__(self, run_id: str, max_events: int = DEFAULT_MAX_EVENTS):
        if max_events < 1:
            raise ValueError("max_events must be >= 1")
        self.run_id = run_id
        self.max_events = max_events
        self._events: deque = deque(maxlen=max_events)
        self._subscribers: set = set()
        self._lock = threading.Lock()
        self._seq = 0

    @property
    def last_seq(self) -> int:
        with self._lock:
            return self._seq

    def publish(self, event: dict) -> dict:
        """Assign the next unique seq, buffer it, and fan out to subscribers.

        Returns the stamped event (callers use its ``event_seq`` to keep
        ``RunState.last_event.seq`` consistent).
        """
        with self._lock:
            self._seq += 1
            stamped = dict(event)
            stamped["event_seq"] = self._seq
            self._events.append(stamped)
            for sub in list(self._subscribers):
                try:
                    sub.put_nowait(stamped)
                except queue.Full:
                    # A slow subscriber can drop transient realtime events;
                    # terminal facts are persisted in the state file and are
                    # re-delivered by a resubscribe (after_seq replay).
                    pass
        return stamped

    def subscribe(self, after_seq: int = 0):
        """Register a subscriber. Returns ``(queue, replay_events)`` where
        ``replay_events`` are buffered events with ``event_seq > after_seq``."""
        q: "queue.Queue[dict]" = queue.Queue(maxsize=DEFAULT_SUBSCRIBER_QUEUE)
        with self._lock:
            self._subscribers.add(q)
            replay = [e for e in self._events if e["event_seq"] > after_seq]
        return q, replay

    def unsubscribe(self, q) -> None:
        with self._lock:
            self._subscribers.discard(q)

    def recent(self, limit: Optional[int] = None) -> list:
        """Return the buffered events (oldest first), optionally limited to the
        most recent ``limit``."""
        with self._lock:
            items = list(self._events)
        if limit is not None:
            items = items[-limit:]
        return items

    def replay_after(self, after_seq: int) -> list:
        with self._lock:
            return [e for e in self._events if e["event_seq"] > after_seq]

    def replay_truncated(self, after_seq: int) -> bool:
        """True when some events with ``event_seq > after_seq`` were evicted
        from the ring buffer, so a reconnect cannot fully replay the stream."""
        with self._lock:
            if not self._events:
                return False
            return self._events[0]["event_seq"] > after_seq + 1
