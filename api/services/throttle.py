"""
Two small helpers shared by the market data clients.

RateLimiter: a sliding window. At most `limit` calls in any `window` seconds, so a
free plan's limit (Polygon: 5 a minute) is never exceeded, even across threads.

TTLCache: remembers results for a while, so repeated requests for the same bars
cost nothing against that limit.
"""

import math
import threading
import time
from collections import OrderedDict, deque
from typing import Any, Callable, Deque, Hashable, Optional


class RateLimited(Exception):
    """Waiting for a free slot would take longer than the caller allowed."""

    def __init__(self, retry_after: float):
        super().__init__(f"Rate limit reached. Retry in {math.ceil(retry_after)} seconds.")
        self.retry_after = retry_after


class RateLimiter:
    def __init__(
        self,
        limit: int,
        window: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self.limit = limit
        self.window = window
        self._clock = clock
        self._sleep = sleep
        self._calls: Deque[float] = deque()
        self._lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self._calls and now - self._calls[0] >= self.window:
            self._calls.popleft()

    def wait_time(self) -> float:
        """Seconds until the next call may go out (0 if one may go out now)."""
        with self._lock:
            now = self._clock()
            self._prune(now)
            if len(self._calls) < self.limit:
                return 0.0
            return max(0.0, self._calls[0] + self.window - now)

    def acquire(self, max_wait: Optional[float] = None) -> float:
        """Take a slot, sleeping until one frees up. Returns seconds waited. Raises RateLimited past `max_wait`."""
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                self._prune(now)
                if len(self._calls) < self.limit:
                    self._calls.append(now)
                    return waited
                delay = max(0.0, self._calls[0] + self.window - now)
            if max_wait is not None and waited + delay > max_wait:
                raise RateLimited(delay)
            self._sleep(delay)
            waited += delay

    def seconds_for(self, calls: int) -> float:
        """Rough time to make `calls` more requests from now, used for "this scan takes about N minutes"."""
        if calls <= 0:
            return 0.0
        with self._lock:
            now = self._clock()
            self._prune(now)
            times = list(self._calls)
        # Call j may go out once call j - limit has left the window. Simulate the queue.
        for _ in range(calls):
            j = len(times)
            times.append(now if j < self.limit else max(now, times[j - self.limit] + self.window))
        return times[-1] - now


class TTLCache:
    """Thread-safe least-recently-used cache where each entry expires after its own TTL."""

    def __init__(self, max_size: int = 256, clock: Callable[[], float] = time.monotonic):
        self.max_size = max_size
        self._clock = clock
        self._items: "OrderedDict[Hashable, tuple]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Optional[Any]:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires, value = item
            if self._clock() >= expires:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return value

    def __contains__(self, key: Hashable) -> bool:
        return self.get(key) is not None

    def set(self, key: Hashable, value: Any, ttl: float) -> None:
        with self._lock:
            self._items[key] = (self._clock() + ttl, value)
            self._items.move_to_end(key)
            while len(self._items) > self.max_size:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
