"""
In-flight request coalescer and micro-TTL cache for high-throughput market data.
Inspired by BeeBots request-coalescing architecture (okx/rest.ts).

Prevents redundant network round-trips when multiple agents (Technical, Red-Team,
Stress Tester, Position Sizer) query the same asset concurrently.
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Callable, Dict, Generic, Optional, Tuple, TypeVar

from src.utils.logger import logger

T = TypeVar("T")


class RequestCoalescer:
    """
    Thread-safe & async-compatible request coalescer with microsecond TTL caching.
    """

    def __init__(self, default_ttl_seconds: float = 3.0):
        self.default_ttl = default_ttl_seconds
        self._cache: Dict[str, Tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self._inflight_events: Dict[str, threading.Event] = {}
        self._inflight_results: Dict[str, Any] = {}
        self._inflight_errors: Dict[str, Exception] = {}

        # Asyncio support
        self._async_lock: Optional[asyncio.Lock] = None
        self._async_inflight: Dict[str, asyncio.Future] = {}

        # Telemetry stats
        self.stats = {
            "cache_hits": 0,
            "coalesced_waits": 0,
            "network_fetches": 0,
        }

    def get_cached(self, key: str, ttl_seconds: Optional[float] = None) -> Optional[Any]:
        """Returns cached item if valid, else None."""
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl
        with self._lock:
            if key in self._cache:
                cached_at, value = self._cache[key]
                if time.time() - cached_at < ttl:
                    self.stats["cache_hits"] += 1
                    return value
                del self._cache[key]
        return None

    def set_cached(self, key: str, value: Any) -> None:
        """Stores item in cache with current timestamp."""
        with self._lock:
            self._cache[key] = (time.time(), value)

    def invalidate(self, key: Optional[str] = None) -> None:
        """Invalidate single key or entire cache."""
        with self._lock:
            if key:
                self._cache.pop(key, None)
            else:
                self._cache.clear()

    def coalesce(
        self,
        key: str,
        fetch_fn: Callable[[], T],
        ttl_seconds: Optional[float] = None,
    ) -> T:
        """
        Thread-safe synchronous coalesce.
        If multiple threads request `key` concurrently, only one runs `fetch_fn()`;
        all others block on the in-flight event and receive the same result.
        """
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl

        # 1. Check hot cache
        cached = self.get_cached(key, ttl)
        if cached is not None:
            return cached

        event_to_wait: Optional[threading.Event] = None
        is_leader = False

        with self._lock:
            # Check cache again inside lock
            if key in self._cache:
                cached_at, value = self._cache[key]
                if time.time() - cached_at < ttl:
                    self.stats["cache_hits"] += 1
                    return value
                del self._cache[key]

            if key in self._inflight_events:
                # Another thread is already fetching this key!
                event_to_wait = self._inflight_events[key]
                self.stats["coalesced_waits"] += 1
            else:
                # We are the leader thread responsible for fetching
                is_leader = True
                event = threading.Event()
                self._inflight_events[key] = event
                self.stats["network_fetches"] += 1

        if not is_leader and event_to_wait:
            # Wait for leader to finish (max 30 seconds safety timeout)
            event_to_wait.wait(timeout=30.0)
            with self._lock:
                if key in self._inflight_errors:
                    raise self._inflight_errors[key]
                if key in self._cache:
                    return self._cache[key][1]
                return self._inflight_results.get(key)

        # Leader execution
        try:
            result = fetch_fn()
            with self._lock:
                self._cache[key] = (time.time(), result)
                self._inflight_results[key] = result
            return result
        except Exception as err:
            with self._lock:
                self._inflight_errors[key] = err
            raise
        finally:
            with self._lock:
                if is_leader and key in self._inflight_events:
                    self._inflight_events[key].set()
                    # Clean up inflight entries after waking waiters
                    del self._inflight_events[key]
                    self._inflight_results.pop(key, None)
                    self._inflight_errors.pop(key, None)

    async def coalesce_async(
        self,
        key: str,
        fetch_coro: Callable[[], Any],
        ttl_seconds: Optional[float] = None,
    ) -> Any:
        """
        Asyncio coalesce.
        If multiple coroutines request `key` concurrently, only one awaits `fetch_coro()`.
        """
        ttl = ttl_seconds if ttl_seconds is not None else self.default_ttl

        # 1. Hot cache check
        cached = self.get_cached(key, ttl)
        if cached is not None:
            return cached

        if self._async_lock is None:
            self._async_lock = asyncio.Lock()

        async with self._async_lock:
            # Recheck cache
            cached = self.get_cached(key, ttl)
            if cached is not None:
                return cached

            if key in self._async_inflight:
                future = self._async_inflight[key]
                self.stats["coalesced_waits"] += 1
            else:
                loop = asyncio.get_running_loop()
                future = loop.create_future()
                self._async_inflight[key] = future
                self.stats["network_fetches"] += 1

                # Schedule background fetch
                asyncio.create_task(self._run_async_fetch(key, fetch_coro, future))

        return await future

    async def _run_async_fetch(
        self,
        key: str,
        fetch_coro: Callable[[], Any],
        future: asyncio.Future,
    ) -> None:
        try:
            result = await fetch_coro()
            self.set_cached(key, result)
            if not future.done():
                future.set_result(result)
        except Exception as e:
            if not future.done():
                future.set_exception(e)
        finally:
            if self._async_lock:
                async with self._async_lock:
                    self._async_inflight.pop(key, None)


# Global singleton instance
market_coalescer = RequestCoalescer(default_ttl_seconds=3.0)
