"""Retry with exponential backoff for rate-limited API calls.

Cohere Trial keys allow only ~10 requests/minute, so a batch job like the eval harness
will hit HTTP 429. Rather than pre-throttle with fixed sleeps (slow and fragile), we wrap
each API call and, on a rate-limit / transient error, back off exponentially and retry.
This is the standard way to ride out 429/503s against any external API — and it keeps the
interactive app resilient too, without the caller having to think about it.
"""
from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")

_TRANSIENT_MARKERS = ("429", "too many requests", "rate limit", "503", "overloaded")


def _is_transient(err: Exception) -> bool:
    if getattr(err, "status_code", None) in (429, 503):
        return True
    msg = str(err).lower()
    return any(m in msg for m in _TRANSIENT_MARKERS)


def with_backoff(fn: Callable[[], T], *, tries: int = 6, base: float = 2.0, cap: float = 60.0) -> T:
    """Call fn(); on a transient error wait base*2**attempt seconds (capped) and retry."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as err:  # noqa: BLE001 - we re-raise anything non-transient
            if attempt == tries - 1 or not _is_transient(err):
                raise
            delay = min(cap, base * (2 ** attempt))
            print(f"  ⏳ rate-limited, retrying in {delay:.0f}s…")
            time.sleep(delay)
    raise RuntimeError("unreachable")
