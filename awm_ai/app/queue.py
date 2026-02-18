"""
AWM AI Queue Module
===================
Thread-safe queue for bar data buffering.
"""

from collections import deque
from threading import Lock
from typing import Optional, List

# Holds latest bars for instant access
BAR_QUEUE = deque(maxlen=1000)
LOCK = Lock()


def push_bar(bar: dict):
    """Add bar to queue (thread-safe)"""
    with LOCK:
        BAR_QUEUE.append(bar)


def get_latest_bar() -> Optional[dict]:
    """Get most recent bar without removing"""
    with LOCK:
        return BAR_QUEUE[-1] if BAR_QUEUE else None


def drain_bars() -> List[dict]:
    """Return all bars and clear queue (for background writer)"""
    with LOCK:
        bars = list(BAR_QUEUE)
        BAR_QUEUE.clear()
        return bars


def get_bar_count() -> int:
    """Get current queue size"""
    with LOCK:
        return len(BAR_QUEUE)