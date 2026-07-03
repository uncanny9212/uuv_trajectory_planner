"""Sliding-window decision memory."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List


class MemoryManager:
    """Keep the latest decision context for ReAct-style feedback."""

    def __init__(self, window_size: int = 5) -> None:
        self.window_size = window_size
        self._items: Deque[Dict[str, Any]] = deque(maxlen=window_size)

    def add(self, item: Dict[str, Any]) -> None:
        self._items.append(item)

    def recent(self) -> List[Dict[str, Any]]:
        return list(self._items)

    def clear(self) -> None:
        self._items.clear()
