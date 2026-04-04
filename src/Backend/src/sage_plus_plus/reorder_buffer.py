"""
Reorder buffer for managing speculative execution results.
"""

import logging
import threading
from typing import Optional, Any

logger = logging.getLogger(__name__)


class ReorderBuffer:
    """Holds one pending speculative result with commit/flush logic.

    Uses threading.Lock instead of asyncio.Lock so it is safe to call
    from both a GIL-free OS thread (store) and the asyncio event loop
    coroutine (commit, flush) without deadlocking.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._pending_tool_name: Optional[str] = None
        self._pending_result: Optional[Any] = None

    def store(self, tool_name: str, result: Any) -> None:
        """Store a speculative result (called from shadow OS thread)."""
        with self._lock:
            self._pending_tool_name = tool_name
            self._pending_result = result

    def commit(self, predicted_tool: str, actual_tool: str) -> Optional[Any]:
        """
        Commit if prediction matches, otherwise return None.

        Args:
            predicted_tool: Tool that was predicted by shadow
            actual_tool: Tool that the WorkerAgent LLM actually chose

        Returns:
            Stored result if hit, None if miss
        """
        with self._lock:
            if self._pending_tool_name == actual_tool == predicted_tool:
                result = self._pending_result
                self._pending_tool_name = None
                self._pending_result = None
                return result
            return None

    def flush(self) -> None:
        """Clear the buffer in O(1)."""
        with self._lock:
            self._pending_tool_name = None
            self._pending_result = None
