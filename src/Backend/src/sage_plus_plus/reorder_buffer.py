
import logging
import threading
from typing import Optional, Any

logger = logging.getLogger(__name__)


class ReorderBuffer:

    def __init__(self):
        self._lock = threading.Lock()
        self._pending_tool_name: Optional[str] = None
        self._pending_args: dict[str, Any] = {}
        self._pending_result: Optional[Any] = None

    def store(self, tool_name: str, result: Any, args: Optional[dict[str, Any]] = None) -> None:
        with self._lock:
            self._pending_tool_name = tool_name
            self._pending_args = args or {}
            self._pending_result = result

    def commit(
        self,
        predicted_tool: str,
        actual_tool: str,
        predicted_args: Optional[dict[str, Any]] = None,
        actual_args: Optional[dict[str, Any]] = None,
    ) -> Optional[Any]:
        with self._lock:
            args_match = (predicted_args or {}) == (actual_args or {}) == self._pending_args
            if self._pending_tool_name == actual_tool == predicted_tool and args_match:
                result = self._pending_result
                self._pending_tool_name = None
                self._pending_args = {}
                self._pending_result = None
                return result
            return None

    def flush(self) -> None:
        with self._lock:
            self._pending_tool_name = None
            self._pending_args = {}
            self._pending_result = None
