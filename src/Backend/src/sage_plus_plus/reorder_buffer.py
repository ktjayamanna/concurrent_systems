
import logging
import threading
from typing import Optional, Any

logger = logging.getLogger(__name__)


class ReorderBuffer:

    def __init__(self):
        self._lock = threading.Lock()
        self._pending_calls: list[tuple[str, dict[str, Any], Any]] = []

    def store(self, tool_name: str, result: Any, args: Optional[dict[str, Any]] = None) -> None:
        self.store_many([(tool_name, args or {}, result)])

    def store_many(self, calls: list[tuple[str, dict[str, Any], Any]]) -> None:
        with self._lock:
            self._pending_calls = [(name, args or {}, result) for name, args, result in calls]

    def commit(
        self,
        predicted_tool: str,
        actual_tool: str,
        predicted_args: Optional[dict[str, Any]] = None,
        actual_args: Optional[dict[str, Any]] = None,
    ) -> Optional[Any]:
        results = self.commit_many([(predicted_tool, predicted_args or {})], [(actual_tool, actual_args or {})])
        return results[0] if results else None

    def commit_many(
        self,
        predicted_calls: list[tuple[str, dict[str, Any]]],
        actual_calls: list[tuple[str, dict[str, Any]]],
    ) -> Optional[list[Any]]:
        with self._lock:
            if len(predicted_calls) != len(actual_calls) or len(self._pending_calls) != len(actual_calls):
                return None
            results = []
            for (stored_name, stored_args, stored_result), (pred_name, pred_args), (actual_name, actual_args) in zip(
                self._pending_calls, predicted_calls, actual_calls
            ):
                if stored_name != pred_name or pred_name != actual_name:
                    return None
                if (stored_args or {}) != (pred_args or {}) or (pred_args or {}) != (actual_args or {}):
                    return None
                results.append(stored_result)
            self._pending_calls = []
            return results

    def flush(self) -> None:
        with self._lock:
            self._pending_calls = []
