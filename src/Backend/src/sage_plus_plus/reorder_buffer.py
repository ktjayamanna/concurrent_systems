"""
Reorder buffer for managing speculative execution results.
"""

import asyncio
import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)


class ReorderBuffer:
    """Holds one pending speculative result with commit/flush logic"""
    
    def __init__(self):
        self._lock = asyncio.Lock()
        self._pending_tool_name: Optional[str] = None
        self._pending_result: Optional[Any] = None
        
    async def store(self, tool_name: str, result: Any) -> None:
        """Store a speculative result"""
        async with self._lock:
            self._pending_tool_name = tool_name
            self._pending_result = result
            
    async def commit(self, predicted_tool: str, actual_tool: str) -> Optional[Any]:
        """
        Commit if prediction matches, otherwise return None.
        
        Args:
            predicted_tool: Tool that was predicted
            actual_tool: Tool that WTP actually chose
            
        Returns:
            Stored result if hit, None if miss
        """
        async with self._lock:
            if self._pending_tool_name == actual_tool == predicted_tool:
                result = self._pending_result
                self._pending_tool_name = None
                self._pending_result = None
                return result
            return None
            
    async def flush(self) -> None:
        """Clear the buffer in O(1)"""
        async with self._lock:
            self._pending_tool_name = None
            self._pending_result = None