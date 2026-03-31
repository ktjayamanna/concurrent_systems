"""
Hazard detection unit for classifying tools as safe/unsafe.
"""

import logging

logger = logging.getLogger(__name__)


class HazardDetectionUnit:
    """Classifies tools as safe (read-only) or unsafe (write/delete operations)"""
    
    def __init__(self):
        # TODO: Load from config file with all 102 SAGE benchmark actions pre-classified
        self._safety_map = {}
        
    def is_safe(self, tool_name: str) -> bool:
        """
        Check if a tool is safe for speculative execution.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if safe (read-only), False if unsafe (writes/deletes)
        """
        return self._safety_map.get(tool_name, False)