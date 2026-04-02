"""
Hazard detection unit for classifying tools as safe/unsafe.
"""

import logging

logger = logging.getLogger(__name__)


class HazardDetectionUnit:
    """Classifies tools as safe (read-only) or unsafe (write/delete operations)"""
    
    def __init__(self):
        '''
        NOTE: Current policy is conservative around financially costly actions.
        In future, we can load from config and support undo/compensation for
        low-risk actions (e.g., some inventory movements).
        '''
        self._unsafe_tools_financial = {
            # Orders / purchasing (direct financial impact)
            "MakeOrder",
            "MakeOrders",
            "AddOrder",
            "AddOrders",
            "CancelOrder",
            "OrderSnack",
        }
        
    def is_safe(self, tool_name: str) -> bool:
        """
        Check if a tool is safe for speculative execution.
        
        Args:
            tool_name: Name of the tool to check
            
        Returns:
            True if safe (read-only), False if unsafe (writes/deletes)
        """
        if not tool_name:
            return False
        return tool_name not in self._unsafe_tools_financial
