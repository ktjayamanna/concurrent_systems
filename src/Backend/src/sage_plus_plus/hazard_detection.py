
class HazardDetectionUnit:
    
    def __init__(self):
        self._unsafe_tools = {
            "MakeOrder",
            "MakeOrders",
            "AddOrder",
            "AddOrders",
            "CancelOrder",
            "OrderSnack",
            "ScheduleMaintenance",
            "ScheduleCleaning",
        }
        
    def is_safe(self, tool_name: str) -> bool:
        if not tool_name:
            return False
        action_name = tool_name.split("--", 1)[1] if "--" in tool_name else tool_name
        if action_name in self._unsafe_tools:
            return False
        unsafe_prefixes = (
            "Book", "CancelOrder", "MakeOrder", "Order", "ScheduleCleaning",
            "ScheduleMaintenance",
        )
        return not action_name.startswith(unsafe_prefixes)
