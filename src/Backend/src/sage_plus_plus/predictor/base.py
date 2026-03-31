"""
Abstract base class for tool predictors.
To be implemented by Rabia.
"""

from abc import ABC, abstractmethod


class BasePredictor(ABC):
    """Abstract interface for tool prediction"""
    
    @abstractmethod
    def predict(self, subtask: str) -> str:
        """
        Predict which tool will be called for a given subtask.
        
        Args:
            subtask: The task description from the orchestrator
            
        Returns:
            Predicted tool name
        """
        pass