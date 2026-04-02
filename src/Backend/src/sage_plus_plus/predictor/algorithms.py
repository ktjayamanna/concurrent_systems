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

class DummyPredictor(BasePredictor):
    """
        Mock predictor to unblock Kaveen.Delete this later.
    """
    def __init__(self, mock_output = 'dummy'):
        self._mock_output = mock_output

    def predict(self, subtask: str) -> str:
        return self._mock_output

class NaiveBayesPredictor(BasePredictor):
    def predict(self, subtask: str) -> str:
        #@TODO: To be implemented by Rabia
        pass

class HabitPredictor(BasePredictor):
    def predict(self, subtask: str) -> str:
        #@TODO: To be implemented by Rabia
        pass

class SmallLLMPredictor(BasePredictor):
    def predict(self, subtask: str) -> str:
        #@TODO: To be implemented by Rabia
        pass