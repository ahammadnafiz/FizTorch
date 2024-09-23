from abc import ABC, abstractmethod
from typing import Any

class BaseModel(ABC):
    @abstractmethod
    def train(self, X: Any, Y: Any, num_iterations: int = None, learning_rate: float = None) -> None:
        """
        Train the model on the given data.

        Args:
            X: Input features
            Y: Target values
            num_iterations: Number of training iterations (optional)
            learning_rate: Learning rate for optimization (optional)
        """
        pass

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """
        Make predictions using the trained model.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        pass
