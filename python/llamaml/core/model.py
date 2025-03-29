"""
Base model interface for LlamaML.

This module defines the base Model class that all LlamaML models should inherit from,
along with the ModelConfig for configuring models.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np


@dataclass
class ModelConfig:
    """Configuration for a model."""
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    
    # Common hyperparameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 10
    
    # Architecture parameters
    hidden_layers: List[int] = field(default_factory=list)
    activation: str = "relu"
    dropout_rate: float = 0.0
    
    # Model-specific parameters
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "hidden_layers": self.hidden_layers,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "params": self.params
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ModelConfig":
        """Create a config from a dictionary."""
        return cls(**config_dict)


class Model(ABC):
    """Base model interface for all LlamaML models."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the model with the given configuration.
        
        Args:
            config: The model configuration.
        """
        self.config = config
        self._model = None
        self.is_fitted = False
        self.feature_names = None
        self.target_names = None
    
    @property
    def name(self) -> str:
        """Get the model name."""
        return self.config.name
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Model":
        """Fit the model to the data.
        
        Args:
            X: The input features.
            y: The target values.
            
        Returns:
            self: The fitted model.
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the model.
        
        Args:
            X: The input features.
            
        Returns:
            Predictions.
        """
        pass
    
    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit the model and make predictions.
        
        Args:
            X: The input features.
            y: The target values.
            
        Returns:
            Predictions.
        """
        self.fit(X, y)
        return self.predict(X)
    
    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on the given data.
        
        Args:
            X: The input features.
            y: The true target values.
            
        Returns:
            Dictionary of metrics.
        """
        pass
    
    def save(self, path: str) -> None:
        """Save the model to the given path.
        
        Args:
            path: The path to save the model to.
        """
        from llamaml.utils.serialization import save_model
        save_model(self, path)
    
    @classmethod
    def load(cls, path: str) -> "Model":
        """Load a model from the given path.
        
        Args:
            path: The path to load the model from.
            
        Returns:
            The loaded model.
        """
        from llamaml.utils.serialization import load_model
        return load_model(path)
    
    def get_params(self) -> Dict[str, Any]:
        """Get the model parameters.
        
        Returns:
            Dictionary of model parameters.
        """
        return self.config.to_dict()
    
    def set_params(self, **params) -> "Model":
        """Set model parameters.
        
        Args:
            **params: Parameters to set.
            
        Returns:
            self: The model with updated parameters.
        """
        for key, value in params.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.config.params[key] = value
        return self
    
    def __repr__(self) -> str:
        """Get string representation of the model."""
        return f"{self.__class__.__name__}(config={self.config})" 