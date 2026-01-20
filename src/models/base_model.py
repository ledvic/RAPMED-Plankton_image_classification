"""
Abstract base class for all models.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any, Optional
import tensorflow as tf
import numpy as np


class BaseModel(ABC):
    """Abstract base class for all models.
    
    Enforces consistent interface across different model architectures.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Args:
            name: Model name
            **kwargs: Additional model parameters
        """
        self.name = name
        self.config = kwargs
        self.model: Optional[tf.keras.Model] = None
        self._is_compiled = False
        self._is_trained = False
    
    @abstractmethod
    def build(self) -> tf.keras.Model:
        """Build the model architecture.
        
        Returns:
            Keras Model instance
        """
        pass
    
    def compile(
        self,
        optimizer: str = 'adam',
        loss: str = 'categorical_crossentropy',
        metrics: list = ['accuracy'],
        **kwargs
    ) -> None:
        """Compile the model.
        
        Args:
            optimizer: Optimizer name or instance
            loss: Loss function
            metrics: List of metrics
            **kwargs: Additional compile arguments
        """
        if self.model is None:
            self.model = self.build()
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics,
            **kwargs
        )
        self._is_compiled = True
    
    def fit(
        self,
        train_data,
        validation_data=None,
        epochs: int = 10,
        callbacks: list = None,
        **kwargs
    ) -> tf.keras.callbacks.History:
        """Train the model.
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset
            epochs: Number of epochs
            callbacks: List of callbacks
            **kwargs: Additional fit arguments
            
        Returns:
            Training history
        """
        if not self._is_compiled:
            raise RuntimeError("Model must be compiled before training")
        
        history = self.model.fit(
            train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=callbacks or [],
            **kwargs
        )
        
        self._is_trained = True
        return history
    
    def predict(self, x: np.ndarray, **kwargs) -> np.ndarray:
        """Make predictions.
        
        Args:
            x: Input data
            **kwargs: Additional predict arguments
            
        Returns:
            Predictions
        """
        if self.model is None:
            raise RuntimeError("Model must be built before prediction")
        
        return self.model.predict(x, **kwargs)
    
    def evaluate(self, test_data, **kwargs) -> Dict[str, float]:
        """Evaluate the model.
        
        Args:
            test_data: Test dataset
            **kwargs: Additional evaluate arguments
            
        Returns:
            Dictionary of metric values
        """
        if not self._is_compiled:
            raise RuntimeError("Model must be compiled before evaluation")
        
        results = self.model.evaluate(test_data, **kwargs)
        
        if isinstance(results, list):
            metric_names = [m.name for m in self.model.metrics]
            return dict(zip(['loss'] + metric_names, results))
        return {'loss': results}
    
    def save(self, filepath: str) -> None:
        """Save the model.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise RuntimeError("Model must be built before saving")
        
        self.model.save(filepath)
    
    @classmethod
    def load(cls, filepath: str) -> "BaseModel":
        """Load a saved model.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded model instance
        """
        model_instance = cls()
        model_instance.model = tf.keras.models.load_model(filepath)
        model_instance._is_compiled = True
        model_instance._is_trained = True
        return model_instance
    
    def summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            self.model = self.build()
        self.model.summary()
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'config': self.config
        }
    
    @property
    def is_compiled(self) -> bool:
        return self._is_compiled
    
    @property
    def is_trained(self) -> bool:
        return self._is_trained
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
