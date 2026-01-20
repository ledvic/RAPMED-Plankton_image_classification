"""
Abstract base class for feature extractors.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any


class BaseFeatureExtractor(ABC):
    """Abstract base class for all feature extractors.
    
    This enforces a consistent interface across all feature extraction methods.
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Args:
            name: Name of the feature extractor
            **kwargs: Additional configuration parameters
        """
        self.name = name
        self.config = kwargs
        self._is_fitted = False
    
    @abstractmethod
    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract features from a single image.
        
        Args:
            image: Input image (H, W, C) or (H, W)
            
        Returns:
            Feature vector
        """
        pass
    
    def extract_batch(self, images: np.ndarray) -> np.ndarray:
        """Extract features from a batch of images.
        
        Args:
            images: Batch of images (N, H, W, C) or (N, H, W)
            
        Returns:
            Feature matrix (N, feature_dim)
        """
        return np.array([self.extract(img) for img in images])
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """Get the dimensionality of the feature vector.
        
        Returns:
            Feature dimension
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get extractor configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'config': self.config
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
