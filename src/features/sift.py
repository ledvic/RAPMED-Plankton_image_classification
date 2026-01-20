"""
SIFT (Scale-Invariant Feature Transform) feature extraction.
"""

import numpy as np
import cv2
from typing import Optional, Tuple


class SIFTFeatureExtractor:
    """Extract SIFT features for object recognition.
    
    SIFT detects keypoints and computes descriptors that are
    invariant to scale and rotation.
    """
    
    def __init__(
        self,
        n_features: int = 100,
        n_octave_layers: int = 3,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10,
        sigma: float = 1.6
    ):
        """
        Args:
            n_features: Number of best features to retain
            n_octave_layers: Number of layers in each octave
            contrast_threshold: Contrast threshold for filtering weak features
            edge_threshold: Threshold for filtering edge-like features
            sigma: Sigma of the Gaussian applied to input image
        """
        self.sift = cv2.SIFT_create(
            nfeatures=n_features,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma
        )
        self.n_features = n_features
    
    def extract(
        self,
        image: np.ndarray,
        aggregate: str = 'mean'
    ) -> np.ndarray:
        """
        Extract SIFT features from an image.
        
        Args:
            image: Input image (grayscale or RGB)
            aggregate: How to aggregate descriptors ('mean', 'max', 'sum')
            
        Returns:
            Aggregated SIFT descriptor (128-dimensional for mean)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        # Handle case when no keypoints are detected
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(128)  # SIFT descriptor is 128-dimensional
        
        # Aggregate descriptors
        if aggregate == 'mean':
            feature = np.mean(descriptors, axis=0)
        elif aggregate == 'max':
            feature = np.max(descriptors, axis=0)
        elif aggregate == 'sum':
            feature = np.sum(descriptors, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregate}")
        
        return feature
    
    def extract_batch(
        self,
        images: np.ndarray,
        aggregate: str = 'mean'
    ) -> np.ndarray:
        """Extract features from a batch of images."""
        return np.array([self.extract(img, aggregate) for img in images])
