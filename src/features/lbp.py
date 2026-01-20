"""
Local Binary Pattern (LBP) feature extraction.
"""

import numpy as np
import cv2
from skimage.feature import local_binary_pattern


class LBPFeatureExtractor:
    """Extract Local Binary Pattern features.
    
    LBP is effective for texture classification and works well
    with microscopic images like plankton.
    """
    
    def __init__(
        self,
        n_points: int = 24,
        radius: int = 3,
        method: str = 'uniform',
        n_bins: int = 26
    ):
        """
        Args:
            n_points: Number of circularly symmetric neighbor points
            radius: Radius of circle
            method: 'default', 'ror', 'uniform', 'nri_uniform', or 'var'
            n_bins: Number of histogram bins
        """
        self.n_points = n_points
        self.radius = radius
        self.method = method
        self.n_bins = n_bins
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract LBP features from an image.
        
        Args:
            image: Input image (grayscale or RGB)
            
        Returns:
            Normalized histogram of LBP patterns
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Compute LBP
        lbp = local_binary_pattern(
            gray,
            self.n_points,
            self.radius,
            method=self.method
        )
        
        # Compute histogram
        hist, _ = np.histogram(
            lbp.ravel(),
            bins=self.n_bins,
            range=(0, self.n_bins),
            density=True
        )
        
        return hist
    
    def extract_batch(self, images: np.ndarray) -> np.ndarray:
        """Extract features from a batch of images."""
        return np.array([self.extract(img) for img in images])
