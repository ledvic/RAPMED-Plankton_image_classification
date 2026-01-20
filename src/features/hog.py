"""
HOG (Histogram of Oriented Gradients) feature extraction.
"""

import numpy as np
import cv2
from skimage.feature import hog


class HOGFeatureExtractor:
    """Extract HOG features for shape description.
    
    HOG captures edge and gradient structure, useful for
    distinguishing plankton species by shape.
    """
    
    def __init__(
        self,
        orientations: int = 9,
        pixels_per_cell: Tuple[int, int] = (8, 8),
        cells_per_block: Tuple[int, int] = (2, 2),
        block_norm: str = 'L2-Hys',
        transform_sqrt: bool = True
    ):
        """
        Args:
            orientations: Number of orientation bins
            pixels_per_cell: Size of a cell (in pixels)
            cells_per_block: Number of cells in each block
            block_norm: Block normalization method
            transform_sqrt: Apply power law compression
        """
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
        self.transform_sqrt = transform_sqrt
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract HOG features from an image.
        
        Args:
            image: Input image (grayscale or RGB)
            
        Returns:
            HOG feature descriptor
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Resize to fixed size for consistent feature dimensions
        # You can adjust this size based on your needs
        gray = cv2.resize(gray, (128, 128))
        
        # Compute HOG features
        features = hog(
            gray,
            orientations=self.orientations,
            pixels_per_cell=self.pixels_per_cell,
            cells_per_block=self.cells_per_block,
            block_norm=self.block_norm,
            transform_sqrt=self.transform_sqrt,
            feature_vector=True
        )
        
        return features
    
    def extract_batch(self, images: np.ndarray) -> np.ndarray:
        """Extract features from a batch of images."""
        return np.array([self.extract(img) for img in images])
