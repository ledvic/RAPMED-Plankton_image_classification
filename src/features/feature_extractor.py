"""
Combined feature extraction pipeline.
"""

import numpy as np
from typing import List, Dict, Optional
import tensorflow as tf

from .gabor import GaborFeatureExtractor
from .lbp import LBPFeatureExtractor
from .sift import SIFTFeatureExtractor
from .hog import HOGFeatureExtractor


class FeatureExtractor:
    """Unified interface for extracting multiple handcrafted features.
    
    Combines Gabor, LBP, SIFT, and HOG features into a single feature vector.
    """
    
    def __init__(
        self,
        use_gabor: bool = True,
        use_lbp: bool = True,
        use_sift: bool = True,
        use_hog: bool = True,
        gabor_config: Optional[Dict] = None,
        lbp_config: Optional[Dict] = None,
        sift_config: Optional[Dict] = None,
        hog_config: Optional[Dict] = None
    ):
        """
        Args:
            use_gabor: Whether to extract Gabor features
            use_lbp: Whether to extract LBP features
            use_sift: Whether to extract SIFT features
            use_hog: Whether to extract HOG features
            *_config: Configuration dictionaries for each extractor
        """
        self.extractors = {}
        
        if use_gabor:
            self.extractors['gabor'] = GaborFeatureExtractor(
                **(gabor_config or {})
            )
        
        if use_lbp:
            self.extractors['lbp'] = LBPFeatureExtractor(
                **(lbp_config or {})
            )
        
        if use_sift:
            self.extractors['sift'] = SIFTFeatureExtractor(
                **(sift_config or {})
            )
        
        if use_hog:
            self.extractors['hog'] = HOGFeatureExtractor(
                **(hog_config or {})
            )
    
    def extract(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all configured features from an image.
        
        Args:
            image: Input image (H, W, C) or (H, W)
            
        Returns:
            Dictionary mapping feature names to feature vectors
        """
        features = {}
        for name, extractor in self.extractors.items():
            features[name] = extractor.extract(image)
        return features
    
    def extract_concatenated(self, image: np.ndarray) -> np.ndarray:
        """
        Extract and concatenate all features into a single vector.
        
        Args:
            image: Input image
            
        Returns:
            Concatenated feature vector
        """
        features = self.extract(image)
        return np.concatenate(list(features.values()))
    
    def extract_batch(
        self,
        images: np.ndarray,
        concatenate: bool = True
    ) -> np.ndarray:
        """
        Extract features from a batch of images.
        
        Args:
            images: Batch of images (N, H, W, C) or (N, H, W)
            concatenate: If True, concatenate all features
            
        Returns:
            Feature matrix (N, total_features) if concatenate=True
            Dictionary of feature matrices otherwise
        """
        if concatenate:
            return np.array([
                self.extract_concatenated(img) for img in images
            ])
        else:
            batch_features = {name: [] for name in self.extractors.keys()}
            for img in images:
                features = self.extract(img)
                for name, feat in features.items():
                    batch_features[name].append(feat)
            return {
                name: np.array(feats)
                for name, feats in batch_features.items()
            }
    
    def get_feature_dims(self) -> Dict[str, int]:
        """Get dimensionality of each feature type."""
        # Extract from a dummy image to get dimensions
        dummy_img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        features = self.extract(dummy_img)
        return {name: len(feat) for name, feat in features.items()}
    
    def get_total_feature_dim(self) -> int:
        """Get total dimensionality of concatenated features."""
        return sum(self.get_feature_dims().values())
