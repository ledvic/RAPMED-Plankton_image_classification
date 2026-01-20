"""
Feature extraction pipeline.
"""

import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
import pickle

from .base_extractor import BaseFeatureExtractor


class FeaturePipeline:
    """Pipeline for extracting and managing multiple feature types.
    
    Uses Strategy pattern to allow dynamic feature extraction configuration.
    """
    
    def __init__(self, extractors: Optional[List[BaseFeatureExtractor]] = None):
        """
        Args:
            extractors: List of feature extractor instances
        """
        self._extractors: Dict[str, BaseFeatureExtractor] = {}
        
        if extractors:
            for extractor in extractors:
                self.add_extractor(extractor)
    
    def add_extractor(self, extractor: BaseFeatureExtractor) -> None:
        """Add a feature extractor to the pipeline.
        
        Args:
            extractor: Feature extractor instance
        """
        if not isinstance(extractor, BaseFeatureExtractor):
            raise TypeError(
                f"Extractor must be instance of BaseFeatureExtractor, "
                f"got {type(extractor)}"
            )
        
        self._extractors[extractor.name] = extractor
    
    def remove_extractor(self, name: str) -> None:
        """Remove a feature extractor from the pipeline.
        
        Args:
            name: Name of the extractor to remove
        """
        if name in self._extractors:
            del self._extractors[name]
    
    def extract(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract all features from an image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary mapping extractor names to feature vectors
        """
        features = {}
        for name, extractor in self._extractors.items():
            features[name] = extractor.extract(image)
        return features
    
    def extract_concatenated(self, image: np.ndarray) -> np.ndarray:
        """Extract and concatenate all features.
        
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
        concatenate: bool = True,
        verbose: bool = False
    ) -> np.ndarray:
        """Extract features from a batch of images.
        
        Args:
            images: Batch of images
            concatenate: Whether to concatenate all features
            verbose: Whether to print progress
            
        Returns:
            Feature matrix or dictionary of feature matrices
        """
        if concatenate:
            features = []
            for i, img in enumerate(images):
                if verbose and i % 100 == 0:
                    print(f"Processing image {i}/{len(images)}")
                features.append(self.extract_concatenated(img))
            return np.array(features)
        else:
            batch_features = {name: [] for name in self._extractors.keys()}
            for img in images:
                img_features = self.extract(img)
                for name, feat in img_features.items():
                    batch_features[name].append(feat)
            return {
                name: np.array(feats)
                for name, feats in batch_features.items()
            }
    
    def get_feature_dims(self) -> Dict[str, int]:
        """Get dimensionality of each feature type."""
        return {
            name: extractor.get_feature_dim()
            for name, extractor in self._extractors.items()
        }
    
    def get_total_feature_dim(self) -> int:
        """Get total dimensionality of concatenated features."""
        return sum(self.get_feature_dims().values())
    
    def get_extractors(self) -> List[str]:
        """Get list of extractor names."""
        return list(self._extractors.keys())
    
    def save(self, path: str) -> None:
        """Save pipeline configuration."""
        config = {
            'extractors': [
                extractor.get_config()
                for extractor in self._extractors.values()
            ]
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(config, f)
    
    @classmethod
    def load(cls, path: str) -> "FeaturePipeline":
        """Load pipeline configuration."""
        with open(path, 'rb') as f:
            config = pickle.load(f)
        # Note: Would need registry to recreate extractors from config
        raise NotImplementedError("Load from config not yet implemented")
    
    def __len__(self) -> int:
        return len(self._extractors)
    
    def __repr__(self) -> str:
        extractors_str = ', '.join(self._extractors.keys())
        return f"FeaturePipeline(extractors=[{extractors_str}])"
