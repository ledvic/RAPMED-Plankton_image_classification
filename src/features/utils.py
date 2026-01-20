"""
Utility functions for feature extraction.
"""

import numpy as np
import os
from pathlib import Path
from typing import List, Dict
import pickle


def save_features(
    features: np.ndarray,
    labels: np.ndarray,
    save_path: str,
    feature_names: List[str] = None
):
    """
    Save extracted features to disk.
    
    Args:
        features: Feature matrix (N, D)
        labels: Label array (N,)
        save_path: Path to save the features
        feature_names: List of feature names
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    data = {
        'features': features,
        'labels': labels,
        'feature_names': feature_names
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Features saved to {save_path}")
    print(f"Shape: {features.shape}")


def load_features(load_path: str) -> Dict:
    """
    Load features from disk.
    
    Args:
        load_path: Path to load features from
        
    Returns:
        Dictionary containing features, labels, and feature names
    """
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Features loaded from {load_path}")
    print(f"Shape: {data['features'].shape}")
    
    return data


def normalize_features(features: np.ndarray, method: str = 'standard') -> np.ndarray:
    """
    Normalize features.
    
    Args:
        features: Feature matrix (N, D)
        method: 'standard' (z-score) or 'minmax'
        
    Returns:
        Normalized features
    """
    if method == 'standard':
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        return (features - mean) / std
    
    elif method == 'minmax':
        min_val = np.min(features, axis=0)
        max_val = np.max(features, axis=0)
        range_val = max_val - min_val
        range_val[range_val == 0] = 1  # Avoid division by zero
        return (features - min_val) / range_val
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
