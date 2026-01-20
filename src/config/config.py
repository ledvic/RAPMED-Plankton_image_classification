"""
Configuration management using OOP principles.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml
from pathlib import Path


@dataclass
class DataConfig:
    """Data configuration."""
    data_dir: str
    img_size: int = 128
    batch_size: int = 32
    validation_split: float = 0.2
    seed: int = 42
    num_classes: int = 21


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    architecture: str = "efficientnetv2-b1"
    num_classes: int = 21
    img_size: int = 128
    dropout_rate: float = 0.5
    dense_units: int = 256
    weights: str = "imagenet"
    trainable_base: bool = False


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    learning_rate: float = 1e-3
    optimizer: str = "adam"
    loss: str = "categorical_crossentropy"
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5
    checkpoint_dir: str = "models/checkpoints"


@dataclass
class FeatureConfig:
    """Feature extraction configuration."""
    use_gabor: bool = True
    use_lbp: bool = True
    use_sift: bool = True
    use_hog: bool = True
    gabor_frequencies: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])
    gabor_orientations: List[float] = field(default_factory=lambda: [0, 45, 90, 135])
    lbp_n_points: int = 24
    lbp_radius: int = 3
    sift_n_features: int = 100
    hog_orientations: int = 9


class Config:
    """Main configuration class that aggregates all configs."""
    
    def __init__(
        self,
        data: DataConfig,
        model: ModelConfig,
        training: TrainingConfig,
        features: Optional[FeatureConfig] = None
    ):
        self.data = data
        self.model = model
        self.training = training
        self.features = features or FeatureConfig()
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        data_config = DataConfig(**config_dict.get('data', {}))
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        features_config = FeatureConfig(**config_dict.get('features', {}))
        
        return cls(data_config, model_config, training_config, features_config)
    
    def to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'features': self.features.__dict__
        }
        
        Path(config_path).parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def __repr__(self) -> str:
        return (
            f"Config(\n"
            f"  data={self.data},\n"
            f"  model={self.model},\n"
            f"  training={self.training},\n"
            f"  features={self.features}\n"
            f")"
        )
