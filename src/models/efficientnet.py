"""
EfficientNet model implementation using OOP.
"""

import tensorflow as tf
from typing import Tuple, Optional
from .base_model import BaseModel


class EfficientNetModel(BaseModel):
    """EfficientNet-based model for plankton classification.
    
    Implements transfer learning with EfficientNetV2.
    """
    
    def __init__(
        self,
        num_classes: int,
        img_size: int = 128,
        variant: str = 'B1',
        weights: str = 'imagenet',
        trainable_base: bool = False,
        dense_units: int = 256,
        dropout_rate: float = 0.5,
        activation: str = 'swish',
        name: str = 'efficientnet_model'
    ):
        """
        Args:
            num_classes: Number of output classes
            img_size: Input image size
            variant: EfficientNet variant (B0, B1, B2, etc.)
            weights: Pretrained weights
            trainable_base: Whether to train base model
            dense_units: Number of dense layer units
            dropout_rate: Dropout rate
            activation: Activation function
            name: Model name
        """
        super().__init__(
            name=name,
            num_classes=num_classes,
            img_size=img_size,
            variant=variant,
            weights=weights,
            trainable_base=trainable_base,
            dense_units=dense_units,
            dropout_rate=dropout_rate,
            activation=activation
        )
        
        self.num_classes = num_classes
        self.img_size = img_size
        self.variant = variant
        self.weights = weights
        self.trainable_base = trainable_base
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.activation = activation
        
        self.base_model: Optional[tf.keras.Model] = None
    
    def build(self) -> tf.keras.Model:
        """Build EfficientNet model."""
        # Get appropriate EfficientNet variant
        efficientnet_class = self._get_efficientnet_class()
        
        # Build base model
        self.base_model = efficientnet_class(
            include_top=False,
            weights=self.weights,
            input_shape=(self.img_size, self.img_size, 3)
        )
        
        # Set trainability
        self.base_model.trainable = self.trainable_base
        
        # Build full model
        inputs = tf.keras.Input(shape=(self.img_size, self.img_size, 3))
        x = self.base_model(inputs, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(self.dense_units, activation=self.activation)(x)
        x = tf.keras.layers.Dropout(self.dropout_rate)(x)
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name=self.name)
        self.model = model
        
        return model
    
    def _get_efficientnet_class(self):
        """Get EfficientNet class based on variant."""
        variant_map = {
            'B0': tf.keras.applications.EfficientNetV2B0,
            'B1': tf.keras.applications.EfficientNetV2B1,
            'B2': tf.keras.applications.EfficientNetV2B2,
            'B3': tf.keras.applications.EfficientNetV2B3,
        }
        
        if self.variant not in variant_map:
            raise ValueError(
                f"Unknown EfficientNet variant: {self.variant}. "
                f"Choose from {list(variant_map.keys())}"
            )
        
        return variant_map[self.variant]
    
    def freeze_base(self) -> None:
        """Freeze base model layers."""
        if self.base_model is not None:
            self.base_model.trainable = False
            self.trainable_base = False
    
    def unfreeze_base(self, from_layer: Optional[int] = None) -> None:
        """Unfreeze base model layers.
        
        Args:
            from_layer: Unfreeze from this layer onwards. If None, unfreeze all.
        """
        if self.base_model is not None:
            if from_layer is None:
                self.base_model.trainable = True
            else:
                for layer in self.base_model.layers[:from_layer]:
                    layer.trainable = False
                for layer in self.base_model.layers[from_layer:]:
                    layer.trainable = True
            
            self.trainable_base = True
            
            # Need to recompile after changing trainability
            if self._is_compiled:
                print("Warning: Model needs to be recompiled after changing trainability")
