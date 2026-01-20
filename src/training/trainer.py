"""
Trainer class for managing model training process.
"""

import tensorflow as tf
from typing import Optional, List, Dict, Any
from pathlib import Path
import json

from ..models.base_model import BaseModel
from ..utils.logger import Logger
from ..utils.checkpoint import CheckpointManager


class Trainer:
    """Trainer class encapsulating the training process.
    
    Manages training loop, callbacks, logging, and checkpointing.
    """
    
    def __init__(
        self,
        model: BaseModel,
        logger: Optional[Logger] = None,
        checkpoint_manager: Optional[CheckpointManager] = None
    ):
        """
        Args:
            model: Model instance to train
            logger: Logger instance
            checkpoint_manager: Checkpoint manager instance
        """
        self.model = model
        self.logger = logger or Logger(name='trainer')
        self.checkpoint_manager = checkpoint_manager
        
        self.history: Optional[tf.keras.callbacks.History] = None
        self.callbacks: List[tf.keras.callbacks.Callback] = []
    
    def add_callback(self, callback: tf.keras.callbacks.Callback) -> None:
        """Add a callback to the trainer.
        
        Args:
            callback: Keras callback instance
        """
        self.callbacks.append(callback)
    
    def setup_callbacks(
        self,
        tensorboard_dir: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        early_stopping_patience: int = 10,
        reduce_lr_patience: int = 5
    ) -> None:
        """Setup standard callbacks.
        
        Args:
            tensorboard_dir: Directory for TensorBoard logs
            checkpoint_dir: Directory for model checkpoints
            early_stopping_patience: Patience for early stopping
            reduce_lr_patience: Patience for learning rate reduction
        """
        # TensorBoard
        if tensorboard_dir:
            tb_callback = tf.keras.callbacks.TensorBoard(
                log_dir=tensorboard_dir,
                histogram_freq=1
            )
            self.add_callback(tb_callback)
        
        # Model Checkpoint
        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{checkpoint_dir}/model_{{epoch:02d}}_{{val_loss:.4f}}.h5",
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            )
            self.add_callback(ckpt_callback)
        
        # Early Stopping
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=early_stopping_patience,
            restore_best_weights=True
        )
        self.add_callback(early_stop)
        
        # Reduce LR on Plateau
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-7
        )
        self.add_callback(reduce_lr)
    
    def train(
        self,
        train_data,
        validation_data=None,
        epochs: int = 100,
        verbose: int = 1
    ) -> tf.keras.callbacks.History:
        """Train the model.
        
        Args:
            train_data: Training dataset
            validation_data: Validation dataset
            epochs: Number of epochs
            verbose: Verbosity level
            
        Returns:
            Training history
        """
        self.logger.info(f"Starting training for {epochs} epochs...")
        
        # Train
        self.history = self.model.fit(
            train_data=train_data,
            validation_data=validation_data,
            epochs=epochs,
            callbacks=self.callbacks,
            verbose=verbose
        )
        
        self.logger.info("Training completed!")
        
        return self.history
    
    def save_history(self, filepath: str) -> None:
        """Save training history to JSON.
        
        Args:
            filepath: Path to save history
        """
        if self.history is None:
            raise RuntimeError("No training history available")
        
        history_dict = {
            key: [float(val) for val in values]
            for key, values in self.history.history.items()
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(history_dict, f, indent=2)
        
        self.logger.info(f"Training history saved to {filepath}")
    
    def get_best_epoch(self, metric: str = 'val_loss', mode: str = 'min') -> int:
        """Get the epoch with the best metric value.
        
        Args:
            metric: Metric name
            mode: 'min' or 'max'
            
        Returns:
            Best epoch number
        """
        if self.history is None:
            raise RuntimeError("No training history available")
        
        values = self.history.history[metric]
        
        if mode == 'min':
            best_idx = values.index(min(values))
        else:
            best_idx = values.index(max(values))
        
        return best_idx + 1  # Epochs are 1-indexed
