"""
Callback system for extensible training hooks.
"""

from abc import ABC
from typing import Dict, Any, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import VAETrainer


class Callback(ABC):
    """Base class for training callbacks.

    Callbacks provide hooks into the training process without modifying
    the core trainer logic. Override the methods you need.

    All methods receive the trainer instance, allowing access to:
    - trainer.model: The model being trained
    - trainer.config: Training configuration
    - trainer.current_epoch: Current epoch number
    - trainer.global_step: Global training step
    - trainer.optimizer: The optimizer
    - trainer.device: Training device
    """

    def on_train_start(self, trainer: 'VAETrainer') -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, trainer: 'VAETrainer') -> None:
        """Called at the end of training."""
        pass

    def on_epoch_start(self, trainer: 'VAETrainer', epoch: int) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(
        self,
        trainer: 'VAETrainer',
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        """Called at the end of each epoch."""
        pass

    def on_batch_start(
        self,
        trainer: 'VAETrainer',
        batch_idx: int,
        batch: Any,
    ) -> None:
        """Called at the beginning of each batch."""
        pass

    def on_batch_end(
        self,
        trainer: 'VAETrainer',
        batch_idx: int,
        batch_metrics: Dict[str, float],
    ) -> None:
        """Called at the end of each batch."""
        pass

    def on_validation_start(self, trainer: 'VAETrainer') -> None:
        """Called at the beginning of validation."""
        pass

    def on_validation_end(
        self,
        trainer: 'VAETrainer',
        val_metrics: Dict[str, float],
    ) -> None:
        """Called at the end of validation."""
        pass

    def on_checkpoint_save(
        self,
        trainer: 'VAETrainer',
        checkpoint_path: str,
    ) -> None:
        """Called after saving a checkpoint."""
        pass

    def on_checkpoint_load(
        self,
        trainer: 'VAETrainer',
        checkpoint_path: str,
    ) -> None:
        """Called after loading a checkpoint."""
        pass


class CallbackList:
    """Container for managing multiple callbacks."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def append(self, callback: Callback) -> None:
        """Add a callback to the list."""
        self.callbacks.append(callback)

    def on_train_start(self, trainer: 'VAETrainer') -> None:
        for callback in self.callbacks:
            callback.on_train_start(trainer)

    def on_train_end(self, trainer: 'VAETrainer') -> None:
        for callback in self.callbacks:
            callback.on_train_end(trainer)

    def on_epoch_start(self, trainer: 'VAETrainer', epoch: int) -> None:
        for callback in self.callbacks:
            callback.on_epoch_start(trainer, epoch)

    def on_epoch_end(
        self,
        trainer: 'VAETrainer',
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        for callback in self.callbacks:
            callback.on_epoch_end(trainer, epoch, train_metrics, val_metrics)

    def on_batch_start(
        self,
        trainer: 'VAETrainer',
        batch_idx: int,
        batch: Any,
    ) -> None:
        for callback in self.callbacks:
            callback.on_batch_start(trainer, batch_idx, batch)

    def on_batch_end(
        self,
        trainer: 'VAETrainer',
        batch_idx: int,
        batch_metrics: Dict[str, float],
    ) -> None:
        for callback in self.callbacks:
            callback.on_batch_end(trainer, batch_idx, batch_metrics)

    def on_validation_start(self, trainer: 'VAETrainer') -> None:
        for callback in self.callbacks:
            callback.on_validation_start(trainer)

    def on_validation_end(
        self,
        trainer: 'VAETrainer',
        val_metrics: Dict[str, float],
    ) -> None:
        for callback in self.callbacks:
            callback.on_validation_end(trainer, val_metrics)

    def on_checkpoint_save(
        self,
        trainer: 'VAETrainer',
        checkpoint_path: str,
    ) -> None:
        for callback in self.callbacks:
            callback.on_checkpoint_save(trainer, checkpoint_path)

    def on_checkpoint_load(
        self,
        trainer: 'VAETrainer',
        checkpoint_path: str,
    ) -> None:
        for callback in self.callbacks:
            callback.on_checkpoint_load(trainer, checkpoint_path)


class ProgressCallback(Callback):
    """Example callback that prints training progress."""

    def on_epoch_end(
        self,
        trainer: 'VAETrainer',
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        msg = f"Epoch {epoch + 1}/{trainer.config.epochs} - "
        msg += f"Train Loss: {train_metrics.get('loss', 0):.4f}"
        if val_metrics:
            msg += f" - Val Loss: {val_metrics.get('loss', 0):.4f}"
        print(msg)


class GradientMonitorCallback(Callback):
    """Callback to monitor gradient statistics."""

    def __init__(self, log_every: int = 100):
        self.log_every = log_every

    def on_batch_end(
        self,
        trainer: 'VAETrainer',
        batch_idx: int,
        batch_metrics: Dict[str, float],
    ) -> None:
        if batch_idx % self.log_every != 0:
            return

        total_norm = 0.0
        for p in trainer.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5

        print(f"Step {trainer.global_step} - Gradient norm: {total_norm:.4f}")
