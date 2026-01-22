"""
Checkpoint management for training.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.optim as optim

if TYPE_CHECKING:
    from .config import TrainingConfig

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages saving and loading of training checkpoints.

    Handles:
    - Saving model, optimizer, scheduler, and training state
    - Loading and restoring training state
    - Managing best model checkpoints
    - Optional checkpoint rotation (keeping only N most recent)
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: Optional[int] = None,
    ):
        """Initialize the checkpoint manager.

        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep (None = keep all)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self._checkpoint_files: List[str] = []

    def save(
        self,
        filename: str,
        model: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        global_step: int,
        config: 'TrainingConfig',
        best_val_loss: float,
        scheduler_state: Optional[dict] = None,
        beta_scheduler_state: Optional[dict] = None,
        early_stopping_state: Optional[dict] = None,
        scaler_state: Optional[dict] = None,
        train_losses: Optional[List[float]] = None,
        val_losses: Optional[List[float]] = None,
        extra_state: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ) -> str:
        """Save a training checkpoint.

        Args:
            filename: Name of the checkpoint file
            model: The model to save
            optimizer: The optimizer to save
            epoch: Current epoch number
            global_step: Current global step
            config: Training configuration
            best_val_loss: Best validation loss seen so far
            scheduler_state: LR scheduler state dict
            beta_scheduler_state: Beta scheduler state
            early_stopping_state: Early stopping state
            scaler_state: AMP scaler state dict
            train_losses: List of training losses per epoch
            val_losses: List of validation losses per epoch
            extra_state: Additional state to save
            is_best: Whether this is the best model so far

        Returns:
            Path to the saved checkpoint
        """
        checkpoint = {
            # Training progress
            'epoch': epoch,
            'global_step': global_step,

            # Model and optimizer
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),

            # Schedulers and training components
            'scheduler_state_dict': scheduler_state,
            'beta_scheduler_state': beta_scheduler_state,
            'early_stopping_state': early_stopping_state,
            'scaler_state_dict': scaler_state,

            # Metrics
            'best_val_loss': best_val_loss,
            'train_losses': train_losses or [],
            'val_losses': val_losses or [],

            # Configuration
            'config': config.to_dict(),
        }

        # Add any extra state
        if extra_state:
            checkpoint.update(extra_state)

        # Save checkpoint
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")

        # Track checkpoint for rotation
        if filename not in ["best_model.pt", "final_model.pt"]:
            self._checkpoint_files.append(str(path))
            self._rotate_checkpoints()

        # Save best model copy
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")

        return str(path)

    def load(
        self,
        path: str,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, Any]:
        """Load a training checkpoint.

        Args:
            path: Path to the checkpoint file
            model: Model to load state into
            optimizer: Optional optimizer to load state into
            device: Device to map tensors to

        Returns:
            Dictionary containing all checkpoint data
        """
        map_location = device if device else 'cpu'
        # weights_only=False is required because checkpoint contains non-tensor data
        # (config dict, loss lists, etc.). Only load checkpoints from trusted sources.
        checkpoint = torch.load(path, map_location=map_location, weights_only=False)

        # Load model state
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded model state from {path}")

        # Load optimizer state if provided
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Loaded optimizer state")

        return checkpoint

    def _rotate_checkpoints(self) -> None:
        """Remove old checkpoints if exceeding max_checkpoints."""
        if self.max_checkpoints is None:
            return

        while len(self._checkpoint_files) > self.max_checkpoints:
            old_checkpoint = self._checkpoint_files.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
                logger.info(f"Removed old checkpoint: {old_checkpoint}")

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the most recent checkpoint.

        Returns:
            Path to the latest checkpoint, or None if no checkpoints exist
        """
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if not checkpoints:
            return None

        # Sort by modification time
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(checkpoints[0])

    def get_best_checkpoint(self) -> Optional[str]:
        """Get the path to the best model checkpoint.

        Returns:
            Path to the best checkpoint, or None if it doesn't exist
        """
        best_path = self.checkpoint_dir / "best_model.pt"
        return str(best_path) if best_path.exists() else None

    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints.

        Returns:
            List of checkpoint paths
        """
        return [str(p) for p in self.checkpoint_dir.glob("*.pt")]


def restore_training_state(
    checkpoint: Dict[str, Any],
    scheduler=None,
    beta_scheduler=None,
    early_stopping=None,
    scaler=None,
) -> Dict[str, Any]:
    """Restore training component states from a checkpoint.

    Args:
        checkpoint: Loaded checkpoint dictionary
        scheduler: LR scheduler to restore
        beta_scheduler: Beta scheduler to restore
        early_stopping: Early stopping handler to restore
        scaler: AMP scaler to restore

    Returns:
        Dictionary with restored values (epoch, global_step, etc.)
    """
    # Restore scheduler
    if scheduler is not None and checkpoint.get('scheduler_state_dict'):
        try:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Restored scheduler state")
        except Exception as e:
            logger.warning(f"Failed to restore scheduler state: {e}")

    # Restore beta scheduler
    if beta_scheduler is not None and checkpoint.get('beta_scheduler_state'):
        beta_scheduler.load_state(checkpoint['beta_scheduler_state'])
        logger.info("Restored beta scheduler state")

    # Restore early stopping
    if early_stopping is not None and checkpoint.get('early_stopping_state'):
        early_stopping.load_state(checkpoint['early_stopping_state'])
        logger.info("Restored early stopping state")

    # Restore scaler
    if scaler is not None and checkpoint.get('scaler_state_dict'):
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        logger.info("Restored AMP scaler state")

    return {
        'epoch': checkpoint.get('epoch', 0),
        'global_step': checkpoint.get('global_step', 0),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf')),
        'train_losses': checkpoint.get('train_losses', []),
        'val_losses': checkpoint.get('val_losses', []),
    }
