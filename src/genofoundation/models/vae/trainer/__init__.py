"""
VAE Training Module with Accelerate and Hydra support.
"""

# Configure logging to output to console by default
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)

"""
Features:
- Multi-GPU / Distributed training via Accelerate
- Mixed precision training (fp16/bf16)
- Gradient accumulation
- Hydra configuration management
- Callback system for extensibility

Usage:
    from trainer import VAETrainer, TrainerConfig

    config = TrainerConfig(
        epochs=100,
        batch_size=256,
        learning_rate=1e-3,
        mixed_precision="fp16",
    )

    trainer = VAETrainer(
        model=your_model,  # You create the model
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    train_losses, val_losses = trainer.train()

Multi-GPU training:
    accelerate launch your_script.py
"""

from .config import (
    TrainerConfig,
    TrainingConfig,  # Backwards compatibility alias
    DataConfig,
    ExperimentConfig,
    Config,
)
from .core import VAETrainer
from .schedulers import BetaScheduler, create_lr_scheduler
from .early_stopping import EarlyStopping
from .logging import WandbLogger, MetricsLogger  # MetricsLogger is alias for backwards compat
from .checkpointing import CheckpointManager
from .callbacks import (
    Callback,
    CallbackList,
    ProgressCallback,
    GradientMonitorCallback,
)

__all__ = [
    # Main classes
    "VAETrainer",
    # Configuration
    "TrainerConfig",
    "TrainingConfig",  # Backwards compatibility
    "DataConfig",
    "ExperimentConfig",
    "Config",
    # Schedulers
    "BetaScheduler",
    "create_lr_scheduler",
    # Training utilities
    "EarlyStopping",
    "WandbLogger",
    "MetricsLogger",  # Backwards compatibility alias
    "CheckpointManager",
    # Callbacks
    "Callback",
    "CallbackList",
    "ProgressCallback",
    "GradientMonitorCallback",
]
