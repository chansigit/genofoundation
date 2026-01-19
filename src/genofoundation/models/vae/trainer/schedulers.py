"""
Learning rate and beta schedulers for VAE training.
"""

import math
from typing import Optional

import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR,
    LinearLR,
    SequentialLR,
    _LRScheduler,
)

from .config import TrainingConfig


class BetaScheduler:
    """KL weight (beta) scheduler for VAE training.

    Supports multiple scheduling strategies:
    - constant: Fixed beta value throughout training
    - linear: Linear warmup from initial to final beta
    - cyclical: Cyclical annealing (https://arxiv.org/abs/1903.10145)
    - sigmoid: Smooth sigmoid warmup
    """

    def __init__(
        self,
        schedule: str = "linear",
        initial_beta: float = 0.0,
        final_beta: float = 1.0,
        warmup_epochs: int = 10,
        cycle_epochs: int = 10,
    ):
        self.schedule = schedule
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        self.warmup_epochs = warmup_epochs
        self.cycle_epochs = cycle_epochs
        self.current_beta = initial_beta

    def step(self, epoch: int) -> float:
        """Update and return the current beta value for the given epoch."""
        if self.schedule == "constant":
            self.current_beta = self.final_beta

        elif self.schedule == "linear":
            if epoch < self.warmup_epochs:
                progress = epoch / max(1, self.warmup_epochs)
                self.current_beta = self.initial_beta + progress * (self.final_beta - self.initial_beta)
            else:
                self.current_beta = self.final_beta

        elif self.schedule == "cyclical":
            # Cyclical annealing: https://arxiv.org/abs/1903.10145
            position = (epoch % self.cycle_epochs) / max(1, self.cycle_epochs)
            self.current_beta = min(self.final_beta, position * self.final_beta)

        elif self.schedule == "sigmoid":
            if epoch < self.warmup_epochs:
                x = (epoch - self.warmup_epochs / 2) / (self.warmup_epochs / 4)
                self.current_beta = self.final_beta / (1 + math.exp(-x))
            else:
                self.current_beta = self.final_beta

        return self.current_beta

    def get_state(self) -> dict:
        """Get scheduler state for checkpointing."""
        return {
            'current_beta': self.current_beta,
            'schedule': self.schedule,
            'initial_beta': self.initial_beta,
            'final_beta': self.final_beta,
            'warmup_epochs': self.warmup_epochs,
            'cycle_epochs': self.cycle_epochs,
        }

    def load_state(self, state: dict):
        """Load scheduler state from checkpoint."""
        self.current_beta = state.get('current_beta', self.initial_beta)

    @classmethod
    def from_config(cls, config: TrainingConfig) -> 'BetaScheduler':
        """Create a BetaScheduler from TrainingConfig."""
        return cls(
            schedule=config.beta_schedule,
            initial_beta=0.0,
            final_beta=config.beta,
            warmup_epochs=config.beta_warmup_epochs,
        )


def create_lr_scheduler(
    optimizer: optim.Optimizer,
    config: TrainingConfig,
    steps_per_epoch: int,
) -> Optional[_LRScheduler]:
    """Create a learning rate scheduler based on configuration.

    Args:
        optimizer: The optimizer to schedule
        config: Training configuration
        steps_per_epoch: Number of training steps per epoch

    Returns:
        A learning rate scheduler or None
    """
    total_epochs = config.epochs
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * config.lr_warmup_epochs

    if config.scheduler == "cosine":
        return CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=config.min_lr,
        )

    elif config.scheduler == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.plateau_factor,
            patience=config.plateau_patience,
            min_lr=config.min_lr,
        )

    elif config.scheduler == "onecycle":
        return OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            total_steps=total_steps,
            pct_start=config.onecycle_pct_start,
        )

    elif config.scheduler == "warmup_cosine":
        warmup = LinearLR(
            optimizer,
            start_factor=config.warmup_start_factor,
            end_factor=1.0,
            total_iters=config.lr_warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - config.lr_warmup_epochs,
            eta_min=config.min_lr,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[config.lr_warmup_epochs],
        )

    elif config.scheduler == "warmup":
        return LinearLR(
            optimizer,
            start_factor=config.warmup_start_factor,
            end_factor=1.0,
            total_iters=config.lr_warmup_epochs,
        )

    return None


def fast_forward_scheduler(
    scheduler: Optional[_LRScheduler],
    target_epoch: int,
    steps_per_epoch: int,
    scheduler_type: str,
) -> None:
    """Fast-forward a scheduler to a target epoch position.

    Used when resuming training to bring the scheduler to the correct state.

    Args:
        scheduler: The scheduler to fast-forward
        target_epoch: Target epoch to reach
        steps_per_epoch: Number of steps per epoch (for step-based schedulers)
        scheduler_type: Type of scheduler (from config.scheduler)
    """
    if scheduler is None or target_epoch <= 0:
        return

    if isinstance(scheduler, (CosineAnnealingLR, LinearLR, SequentialLR)):
        # Epoch-based schedulers
        for _ in range(target_epoch):
            scheduler.step()

    elif isinstance(scheduler, OneCycleLR):
        # Step-based scheduler
        target_steps = target_epoch * steps_per_epoch
        for _ in range(target_steps):
            scheduler.step()

    elif isinstance(scheduler, ReduceLROnPlateau):
        # Stateful scheduler - state should be loaded from checkpoint
        pass

    else:
        # Generic fallback - assume epoch-based
        for _ in range(target_epoch):
            scheduler.step()
