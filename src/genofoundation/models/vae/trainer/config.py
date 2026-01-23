"""
Training configuration with Hydra support.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Any
from omegaconf import DictConfig, OmegaConf

VALID_OPTIMIZERS = ["adam", "adamw", "sgd"]
VALID_SCHEDULERS = ["cosine", "plateau", "onecycle", "warmup_cosine", "warmup", None]
VALID_BETA_SCHEDULES = ["linear", "cyclical", "constant", "sigmoid"]
VALID_MIXED_PRECISION = ["no", "fp16", "bf16"]
VALID_WANDB_MODES = ["online", "offline", "disabled"]


@dataclass
class TrainerConfig:
    """Trainer configuration (Hydra-compatible)."""

    # Basic training parameters
    epochs: int = 100
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5

    # VAE-specific parameters
    beta: float = 1.0
    beta_warmup_epochs: int = 10
    beta_schedule: str = "linear"

    # Optimizer
    optimizer: str = "adamw"
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])

    # Learning rate scheduling
    scheduler: Optional[str] = "cosine"
    lr_warmup_epochs: int = 5
    min_lr: float = 1e-6

    # Gradient settings
    grad_clip_norm: float = 50.0
    gradient_accumulation_steps: int = 1
    record_grad_norm: bool = True

    # Scheduler-specific parameters
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    onecycle_pct_start: float = 0.1
    warmup_start_factor: float = 0.01

    # Early stopping
    early_stopping: bool = True
    patience: int = 15
    min_delta: float = 1e-4

    # Checkpointing
    save_every: int = 10
    save_best: bool = True
    max_checkpoints: Optional[int] = 5
    save_after_epoch: int = 20  # Skip saving checkpoints during warmup

    # Logging (Weights & Biases)
    log_every: int = 100
    wandb_mode: str = "disabled"  # "online", "offline", or "disabled"
    wandb_project: str = "vae_training"
    wandb_entity: Optional[str] = None  # wandb username or team name
    wandb_run_name: Optional[str] = None  # auto-generated if None
    wandb_tags: Optional[List[str]] = None
    wandb_notes: Optional[str] = None
    wandb_dir: Optional[str] = None  # offline log directory (uses output_dir/wandb if None)

    # Accelerate settings
    mixed_precision: str = "no"

    # DataLoader settings
    num_workers: int = 4

    def __post_init__(self):
        """Validate configuration after initialization."""
        # Convert OmegaConf ListConfig to regular list if needed
        if hasattr(self.betas, '_iter_ex'):
            self.betas = list(self.betas)
        self._validate()

    def _validate(self):
        """Validate all configuration values."""
        if self.epochs <= 0:
            raise ValueError(f"epochs must be positive, got {self.epochs}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {self.weight_decay}")
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        if self.beta_schedule not in VALID_BETA_SCHEDULES:
            raise ValueError(f"beta_schedule must be one of {VALID_BETA_SCHEDULES}, got {self.beta_schedule}")
        if self.optimizer not in VALID_OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {VALID_OPTIMIZERS}, got {self.optimizer}")
        if self.scheduler is not None and self.scheduler not in VALID_SCHEDULERS:
            raise ValueError(f"scheduler must be one of {VALID_SCHEDULERS}, got {self.scheduler}")
        if self.mixed_precision not in VALID_MIXED_PRECISION:
            raise ValueError(f"mixed_precision must be one of {VALID_MIXED_PRECISION}, got {self.mixed_precision}")
        if self.gradient_accumulation_steps < 1:
            raise ValueError(f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}")
        if self.wandb_mode not in VALID_WANDB_MODES:
            raise ValueError(f"wandb_mode must be one of {VALID_WANDB_MODES}, got {self.wandb_mode}")

    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'beta': self.beta,
            'beta_warmup_epochs': self.beta_warmup_epochs,
            'beta_schedule': self.beta_schedule,
            'optimizer': self.optimizer,
            'betas': list(self.betas),
            'scheduler': self.scheduler,
            'lr_warmup_epochs': self.lr_warmup_epochs,
            'min_lr': self.min_lr,
            'grad_clip_norm': self.grad_clip_norm,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'record_grad_norm': self.record_grad_norm,
            'plateau_factor': self.plateau_factor,
            'plateau_patience': self.plateau_patience,
            'onecycle_pct_start': self.onecycle_pct_start,
            'warmup_start_factor': self.warmup_start_factor,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'save_every': self.save_every,
            'save_best': self.save_best,
            'max_checkpoints': self.max_checkpoints,
            'save_after_epoch': self.save_after_epoch,
            'log_every': self.log_every,
            'wandb_mode': self.wandb_mode,
            'wandb_project': self.wandb_project,
            'wandb_entity': self.wandb_entity,
            'wandb_run_name': self.wandb_run_name,
            'wandb_tags': self.wandb_tags,
            'wandb_notes': self.wandb_notes,
            'wandb_dir': self.wandb_dir,
            'mixed_precision': self.mixed_precision,
            'num_workers': self.num_workers,
        }


@dataclass
class DataConfig:
    """Data configuration."""
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    val_split: float = 0.1
    normalize: bool = True


@dataclass
class ExperimentConfig:
    """Experiment settings."""
    name: str = "vae_training"
    seed: int = 42
    output_dir: str = "./outputs"


@dataclass
class Config:
    """Root configuration combining all sub-configs.

    Note: Model configuration is not included here.
    Users should create and manage their own models.
    """
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'Config':
        """Create Config from Hydra DictConfig."""
        return cls(
            experiment=ExperimentConfig(**OmegaConf.to_container(cfg.experiment, resolve=True)),
            trainer=TrainerConfig(**OmegaConf.to_container(cfg.trainer, resolve=True)),
            data=DataConfig(**OmegaConf.to_container(cfg.data, resolve=True)),
        )


# Backwards compatibility alias
TrainingConfig = TrainerConfig
