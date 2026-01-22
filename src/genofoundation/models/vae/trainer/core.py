"""
VAE Trainer with Accelerate support.

Supports:
- Multi-GPU / Distributed training via Accelerate
- Mixed precision training (fp16/bf16)
- Gradient accumulation
- Hydra configuration
- Callbacks system
"""

import logging
from typing import Optional, Dict, Any, List, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from accelerate.utils import set_seed

from .config import TrainerConfig, TrainingConfig
from .schedulers import BetaScheduler, create_lr_scheduler
from .early_stopping import EarlyStopping
from .logging import MetricsLogger
from .checkpointing import CheckpointManager, restore_training_state
from .callbacks import Callback, CallbackList

logger = logging.getLogger(__name__)


class VAETrainer:
    """VAE Trainer with Accelerate for distributed training.

    Features:
    - Multi-GPU and distributed training via Accelerate
    - Mixed precision training (fp16/bf16)
    - Gradient accumulation
    - Learning rate scheduling
    - Beta (KL weight) scheduling
    - Early stopping
    - Checkpoint save/restore
    - TensorBoard / WandB logging
    - Callback system for extensibility
    """

    def __init__(
        self,
        model: nn.Module,
        config: Union[TrainerConfig, dict],
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        callbacks: Optional[List[Callback]] = None,
        output_dir: str = "./outputs",
        seed: Optional[int] = None,
    ):
        """Initialize the trainer.

        Args:
            model: VAE model to train
            config: Training configuration (TrainerConfig or dict)
            train_loader: Training data loader (preferred over train_dataset)
            val_loader: Validation data loader (preferred over val_dataset)
            train_dataset: Training dataset (used if train_loader not provided)
            val_dataset: Validation dataset (used if val_loader not provided)
            optimizer: Custom optimizer (created from config if not provided)
            scheduler: Custom LR scheduler (created from config if not provided)
            callbacks: List of training callbacks
            output_dir: Output directory for checkpoints and logs
            seed: Random seed for reproducibility
        """
        # Parse config
        if isinstance(config, dict):
            self.config = TrainerConfig(**config)
        else:
            self.config = config

        self.output_dir = output_dir

        # Initialize Accelerator
        self.accelerator = Accelerator(
            mixed_precision=self.config.mixed_precision,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            project_dir=output_dir,
        )

        # Set seed for reproducibility
        if seed is not None:
            set_seed(seed)

        self.model = model

        # Create data loaders if not provided
        if train_loader is None and train_dataset is not None:
            train_loader = self._create_data_loader(train_dataset, shuffle=True)
        if val_loader is None and val_dataset is not None:
            val_loader = self._create_data_loader(val_dataset, shuffle=False)

        if train_loader is None:
            raise ValueError("Either train_loader or train_dataset must be provided")

        self.train_loader = train_loader
        self.val_loader = val_loader

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []

        # Set up optimizer
        self._custom_optimizer = optimizer is not None
        self.optimizer = optimizer or self._create_optimizer()

        # Set up LR scheduler
        self._custom_scheduler = scheduler is not None
        self.scheduler = scheduler or create_lr_scheduler(
            self.optimizer,
            self.config,
            steps_per_epoch=len(self.train_loader),
        )

        # Prepare with Accelerator (handles device placement, DDP wrapping, etc.)
        self.model, self.optimizer, self.train_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader
        )

        if self.val_loader is not None:
            self.val_loader = self.accelerator.prepare(self.val_loader)

        if self.scheduler is not None:
            self.scheduler = self.accelerator.prepare(self.scheduler)

        # Set up beta scheduler (not device-dependent)
        self.beta_scheduler = BetaScheduler.from_config(self.config)

        # Set up early stopping
        self.early_stopping = EarlyStopping.from_config(self.config)

        # Logging is initialized lazily in train() to support resume
        self.metrics_logger = None
        self._wandb_run_id = None  # Will be set when wandb is initialized or loaded from checkpoint

        # Set up checkpointing
        self.checkpoint_manager = CheckpointManager(
            output_dir,
            max_checkpoints=self.config.max_checkpoints,
        )

        # Set up callbacks
        self.callbacks = CallbackList(callbacks)

        # Warn if early stopping is enabled but no validation loader
        if self.early_stopping and self.val_loader is None:
            logger.warning(
                "Early stopping is enabled but no validation loader provided. "
                "Early stopping will be ineffective."
            )

    @property
    def device(self) -> torch.device:
        """Get the current device from Accelerator."""
        return self.accelerator.device

    def _create_data_loader(
        self,
        dataset: Optional[Dataset],
        shuffle: bool,
    ) -> Optional[DataLoader]:
        """Create a DataLoader from a dataset."""
        if dataset is None:
            return None

        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=shuffle,
        )

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer from configuration."""
        params = self.model.parameters()
        betas = tuple(self.config.betas)

        if self.config.optimizer == "adam":
            return optim.Adam(
                params,
                lr=self.config.learning_rate,
                betas=betas,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adamw":
            return optim.AdamW(
                params,
                lr=self.config.learning_rate,
                betas=betas,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "sgd":
            return optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _unpack_batch(self, batch: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Unpack a batch into input and optional condition.

        Args:
            batch: A tensor, tuple, or list from the data loader

        Returns:
            Tuple of (input_tensor, condition_tensor or None)
        """
        if isinstance(batch, (list, tuple)):
            x = batch[0]
            condition = batch[1] if len(batch) > 1 else None
        else:
            x = batch
            condition = None
        return x, condition

    def print(self, *args, **kwargs):
        """Print only on main process."""
        self.accelerator.print(*args, **kwargs)

    def log_info(self, msg: str):
        """Log info only on main process."""
        if self.accelerator.is_main_process:
            logger.info(msg)

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of average metrics for the epoch
        """
        self.model.train()

        # Use tensors for accumulation to reduce GPU-CPU sync overhead.
        # Only sync once at epoch end instead of every batch.
        total_loss = torch.tensor(0.0, device=self.device)
        total_recon_loss = torch.tensor(0.0, device=self.device)
        total_kl_loss = torch.tensor(0.0, device=self.device)
        total_grad_norm = 0.0
        num_batches = 0
        num_grad_steps = 0

        current_beta = self.beta_scheduler.step(self.current_epoch)

        for batch_idx, batch in enumerate(self.train_loader):
            # Callback: batch start
            self.callbacks.on_batch_start(self, batch_idx, batch)

            # Unpack batch (already on correct device via Accelerator)
            x, condition = self._unpack_batch(batch)

            # Use Accelerator's gradient accumulation context
            with self.accelerator.accumulate(self.model):
                # Forward pass (autocast handled by Accelerator)
                outputs = self.model(x, condition=condition)
                losses = self.model.loss(x, outputs, beta=current_beta)
                loss = losses['total']

                # Backward pass
                self.accelerator.backward(loss)

                # Gradient clipping and optional norm recording
                if self.accelerator.sync_gradients:
                    grad_norm = self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip_norm,
                    )
                    if self.config.record_grad_norm:
                        total_grad_norm += grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm
                        num_grad_steps += 1

                self.optimizer.step()
                self.optimizer.zero_grad()

            # Accumulate metrics on GPU (no sync per batch)
            total_loss = total_loss + loss.detach()
            total_recon_loss = total_recon_loss + losses['reconstruction'].detach()
            total_kl_loss = total_kl_loss + losses['kl'].detach()
            num_batches += 1

            # Update global step when gradients are synced
            if self.accelerator.sync_gradients:
                self.global_step += 1

                # OneCycle scheduler steps per batch
                if self.config.scheduler == "onecycle" and self.scheduler is not None:
                    self.scheduler.step()

            # Batch metrics for callback (sync here for callback compatibility)
            batch_metrics = {
                'loss': loss.item(),
                'recon_loss': losses['reconstruction'].item(),
                'kl_loss': losses['kl'].item(),
                'beta': current_beta,
            }
            if self.config.record_grad_norm and self.accelerator.sync_gradients:
                batch_metrics['grad_norm'] = grad_norm.item() if hasattr(grad_norm, 'item') else grad_norm

            # Callback: batch end
            self.callbacks.on_batch_end(self, batch_idx, batch_metrics)

            # Logging (only on main process, reuse batch_metrics to avoid extra sync)
            if batch_idx % self.config.log_every == 0:
                self.log_info(
                    f"Epoch [{self.current_epoch + 1}/{self.config.epochs}] "
                    f"Step [{batch_idx}/{len(self.train_loader)}] "
                    f"Loss: {batch_metrics['loss']:.4f} "
                    f"Recon: {batch_metrics['recon_loss']:.4f} "
                    f"KL: {batch_metrics['kl_loss']:.4f} "
                    f"Beta: {current_beta:.4f}"
                )

        # Reduce across all processes (single sync point for epoch metrics)
        total_loss = self.accelerator.reduce(total_loss, reduction="sum")
        total_recon_loss = self.accelerator.reduce(total_recon_loss, reduction="sum")
        total_kl_loss = self.accelerator.reduce(total_kl_loss, reduction="sum")

        # Global batch count = local batches * number of processes
        global_num_batches = num_batches * self.accelerator.num_processes

        # Compute average metrics (single .item() calls at epoch end)
        avg_metrics = {
            'loss': (total_loss / global_num_batches).item(),
            'recon_loss': (total_recon_loss / global_num_batches).item(),
            'kl_loss': (total_kl_loss / global_num_batches).item(),
            'beta': current_beta,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
        if self.config.record_grad_norm and num_grad_steps > 0:
            avg_metrics['grad_norm'] = total_grad_norm / num_grad_steps

        return avg_metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation.

        Returns:
            Dictionary of validation metrics, or empty dict if no val_loader
        """
        if self.val_loader is None:
            return {}

        self.callbacks.on_validation_start(self)
        self.model.eval()

        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        num_batches = 0

        for batch in self.val_loader:
            x, condition = self._unpack_batch(batch)

            outputs = self.model(x, condition=condition)
            losses = self.model.loss(x, outputs, beta=self.config.beta)

            # Gather across processes
            total_loss += self.accelerator.gather(losses['total']).mean().item()
            total_recon_loss += self.accelerator.gather(losses['reconstruction']).mean().item()
            total_kl_loss += self.accelerator.gather(losses['kl']).mean().item()
            num_batches += 1

        val_metrics = {
            'loss': total_loss / num_batches,
            'recon_loss': total_recon_loss / num_batches,
            'kl_loss': total_kl_loss / num_batches,
        }

        self.callbacks.on_validation_end(self, val_metrics)
        return val_metrics

    def save_checkpoint(self, filename: str, is_best: bool = False) -> str:
        """Save a training checkpoint.

        Args:
            filename: Name of the checkpoint file
            is_best: Whether this is the best model so far

        Returns:
            Path to the saved checkpoint
        """
        # Wait for all processes
        self.accelerator.wait_for_everyone()

        # Only save on main process
        if not self.accelerator.is_main_process:
            return ""

        # Unwrap model for saving
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Get wandb run ID if available
        wandb_run_id = None
        if self.metrics_logger and self.metrics_logger.wandb_run:
            wandb_run_id = self.metrics_logger.wandb_run.id

        path = self.checkpoint_manager.save(
            filename=filename,
            model=unwrapped_model,
            optimizer=self.optimizer,
            epoch=self.current_epoch + 1,  # Save next epoch to resume from
            global_step=self.global_step,
            config=self.config,
            best_val_loss=self.best_val_loss,
            scheduler_state=self.scheduler.state_dict() if self.scheduler else None,
            beta_scheduler_state=self.beta_scheduler.get_state(),
            early_stopping_state=self.early_stopping.get_state() if self.early_stopping else None,
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            is_best=is_best,
            extra_state={'wandb_run_id': wandb_run_id} if wandb_run_id else None,
        )

        self.callbacks.on_checkpoint_save(self, path)
        return path

    def load_checkpoint(self, path: str) -> None:
        """Load a training checkpoint.

        Args:
            path: Path to the checkpoint file
        """
        # Unwrap model for loading
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        checkpoint = self.checkpoint_manager.load(
            path=path,
            model=unwrapped_model,
            optimizer=self.optimizer,
            device=self.device,
        )

        # Restore training state
        state = restore_training_state(
            checkpoint=checkpoint,
            scheduler=self.scheduler if not self._custom_scheduler else None,
            beta_scheduler=self.beta_scheduler,
            early_stopping=self.early_stopping,
        )

        self.current_epoch = state['epoch']
        self.global_step = state['global_step']
        self.best_val_loss = state['best_val_loss']
        self.train_losses = state['train_losses']
        self.val_losses = state['val_losses']

        # Restore wandb run ID if available
        if 'wandb_run_id' in checkpoint:
            self._wandb_run_id = checkpoint['wandb_run_id']
            self.log_info(f"Found wandb run ID: {self._wandb_run_id}")

        self.callbacks.on_checkpoint_load(self, path)
        self.log_info(f"Resumed from epoch {self.current_epoch}")

    def _init_wandb_logger(self) -> None:
        """Initialize wandb logger with optional resume support.

        Should be called after load_checkpoint() if resuming, so that
        the wandb run ID can be restored from the checkpoint.
        """
        if not self.accelerator.is_main_process:
            return

        if self.config.wandb_mode == "disabled":
            return

        # If we already have a logger, don't reinitialize
        if self.metrics_logger is not None:
            return

        self.metrics_logger = MetricsLogger(
            self.config,
            output_dir=self.output_dir,
            resume_run_id=self._wandb_run_id,
        )

        # Store the run ID for future checkpoints
        if self.metrics_logger.wandb_run:
            self._wandb_run_id = self.metrics_logger.wandb_run.id

    def train(self, resume_from: Optional[str] = None) -> Tuple[List[float], List[float]]:
        """Run the complete training loop.

        Args:
            resume_from: Optional path to checkpoint to resume from

        Returns:
            Tuple of (train_losses, val_losses) per epoch
        """
        # Load checkpoint if resuming (this will restore wandb_run_id)
        if resume_from:
            self.load_checkpoint(resume_from)

        # Initialize wandb logger after checkpoint load (supports resume)
        self._init_wandb_logger()

        self.log_info(f"Starting training on {self.device}")
        self.log_info(f"Number of processes: {self.accelerator.num_processes}")
        self.log_info(f"Mixed precision: {self.config.mixed_precision}")
        self.log_info(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        self.log_info(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )
        self.log_info(f"Training from epoch {self.current_epoch} to {self.config.epochs}")

        self.callbacks.on_train_start(self)

        # Training loop
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            self.callbacks.on_epoch_start(self, epoch)

            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['loss'])

            if self.metrics_logger:
                self.metrics_logger.log_metrics(train_metrics, epoch, prefix="train")

            # Validate
            val_metrics = self.validate()
            is_best = False

            if val_metrics:
                self.val_losses.append(val_metrics['loss'])
                if self.metrics_logger:
                    self.metrics_logger.log_metrics(val_metrics, epoch, prefix="val")

                val_loss = val_metrics['loss']
                is_best = val_loss < self.best_val_loss

                if is_best:
                    self.best_val_loss = val_loss

                # Plateau scheduler uses validation loss
                if self.config.scheduler == "plateau" and self.scheduler:
                    self.scheduler.step(val_loss)

            # Other schedulers step per epoch
            if self.scheduler and self.config.scheduler not in ["plateau", "onecycle"]:
                self.scheduler.step()

            # Callback: epoch end
            self.callbacks.on_epoch_end(self, epoch, train_metrics, val_metrics)

            # Log epoch summary
            log_msg = (
                f"Epoch [{epoch + 1}/{self.config.epochs}] "
                f"Train Loss: {train_metrics['loss']:.4f}"
            )
            if val_metrics:
                log_msg += (
                    f" Val Loss: {val_metrics['loss']:.4f}"
                    f" Val Recon: {val_metrics['recon_loss']:.4f}"
                )
            self.log_info(log_msg)

            # Save checkpoints
            if self.config.save_best and is_best:
                self.save_checkpoint("best_model.pt", is_best=True)

            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

            # Early stopping (only check on main process, then broadcast)
            if self.early_stopping and val_metrics:
                should_stop = self.early_stopping(val_metrics['loss'])
                # Broadcast early stopping decision to all processes
                should_stop = self.accelerator.gather(
                    torch.tensor([should_stop], device=self.device)
                )[0].item()

                if should_stop:
                    self.log_info(f"Early stopping at epoch {epoch + 1}")
                    break

        # Save final model
        self.save_checkpoint("final_model.pt")

        # Cleanup
        self.callbacks.on_train_end(self)
        if self.metrics_logger:
            self.metrics_logger.close()

        self.log_info("Training completed!")
        return self.train_losses, self.val_losses
