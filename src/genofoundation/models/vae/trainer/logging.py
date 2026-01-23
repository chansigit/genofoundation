"""
Weights & Biases logging utilities for training.

Supports two modes:
- online: Logs directly to wandb servers (requires authentication)
- offline: Logs locally, can be synced later with `wandb sync`
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


class WandbLogger:
    """Weights & Biases logger with online/offline mode support.

    Online mode:
        - Validates wandb authentication at initialization
        - Logs metrics directly to wandb servers

    Offline mode:
        - Validates log directory exists at initialization
        - Stores logs locally for later upload via `wandb sync <dir>`

    Example:
        # Online mode
        logger = WandbLogger(config, output_dir="./outputs")

        # Offline mode - sync later with:
        # wandb sync ./outputs/wandb/offline-run-*
    """

    def __init__(
        self,
        config,
        output_dir: Optional[str] = None,
        resume_run_id: Optional[str] = None,
        external_run: Optional[Any] = None,
    ):
        """Initialize the wandb logger.

        Args:
            config: Training configuration containing wandb settings
            output_dir: Output directory for offline logs
            resume_run_id: Optional wandb run ID to resume (from checkpoint)
            external_run: Optional pre-initialized wandb run to use instead of creating new

        Raises:
            RuntimeError: If online mode and authentication fails
            ValueError: If offline mode and log directory doesn't exist
            ImportError: If wandb is not installed
        """
        self.config = config
        self.output_dir = output_dir or "./outputs"
        self.wandb_run = None
        self._enabled = config.wandb_mode != "disabled"
        self._resume_run_id = resume_run_id
        self._external_run = external_run

        if self._enabled:
            self._setup()

    def _setup(self) -> None:
        """Set up wandb based on mode."""
        # If external run is provided, use it directly (skip setup)
        if self._external_run is not None:
            self.wandb_run = self._external_run
            logger.info(
                f"Using external wandb run: {self.wandb_run.name} (id: {self.wandb_run.id})"
            )
            return

        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb is required for logging. Install with: pip install wandb"
            )

        mode = self.config.wandb_mode

        if mode == "online":
            self._validate_online_auth()
        elif mode == "offline":
            self._validate_offline_dir()

        self._init_wandb(mode)

    def _validate_online_auth(self) -> None:
        """Validate wandb authentication for online mode.

        Raises:
            RuntimeError: If authentication is invalid or not configured
        """
        import wandb

        # Check if logged in
        if wandb.api.api_key is None:
            raise RuntimeError(
                "wandb authentication required for online mode. "
                "Run 'wandb login' or set WANDB_API_KEY environment variable."
            )

        # Validate entity if specified
        if self.config.wandb_entity:
            try:
                api = wandb.Api()
                # Try to access entity to validate it exists and we have access
                api.viewer
                logger.info(f"wandb authenticated for entity: {self.config.wandb_entity}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to validate wandb entity '{self.config.wandb_entity}': {e}"
                )
        else:
            logger.info("wandb authenticated (using default entity)")

    def _validate_offline_dir(self) -> None:
        """Validate offline log directory exists.

        Raises:
            ValueError: If directory doesn't exist and can't be created
        """
        base_wandb_dir = self.config.wandb_dir or os.path.join(self.output_dir, "wandb")
        run_name = self.config.wandb_run_name or "default"
        run_dir = os.path.join(base_wandb_dir, run_name)

        try:
            Path(run_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(
                f"Failed to create wandb offline directory '{run_dir}': {e}"
            )

        # Print informative message about offline mode
        self._print_offline_mode_info(run_dir)

    def _print_offline_mode_info(self, run_dir: str) -> None:
        """Print informative message about offline mode and sync instructions."""
        border = "=" * 70
        logger.info(border)
        logger.info("WANDB OFFLINE MODE")
        logger.info(border)
        logger.info(f"Logs directory: {os.path.abspath(run_dir)}")
        logger.info("")
        logger.info("To upload logs to wandb after training, run:")
        logger.info(f"  wandb sync {os.path.abspath(run_dir)}")
        logger.info("")
        logger.info("Other useful commands:")
        logger.info(f"  wandb sync --sync-all {os.path.abspath(run_dir)}  # Sync all runs")
        logger.info(f"  wandb sync --clean {os.path.abspath(run_dir)}     # Sync and remove local")
        logger.info(border)

    def _init_wandb(self, mode: str) -> None:
        """Initialize wandb run.

        Args:
            mode: "online" or "offline"
        """
        import wandb

        # Determine base wandb directory
        base_wandb_dir = self.config.wandb_dir or os.path.join(self.output_dir, "wandb")

        # Create run-specific directory based on run name (not timestamp)
        # This makes it easier to find and organize runs
        run_name = self.config.wandb_run_name or "default"
        run_dir = os.path.join(base_wandb_dir, run_name)
        Path(run_dir).mkdir(parents=True, exist_ok=True)

        # Set environment variable for offline mode
        if mode == "offline":
            os.environ["WANDB_MODE"] = "offline"

        # Prepare init kwargs
        init_kwargs = {
            "project": self.config.wandb_project,
            "name": self.config.wandb_run_name,
            "config": self.config.to_dict(),
            "dir": run_dir,
        }

        # Handle resume: if we have a run ID from checkpoint, use it
        if self._resume_run_id:
            init_kwargs["id"] = self._resume_run_id
            init_kwargs["resume"] = "must"  # Must resume, fail if can't
            logger.info(f"Resuming wandb run: {self._resume_run_id}")
        else:
            init_kwargs["resume"] = "allow"  # Allow resume if run exists

        # Add optional parameters
        if self.config.wandb_entity:
            init_kwargs["entity"] = self.config.wandb_entity
        if self.config.wandb_tags:
            init_kwargs["tags"] = list(self.config.wandb_tags)
        if self.config.wandb_notes:
            init_kwargs["notes"] = self.config.wandb_notes

        try:
            self.wandb_run = wandb.init(**init_kwargs)
            self._run_dir = run_dir  # Store for later reference
            resumed = " (resumed)" if self._resume_run_id else ""
            logger.info(
                f"wandb initialized in {mode} mode{resumed} - "
                f"project: {self.config.wandb_project}, "
                f"run: {self.wandb_run.name} (id: {self.wandb_run.id})"
            )
            logger.info(f"wandb logs directory: {run_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize wandb: {e}")

    @property
    def enabled(self) -> bool:
        """Check if logging is enabled."""
        return self._enabled and self.wandb_run is not None

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "train",
    ) -> None:
        """Log metrics to wandb.

        Args:
            metrics: Dictionary of metric names to values
            step: Current training step or epoch
            prefix: Prefix for metric names (e.g., 'train', 'val')
        """
        if not self.enabled:
            return

        try:
            import wandb
            wandb.log(
                {f"{prefix}/{k}": v for k, v in metrics.items()},
                step=step,
            )
        except Exception as e:
            logger.warning(f"Failed to log metrics to wandb: {e}")

    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """Log/update hyperparameters.

        Args:
            hparams: Dictionary of hyperparameter names to values
        """
        if not self.enabled:
            return

        try:
            import wandb
            wandb.config.update(hparams, allow_val_change=True)
        except Exception as e:
            logger.warning(f"Failed to log hyperparameters to wandb: {e}")

    def log_histogram(
        self,
        tag: str,
        values,
        step: int,
    ) -> None:
        """Log a histogram of values.

        Args:
            tag: Name for the histogram
            values: Values to create histogram from (numpy array or tensor)
            step: Current training step
        """
        if not self.enabled:
            return

        try:
            import wandb
            import numpy as np

            if hasattr(values, 'cpu'):
                values = values.cpu().detach().numpy()
            elif not isinstance(values, np.ndarray):
                values = np.array(values)

            wandb.log({tag: wandb.Histogram(values)}, step=step)
        except Exception as e:
            logger.warning(f"Failed to log histogram to wandb: {e}")

    def log_image(
        self,
        tag: str,
        image,
        step: int,
        caption: Optional[str] = None,
    ) -> None:
        """Log an image.

        Args:
            tag: Name for the image
            image: Image tensor (C, H, W), (H, W), or numpy array
            step: Current training step
            caption: Optional caption for the image
        """
        if not self.enabled:
            return

        try:
            import wandb

            if hasattr(image, 'cpu'):
                image = image.cpu().detach().numpy()

            wandb.log(
                {tag: wandb.Image(image, caption=caption)},
                step=step,
            )
        except Exception as e:
            logger.warning(f"Failed to log image to wandb: {e}")

    def log_table(
        self,
        tag: str,
        columns: list,
        data: list,
        step: Optional[int] = None,
    ) -> None:
        """Log a table.

        Args:
            tag: Name for the table
            columns: List of column names
            data: List of rows (each row is a list of values)
            step: Optional training step
        """
        if not self.enabled:
            return

        try:
            import wandb

            table = wandb.Table(columns=columns, data=data)
            log_dict = {tag: table}
            if step is not None:
                wandb.log(log_dict, step=step)
            else:
                wandb.log(log_dict)
        except Exception as e:
            logger.warning(f"Failed to log table to wandb: {e}")

    def watch_model(self, model, log: str = "gradients", log_freq: int = 100) -> None:
        """Watch model gradients and parameters.

        Args:
            model: PyTorch model to watch
            log: What to log - "gradients", "parameters", "all", or None
            log_freq: Logging frequency (steps)
        """
        if not self.enabled:
            return

        try:
            import wandb
            wandb.watch(model, log=log, log_freq=log_freq)
            logger.info(f"wandb watching model (log={log}, freq={log_freq})")
        except Exception as e:
            logger.warning(f"Failed to watch model with wandb: {e}")

    @property
    def run_dir(self) -> Optional[str]:
        """Get the directory where wandb logs are stored."""
        return getattr(self, '_run_dir', None)

    def finish(self) -> None:
        """Finish the wandb run and clean up."""
        if self.wandb_run is not None:
            try:
                import wandb

                # Log final summary if offline
                if self.config.wandb_mode == "offline" and self.run_dir:
                    self._print_offline_finish_info()

                wandb.finish()
                self.wandb_run = None
                logger.info("wandb run finished")
            except Exception as e:
                logger.warning(f"Error finishing wandb run: {e}")

    def _print_offline_finish_info(self) -> None:
        """Print instructions for syncing offline logs after training."""
        border = "=" * 70
        logger.info("")
        logger.info(border)
        logger.info("TRAINING COMPLETE - WANDB OFFLINE LOGS READY TO SYNC")
        logger.info(border)
        logger.info(f"Logs saved to: {os.path.abspath(self.run_dir)}")
        logger.info("")
        logger.info("To upload to wandb, run one of the following:")
        logger.info("")
        logger.info("  1. Sync this run:")
        logger.info(f"     wandb sync {os.path.abspath(self.run_dir)}")
        logger.info("")
        logger.info("  2. Sync and delete local logs:")
        logger.info(f"     wandb sync --clean {os.path.abspath(self.run_dir)}")
        logger.info("")
        logger.info("  3. Preview without uploading:")
        logger.info(f"     wandb sync --dry-run {os.path.abspath(self.run_dir)}")
        logger.info(border)

    def close(self) -> None:
        """Alias for finish() for backwards compatibility."""
        self.finish()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.finish()
        return False


# Backwards compatibility alias
MetricsLogger = WandbLogger
