#!/usr/bin/env python
"""
VAE Training CLI with Hydra configuration.

This script demonstrates how to use the trainer with Hydra.
You should create your own training script that:
1. Creates your model
2. Creates your dataloaders
3. Passes them to VAETrainer

Usage:
    # Basic training
    python train.py

    # Override trainer parameters
    python train.py trainer.learning_rate=1e-4 trainer.epochs=200

    # Use fast config for debugging
    python train.py trainer=fast

    # Multi-GPU training with Accelerate
    accelerate launch train.py trainer.batch_size=256

    # Resume training
    python train.py resume_from=outputs/checkpoint_epoch_50.pt

Example of creating your own training script:

    from train import run_training
    from my_models import MyVAE
    from my_data import create_my_dataloaders

    model = MyVAE(input_dim=1000, latent_dim=128)
    train_loader, val_loader = create_my_dataloaders()

    run_training(model, train_loader, val_loader)
"""

import logging
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from trainer import VAETrainer
from trainer.config import Config, TrainerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    config: Optional[TrainerConfig] = None,
    output_dir: str = "./outputs",
    seed: int = 42,
    resume_from: Optional[str] = None,
):
    """Run training with the given model and dataloaders.

    This is the main API for training. Users should:
    1. Create their own model
    2. Create their own dataloaders
    3. Call this function

    Args:
        model: Your VAE model (must have forward() and loss() methods)
        train_loader: Training dataloader
        val_loader: Optional validation dataloader
        config: TrainerConfig (uses defaults if None)
        output_dir: Directory for checkpoints and logs
        seed: Random seed
        resume_from: Path to checkpoint for resuming

    Returns:
        Tuple of (train_losses, val_losses)

    Example:
        model = MyVAE(input_dim=784, latent_dim=64)
        train_loader = DataLoader(train_dataset, batch_size=128)
        val_loader = DataLoader(val_dataset, batch_size=128)

        train_losses, val_losses = run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=TrainerConfig(epochs=100, learning_rate=1e-3),
        )
    """
    if config is None:
        config = TrainerConfig()

    trainer = VAETrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
        seed=seed,
    )

    return trainer.train(resume_from=resume_from)


# ============================================================================
# Demo with dummy model (for testing the pipeline only)
# ============================================================================

class _DemoVAE(nn.Module):
    """Demo VAE for testing. Replace with your actual model."""

    def __init__(self, input_dim: int = 784, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x, condition=None):
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)
        return {'reconstruction': recon, 'mu': mu, 'log_var': log_var, 'z': z}

    def loss(self, x, outputs, beta=1.0):
        recon_loss = nn.functional.mse_loss(outputs['reconstruction'], x)
        kl_loss = -0.5 * torch.mean(1 + outputs['log_var'] - outputs['mu'].pow(2) - outputs['log_var'].exp())
        return {'total': recon_loss + beta * kl_loss, 'reconstruction': recon_loss, 'kl': kl_loss}


def _create_demo_dataloaders(batch_size: int = 128):
    """Create demo dataloaders for testing."""
    from torch.utils.data import TensorDataset, random_split

    # Dummy data
    data = torch.randn(10000, 784)
    dataset = TensorDataset(data)

    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Demo training with Hydra configuration.

    This is a demo showing how to use Hydra with the trainer.
    In your actual code, replace _DemoVAE and _create_demo_dataloaders
    with your own model and data loading logic.
    """
    logger.info("=" * 60)
    logger.info("DEMO MODE: Using dummy model and data")
    logger.info("Replace _DemoVAE and _create_demo_dataloaders with your own!")
    logger.info("=" * 60)

    # Print config
    logger.info("Configuration:")
    logger.info(OmegaConf.to_yaml(cfg))

    # Convert Hydra config to dataclass
    config = Config.from_hydra(cfg)

    # Get output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger.info(f"Output directory: {output_dir}")

    # =========================================
    # YOUR CODE HERE: Create your own model
    # =========================================
    # model = YourVAE(...)
    model = _DemoVAE(input_dim=784, latent_dim=64)

    # =========================================
    # YOUR CODE HERE: Create your own dataloaders
    # =========================================
    # train_loader, val_loader = create_your_dataloaders(...)
    train_loader, val_loader = _create_demo_dataloaders(config.trainer.batch_size)

    # Run training
    train_losses, val_losses = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config.trainer,
        output_dir=output_dir,
        seed=config.experiment.seed,
        resume_from=cfg.get('resume_from', None),
    )

    # Log results
    logger.info(f"Training completed!")
    logger.info(f"Final train loss: {train_losses[-1]:.4f}")
    if val_losses:
        logger.info(f"Final val loss: {val_losses[-1]:.4f}")


if __name__ == "__main__":
    main()
