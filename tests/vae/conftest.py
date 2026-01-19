"""
Pytest fixtures for VAE trainer tests.
"""

import os
import sys
import tempfile
import shutil
from typing import Dict, Any

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Add the vae module to path for imports
vae_path = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "src", "genofoundation", "models", "vae"
)
sys.path.insert(0, os.path.abspath(vae_path))

from trainer import (
    TrainerConfig,
    BetaScheduler,
    EarlyStopping,
    CheckpointManager,
    Callback,
)


class SimpleVAE(nn.Module):
    """Simple VAE model for testing."""

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
        self.latent_dim = latent_dim

    def forward(self, x, condition=None):
        h = self.encoder(x)
        mu, log_var = h.chunk(2, dim=-1)
        std = torch.exp(0.5 * log_var)
        z = mu + std * torch.randn_like(std)
        recon = self.decoder(z)
        return {
            'reconstruction': recon,
            'mu': mu,
            'log_var': log_var,
            'z': z,
        }

    def loss(self, x, outputs, beta=1.0):
        recon_loss = nn.functional.mse_loss(outputs['reconstruction'], x)
        kl_loss = -0.5 * torch.mean(
            1 + outputs['log_var'] - outputs['mu'].pow(2) - outputs['log_var'].exp()
        )
        return {
            'total': recon_loss + beta * kl_loss,
            'reconstruction': recon_loss,
            'kl': kl_loss,
        }


class SimpleDataset(Dataset):
    """Simple dataset for testing."""

    def __init__(self, num_samples: int = 200, input_dim: int = 784, seed: int = 42):
        torch.manual_seed(seed)
        self.data = torch.randn(num_samples, input_dim)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class ConditionalDataset(Dataset):
    """Dataset that returns (data, condition) tuples for testing conditional VAE."""

    def __init__(self, num_samples: int = 200, input_dim: int = 784, num_classes: int = 10, seed: int = 42):
        torch.manual_seed(seed)
        self.data = torch.randn(num_samples, input_dim)
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test artifacts."""
    tmpdir = tempfile.mkdtemp(prefix="vae_test_")
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def simple_vae():
    """Create a simple VAE model for testing."""
    return SimpleVAE(input_dim=784, latent_dim=64)


@pytest.fixture
def small_vae():
    """Create a smaller VAE model for faster tests."""
    return SimpleVAE(input_dim=128, latent_dim=16)


@pytest.fixture
def train_dataset():
    """Create a training dataset."""
    return SimpleDataset(num_samples=200, input_dim=784, seed=42)


@pytest.fixture
def val_dataset():
    """Create a validation dataset."""
    return SimpleDataset(num_samples=50, input_dim=784, seed=123)


@pytest.fixture
def small_train_dataset():
    """Create a smaller training dataset for faster tests."""
    return SimpleDataset(num_samples=64, input_dim=128, seed=42)


@pytest.fixture
def small_val_dataset():
    """Create a smaller validation dataset for faster tests."""
    return SimpleDataset(num_samples=16, input_dim=128, seed=123)


@pytest.fixture
def train_loader(train_dataset):
    """Create a training data loader."""
    return DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)


@pytest.fixture
def val_loader(val_dataset):
    """Create a validation data loader."""
    return DataLoader(val_dataset, batch_size=32, shuffle=False)


@pytest.fixture
def small_train_loader(small_train_dataset):
    """Create a smaller training data loader for faster tests."""
    return DataLoader(small_train_dataset, batch_size=16, shuffle=True, drop_last=True)


@pytest.fixture
def small_val_loader(small_val_dataset):
    """Create a smaller validation data loader for faster tests."""
    return DataLoader(small_val_dataset, batch_size=16, shuffle=False)


@pytest.fixture
def conditional_train_loader():
    """Create a conditional training data loader."""
    dataset = ConditionalDataset(num_samples=100, input_dim=784)
    return DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)


@pytest.fixture
def basic_config(temp_dir):
    """Create a basic trainer configuration."""
    return TrainerConfig(
        epochs=10,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-5,
        beta=1.0,
        beta_warmup_epochs=3,
        beta_schedule="linear",
        optimizer="adamw",
        scheduler="cosine",
        lr_warmup_epochs=2,
        min_lr=1e-6,
        grad_clip_norm=1.0,
        gradient_accumulation_steps=1,
        early_stopping=True,
        patience=5,
        min_delta=1e-4,
        save_every=5,
        save_best=True,
        max_checkpoints=3,
        log_every=10,
        wandb_mode="disabled",
        mixed_precision="no",
    )


@pytest.fixture
def fast_config(temp_dir):
    """Create a fast configuration for quick integration tests."""
    return TrainerConfig(
        epochs=2,
        batch_size=16,
        learning_rate=1e-3,
        beta=1.0,
        beta_warmup_epochs=1,
        beta_schedule="constant",
        optimizer="adam",
        scheduler=None,
        early_stopping=False,
        save_every=1,
        save_best=True,
        log_every=1,
        wandb_mode="disabled",
        mixed_precision="no",
    )


@pytest.fixture
def beta_scheduler():
    """Create a beta scheduler with linear warmup."""
    return BetaScheduler(
        schedule="linear",
        initial_beta=0.0,
        final_beta=1.0,
        warmup_epochs=10,
    )


@pytest.fixture
def early_stopping():
    """Create an early stopping handler."""
    return EarlyStopping(patience=5, min_delta=1e-4, mode="min")


@pytest.fixture
def checkpoint_manager(temp_dir):
    """Create a checkpoint manager."""
    return CheckpointManager(checkpoint_dir=temp_dir, max_checkpoints=3)


class MockCallback(Callback):
    """Mock callback for testing callback system."""

    def __init__(self):
        self.train_start_called = False
        self.train_end_called = False
        self.epoch_starts = []
        self.epoch_ends = []
        self.batch_starts = []
        self.batch_ends = []
        self.validation_starts = 0
        self.validation_ends = []
        self.checkpoint_saves = []
        self.checkpoint_loads = []

    def on_train_start(self, trainer):
        self.train_start_called = True

    def on_train_end(self, trainer):
        self.train_end_called = True

    def on_epoch_start(self, trainer, epoch):
        self.epoch_starts.append(epoch)

    def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics=None):
        self.epoch_ends.append({
            'epoch': epoch,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
        })

    def on_batch_start(self, trainer, batch_idx, batch):
        self.batch_starts.append(batch_idx)

    def on_batch_end(self, trainer, batch_idx, batch_metrics):
        self.batch_ends.append({
            'batch_idx': batch_idx,
            'metrics': batch_metrics,
        })

    def on_validation_start(self, trainer):
        self.validation_starts += 1

    def on_validation_end(self, trainer, val_metrics):
        self.validation_ends.append(val_metrics)

    def on_checkpoint_save(self, trainer, checkpoint_path):
        self.checkpoint_saves.append(checkpoint_path)

    def on_checkpoint_load(self, trainer, checkpoint_path):
        self.checkpoint_loads.append(checkpoint_path)


@pytest.fixture
def mock_callback():
    """Create a mock callback for testing."""
    return MockCallback()
