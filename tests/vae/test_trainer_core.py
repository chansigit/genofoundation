"""
Integration tests for VAETrainer core functionality.
"""

import os
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from trainer import VAETrainer, TrainerConfig


class TestVAETrainerInitialization:
    """Test cases for VAETrainer initialization."""

    def test_basic_initialization(self, small_vae, small_train_loader, fast_config, temp_dir):
        """Test basic trainer initialization."""
        trainer = VAETrainer(
            model=small_vae,
            config=fast_config,
            train_loader=small_train_loader,
            output_dir=temp_dir,
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.train_loader is not None
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0

    def test_initialization_with_val_loader(self, small_vae, small_train_loader, small_val_loader, fast_config, temp_dir):
        """Test initialization with validation loader."""
        trainer = VAETrainer(
            model=small_vae,
            config=fast_config,
            train_loader=small_train_loader,
            val_loader=small_val_loader,
            output_dir=temp_dir,
        )

        assert trainer.val_loader is not None

    def test_initialization_with_dataset(self, small_vae, small_train_dataset, fast_config, temp_dir):
        """Test initialization with dataset instead of loader."""
        trainer = VAETrainer(
            model=small_vae,
            config=fast_config,
            train_dataset=small_train_dataset,
            output_dir=temp_dir,
        )

        assert trainer.train_loader is not None

    def test_initialization_with_dict_config(self, small_vae, small_train_loader, temp_dir):
        """Test initialization with dict config."""
        config_dict = {
            'epochs': 2,
            'batch_size': 16,
            'learning_rate': 1e-3,
            'mixed_precision': 'no',
            'wandb_mode': 'disabled',
        }

        trainer = VAETrainer(
            model=small_vae,
            config=config_dict,
            train_loader=small_train_loader,
            output_dir=temp_dir,
        )

        assert trainer.config.epochs == 2

    def test_initialization_requires_train_data(self, small_vae, fast_config, temp_dir):
        """Test that initialization requires training data."""
        with pytest.raises(ValueError, match="Either train_loader or train_dataset must be provided"):
            VAETrainer(
                model=small_vae,
                config=fast_config,
                output_dir=temp_dir,
            )

    def test_initialization_with_seed(self, small_vae, small_train_loader, fast_config, temp_dir):
        """Test initialization with seed."""
        trainer = VAETrainer(
            model=small_vae,
            config=fast_config,
            train_loader=small_train_loader,
            output_dir=temp_dir,
            seed=42,
        )

        assert trainer is not None

    def test_initialization_with_custom_optimizer(self, small_vae, small_train_loader, fast_config, temp_dir):
        """Test initialization with custom optimizer."""
        custom_optimizer = torch.optim.SGD(small_vae.parameters(), lr=0.01)

        trainer = VAETrainer(
            model=small_vae,
            config=fast_config,
            train_loader=small_train_loader,
            optimizer=custom_optimizer,
            output_dir=temp_dir,
        )

        # The optimizer should be prepared by accelerator, so it's the same optimizer wrapped
        assert trainer._custom_optimizer is True


class TestVAETrainerOptimizers:
    """Test cases for optimizer creation."""

    def test_adam_optimizer(self, small_vae, small_train_loader, temp_dir):
        """Test Adam optimizer creation."""
        config = TrainerConfig(
            optimizer="adam",
            epochs=1,
            mixed_precision="no",
            wandb_mode="disabled",
        )

        trainer = VAETrainer(
            model=small_vae,
            config=config,
            train_loader=small_train_loader,
            output_dir=temp_dir,
        )

        # Note: optimizer is wrapped by accelerator, check the underlying type
        assert trainer.optimizer is not None

    def test_adamw_optimizer(self, small_vae, small_train_loader, temp_dir):
        """Test AdamW optimizer creation."""
        config = TrainerConfig(
            optimizer="adamw",
            epochs=1,
            mixed_precision="no",
            wandb_mode="disabled",
        )

        trainer = VAETrainer(
            model=small_vae,
            config=config,
            train_loader=small_train_loader,
            output_dir=temp_dir,
        )

        assert trainer.optimizer is not None

    def test_sgd_optimizer(self, small_vae, small_train_loader, temp_dir):
        """Test SGD optimizer creation."""
        config = TrainerConfig(
            optimizer="sgd",
            epochs=1,
            mixed_precision="no",
            wandb_mode="disabled",
        )

        trainer = VAETrainer(
            model=small_vae,
            config=config,
            train_loader=small_train_loader,
            output_dir=temp_dir,
        )

        assert trainer.optimizer is not None


class TestVAETrainerBatchUnpacking:
    """Test cases for batch unpacking."""

    def test_unpack_tensor_batch(self, small_vae, small_train_loader, fast_config, temp_dir):
        """Test unpacking a tensor batch."""
        trainer = VAETrainer(
            model=small_vae,
            config=fast_config,
            train_loader=small_train_loader,
            output_dir=temp_dir,
        )

        batch = torch.randn(16, 128)
        x, condition = trainer._unpack_batch(batch)

        assert x.shape == (16, 128)
        assert condition is None

    def test_unpack_tuple_batch(self, small_vae, small_train_loader, fast_config, temp_dir):
        """Test unpacking a tuple batch."""
        trainer = VAETrainer(
            model=small_vae,
            config=fast_config,
            train_loader=small_train_loader,
            output_dir=temp_dir,
        )

        data = torch.randn(16, 128)
        labels = torch.randint(0, 10, (16,))
        batch = (data, labels)

        x, condition = trainer._unpack_batch(batch)

        assert x.shape == (16, 128)
        assert condition.shape == (16,)

    def test_unpack_list_batch(self, small_vae, small_train_loader, fast_config, temp_dir):
        """Test unpacking a list batch."""
        trainer = VAETrainer(
            model=small_vae,
            config=fast_config,
            train_loader=small_train_loader,
            output_dir=temp_dir,
        )

        data = torch.randn(16, 128)
        labels = torch.randint(0, 10, (16,))
        batch = [data, labels]

        x, condition = trainer._unpack_batch(batch)

        assert x.shape == (16, 128)
        assert condition.shape == (16,)


class TestVAETrainerTraining:
    """Test cases for training functionality."""

    def test_train_epoch(self, small_vae, small_train_loader, fast_config, temp_dir):
        """Test training for one epoch."""
        trainer = VAETrainer(
            model=small_vae,
            config=fast_config,
            train_loader=small_train_loader,
            output_dir=temp_dir,
        )

        metrics = trainer.train_epoch()

        assert 'loss' in metrics
        assert 'recon_loss' in metrics
        assert 'kl_loss' in metrics
        assert 'beta' in metrics
        assert 'lr' in metrics
        assert metrics['loss'] > 0

    def test_train_epoch_with_grad_norm_recording(self, small_vae, small_train_loader, temp_dir):
        """Test gradient norm is recorded when enabled (default)."""
        config = TrainerConfig(
            epochs=1,
            batch_size=16,
            mixed_precision="no",
            wandb_mode="disabled",
            record_grad_norm=True,
        )

        trainer = VAETrainer(
            model=small_vae,
            config=config,
            train_loader=small_train_loader,
            output_dir=temp_dir,
        )

        metrics = trainer.train_epoch()

        assert 'grad_norm' in metrics
        assert metrics['grad_norm'] > 0

    def test_train_epoch_without_grad_norm_recording(self, small_vae, small_train_loader, temp_dir):
        """Test gradient norm is not recorded when disabled."""
        config = TrainerConfig(
            epochs=1,
            batch_size=16,
            mixed_precision="no",
            wandb_mode="disabled",
            record_grad_norm=False,
        )

        trainer = VAETrainer(
            model=small_vae,
            config=config,
            train_loader=small_train_loader,
            output_dir=temp_dir,
        )

        metrics = trainer.train_epoch()

        assert 'grad_norm' not in metrics

    def test_validate(self, small_vae, small_train_loader, small_val_loader, fast_config, temp_dir):
        """Test validation."""
        trainer = VAETrainer(
            model=small_vae,
            config=fast_config,
            train_loader=small_train_loader,
            val_loader=small_val_loader,
            output_dir=temp_dir,
        )

        metrics = trainer.validate()

        assert 'loss' in metrics
        assert 'recon_loss' in metrics
        assert 'kl_loss' in metrics
        assert metrics['loss'] > 0

    def test_validate_without_val_loader(self, small_vae, small_train_loader, fast_config, temp_dir):
        """Test validation without validation loader."""
        trainer = VAETrainer(
            model=small_vae,
            config=fast_config,
            train_loader=small_train_loader,
            output_dir=temp_dir,
        )

        metrics = trainer.validate()

        assert metrics == {}

    def test_full_training_loop(self, small_vae, small_train_loader, small_val_loader, fast_config, temp_dir):
        """Test full training loop."""
        trainer = VAETrainer(
            model=small_vae,
            config=fast_config,
            train_loader=small_train_loader,
            val_loader=small_val_loader,
            output_dir=temp_dir,
        )

        train_losses, val_losses = trainer.train()

        assert len(train_losses) == 2  # 2 epochs
        assert len(val_losses) == 2
        assert all(loss > 0 for loss in train_losses)
        assert all(loss > 0 for loss in val_losses)


class TestVAETrainerCheckpointing:
    """Test cases for checkpointing functionality."""

    def test_save_checkpoint(self, small_vae, small_train_loader, fast_config, temp_dir):
        """Test saving checkpoint."""
        trainer = VAETrainer(
            model=small_vae,
            config=fast_config,
            train_loader=small_train_loader,
            output_dir=temp_dir,
        )

        trainer.current_epoch = 5
        trainer.global_step = 100
        trainer.best_val_loss = 0.5

        path = trainer.save_checkpoint("test_checkpoint.pt")

        # Only main process saves
        if trainer.accelerator.is_main_process:
            assert os.path.exists(path)

    def test_load_checkpoint(self, small_vae, small_train_loader, fast_config, temp_dir):
        """Test loading checkpoint."""
        trainer = VAETrainer(
            model=small_vae,
            config=fast_config,
            train_loader=small_train_loader,
            output_dir=temp_dir,
        )

        trainer.current_epoch = 5
        trainer.global_step = 100
        trainer.best_val_loss = 0.5
        trainer.train_losses = [1.0, 0.9]
        trainer.val_losses = [1.1, 0.95]

        path = trainer.save_checkpoint("resume_test.pt")

        if trainer.accelerator.is_main_process:
            # Create new trainer and load
            new_trainer = VAETrainer(
                model=small_vae,
                config=fast_config,
                train_loader=small_train_loader,
                output_dir=temp_dir,
            )

            new_trainer.load_checkpoint(path)

            assert new_trainer.current_epoch == 5
            assert new_trainer.global_step == 100
            assert new_trainer.best_val_loss == 0.5

    def test_resume_training(self, small_train_loader, fast_config, temp_dir):
        """Test resuming training from checkpoint."""
        import torch
        import torch.nn as nn

        # Simple VAE for testing resume - same architecture used for both save and load
        class ResumeTestVAE(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32))
                self.decoder = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 128))

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

        # First training run
        model1 = ResumeTestVAE()
        trainer1 = VAETrainer(
            model=model1,
            config=fast_config,
            train_loader=small_train_loader,
            output_dir=temp_dir,
        )

        trainer1.train_epoch()
        trainer1.current_epoch = 1
        path = trainer1.save_checkpoint("resume_checkpoint.pt")

        if trainer1.accelerator.is_main_process:
            # Resume training with a new model instance (same architecture)
            model2 = ResumeTestVAE()

            trainer2 = VAETrainer(
                model=model2,
                config=fast_config,
                train_loader=small_train_loader,
                output_dir=temp_dir,
            )

            train_losses, val_losses = trainer2.train(resume_from=path)

            # Should have trained for remaining epochs
            assert len(train_losses) >= 1


class TestVAETrainerCallbacks:
    """Test cases for callback integration."""

    def test_callbacks_are_called(self, small_vae, small_train_loader, small_val_loader, fast_config, temp_dir, mock_callback):
        """Test that callbacks are called during training."""
        trainer = VAETrainer(
            model=small_vae,
            config=fast_config,
            train_loader=small_train_loader,
            val_loader=small_val_loader,
            callbacks=[mock_callback],
            output_dir=temp_dir,
        )

        trainer.train()

        assert mock_callback.train_start_called
        assert mock_callback.train_end_called
        assert len(mock_callback.epoch_starts) == 2  # 2 epochs
        assert len(mock_callback.epoch_ends) == 2
        assert len(mock_callback.batch_starts) > 0
        assert len(mock_callback.batch_ends) > 0
        assert mock_callback.validation_starts >= 2
        assert len(mock_callback.validation_ends) >= 2


class TestVAETrainerDevice:
    """Test cases for device handling."""

    def test_device_property(self, small_vae, small_train_loader, fast_config, temp_dir):
        """Test device property."""
        trainer = VAETrainer(
            model=small_vae,
            config=fast_config,
            train_loader=small_train_loader,
            output_dir=temp_dir,
        )

        device = trainer.device

        assert device is not None
        assert isinstance(device, torch.device)


class TestVAETrainerSchedulers:
    """Test cases for scheduler integration."""

    def test_beta_scheduler_updates(self, small_vae, small_train_loader, temp_dir):
        """Test beta scheduler updates during training."""
        config = TrainerConfig(
            epochs=3,
            beta=1.0,
            beta_schedule="linear",
            beta_warmup_epochs=3,
            mixed_precision="no",
            wandb_mode="disabled",
            scheduler=None,
            early_stopping=False,
        )

        trainer = VAETrainer(
            model=small_vae,
            config=config,
            train_loader=small_train_loader,
            output_dir=temp_dir,
        )

        # Record beta values during training
        beta_values = []
        original_train_epoch = trainer.train_epoch

        def wrapped_train_epoch():
            result = original_train_epoch()
            beta_values.append(result['beta'])
            return result

        trainer.train_epoch = wrapped_train_epoch
        trainer.train()

        # Beta should increase during warmup
        assert len(beta_values) == 3
        assert beta_values[-1] >= beta_values[0]

    def test_lr_scheduler_steps(self, small_vae, small_train_loader, temp_dir):
        """Test LR scheduler steps during training."""
        config = TrainerConfig(
            epochs=3,
            scheduler="cosine",
            mixed_precision="no",
            wandb_mode="disabled",
            early_stopping=False,
        )

        trainer = VAETrainer(
            model=small_vae,
            config=config,
            train_loader=small_train_loader,
            output_dir=temp_dir,
        )

        initial_lr = trainer.optimizer.param_groups[0]['lr']

        trainer.train()

        final_lr = trainer.optimizer.param_groups[0]['lr']

        # LR should have changed with cosine scheduler
        assert final_lr != initial_lr
