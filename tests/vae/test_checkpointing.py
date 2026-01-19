"""
Unit tests for checkpoint management.
"""

import os
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from trainer import CheckpointManager, TrainerConfig
from trainer.checkpointing import restore_training_state


class TestCheckpointManager:
    """Test cases for CheckpointManager."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return nn.Linear(10, 10)

    @pytest.fixture
    def optimizer(self, simple_model):
        """Create an optimizer for testing."""
        return optim.AdamW(simple_model.parameters(), lr=1e-3)

    @pytest.fixture
    def config(self):
        """Create a config for testing."""
        return TrainerConfig(epochs=10, batch_size=32)

    def test_initialization(self, temp_dir):
        """Test CheckpointManager initialization."""
        manager = CheckpointManager(temp_dir, max_checkpoints=5)

        assert manager.checkpoint_dir.exists()
        assert manager.max_checkpoints == 5

    def test_initialization_creates_directory(self, temp_dir):
        """Test that initialization creates the checkpoint directory."""
        new_dir = os.path.join(temp_dir, "checkpoints", "new")
        manager = CheckpointManager(new_dir)

        assert os.path.exists(new_dir)

    def test_save_checkpoint(self, temp_dir, simple_model, optimizer, config):
        """Test saving a checkpoint."""
        manager = CheckpointManager(temp_dir)

        path = manager.save(
            filename="test_checkpoint.pt",
            model=simple_model,
            optimizer=optimizer,
            epoch=5,
            global_step=100,
            config=config,
            best_val_loss=0.5,
        )

        assert os.path.exists(path)

        checkpoint = torch.load(path)
        assert checkpoint['epoch'] == 5
        assert checkpoint['global_step'] == 100
        assert checkpoint['best_val_loss'] == 0.5
        assert 'model_state_dict' in checkpoint
        assert 'optimizer_state_dict' in checkpoint
        assert 'config' in checkpoint

    def test_save_checkpoint_with_scheduler_state(self, temp_dir, simple_model, optimizer, config):
        """Test saving checkpoint with scheduler state."""
        manager = CheckpointManager(temp_dir)

        scheduler_state = {'step': 10, 'last_epoch': 5}

        path = manager.save(
            filename="scheduler_checkpoint.pt",
            model=simple_model,
            optimizer=optimizer,
            epoch=5,
            global_step=100,
            config=config,
            best_val_loss=0.5,
            scheduler_state=scheduler_state,
        )

        checkpoint = torch.load(path)
        assert checkpoint['scheduler_state_dict'] == scheduler_state

    def test_save_checkpoint_with_extra_state(self, temp_dir, simple_model, optimizer, config):
        """Test saving checkpoint with extra state."""
        manager = CheckpointManager(temp_dir)

        extra = {'custom_metric': 0.95}

        path = manager.save(
            filename="extra_checkpoint.pt",
            model=simple_model,
            optimizer=optimizer,
            epoch=5,
            global_step=100,
            config=config,
            best_val_loss=0.5,
            extra_state=extra,
        )

        checkpoint = torch.load(path)
        assert checkpoint['custom_metric'] == 0.95

    def test_save_best_checkpoint(self, temp_dir, simple_model, optimizer, config):
        """Test saving best checkpoint."""
        manager = CheckpointManager(temp_dir)

        manager.save(
            filename="regular.pt",
            model=simple_model,
            optimizer=optimizer,
            epoch=5,
            global_step=100,
            config=config,
            best_val_loss=0.5,
            is_best=True,
        )

        best_path = os.path.join(temp_dir, "best_model.pt")
        assert os.path.exists(best_path)

    def test_load_checkpoint(self, temp_dir, simple_model, optimizer, config):
        """Test loading a checkpoint."""
        manager = CheckpointManager(temp_dir)

        # Save checkpoint
        path = manager.save(
            filename="load_test.pt",
            model=simple_model,
            optimizer=optimizer,
            epoch=5,
            global_step=100,
            config=config,
            best_val_loss=0.5,
        )

        # Modify model
        for param in simple_model.parameters():
            param.data.fill_(999.0)

        # Load checkpoint
        checkpoint = manager.load(path, simple_model, optimizer)

        assert checkpoint['epoch'] == 5
        assert checkpoint['global_step'] == 100

        # Model should be restored
        for param in simple_model.parameters():
            assert not torch.all(param.data == 999.0)

    def test_checkpoint_rotation(self, temp_dir, simple_model, optimizer, config):
        """Test checkpoint rotation keeps only max_checkpoints."""
        manager = CheckpointManager(temp_dir, max_checkpoints=3)

        # Save 5 checkpoints
        for i in range(5):
            manager.save(
                filename=f"checkpoint_epoch_{i}.pt",
                model=simple_model,
                optimizer=optimizer,
                epoch=i,
                global_step=i * 10,
                config=config,
                best_val_loss=1.0 - i * 0.1,
            )

        # Only the last 3 should remain (plus best_model.pt is not counted)
        epoch_checkpoints = [
            f for f in os.listdir(temp_dir)
            if f.startswith("checkpoint_epoch_")
        ]
        assert len(epoch_checkpoints) == 3

    def test_best_and_final_not_rotated(self, temp_dir, simple_model, optimizer, config):
        """Test that best_model.pt and final_model.pt are not rotated."""
        manager = CheckpointManager(temp_dir, max_checkpoints=2)

        # Save best model
        manager.save(
            filename="best_model.pt",
            model=simple_model,
            optimizer=optimizer,
            epoch=0,
            global_step=0,
            config=config,
            best_val_loss=0.5,
        )

        # Save final model
        manager.save(
            filename="final_model.pt",
            model=simple_model,
            optimizer=optimizer,
            epoch=10,
            global_step=100,
            config=config,
            best_val_loss=0.3,
        )

        # Save more checkpoints
        for i in range(5):
            manager.save(
                filename=f"checkpoint_epoch_{i}.pt",
                model=simple_model,
                optimizer=optimizer,
                epoch=i,
                global_step=i * 10,
                config=config,
                best_val_loss=1.0,
            )

        # Best and final should still exist
        assert os.path.exists(os.path.join(temp_dir, "best_model.pt"))
        assert os.path.exists(os.path.join(temp_dir, "final_model.pt"))

    def test_get_latest_checkpoint(self, temp_dir, simple_model, optimizer, config):
        """Test getting latest checkpoint."""
        manager = CheckpointManager(temp_dir)

        # Initially no checkpoints
        assert manager.get_latest_checkpoint() is None

        # Save checkpoints
        for i in range(3):
            manager.save(
                filename=f"checkpoint_epoch_{i}.pt",
                model=simple_model,
                optimizer=optimizer,
                epoch=i,
                global_step=i * 10,
                config=config,
                best_val_loss=1.0,
            )

        latest = manager.get_latest_checkpoint()
        assert latest is not None
        assert "checkpoint_epoch_" in latest

    def test_get_best_checkpoint(self, temp_dir, simple_model, optimizer, config):
        """Test getting best checkpoint."""
        manager = CheckpointManager(temp_dir)

        # Initially no best checkpoint
        assert manager.get_best_checkpoint() is None

        # Save best checkpoint
        manager.save(
            filename="some_checkpoint.pt",
            model=simple_model,
            optimizer=optimizer,
            epoch=5,
            global_step=50,
            config=config,
            best_val_loss=0.5,
            is_best=True,
        )

        best = manager.get_best_checkpoint()
        assert best is not None
        assert "best_model.pt" in best

    def test_list_checkpoints(self, temp_dir, simple_model, optimizer, config):
        """Test listing all checkpoints."""
        manager = CheckpointManager(temp_dir)

        # Save a few checkpoints
        for i in range(3):
            manager.save(
                filename=f"checkpoint_{i}.pt",
                model=simple_model,
                optimizer=optimizer,
                epoch=i,
                global_step=i * 10,
                config=config,
                best_val_loss=1.0,
            )

        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 3


class TestRestoreTrainingState:
    """Test cases for restore_training_state function."""

    def test_restore_basic_state(self):
        """Test restoring basic training state."""
        checkpoint = {
            'epoch': 5,
            'global_step': 100,
            'best_val_loss': 0.5,
            'train_losses': [1.0, 0.9, 0.8],
            'val_losses': [1.1, 0.95, 0.85],
        }

        state = restore_training_state(checkpoint)

        assert state['epoch'] == 5
        assert state['global_step'] == 100
        assert state['best_val_loss'] == 0.5
        assert state['train_losses'] == [1.0, 0.9, 0.8]
        assert state['val_losses'] == [1.1, 0.95, 0.85]

    def test_restore_with_scheduler(self):
        """Test restoring scheduler state."""
        optimizer = optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        # Step scheduler a few times
        for _ in range(5):
            scheduler.step()

        scheduler_state = scheduler.state_dict()

        checkpoint = {
            'epoch': 5,
            'global_step': 100,
            'best_val_loss': 0.5,
            'scheduler_state_dict': scheduler_state,
        }

        # Create new scheduler
        new_optimizer = optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
        new_scheduler = torch.optim.lr_scheduler.StepLR(new_optimizer, step_size=10)

        restore_training_state(checkpoint, scheduler=new_scheduler)

        assert new_scheduler.last_epoch == scheduler.last_epoch

    def test_restore_with_beta_scheduler(self):
        """Test restoring beta scheduler state."""
        from trainer import BetaScheduler

        beta_scheduler = BetaScheduler(schedule="linear", final_beta=1.0, warmup_epochs=10)

        checkpoint = {
            'epoch': 5,
            'global_step': 100,
            'best_val_loss': 0.5,
            'beta_scheduler_state': {'current_beta': 0.5},
        }

        restore_training_state(checkpoint, beta_scheduler=beta_scheduler)

        assert beta_scheduler.current_beta == 0.5

    def test_restore_with_early_stopping(self):
        """Test restoring early stopping state."""
        from trainer import EarlyStopping

        early_stopping = EarlyStopping(patience=10)

        checkpoint = {
            'epoch': 5,
            'global_step': 100,
            'best_val_loss': 0.5,
            'early_stopping_state': {
                'counter': 3,
                'best_score': 0.6,
                'should_stop': False,
            },
        }

        restore_training_state(checkpoint, early_stopping=early_stopping)

        assert early_stopping.counter == 3
        assert early_stopping.best_score == 0.6

    def test_restore_with_missing_fields(self):
        """Test restoring with missing optional fields."""
        checkpoint = {}  # Empty checkpoint

        state = restore_training_state(checkpoint)

        # Should use defaults
        assert state['epoch'] == 0
        assert state['global_step'] == 0
        assert state['best_val_loss'] == float('inf')
        assert state['train_losses'] == []
        assert state['val_losses'] == []
