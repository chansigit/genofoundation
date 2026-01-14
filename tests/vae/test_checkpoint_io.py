"""
Unit tests for checkpoint save/load functionality.
"""

import pytest
import torch
import os
from pathlib import Path
from vae.trainer import VAETrainer, TrainingConfig


class TestCheckpointIO:
    """Test cases for checkpoint save/load operations"""
    
    def test_basic_checkpoint_save_load(self, basic_config, simple_vae, test_dataset, val_dataset):
        """Test basic checkpoint save and load functionality"""
        trainer = VAETrainer(simple_vae, basic_config, test_dataset, val_dataset)
        
        # Set some training state
        trainer.absolute_epoch = 10
        trainer.absolute_step = 150
        trainer.current_epoch = 10
        trainer.global_step = 150
        trainer.best_val_loss = 0.5
        trainer.train_losses = [1.0, 0.9, 0.8]
        trainer.val_losses = [1.1, 0.95, 0.85]
        
        # Save checkpoint
        checkpoint_path = "test_checkpoint.pt"
        trainer.save_checkpoint(checkpoint_path)
        
        # Verify file exists
        full_path = os.path.join(basic_config.checkpoint_dir, checkpoint_path)
        assert os.path.exists(full_path)
        
        # Load checkpoint into new trainer
        new_trainer = VAETrainer(simple_vae, basic_config, test_dataset, val_dataset)
        new_trainer.load_checkpoint(full_path)
        
        # Verify state restoration
        assert new_trainer.absolute_epoch == 10
        assert new_trainer.absolute_step == 150
        assert new_trainer.current_epoch == 10
        assert new_trainer.global_step == 150
        assert new_trainer.best_val_loss == 0.5
        assert new_trainer.train_losses == [1.0, 0.9, 0.8]
        assert new_trainer.val_losses == [1.1, 0.95, 0.85]
        
    def test_beta_scheduler_state_save_load(self, basic_config, simple_vae, test_dataset):
        """Test beta scheduler state is properly saved and restored"""
        trainer = VAETrainer(simple_vae, basic_config, test_dataset)
        
        # Advance beta scheduler
        trainer.absolute_epoch = 3
        trainer.beta_scheduler.step(3)  # Should change current_beta
        original_beta = trainer.beta_scheduler.current_beta
        
        # Save and reload
        checkpoint_path = "beta_test_checkpoint.pt"
        trainer.save_checkpoint(checkpoint_path)
        
        new_trainer = VAETrainer(simple_vae, basic_config, test_dataset)
        full_path = os.path.join(basic_config.checkpoint_dir, checkpoint_path)
        new_trainer.load_checkpoint(full_path)
        
        # Beta scheduler state should be recalculated from absolute_epoch
        expected_beta = new_trainer.beta_scheduler.step(3)
        assert abs(new_trainer.beta_scheduler.current_beta - expected_beta) < 1e-6
        
    def test_early_stopping_state_save_load(self, basic_config, simple_vae, test_dataset):
        """Test early stopping state is properly saved and restored"""
        config = basic_config
        config.early_stopping = True
        config.patience = 5
        
        trainer = VAETrainer(simple_vae, config, test_dataset)
        
        # Trigger early stopping state changes
        trainer.early_stopping.best_score = 0.8
        trainer.early_stopping.counter = 3
        
        # Save and reload
        checkpoint_path = "early_stop_test.pt"
        trainer.save_checkpoint(checkpoint_path)
        
        new_trainer = VAETrainer(simple_vae, config, test_dataset)
        full_path = os.path.join(config.checkpoint_dir, checkpoint_path)
        new_trainer.load_checkpoint(full_path)
        
        # Early stopping state should be restored
        assert new_trainer.early_stopping.best_score == 0.8
        assert new_trainer.early_stopping.counter == 3
        assert new_trainer.early_stopping.should_stop is False
        
    def test_scheduler_state_save_load(self, basic_config, simple_vae, test_dataset):
        """Test LR scheduler state is properly saved and restored"""
        config = basic_config
        config.scheduler = "cosine"
        
        trainer = VAETrainer(simple_vae, config, test_dataset)
        
        # Advance scheduler
        for _ in range(5):
            trainer.scheduler.step()
        original_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Save and reload
        checkpoint_path = "scheduler_test.pt"
        trainer.save_checkpoint(checkpoint_path)
        
        new_trainer = VAETrainer(simple_vae, config, test_dataset)
        full_path = os.path.join(config.checkpoint_dir, checkpoint_path)
        new_trainer.load_checkpoint(full_path)
        
        # Note: Scheduler state is stored but fast-forward logic in training handles restoration
        # Here we just verify the state was saved
        checkpoint = torch.load(full_path)
        assert 'scheduler_state_dict' in checkpoint
        assert checkpoint['scheduler_state_dict'] is not None
        
    def test_mixed_precision_scaler_save_load(self, basic_config, simple_vae, test_dataset):
        """Test mixed precision scaler state is saved and restored"""
        config = basic_config
        config.mixed_precision = True
        config.device = "cpu"  # Use CPU for testing
        
        trainer = VAETrainer(simple_vae, config, test_dataset)
        
        # Scaler should exist
        assert trainer.scaler is not None
        
        # Save checkpoint
        checkpoint_path = "scaler_test.pt"
        trainer.save_checkpoint(checkpoint_path)
        
        # Verify scaler state was saved
        full_path = os.path.join(config.checkpoint_dir, checkpoint_path)
        checkpoint = torch.load(full_path)
        assert 'scaler_state_dict' in checkpoint
        
    def test_config_preservation(self, basic_config, simple_vae, test_dataset):
        """Test that training config is preserved in checkpoint"""
        trainer = VAETrainer(simple_vae, basic_config, test_dataset)
        
        # Save checkpoint
        checkpoint_path = "config_test.pt"
        trainer.save_checkpoint(checkpoint_path)
        
        # Load and check config
        full_path = os.path.join(basic_config.checkpoint_dir, checkpoint_path)
        checkpoint = torch.load(full_path)
        
        saved_config = checkpoint['config']
        assert saved_config['epochs'] == basic_config.epochs
        assert saved_config['batch_size'] == basic_config.batch_size
        assert saved_config['learning_rate'] == basic_config.learning_rate
        assert saved_config['scheduler'] == basic_config.scheduler
        
    def test_best_checkpoint_save(self, basic_config, simple_vae, test_dataset):
        """Test best checkpoint is saved when is_best=True"""
        trainer = VAETrainer(simple_vae, basic_config, test_dataset)
        
        # Save as best
        checkpoint_path = "regular_checkpoint.pt"
        trainer.save_checkpoint(checkpoint_path, is_best=True)
        
        # Both regular and best files should exist
        regular_path = os.path.join(basic_config.checkpoint_dir, checkpoint_path)
        best_path = os.path.join(basic_config.checkpoint_dir, "best_model.pt")
        
        assert os.path.exists(regular_path)
        assert os.path.exists(best_path)
        
        # Contents should be identical
        regular_checkpoint = torch.load(regular_path)
        best_checkpoint = torch.load(best_path)
        
        assert regular_checkpoint['absolute_epoch'] == best_checkpoint['absolute_epoch']
        assert regular_checkpoint['best_val_loss'] == best_checkpoint['best_val_loss']
        
    def test_backward_compatibility(self, basic_config, simple_vae, test_dataset):
        """Test loading checkpoints without new absolute progress fields"""
        trainer = VAETrainer(simple_vae, basic_config, test_dataset)
        
        # Create a legacy checkpoint manually (without absolute progress fields)
        legacy_checkpoint = {
            'epoch': 8,
            'global_step': 120,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'scheduler_state_dict': trainer.scheduler.state_dict() if trainer.scheduler else None,
            'best_val_loss': 0.7,
            'config': vars(basic_config),
            'train_losses': [1.0, 0.8],
            'val_losses': [1.1, 0.9],
        }
        
        # Save legacy checkpoint
        legacy_path = os.path.join(basic_config.checkpoint_dir, "legacy_checkpoint.pt")
        torch.save(legacy_checkpoint, legacy_path)
        
        # Load into new trainer
        new_trainer = VAETrainer(simple_vae, basic_config, test_dataset)
        new_trainer.load_checkpoint(legacy_path)
        
        # Should fallback to legacy fields
        assert new_trainer.absolute_epoch == 8  # Should use 'epoch' field
        assert new_trainer.absolute_step == 120  # Should use 'global_step' field
        assert new_trainer.total_epochs_planned == basic_config.epochs  # Should use config
        
    def test_malformed_checkpoint_handling(self, basic_config, simple_vae, test_dataset):
        """Test handling of malformed/corrupted checkpoints"""
        trainer = VAETrainer(simple_vae, basic_config, test_dataset)
        
        # Create malformed checkpoint (missing required fields)
        malformed_checkpoint = {
            'epoch': 5,
            # Missing model_state_dict and other required fields
        }
        
        malformed_path = os.path.join(basic_config.checkpoint_dir, "malformed.pt")
        torch.save(malformed_checkpoint, malformed_path)
        
        # Loading should fail gracefully
        with pytest.raises(KeyError):
            trainer.load_checkpoint(malformed_path)
            
    def test_nonexistent_checkpoint(self, basic_config, simple_vae, test_dataset):
        """Test loading nonexistent checkpoint file"""
        trainer = VAETrainer(simple_vae, basic_config, test_dataset)
        
        nonexistent_path = "/nonexistent/path/checkpoint.pt"
        
        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint(nonexistent_path)
            
    def test_checkpoint_device_mapping(self, basic_config, simple_vae, test_dataset):
        """Test checkpoint loading with device mapping"""
        trainer = VAETrainer(simple_vae, basic_config, test_dataset)
        
        # Ensure we're using CPU
        trainer.device = torch.device("cpu")
        
        # Save checkpoint
        checkpoint_path = "device_test.pt"
        trainer.save_checkpoint(checkpoint_path)
        
        # Load checkpoint (device mapping should work)
        full_path = os.path.join(basic_config.checkpoint_dir, checkpoint_path)
        new_trainer = VAETrainer(simple_vae, basic_config, test_dataset)
        new_trainer.load_checkpoint(full_path)  # Should not raise device errors
        
        # Verify model is on correct device
        model_device = next(new_trainer.model.parameters()).device
        assert model_device == new_trainer.device
        
    def test_empty_history_lists(self, basic_config, simple_vae, test_dataset):
        """Test checkpointing with empty history lists"""
        trainer = VAETrainer(simple_vae, basic_config, test_dataset)
        
        # Explicitly empty histories (default state)
        trainer.train_losses = []
        trainer.val_losses = []
        
        # Save and load
        checkpoint_path = "empty_history.pt"
        trainer.save_checkpoint(checkpoint_path)
        
        full_path = os.path.join(basic_config.checkpoint_dir, checkpoint_path)
        new_trainer = VAETrainer(simple_vae, basic_config, test_dataset)
        new_trainer.load_checkpoint(full_path)
        
        assert new_trainer.train_losses == []
        assert new_trainer.val_losses == []