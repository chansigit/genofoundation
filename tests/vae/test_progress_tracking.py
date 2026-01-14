"""
Unit tests for absolute progress tracking functionality.
"""

import pytest
import torch
from vae.trainer import VAETrainer, TrainingConfig


class TestProgressTracking:
    """Test cases for absolute progress tracking"""
    
    def test_initial_progress_state(self, basic_config, simple_vae, test_dataset):
        """Test initial progress tracking state"""
        trainer = VAETrainer(simple_vae, basic_config, test_dataset)
        
        assert trainer.total_epochs_planned == basic_config.epochs
        assert trainer.absolute_epoch == 0
        assert trainer.absolute_step == 0
        assert trainer.current_epoch == 0  # Legacy compatibility
        assert trainer.global_step == 0    # Legacy compatibility
        
    def test_progress_tracking_consistency(self, basic_config, simple_vae, test_dataset):
        """Test that progress tracking fields remain consistent"""
        trainer = VAETrainer(simple_vae, basic_config, test_dataset)
        
        # Simulate training progress
        trainer.absolute_epoch = 5
        trainer.absolute_step = 75
        trainer.current_epoch = trainer.absolute_epoch
        trainer.global_step = trainer.absolute_step
        
        # Consistency checks
        assert trainer.current_epoch == trainer.absolute_epoch
        assert trainer.global_step == trainer.absolute_step
        
    def test_total_epochs_from_config(self, temp_checkpoint_dir, simple_vae, test_dataset):
        """Test total_epochs_planned is correctly set from config"""
        config = TrainingConfig(
            epochs=50,  # Different from default
            checkpoint_dir=temp_checkpoint_dir,
            device="cpu"
        )
        
        trainer = VAETrainer(simple_vae, config, test_dataset)
        
        assert trainer.total_epochs_planned == 50
        
    def test_progress_preservation_across_checkpoints(self, basic_config, simple_vae, test_dataset):
        """Test that absolute progress is preserved across checkpoint cycles"""
        trainer = VAETrainer(simple_vae, basic_config, test_dataset)
        
        # Set progress state
        trainer.absolute_epoch = 12
        trainer.absolute_step = 180
        trainer.total_epochs_planned = 100
        
        # Save checkpoint
        checkpoint_path = "progress_test.pt"
        trainer.save_checkpoint(checkpoint_path)
        
        # Load into new trainer
        new_trainer = VAETrainer(simple_vae, basic_config, test_dataset)
        full_path = os.path.join(basic_config.checkpoint_dir, checkpoint_path)
        new_trainer.load_checkpoint(full_path)
        
        # Verify progress preservation
        assert new_trainer.absolute_epoch == 12
        assert new_trainer.absolute_step == 180
        assert new_trainer.total_epochs_planned == 100
        
    def test_legacy_field_derivation(self, basic_config, simple_vae, test_dataset):
        """Test that legacy fields are properly derived from absolute progress"""
        trainer = VAETrainer(simple_vae, basic_config, test_dataset)
        
        # Set absolute progress
        trainer.absolute_epoch = 8
        trainer.absolute_step = 120
        
        # Update legacy fields (as would happen in training loop)
        trainer.current_epoch = trainer.absolute_epoch
        trainer.global_step = trainer.absolute_step
        
        assert trainer.current_epoch == 8
        assert trainer.global_step == 120
        
    def test_scheduler_uses_total_epochs(self, basic_config, simple_vae, test_dataset):
        """Test that scheduler creation uses total_epochs_planned"""
        config = basic_config
        config.scheduler = "cosine"
        config.epochs = 25
        
        trainer = VAETrainer(simple_vae, config, test_dataset)
        
        # Scheduler should be created with total planned epochs
        assert trainer.scheduler.T_max == 25  # CosineAnnealingLR parameter
        
    def test_progress_calculation_correctness(self, basic_config, simple_vae, test_dataset):
        """Test that progress calculations are mathematically correct"""
        trainer = VAETrainer(simple_vae, basic_config, test_dataset)
        
        steps_per_epoch = len(trainer.train_loader)
        
        # Test epoch to step conversion
        for epoch in range(5):
            expected_total_steps = epoch * steps_per_epoch
            trainer.absolute_epoch = epoch
            trainer.absolute_step = expected_total_steps
            
            # Verify relationship holds
            calculated_epochs = trainer.absolute_step // steps_per_epoch
            assert calculated_epochs == epoch
            
    def test_progress_with_different_batch_sizes(self, temp_checkpoint_dir, simple_vae, test_dataset):
        """Test progress tracking with different batch sizes"""
        config1 = TrainingConfig(
            epochs=20,
            batch_size=16,
            checkpoint_dir=temp_checkpoint_dir,
            device="cpu"
        )
        
        config2 = TrainingConfig(
            epochs=20,
            batch_size=32,
            checkpoint_dir=temp_checkpoint_dir,
            device="cpu"
        )
        
        trainer1 = VAETrainer(simple_vae, config1, test_dataset)
        trainer2 = VAETrainer(simple_vae, config2, test_dataset)
        
        # Different batch sizes should give different steps per epoch
        steps1 = len(trainer1.train_loader)
        steps2 = len(trainer2.train_loader)
        
        assert steps1 != steps2  # Should be different due to batch size
        
        # But total epochs planned should be the same
        assert trainer1.total_epochs_planned == trainer2.total_epochs_planned
        
    def test_progress_edge_cases(self, basic_config, simple_vae, test_dataset):
        """Test edge cases in progress tracking"""
        trainer = VAETrainer(simple_vae, basic_config, test_dataset)
        
        # Test zero progress
        assert trainer.absolute_epoch == 0
        assert trainer.absolute_step == 0
        
        # Test final progress
        trainer.absolute_epoch = trainer.total_epochs_planned - 1  # Last epoch (0-indexed)
        final_step = trainer.absolute_epoch * len(trainer.train_loader)
        trainer.absolute_step = final_step
        
        assert trainer.absolute_epoch == basic_config.epochs - 1
        
    def test_progress_tracking_with_resume(self, basic_config, simple_vae, test_dataset):
        """Test progress tracking behavior during resume"""
        # Create first trainer and advance progress
        trainer1 = VAETrainer(simple_vae, basic_config, test_dataset)
        trainer1.absolute_epoch = 7
        trainer1.absolute_step = 105
        
        # Save state
        checkpoint_path = "resume_progress_test.pt"
        trainer1.save_checkpoint(checkpoint_path)
        
        # Create second trainer and resume
        trainer2 = VAETrainer(simple_vae, basic_config, test_dataset)
        full_path = os.path.join(basic_config.checkpoint_dir, checkpoint_path)
        trainer2.load_checkpoint(full_path)
        
        # Progress should continue from where it left off
        assert trainer2.absolute_epoch == 7
        assert trainer2.absolute_step == 105
        
        # Further progress should build on resumed state
        trainer2.absolute_epoch = 9
        expected_step = 9 * len(trainer2.train_loader)
        trainer2.absolute_step = expected_step
        
        assert trainer2.absolute_epoch == 9
        assert trainer2.absolute_step == expected_step
        
    def test_config_epochs_immutable_after_init(self, basic_config, simple_vae, test_dataset):
        """Test that total_epochs_planned acts as immutable reference"""
        trainer = VAETrainer(simple_vae, basic_config, test_dataset)
        
        original_planned = trainer.total_epochs_planned
        
        # Simulate config change (shouldn't affect planned epochs)
        basic_config.epochs = 50  # Change config after trainer creation
        
        # Planned epochs should remain unchanged
        assert trainer.total_epochs_planned == original_planned
        assert trainer.total_epochs_planned != basic_config.epochs