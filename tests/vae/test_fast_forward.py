"""
Unit tests for scheduler fast-forward logic.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, OneCycleLR, ReduceLROnPlateau, SequentialLR
from vae.trainer import VAETrainer


class TestSchedulerFastForward:
    """Test cases for scheduler fast-forward functionality"""
    
    def test_fast_forward_cosine_scheduler(self, basic_config, simple_vae, test_dataset):
        """Test fast-forwarding CosineAnnealingLR scheduler"""
        config = basic_config
        config.scheduler = "cosine"
        
        trainer = VAETrainer(simple_vae, config, test_dataset)
        
        # Get initial LR
        initial_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Fast-forward 5 epochs
        trainer._fast_forward_scheduler(5)
        fast_forward_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Create reference scheduler and step manually
        ref_optimizer = optim.AdamW(simple_vae.parameters(), lr=config.learning_rate)
        ref_scheduler = CosineAnnealingLR(ref_optimizer, T_max=config.epochs, eta_min=config.min_lr)
        
        for _ in range(5):
            ref_scheduler.step()
        reference_lr = ref_optimizer.param_groups[0]['lr']
        
        # Should match manual stepping
        assert abs(fast_forward_lr - reference_lr) < 1e-6
        assert fast_forward_lr != initial_lr  # Should have changed
        
    def test_fast_forward_linear_scheduler(self, basic_config, simple_vae, test_dataset):
        """Test fast-forwarding LinearLR scheduler"""
        config = basic_config
        config.scheduler = "warmup" 
        config.lr_warmup_epochs = 10
        
        trainer = VAETrainer(simple_vae, config, test_dataset)
        
        # Fast-forward 3 epochs (within warmup)
        trainer._fast_forward_scheduler(3)
        fast_forward_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Create reference
        ref_optimizer = optim.AdamW(simple_vae.parameters(), lr=config.learning_rate)
        ref_scheduler = LinearLR(ref_optimizer, start_factor=0.01, end_factor=1.0, total_iters=10)
        
        for _ in range(3):
            ref_scheduler.step()
        reference_lr = ref_optimizer.param_groups[0]['lr']
        
        assert abs(fast_forward_lr - reference_lr) < 1e-6
        
    def test_fast_forward_onecycle_scheduler(self, basic_config, simple_vae, test_dataset):
        """Test fast-forwarding OneCycleLR scheduler"""
        config = basic_config
        config.scheduler = "onecycle"
        
        trainer = VAETrainer(simple_vae, config, test_dataset)
        
        # Fast-forward 2 epochs
        target_steps = 2 * len(trainer.train_loader)
        trainer._fast_forward_scheduler(2)
        fast_forward_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Create reference
        ref_optimizer = optim.AdamW(simple_vae.parameters(), lr=config.learning_rate)
        total_steps = len(trainer.train_loader) * config.epochs
        ref_scheduler = OneCycleLR(ref_optimizer, max_lr=config.learning_rate, 
                                  total_steps=total_steps, pct_start=0.1)
        
        for _ in range(target_steps):
            ref_scheduler.step()
        reference_lr = ref_optimizer.param_groups[0]['lr']
        
        assert abs(fast_forward_lr - reference_lr) < 1e-6
        
    def test_fast_forward_sequential_scheduler(self, basic_config, simple_vae, test_dataset):
        """Test fast-forwarding SequentialLR scheduler"""
        config = basic_config
        config.scheduler = "warmup_cosine"
        config.lr_warmup_epochs = 3
        
        trainer = VAETrainer(simple_vae, config, test_dataset)
        
        # Fast-forward to middle of warmup (epoch 1)
        trainer._fast_forward_scheduler(1)
        warmup_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Fast-forward to after warmup (epoch 5)
        trainer = VAETrainer(simple_vae, config, test_dataset)  # Reset
        trainer._fast_forward_scheduler(5)
        cosine_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Warmup LR should be different from post-warmup LR
        assert warmup_lr != cosine_lr
        
        # Verify against manual reference
        ref_optimizer = optim.AdamW(simple_vae.parameters(), lr=config.learning_rate)
        warmup_sched = LinearLR(ref_optimizer, start_factor=0.01, end_factor=1.0,
                               total_iters=len(trainer.train_loader) * 3)
        cosine_sched = CosineAnnealingLR(ref_optimizer, T_max=config.epochs - 3, eta_min=config.min_lr)
        ref_scheduler = SequentialLR(ref_optimizer, [warmup_sched, cosine_sched], milestones=[3])
        
        for _ in range(5):
            ref_scheduler.step()
        reference_lr = ref_optimizer.param_groups[0]['lr']
        
        assert abs(cosine_lr - reference_lr) < 1e-6
        
    def test_fast_forward_plateau_scheduler(self, basic_config, simple_vae, test_dataset):
        """Test fast-forwarding ReduceLROnPlateau scheduler (should be no-op)"""
        config = basic_config
        config.scheduler = "plateau"
        
        trainer = VAETrainer(simple_vae, config, test_dataset)
        
        # Get initial LR
        initial_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Fast-forward should be no-op for plateau scheduler
        trainer._fast_forward_scheduler(5)
        fast_forward_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Should remain unchanged (plateau scheduler is stateful and metric-driven)
        assert fast_forward_lr == initial_lr
        
    def test_fast_forward_zero_epochs(self, basic_config, simple_vae, test_dataset):
        """Test fast-forwarding zero epochs (should be no-op)"""
        config = basic_config
        config.scheduler = "cosine"
        
        trainer = VAETrainer(simple_vae, config, test_dataset)
        
        initial_lr = trainer.optimizer.param_groups[0]['lr']
        trainer._fast_forward_scheduler(0)
        final_lr = trainer.optimizer.param_groups[0]['lr']
        
        assert final_lr == initial_lr
        
    def test_fast_forward_negative_epochs(self, basic_config, simple_vae, test_dataset):
        """Test fast-forwarding negative epochs (should be no-op)"""
        config = basic_config
        config.scheduler = "cosine"
        
        trainer = VAETrainer(simple_vae, config, test_dataset)
        
        initial_lr = trainer.optimizer.param_groups[0]['lr']
        trainer._fast_forward_scheduler(-5)
        final_lr = trainer.optimizer.param_groups[0]['lr']
        
        assert final_lr == initial_lr
        
    def test_fast_forward_no_scheduler(self, basic_config, simple_vae, test_dataset):
        """Test fast-forwarding with no scheduler (should be no-op)"""
        config = basic_config
        config.scheduler = None  # This should result in no scheduler
        
        # Manually set scheduler to None to test edge case
        trainer = VAETrainer(simple_vae, config, test_dataset)
        trainer.scheduler = None
        
        initial_lr = trainer.optimizer.param_groups[0]['lr']
        trainer._fast_forward_scheduler(5)
        final_lr = trainer.optimizer.param_groups[0]['lr']
        
        assert final_lr == initial_lr
        
    def test_fast_forward_beyond_warmup(self, basic_config, simple_vae, test_dataset):
        """Test fast-forwarding beyond warmup period for warmup scheduler"""
        config = basic_config
        config.scheduler = "warmup"
        config.lr_warmup_epochs = 3
        
        trainer = VAETrainer(simple_vae, config, test_dataset)
        
        # Fast-forward beyond warmup period
        trainer._fast_forward_scheduler(5)
        fast_forward_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Should be at target learning rate (warmup completed)
        expected_lr = config.learning_rate  # Should reach full LR after warmup
        
        # Allow some tolerance for scheduler implementation details
        assert abs(fast_forward_lr - expected_lr) < 1e-4
        
    def test_fast_forward_consistency(self, basic_config, simple_vae, test_dataset):
        """Test that fast-forwarding gives consistent results"""
        config = basic_config
        config.scheduler = "cosine"
        
        # Create two identical trainers
        trainer1 = VAETrainer(simple_vae, config, test_dataset)
        trainer2 = VAETrainer(simple_vae, config, test_dataset)
        
        # Fast-forward both to same epoch
        trainer1._fast_forward_scheduler(7)
        trainer2._fast_forward_scheduler(7)
        
        lr1 = trainer1.optimizer.param_groups[0]['lr']
        lr2 = trainer2.optimizer.param_groups[0]['lr']
        
        assert abs(lr1 - lr2) < 1e-10  # Should be identical
        
    def test_fast_forward_incremental_vs_direct(self, basic_config, simple_vae, test_dataset):
        """Test that incremental fast-forward equals direct fast-forward"""
        config = basic_config
        config.scheduler = "cosine"
        
        # Direct fast-forward
        trainer1 = VAETrainer(simple_vae, config, test_dataset)
        trainer1._fast_forward_scheduler(5)
        direct_lr = trainer1.optimizer.param_groups[0]['lr']
        
        # Incremental fast-forward
        trainer2 = VAETrainer(simple_vae, config, test_dataset)
        trainer2._fast_forward_scheduler(3)
        trainer2._fast_forward_scheduler(2)  # 3 + 2 = 5 total
        incremental_lr = trainer2.optimizer.param_groups[0]['lr']
        
        # Results should be the same
        assert abs(direct_lr - incremental_lr) < 1e-10