"""
Unit tests for learning rate scheduler creation.
"""

import pytest
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    OneCycleLR,
    LinearLR,
    SequentialLR,
)

from trainer import TrainerConfig, create_lr_scheduler
from trainer.schedulers import fast_forward_scheduler


class TestCreateLRScheduler:
    """Test cases for create_lr_scheduler function."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return torch.nn.Linear(10, 10)

    @pytest.fixture
    def optimizer(self, simple_model):
        """Create an optimizer for testing."""
        return optim.AdamW(simple_model.parameters(), lr=1e-3)

    def test_cosine_scheduler(self, optimizer):
        """Test creating cosine annealing scheduler."""
        config = TrainerConfig(
            scheduler="cosine",
            epochs=100,
            min_lr=1e-6,
        )

        scheduler = create_lr_scheduler(optimizer, config, steps_per_epoch=100)

        assert scheduler is not None
        assert isinstance(scheduler, CosineAnnealingLR)
        assert scheduler.T_max == 100
        assert scheduler.eta_min == 1e-6

    def test_plateau_scheduler(self, optimizer):
        """Test creating reduce on plateau scheduler."""
        config = TrainerConfig(
            scheduler="plateau",
            min_lr=1e-6,
            plateau_factor=0.5,
            plateau_patience=10,
        )

        scheduler = create_lr_scheduler(optimizer, config, steps_per_epoch=100)

        assert scheduler is not None
        assert isinstance(scheduler, ReduceLROnPlateau)
        assert scheduler.factor == 0.5
        assert scheduler.patience == 10

    def test_onecycle_scheduler(self, optimizer):
        """Test creating one cycle scheduler."""
        config = TrainerConfig(
            scheduler="onecycle",
            epochs=100,
            learning_rate=1e-3,
            onecycle_pct_start=0.1,
        )

        scheduler = create_lr_scheduler(optimizer, config, steps_per_epoch=100)

        assert scheduler is not None
        assert isinstance(scheduler, OneCycleLR)
        assert scheduler.total_steps == 100 * 100

    def test_warmup_cosine_scheduler(self, optimizer):
        """Test creating warmup + cosine scheduler."""
        config = TrainerConfig(
            scheduler="warmup_cosine",
            epochs=100,
            lr_warmup_epochs=10,
            min_lr=1e-6,
            warmup_start_factor=0.01,
        )

        scheduler = create_lr_scheduler(optimizer, config, steps_per_epoch=100)

        assert scheduler is not None
        assert isinstance(scheduler, SequentialLR)
        assert len(scheduler._schedulers) == 2

    def test_warmup_scheduler(self, optimizer):
        """Test creating warmup only scheduler."""
        config = TrainerConfig(
            scheduler="warmup",
            lr_warmup_epochs=10,
            warmup_start_factor=0.01,
        )

        scheduler = create_lr_scheduler(optimizer, config, steps_per_epoch=100)

        assert scheduler is not None
        assert isinstance(scheduler, LinearLR)

    def test_no_scheduler(self, optimizer):
        """Test no scheduler when scheduler is None."""
        config = TrainerConfig(scheduler=None)

        scheduler = create_lr_scheduler(optimizer, config, steps_per_epoch=100)

        assert scheduler is None

    def test_cosine_scheduler_lr_decay(self, optimizer):
        """Test that cosine scheduler actually decays LR."""
        config = TrainerConfig(
            scheduler="cosine",
            epochs=10,
            min_lr=1e-6,
        )

        scheduler = create_lr_scheduler(optimizer, config, steps_per_epoch=100)

        initial_lr = optimizer.param_groups[0]['lr']

        for _ in range(5):
            scheduler.step()

        mid_lr = optimizer.param_groups[0]['lr']

        for _ in range(5):
            scheduler.step()

        final_lr = optimizer.param_groups[0]['lr']

        assert mid_lr < initial_lr
        assert final_lr < mid_lr

    def test_warmup_scheduler_lr_increase(self, optimizer):
        """Test that warmup scheduler increases LR."""
        config = TrainerConfig(
            scheduler="warmup",
            lr_warmup_epochs=10,
            warmup_start_factor=0.01,
            learning_rate=1e-3,
        )

        # Reset optimizer LR
        for pg in optimizer.param_groups:
            pg['lr'] = 1e-3

        scheduler = create_lr_scheduler(optimizer, config, steps_per_epoch=100)

        # LinearLR starts at start_factor * lr, then increases
        initial_lr = optimizer.param_groups[0]['lr']

        for _ in range(5):
            scheduler.step()

        mid_lr = optimizer.param_groups[0]['lr']

        for _ in range(5):
            scheduler.step()

        final_lr = optimizer.param_groups[0]['lr']

        # During warmup, LR should increase from start_factor towards 1.0
        # initial_lr is already scaled by start_factor (0.01 * 1e-3 = 1e-5)
        assert mid_lr > initial_lr  # Mid should be higher than initial
        assert final_lr >= mid_lr  # Final should be at or near target


class TestFastForwardScheduler:
    """Test cases for fast_forward_scheduler function."""

    @pytest.fixture
    def simple_model(self):
        """Create a simple model for testing."""
        return torch.nn.Linear(10, 10)

    @pytest.fixture
    def optimizer(self, simple_model):
        """Create an optimizer for testing."""
        return optim.AdamW(simple_model.parameters(), lr=1e-3)

    def test_fast_forward_cosine(self, optimizer):
        """Test fast forwarding cosine scheduler."""
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

        fast_forward_scheduler(scheduler, target_epoch=10, steps_per_epoch=100, scheduler_type="cosine")

        # Compare with manually stepped scheduler
        ref_optimizer = optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
        ref_scheduler = CosineAnnealingLR(ref_optimizer, T_max=100, eta_min=1e-6)

        for _ in range(10):
            ref_scheduler.step()

        assert abs(optimizer.param_groups[0]['lr'] - ref_optimizer.param_groups[0]['lr']) < 1e-10

    def test_fast_forward_onecycle(self, optimizer):
        """Test fast forwarding one cycle scheduler."""
        scheduler = OneCycleLR(optimizer, max_lr=1e-3, total_steps=1000, pct_start=0.1)

        fast_forward_scheduler(scheduler, target_epoch=2, steps_per_epoch=100, scheduler_type="onecycle")

        # Should have stepped 200 times
        # Compare with manually stepped scheduler
        ref_optimizer = optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
        ref_scheduler = OneCycleLR(ref_optimizer, max_lr=1e-3, total_steps=1000, pct_start=0.1)

        for _ in range(200):
            ref_scheduler.step()

        assert abs(optimizer.param_groups[0]['lr'] - ref_optimizer.param_groups[0]['lr']) < 1e-10

    def test_fast_forward_plateau(self, optimizer):
        """Test fast forwarding plateau scheduler (should be no-op)."""
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

        initial_lr = optimizer.param_groups[0]['lr']

        fast_forward_scheduler(scheduler, target_epoch=10, steps_per_epoch=100, scheduler_type="plateau")

        # Plateau scheduler should not change
        assert optimizer.param_groups[0]['lr'] == initial_lr

    def test_fast_forward_zero_epochs(self, optimizer):
        """Test fast forwarding zero epochs."""
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

        initial_lr = optimizer.param_groups[0]['lr']

        fast_forward_scheduler(scheduler, target_epoch=0, steps_per_epoch=100, scheduler_type="cosine")

        assert optimizer.param_groups[0]['lr'] == initial_lr

    def test_fast_forward_negative_epochs(self, optimizer):
        """Test fast forwarding negative epochs (should be no-op)."""
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

        initial_lr = optimizer.param_groups[0]['lr']

        fast_forward_scheduler(scheduler, target_epoch=-5, steps_per_epoch=100, scheduler_type="cosine")

        assert optimizer.param_groups[0]['lr'] == initial_lr

    def test_fast_forward_none_scheduler(self, optimizer):
        """Test fast forwarding None scheduler."""
        # Should not raise
        fast_forward_scheduler(None, target_epoch=10, steps_per_epoch=100, scheduler_type="cosine")
