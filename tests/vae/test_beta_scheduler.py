"""
Unit tests for BetaScheduler class.
"""

import pytest
import math
import numpy as np
from trainer import BetaScheduler, TrainerConfig


class TestBetaScheduler:
    """Test cases for BetaScheduler."""

    def test_constant_schedule(self):
        """Test constant beta schedule."""
        scheduler = BetaScheduler(
            schedule="constant",
            initial_beta=0.0,
            final_beta=1.5,
            warmup_epochs=10,
        )

        assert scheduler.step(0) == 1.5
        assert scheduler.step(5) == 1.5
        assert scheduler.step(10) == 1.5
        assert scheduler.step(100) == 1.5

    def test_linear_schedule_warmup(self):
        """Test linear schedule during warmup period."""
        scheduler = BetaScheduler(
            schedule="linear",
            initial_beta=0.0,
            final_beta=2.0,
            warmup_epochs=10,
        )

        assert scheduler.step(0) == 0.0
        assert scheduler.step(5) == 1.0  # Halfway
        assert scheduler.step(10) == 2.0  # End of warmup

        expected_at_3 = 0.0 + (3 / 10) * 2.0
        assert abs(scheduler.step(3) - expected_at_3) < 1e-6

    def test_linear_schedule_post_warmup(self):
        """Test linear schedule after warmup completion."""
        scheduler = BetaScheduler(
            schedule="linear",
            initial_beta=0.0,
            final_beta=1.0,
            warmup_epochs=5,
        )

        assert scheduler.step(5) == 1.0
        assert scheduler.step(10) == 1.0
        assert scheduler.step(100) == 1.0

    def test_cyclical_schedule(self):
        """Test cyclical beta schedule."""
        scheduler = BetaScheduler(
            schedule="cyclical",
            initial_beta=0.0,
            final_beta=1.0,
            warmup_epochs=10,
            cycle_epochs=4,
        )

        # First cycle (epochs 0-3)
        assert scheduler.step(0) == 0.0
        assert abs(scheduler.step(2) - 0.5) < 1e-6
        # At epoch 3: position = 3/4 = 0.75, beta = min(1.0, 0.75) = 0.75
        assert abs(scheduler.step(3) - 0.75) < 1e-6

        # Second cycle (epochs 4-7)
        assert scheduler.step(4) == 0.0  # Start of new cycle
        assert abs(scheduler.step(6) - 0.5) < 1e-6

    def test_sigmoid_schedule_warmup(self):
        """Test sigmoid schedule during warmup."""
        scheduler = BetaScheduler(
            schedule="sigmoid",
            initial_beta=0.0,
            final_beta=1.0,
            warmup_epochs=10,
        )

        beta_start = scheduler.step(0)
        beta_mid = scheduler.step(5)
        beta_end = scheduler.step(10)

        # Should be monotonically increasing
        assert beta_start < beta_mid < beta_end

        # Should approach final_beta at the end
        assert abs(beta_end - 1.0) < 0.1

    def test_sigmoid_schedule_post_warmup(self):
        """Test sigmoid schedule after warmup."""
        scheduler = BetaScheduler(
            schedule="sigmoid",
            initial_beta=0.0,
            final_beta=2.0,
            warmup_epochs=5,
        )

        assert scheduler.step(10) == 2.0
        assert scheduler.step(100) == 2.0

    def test_unknown_schedule(self):
        """Test unknown schedule doesn't change beta."""
        scheduler = BetaScheduler(
            schedule="unknown_schedule",
            initial_beta=0.5,
            final_beta=1.0,
            warmup_epochs=5,
        )

        initial_beta = scheduler.current_beta
        scheduler.step(0)
        assert scheduler.current_beta == initial_beta

    def test_zero_warmup_epochs(self):
        """Test linear schedule with zero warmup epochs."""
        scheduler = BetaScheduler(
            schedule="linear",
            initial_beta=0.0,
            final_beta=1.0,
            warmup_epochs=0,
        )
        # With 0 warmup, epoch 0 should still be before warmup_epochs
        # So it should still be at initial or interpolated value
        # Actually with warmup=0, epoch 0 is NOT < 0, so goes to final
        assert scheduler.step(0) == 1.0

    def test_single_warmup_epoch(self):
        """Test linear schedule with single warmup epoch."""
        scheduler = BetaScheduler(
            schedule="linear",
            initial_beta=0.0,
            final_beta=1.0,
            warmup_epochs=1,
        )
        assert scheduler.step(0) == 0.0
        assert scheduler.step(1) == 1.0

    def test_negative_epoch(self):
        """Test behavior with negative epochs."""
        scheduler = BetaScheduler(
            schedule="linear",
            initial_beta=0.0,
            final_beta=1.0,
            warmup_epochs=5,
        )

        beta = scheduler.step(-1)
        # Progress = -1/5 = -0.2, beta = 0 + (-0.2) * 1 = -0.2 (but might be clamped)
        assert beta <= 0.0

    def test_current_beta_tracking(self):
        """Test that current_beta is properly tracked."""
        scheduler = BetaScheduler(
            schedule="linear",
            initial_beta=0.2,
            final_beta=1.8,
            warmup_epochs=4,
        )

        assert scheduler.current_beta == 0.2

        beta_2 = scheduler.step(2)
        assert scheduler.current_beta == beta_2

        beta_4 = scheduler.step(4)
        assert scheduler.current_beta == beta_4
        assert beta_4 == 1.8

    def test_multiple_steps_consistency(self):
        """Test that multiple calls to step() with same epoch give same result."""
        scheduler = BetaScheduler(
            schedule="linear",
            initial_beta=0.0,
            final_beta=1.0,
            warmup_epochs=10,
        )

        beta1 = scheduler.step(5)
        beta2 = scheduler.step(5)
        beta3 = scheduler.step(5)

        assert beta1 == beta2 == beta3

    def test_schedule_parameters_preserved(self):
        """Test that schedule parameters are preserved correctly."""
        scheduler = BetaScheduler(
            schedule="cyclical",
            initial_beta=0.1,
            final_beta=2.5,
            warmup_epochs=8,
            cycle_epochs=6,
        )

        assert scheduler.schedule == "cyclical"
        assert scheduler.initial_beta == 0.1
        assert scheduler.final_beta == 2.5
        assert scheduler.warmup_epochs == 8
        assert scheduler.cycle_epochs == 6

    def test_cyclical_multiple_cycles(self):
        """Test cyclical schedule across multiple cycles."""
        scheduler = BetaScheduler(
            schedule="cyclical",
            initial_beta=0.0,
            final_beta=1.0,
            cycle_epochs=3,
        )

        cycle1_values = [scheduler.step(i) for i in range(3)]
        cycle2_values = [scheduler.step(i + 3) for i in range(3)]
        cycle3_values = [scheduler.step(i + 6) for i in range(3)]

        np.testing.assert_array_almost_equal(cycle1_values, cycle2_values)
        np.testing.assert_array_almost_equal(cycle2_values, cycle3_values)

    def test_get_state(self):
        """Test get_state method for checkpointing."""
        scheduler = BetaScheduler(
            schedule="linear",
            initial_beta=0.0,
            final_beta=1.0,
            warmup_epochs=10,
            cycle_epochs=5,
        )

        scheduler.step(5)

        state = scheduler.get_state()

        assert state['current_beta'] == scheduler.current_beta
        assert state['schedule'] == "linear"
        assert state['initial_beta'] == 0.0
        assert state['final_beta'] == 1.0
        assert state['warmup_epochs'] == 10
        assert state['cycle_epochs'] == 5

    def test_load_state(self):
        """Test load_state method for checkpoint restoration."""
        scheduler = BetaScheduler(
            schedule="linear",
            initial_beta=0.0,
            final_beta=1.0,
            warmup_epochs=10,
        )

        state = {'current_beta': 0.75}
        scheduler.load_state(state)

        assert scheduler.current_beta == 0.75

    def test_from_config(self):
        """Test creating BetaScheduler from config."""
        config = TrainerConfig(
            beta=2.0,
            beta_schedule="sigmoid",
            beta_warmup_epochs=15,
        )

        scheduler = BetaScheduler.from_config(config)

        assert scheduler.schedule == "sigmoid"
        assert scheduler.initial_beta == 0.0
        assert scheduler.final_beta == 2.0
        assert scheduler.warmup_epochs == 15

    def test_linear_interpolation_precision(self):
        """Test linear interpolation precision."""
        scheduler = BetaScheduler(
            schedule="linear",
            initial_beta=0.0,
            final_beta=1.0,
            warmup_epochs=100,
        )

        for epoch in range(101):
            expected = epoch / 100.0
            actual = scheduler.step(epoch)
            assert abs(actual - expected) < 1e-10, f"Epoch {epoch}: expected {expected}, got {actual}"
