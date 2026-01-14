"""
Unit tests for BetaScheduler class.
"""

import pytest
import numpy as np
import math
from vae.trainer import BetaScheduler


class TestBetaScheduler:
    """Test cases for BetaScheduler"""
    
    def test_constant_schedule(self):
        """Test constant beta schedule"""
        scheduler = BetaScheduler(
            schedule="constant",
            initial_beta=0.0,
            final_beta=1.5,
            warmup_epochs=10
        )
        
        # Should always return final_beta regardless of epoch
        assert scheduler.step(0) == 1.5
        assert scheduler.step(5) == 1.5
        assert scheduler.step(10) == 1.5
        assert scheduler.step(100) == 1.5
        
    def test_linear_schedule_warmup(self):
        """Test linear schedule during warmup period"""
        scheduler = BetaScheduler(
            schedule="linear",
            initial_beta=0.0,
            final_beta=2.0,
            warmup_epochs=10
        )
        
        # Test specific points during warmup
        assert scheduler.step(0) == 0.0
        assert scheduler.step(5) == 1.0  # Halfway through warmup
        assert scheduler.step(10) == 2.0  # End of warmup
        
        # Test precise interpolation
        expected_at_3 = 0.0 + (3/10) * 2.0  # 0.6
        assert abs(scheduler.step(3) - expected_at_3) < 1e-6
        
    def test_linear_schedule_post_warmup(self):
        """Test linear schedule after warmup completion"""
        scheduler = BetaScheduler(
            schedule="linear",
            initial_beta=0.0,
            final_beta=1.0,
            warmup_epochs=5
        )
        
        # After warmup, should stay at final_beta
        assert scheduler.step(5) == 1.0
        assert scheduler.step(10) == 1.0
        assert scheduler.step(100) == 1.0
        
    def test_cyclical_schedule(self):
        """Test cyclical beta schedule"""
        scheduler = BetaScheduler(
            schedule="cyclical",
            initial_beta=0.0,
            final_beta=1.0,
            warmup_epochs=10,  # Not used in cyclical
            cycle_epochs=4
        )
        
        # Test first cycle (epochs 0-3)
        assert scheduler.step(0) == 0.0  # Start of cycle
        assert abs(scheduler.step(2) - 0.5) < 1e-6  # Mid cycle
        assert scheduler.step(3) == 1.0  # End of cycle (but capped)
        
        # Test second cycle (epochs 4-7)  
        assert scheduler.step(4) == 0.0  # Start of new cycle
        assert abs(scheduler.step(6) - 0.5) < 1e-6  # Mid second cycle
        
    def test_sigmoid_schedule_warmup(self):
        """Test sigmoid schedule during warmup"""
        scheduler = BetaScheduler(
            schedule="sigmoid",
            initial_beta=0.0,
            final_beta=1.0,
            warmup_epochs=10
        )
        
        # Test sigmoid properties
        beta_start = scheduler.step(0)
        beta_mid = scheduler.step(5)
        beta_end = scheduler.step(10)
        
        # Should be monotonically increasing
        assert beta_start < beta_mid < beta_end
        
        # Should approach final_beta at the end
        assert abs(beta_end - 1.0) < 0.1
        
    def test_sigmoid_schedule_post_warmup(self):
        """Test sigmoid schedule after warmup"""
        scheduler = BetaScheduler(
            schedule="sigmoid",
            initial_beta=0.0,
            final_beta=2.0,
            warmup_epochs=5
        )
        
        # After warmup, should stay at final_beta
        assert scheduler.step(10) == 2.0
        assert scheduler.step(100) == 2.0
        
    def test_unknown_schedule(self):
        """Test unknown schedule defaults to constant"""
        scheduler = BetaScheduler(
            schedule="unknown_schedule",
            initial_beta=0.5,
            final_beta=1.0,
            warmup_epochs=5
        )
        
        # Unknown schedule should not change current_beta from initial
        initial_beta = scheduler.current_beta
        scheduler.step(0)
        assert scheduler.current_beta == initial_beta
        
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Zero warmup epochs
        scheduler = BetaScheduler(
            schedule="linear",
            initial_beta=0.0,
            final_beta=1.0,
            warmup_epochs=0
        )
        assert scheduler.step(0) == 1.0  # Should immediately be at final
        
        # Single warmup epoch
        scheduler = BetaScheduler(
            schedule="linear", 
            initial_beta=0.0,
            final_beta=1.0,
            warmup_epochs=1
        )
        assert scheduler.step(0) == 0.0
        assert scheduler.step(1) == 1.0
        
    def test_negative_epoch(self):
        """Test behavior with negative epochs (edge case)"""
        scheduler = BetaScheduler(
            schedule="linear",
            initial_beta=0.0,
            final_beta=1.0,
            warmup_epochs=5
        )
        
        # Should handle gracefully (likely clamp to 0)
        beta = scheduler.step(-1)
        assert beta >= 0.0
        
    def test_current_beta_tracking(self):
        """Test that current_beta is properly tracked"""
        scheduler = BetaScheduler(
            schedule="linear",
            initial_beta=0.2,
            final_beta=1.8,
            warmup_epochs=4
        )
        
        # Initial state
        assert scheduler.current_beta == 0.2
        
        # After stepping
        beta_2 = scheduler.step(2)
        assert scheduler.current_beta == beta_2
        
        beta_4 = scheduler.step(4) 
        assert scheduler.current_beta == beta_4
        assert beta_4 == 1.8
        
    def test_multiple_steps_consistency(self):
        """Test that multiple calls to step() with same epoch give same result"""
        scheduler = BetaScheduler(
            schedule="linear",
            initial_beta=0.0,
            final_beta=1.0,
            warmup_epochs=10
        )
        
        # Multiple calls should be consistent
        beta1 = scheduler.step(5)
        beta2 = scheduler.step(5) 
        beta3 = scheduler.step(5)
        
        assert beta1 == beta2 == beta3
        
    def test_schedule_parameters_preserved(self):
        """Test that schedule parameters are preserved correctly"""
        scheduler = BetaScheduler(
            schedule="cyclical",
            initial_beta=0.1,
            final_beta=2.5,
            warmup_epochs=8,
            cycle_epochs=6
        )
        
        assert scheduler.schedule == "cyclical"
        assert scheduler.initial_beta == 0.1
        assert scheduler.final_beta == 2.5
        assert scheduler.warmup_epochs == 8
        assert scheduler.cycle_epochs == 6
        
    def test_cyclical_multiple_cycles(self):
        """Test cyclical schedule across multiple cycles"""
        scheduler = BetaScheduler(
            schedule="cyclical",
            initial_beta=0.0,
            final_beta=1.0,
            cycle_epochs=3
        )
        
        # Test pattern repeats across cycles
        cycle1_values = [scheduler.step(i) for i in range(3)]
        cycle2_values = [scheduler.step(i + 3) for i in range(3)]
        cycle3_values = [scheduler.step(i + 6) for i in range(3)]
        
        np.testing.assert_array_almost_equal(cycle1_values, cycle2_values)
        np.testing.assert_array_almost_equal(cycle2_values, cycle3_values)