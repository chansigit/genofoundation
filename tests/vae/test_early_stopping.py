"""
Unit tests for EarlyStopping class.
"""

import pytest
from vae.trainer import EarlyStopping


class TestEarlyStopping:
    """Test cases for EarlyStopping"""
    
    def test_initialization(self):
        """Test EarlyStopping initialization"""
        es = EarlyStopping(patience=5, min_delta=0.01, mode="min")
        
        assert es.patience == 5
        assert es.min_delta == 0.01
        assert es.mode == "min"
        assert es.counter == 0
        assert es.best_score is None
        assert es.should_stop is False
        
    def test_first_score_min_mode(self):
        """Test first score handling in min mode"""
        es = EarlyStopping(patience=3, min_delta=0.01, mode="min")
        
        should_stop = es(1.0)
        
        assert es.best_score == 1.0
        assert es.counter == 0
        assert should_stop is False
        assert es.should_stop is False
        
    def test_first_score_max_mode(self):
        """Test first score handling in max mode"""
        es = EarlyStopping(patience=3, min_delta=0.01, mode="max")
        
        should_stop = es(0.8)
        
        assert es.best_score == 0.8
        assert es.counter == 0
        assert should_stop is False
        
    def test_improvement_min_mode(self):
        """Test improvement detection in min mode"""
        es = EarlyStopping(patience=3, min_delta=0.01, mode="min")
        
        # Initial score
        es(1.0)
        
        # Improvement (lower is better in min mode)
        should_stop = es(0.95)  # Improvement > min_delta
        
        assert es.best_score == 0.95
        assert es.counter == 0
        assert should_stop is False
        
    def test_improvement_max_mode(self):
        """Test improvement detection in max mode"""
        es = EarlyStopping(patience=3, min_delta=0.01, mode="max")
        
        # Initial score
        es(0.8)
        
        # Improvement (higher is better in max mode)
        should_stop = es(0.85)  # Improvement > min_delta
        
        assert es.best_score == 0.85
        assert es.counter == 0
        assert should_stop is False
        
    def test_no_improvement_min_mode(self):
        """Test no improvement detection in min mode"""
        es = EarlyStopping(patience=3, min_delta=0.01, mode="min")
        
        # Initial score
        es(1.0)
        
        # No significant improvement
        should_stop = es(1.005)  # Improvement < min_delta
        
        assert es.best_score == 1.0  # No update
        assert es.counter == 1
        assert should_stop is False
        
    def test_no_improvement_max_mode(self):
        """Test no improvement detection in max mode"""
        es = EarlyStopping(patience=3, min_delta=0.01, mode="max")
        
        # Initial score
        es(0.8)
        
        # No significant improvement  
        should_stop = es(0.805)  # Improvement < min_delta
        
        assert es.best_score == 0.8  # No update
        assert es.counter == 1
        assert should_stop is False
        
    def test_early_stopping_triggered(self):
        """Test early stopping trigger after patience exhausted"""
        es = EarlyStopping(patience=2, min_delta=0.01, mode="min")
        
        # Initial score
        es(1.0)
        
        # Two consecutive non-improvements
        es(1.005)  # counter = 1
        should_stop = es(1.002)  # counter = 2, should trigger
        
        assert es.counter == 2
        assert should_stop is True
        assert es.should_stop is True
        
    def test_patience_counting(self):
        """Test patience counter increments correctly"""
        es = EarlyStopping(patience=4, min_delta=0.01, mode="min")
        
        # Initial
        es(1.0)
        
        # Series of non-improvements
        es(1.005)  # counter = 1
        assert es.counter == 1 and not es.should_stop
        
        es(1.003)  # counter = 2
        assert es.counter == 2 and not es.should_stop
        
        es(1.001)  # counter = 3
        assert es.counter == 3 and not es.should_stop
        
        es(1.002)  # counter = 4, trigger
        assert es.counter == 4 and es.should_stop
        
    def test_patience_reset_on_improvement(self):
        """Test patience counter resets when improvement occurs"""
        es = EarlyStopping(patience=3, min_delta=0.01, mode="min")
        
        # Initial
        es(1.0)
        
        # Non-improvements
        es(1.005)  # counter = 1
        es(1.003)  # counter = 2
        
        # Improvement resets counter
        es(0.95)  # Significant improvement
        assert es.counter == 0
        assert es.best_score == 0.95
        assert not es.should_stop
        
        # Continue counting from reset
        es(0.955)  # counter = 1
        assert es.counter == 1
        
    def test_min_delta_boundary(self):
        """Test min_delta boundary conditions"""
        es = EarlyStopping(patience=2, min_delta=0.1, mode="min")
        
        es(1.0)  # Initial
        
        # Exactly at boundary (should count as no improvement)
        es(0.9)  # Improvement exactly equals min_delta
        assert es.counter == 1  # Should count as no improvement
        
        # Just over boundary (should count as improvement)
        es(0.89)  # Improvement > min_delta
        assert es.counter == 0  # Should reset
        assert es.best_score == 0.89
        
    def test_zero_min_delta(self):
        """Test behavior with zero min_delta"""
        es = EarlyStopping(patience=2, min_delta=0.0, mode="min")
        
        es(1.0)
        
        # Any improvement should reset counter
        es(0.999)  # Tiny improvement
        assert es.counter == 0
        assert es.best_score == 0.999
        
    def test_large_min_delta(self):
        """Test behavior with large min_delta"""
        es = EarlyStopping(patience=2, min_delta=0.5, mode="min")
        
        es(1.0)
        
        # Small improvement insufficient
        es(0.9)  # 0.1 improvement < 0.5 min_delta
        assert es.counter == 1
        
        # Large improvement sufficient
        es(0.4)  # 0.6 improvement > 0.5 min_delta  
        assert es.counter == 0
        assert es.best_score == 0.4
        
    def test_negative_scores(self):
        """Test handling of negative scores"""
        es = EarlyStopping(patience=2, min_delta=0.1, mode="min")
        
        es(-0.5)  # Negative initial score
        
        # Improvement in negative range
        es(-0.7)  # Better (more negative)
        assert es.counter == 0
        assert es.best_score == -0.7
        
        # No improvement 
        es(-0.65)  # Worse (less negative)
        assert es.counter == 1
        
    def test_equal_scores(self):
        """Test handling of equal scores"""
        es = EarlyStopping(patience=2, min_delta=0.01, mode="min")
        
        es(1.0)
        
        # Exactly equal score (no improvement)
        es(1.0)
        assert es.counter == 1
        
        es(1.0)  # Still equal
        assert es.counter == 2 and es.should_stop
        
    def test_max_mode_comprehensive(self):
        """Comprehensive test of max mode behavior"""
        es = EarlyStopping(patience=3, min_delta=0.05, mode="max")
        
        es(0.7)  # Initial
        
        # Improvement
        es(0.8)  # +0.1 > 0.05
        assert es.counter == 0 and es.best_score == 0.8
        
        # Insufficient improvement  
        es(0.82)  # +0.02 < 0.05
        assert es.counter == 1
        
        # Degradation
        es(0.75)  # -0.05 (much worse)
        assert es.counter == 2
        
        # Still no good improvement
        es(0.81)  # +0.01 < 0.05
        assert es.counter == 3 and es.should_stop