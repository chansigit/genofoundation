"""
Unit tests for EarlyStopping class.
"""

import pytest
from trainer import EarlyStopping, TrainerConfig


class TestEarlyStopping:
    """Test cases for EarlyStopping."""

    def test_initialization(self):
        """Test EarlyStopping initialization."""
        es = EarlyStopping(patience=5, min_delta=0.01, mode="min")

        assert es.patience == 5
        assert es.min_delta == 0.01
        assert es.mode == "min"
        assert es.counter == 0
        assert es.best_score is None
        assert es.should_stop is False

    def test_invalid_patience_raises_error(self):
        """Test invalid patience raises error."""
        with pytest.raises(ValueError, match="patience must be positive"):
            EarlyStopping(patience=0)

        with pytest.raises(ValueError, match="patience must be positive"):
            EarlyStopping(patience=-1)

    def test_invalid_mode_raises_error(self):
        """Test invalid mode raises error."""
        with pytest.raises(ValueError, match="mode must be 'min' or 'max'"):
            EarlyStopping(mode="invalid")

    def test_first_score_min_mode(self):
        """Test first score handling in min mode."""
        es = EarlyStopping(patience=3, min_delta=0.01, mode="min")

        should_stop = es(1.0)

        assert es.best_score == 1.0
        assert es.counter == 0
        assert should_stop is False

    def test_first_score_max_mode(self):
        """Test first score handling in max mode."""
        es = EarlyStopping(patience=3, min_delta=0.01, mode="max")

        should_stop = es(0.8)

        assert es.best_score == 0.8
        assert es.counter == 0
        assert should_stop is False

    def test_improvement_min_mode(self):
        """Test improvement detection in min mode."""
        es = EarlyStopping(patience=3, min_delta=0.01, mode="min")

        es(1.0)
        should_stop = es(0.95)  # Improvement > min_delta

        assert es.best_score == 0.95
        assert es.counter == 0
        assert should_stop is False

    def test_improvement_max_mode(self):
        """Test improvement detection in max mode."""
        es = EarlyStopping(patience=3, min_delta=0.01, mode="max")

        es(0.8)
        should_stop = es(0.85)  # Improvement > min_delta

        assert es.best_score == 0.85
        assert es.counter == 0
        assert should_stop is False

    def test_no_improvement_min_mode(self):
        """Test no improvement detection in min mode."""
        es = EarlyStopping(patience=3, min_delta=0.01, mode="min")

        es(1.0)
        should_stop = es(1.005)  # Improvement < min_delta

        assert es.best_score == 1.0
        assert es.counter == 1
        assert should_stop is False

    def test_no_improvement_max_mode(self):
        """Test no improvement detection in max mode."""
        es = EarlyStopping(patience=3, min_delta=0.01, mode="max")

        es(0.8)
        should_stop = es(0.805)  # Improvement < min_delta

        assert es.best_score == 0.8
        assert es.counter == 1
        assert should_stop is False

    def test_early_stopping_triggered(self):
        """Test early stopping trigger after patience exhausted."""
        es = EarlyStopping(patience=2, min_delta=0.01, mode="min")

        es(1.0)
        es(1.005)  # counter = 1
        should_stop = es(1.002)  # counter = 2, should trigger

        assert es.counter == 2
        assert should_stop is True
        assert es.should_stop is True

    def test_patience_counting(self):
        """Test patience counter increments correctly."""
        es = EarlyStopping(patience=4, min_delta=0.01, mode="min")

        es(1.0)

        es(1.005)  # counter = 1
        assert es.counter == 1 and not es.should_stop

        es(1.003)  # counter = 2
        assert es.counter == 2 and not es.should_stop

        es(1.001)  # counter = 3
        assert es.counter == 3 and not es.should_stop

        es(1.002)  # counter = 4, trigger
        assert es.counter == 4 and es.should_stop

    def test_patience_reset_on_improvement(self):
        """Test patience counter resets when improvement occurs."""
        es = EarlyStopping(patience=3, min_delta=0.01, mode="min")

        es(1.0)
        es(1.005)  # counter = 1
        es(1.003)  # counter = 2

        es(0.95)  # Significant improvement
        assert es.counter == 0
        assert es.best_score == 0.95
        assert not es.should_stop

        es(0.955)  # counter = 1
        assert es.counter == 1

    def test_min_delta_boundary(self):
        """Test min_delta boundary conditions."""
        es = EarlyStopping(patience=2, min_delta=0.1, mode="min")

        es(1.0)

        es(0.9)  # Improvement exactly equals min_delta
        assert es.counter == 1  # Should count as no improvement

        es(0.89)  # Improvement > min_delta
        assert es.counter == 0
        assert es.best_score == 0.89

    def test_zero_min_delta(self):
        """Test behavior with zero min_delta."""
        es = EarlyStopping(patience=2, min_delta=0.0, mode="min")

        es(1.0)

        es(0.999)  # Tiny improvement
        assert es.counter == 0
        assert es.best_score == 0.999

    def test_large_min_delta(self):
        """Test behavior with large min_delta."""
        es = EarlyStopping(patience=2, min_delta=0.5, mode="min")

        es(1.0)

        es(0.9)  # 0.1 improvement < 0.5 min_delta
        assert es.counter == 1

        es(0.4)  # 0.6 improvement > 0.5 min_delta
        assert es.counter == 0
        assert es.best_score == 0.4

    def test_negative_scores(self):
        """Test handling of negative scores."""
        es = EarlyStopping(patience=2, min_delta=0.1, mode="min")

        es(-0.5)

        es(-0.7)  # Better (more negative)
        assert es.counter == 0
        assert es.best_score == -0.7

        es(-0.65)  # Worse (less negative)
        assert es.counter == 1

    def test_equal_scores(self):
        """Test handling of equal scores."""
        es = EarlyStopping(patience=2, min_delta=0.01, mode="min")

        es(1.0)

        es(1.0)  # Equal score
        assert es.counter == 1

        es(1.0)  # Still equal
        assert es.counter == 2 and es.should_stop

    def test_max_mode_comprehensive(self):
        """Comprehensive test of max mode behavior."""
        es = EarlyStopping(patience=3, min_delta=0.05, mode="max")

        es(0.7)  # Initial

        es(0.8)  # +0.1 > 0.05
        assert es.counter == 0 and es.best_score == 0.8

        es(0.82)  # +0.02 < 0.05
        assert es.counter == 1

        es(0.75)  # Much worse
        assert es.counter == 2

        es(0.81)  # +0.01 < 0.05
        assert es.counter == 3 and es.should_stop

    def test_reset(self):
        """Test reset method."""
        es = EarlyStopping(patience=3, min_delta=0.01, mode="min")

        es(1.0)
        es(1.1)
        es(1.2)

        es.reset()

        assert es.counter == 0
        assert es.best_score is None
        assert es.should_stop is False

    def test_get_state(self):
        """Test get_state method for checkpointing."""
        es = EarlyStopping(patience=5, min_delta=0.01, mode="min")

        es(1.0)
        es(1.1)  # counter = 1

        state = es.get_state()

        assert state['counter'] == 1
        assert state['best_score'] == 1.0
        assert state['should_stop'] is False
        assert state['patience'] == 5
        assert state['min_delta'] == 0.01
        assert state['mode'] == "min"

    def test_load_state(self):
        """Test load_state method for checkpoint restoration."""
        es = EarlyStopping(patience=5, min_delta=0.01, mode="min")

        state = {
            'counter': 3,
            'best_score': 0.5,
            'should_stop': False,
        }

        es.load_state(state)

        assert es.counter == 3
        assert es.best_score == 0.5
        assert es.should_stop is False

    def test_from_config_enabled(self):
        """Test creating EarlyStopping from config when enabled."""
        config = TrainerConfig(
            early_stopping=True,
            patience=10,
            min_delta=0.001,
        )

        es = EarlyStopping.from_config(config)

        assert es is not None
        assert es.patience == 10
        assert es.min_delta == 0.001
        assert es.mode == "min"

    def test_from_config_disabled(self):
        """Test creating EarlyStopping from config when disabled."""
        config = TrainerConfig(early_stopping=False)

        es = EarlyStopping.from_config(config)

        assert es is None
