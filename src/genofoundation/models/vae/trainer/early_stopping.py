"""
Early stopping handler for training.
"""

from typing import Optional

from .config import TrainingConfig


class EarlyStopping:
    """Early stopping handler.

    Monitors a metric and signals when training should stop if no improvement
    is seen for a specified number of epochs (patience).

    Args:
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change to qualify as an improvement
        mode: 'min' for metrics that should decrease, 'max' for metrics that should increase
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
    ):
        if patience <= 0:
            raise ValueError(f"patience must be positive, got {patience}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop based on the given score.

        Args:
            score: The metric value to evaluate

        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def _is_improvement(self, score: float) -> bool:
        """Check if the score is an improvement over the best score."""
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def get_state(self) -> dict:
        """Get early stopping state for checkpointing."""
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'should_stop': self.should_stop,
            'patience': self.patience,
            'min_delta': self.min_delta,
            'mode': self.mode,
        }

    def load_state(self, state: dict):
        """Load early stopping state from checkpoint."""
        self.counter = state.get('counter', 0)
        self.best_score = state.get('best_score')
        self.should_stop = state.get('should_stop', False)

    @classmethod
    def from_config(cls, config: TrainingConfig) -> Optional['EarlyStopping']:
        """Create an EarlyStopping instance from TrainingConfig.

        Returns None if early stopping is disabled in config.
        """
        if not config.early_stopping:
            return None

        return cls(
            patience=config.patience,
            min_delta=config.min_delta,
            mode="min",
        )
