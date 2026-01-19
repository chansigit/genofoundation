"""
Unit tests for callback system.
"""

import pytest
from typing import Dict, Any, Optional

from trainer import (
    Callback,
    CallbackList,
    ProgressCallback,
    GradientMonitorCallback,
)


class TestCallback:
    """Test cases for base Callback class."""

    def test_default_methods_are_noop(self):
        """Test that default callback methods do nothing."""
        callback = Callback()

        # All these should not raise
        callback.on_train_start(None)
        callback.on_train_end(None)
        callback.on_epoch_start(None, 0)
        callback.on_epoch_end(None, 0, {})
        callback.on_batch_start(None, 0, None)
        callback.on_batch_end(None, 0, {})
        callback.on_validation_start(None)
        callback.on_validation_end(None, {})
        callback.on_checkpoint_save(None, "path")
        callback.on_checkpoint_load(None, "path")


class TestCallbackList:
    """Test cases for CallbackList class."""

    def test_empty_initialization(self):
        """Test CallbackList with no callbacks."""
        callback_list = CallbackList()

        assert callback_list.callbacks == []

    def test_initialization_with_callbacks(self, mock_callback):
        """Test CallbackList with initial callbacks."""
        callback_list = CallbackList([mock_callback])

        assert len(callback_list.callbacks) == 1
        assert callback_list.callbacks[0] is mock_callback

    def test_append_callback(self, mock_callback):
        """Test appending a callback."""
        callback_list = CallbackList()
        callback_list.append(mock_callback)

        assert len(callback_list.callbacks) == 1

    def test_on_train_start(self, mock_callback):
        """Test on_train_start is called for all callbacks."""
        callback_list = CallbackList([mock_callback])

        callback_list.on_train_start(None)

        assert mock_callback.train_start_called is True

    def test_on_train_end(self, mock_callback):
        """Test on_train_end is called for all callbacks."""
        callback_list = CallbackList([mock_callback])

        callback_list.on_train_end(None)

        assert mock_callback.train_end_called is True

    def test_on_epoch_start(self, mock_callback):
        """Test on_epoch_start is called for all callbacks."""
        callback_list = CallbackList([mock_callback])

        callback_list.on_epoch_start(None, 5)

        assert 5 in mock_callback.epoch_starts

    def test_on_epoch_end(self, mock_callback):
        """Test on_epoch_end is called for all callbacks."""
        callback_list = CallbackList([mock_callback])

        train_metrics = {'loss': 0.5}
        val_metrics = {'loss': 0.6}

        callback_list.on_epoch_end(None, 5, train_metrics, val_metrics)

        assert len(mock_callback.epoch_ends) == 1
        assert mock_callback.epoch_ends[0]['epoch'] == 5
        assert mock_callback.epoch_ends[0]['train_metrics'] == train_metrics
        assert mock_callback.epoch_ends[0]['val_metrics'] == val_metrics

    def test_on_batch_start(self, mock_callback):
        """Test on_batch_start is called for all callbacks."""
        callback_list = CallbackList([mock_callback])

        callback_list.on_batch_start(None, 10, None)

        assert 10 in mock_callback.batch_starts

    def test_on_batch_end(self, mock_callback):
        """Test on_batch_end is called for all callbacks."""
        callback_list = CallbackList([mock_callback])

        batch_metrics = {'loss': 0.5}
        callback_list.on_batch_end(None, 10, batch_metrics)

        assert len(mock_callback.batch_ends) == 1
        assert mock_callback.batch_ends[0]['batch_idx'] == 10
        assert mock_callback.batch_ends[0]['metrics'] == batch_metrics

    def test_on_validation_start(self, mock_callback):
        """Test on_validation_start is called for all callbacks."""
        callback_list = CallbackList([mock_callback])

        callback_list.on_validation_start(None)

        assert mock_callback.validation_starts == 1

    def test_on_validation_end(self, mock_callback):
        """Test on_validation_end is called for all callbacks."""
        callback_list = CallbackList([mock_callback])

        val_metrics = {'loss': 0.5}
        callback_list.on_validation_end(None, val_metrics)

        assert len(mock_callback.validation_ends) == 1
        assert mock_callback.validation_ends[0] == val_metrics

    def test_on_checkpoint_save(self, mock_callback):
        """Test on_checkpoint_save is called for all callbacks."""
        callback_list = CallbackList([mock_callback])

        callback_list.on_checkpoint_save(None, "/path/to/checkpoint.pt")

        assert "/path/to/checkpoint.pt" in mock_callback.checkpoint_saves

    def test_on_checkpoint_load(self, mock_callback):
        """Test on_checkpoint_load is called for all callbacks."""
        callback_list = CallbackList([mock_callback])

        callback_list.on_checkpoint_load(None, "/path/to/checkpoint.pt")

        assert "/path/to/checkpoint.pt" in mock_callback.checkpoint_loads

    def test_multiple_callbacks(self, mock_callback):
        """Test multiple callbacks are all called."""
        # Create a second mock callback inline
        class MockCallback2(Callback):
            def __init__(self):
                self.train_start_called = False

            def on_train_start(self, trainer):
                self.train_start_called = True

        callback2 = MockCallback2()
        callback_list = CallbackList([mock_callback, callback2])

        callback_list.on_train_start(None)

        assert mock_callback.train_start_called is True
        assert callback2.train_start_called is True


class TestProgressCallback:
    """Test cases for ProgressCallback."""

    def test_on_epoch_end_prints(self, capsys):
        """Test ProgressCallback prints progress."""

        class MockTrainer:
            class config:
                epochs = 10

        callback = ProgressCallback()

        train_metrics = {'loss': 0.5}
        val_metrics = {'loss': 0.6}

        callback.on_epoch_end(MockTrainer(), 5, train_metrics, val_metrics)

        captured = capsys.readouterr()
        assert "Epoch 6/10" in captured.out
        assert "Train Loss: 0.5000" in captured.out
        assert "Val Loss: 0.6000" in captured.out

    def test_on_epoch_end_without_val_metrics(self, capsys):
        """Test ProgressCallback without validation metrics."""

        class MockTrainer:
            class config:
                epochs = 10

        callback = ProgressCallback()

        train_metrics = {'loss': 0.5}

        callback.on_epoch_end(MockTrainer(), 5, train_metrics)

        captured = capsys.readouterr()
        assert "Epoch 6/10" in captured.out
        assert "Train Loss: 0.5000" in captured.out
        assert "Val Loss" not in captured.out


class TestGradientMonitorCallback:
    """Test cases for GradientMonitorCallback."""

    def test_initialization(self):
        """Test GradientMonitorCallback initialization."""
        callback = GradientMonitorCallback(log_every=50)

        assert callback.log_every == 50

    def test_skips_non_log_batches(self, capsys):
        """Test callback skips batches that aren't at log interval."""

        class MockTrainer:
            class model:
                @staticmethod
                def parameters():
                    return []

        callback = GradientMonitorCallback(log_every=100)

        callback.on_batch_end(MockTrainer(), 50, {})

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_logs_at_interval(self, capsys):
        """Test callback logs at correct intervals."""
        import torch

        class MockTrainer:
            global_step = 100

            class model:
                @staticmethod
                def parameters():
                    p = torch.nn.Parameter(torch.zeros(10))
                    p.grad = torch.ones(10)
                    return [p]

        callback = GradientMonitorCallback(log_every=100)

        callback.on_batch_end(MockTrainer(), 100, {})

        captured = capsys.readouterr()
        assert "Step 100" in captured.out
        assert "Gradient norm" in captured.out


class TestCustomCallback:
    """Test cases for custom callback implementation."""

    def test_custom_callback_integration(self):
        """Test a custom callback can be created and used."""

        class MetricAccumulator(Callback):
            def __init__(self):
                self.train_losses = []
                self.val_losses = []

            def on_epoch_end(self, trainer, epoch, train_metrics, val_metrics=None):
                self.train_losses.append(train_metrics.get('loss', 0))
                if val_metrics:
                    self.val_losses.append(val_metrics.get('loss', 0))

        callback = MetricAccumulator()
        callback_list = CallbackList([callback])

        # Simulate training loop
        for epoch in range(5):
            callback_list.on_epoch_end(
                None,
                epoch,
                {'loss': 1.0 - epoch * 0.1},
                {'loss': 1.1 - epoch * 0.1},
            )

        assert len(callback.train_losses) == 5
        assert len(callback.val_losses) == 5
        assert callback.train_losses[0] > callback.train_losses[-1]
