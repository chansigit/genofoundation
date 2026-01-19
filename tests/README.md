# VAE Trainer Test Suite

This directory contains comprehensive unit tests and integration tests for the VAE trainer module.

## Test Files Structure

```
tests/
├── __init__.py                      # Test package marker
├── README.md                        # This file
└── vae/
    ├── __init__.py                  # VAE test package marker
    ├── conftest.py                  # Test fixtures and shared utilities
    ├── test_config.py               # Configuration classes tests
    ├── test_early_stopping.py       # EarlyStopping component tests
    ├── test_beta_scheduler.py       # BetaScheduler component tests
    ├── test_lr_schedulers.py        # LR scheduler creation tests
    ├── test_callbacks.py            # Callback system tests
    ├── test_checkpointing.py        # Checkpoint save/load tests
    └── test_trainer_core.py         # VAETrainer integration tests
```

## Test Coverage

### **Configuration Tests** (`test_config.py`)
- TrainerConfig initialization and validation
- DataConfig and ExperimentConfig
- Invalid parameter handling

### **Early Stopping Tests** (`test_early_stopping.py`)
- Patience logic and counter behavior
- Min/max mode handling
- Improvement detection with min_delta
- State save/restore for checkpointing

### **Beta Scheduler Tests** (`test_beta_scheduler.py`)
- All schedule types: linear, cyclical, sigmoid, constant
- Warmup behavior
- State save/restore for checkpointing
- Edge cases and boundary conditions

### **LR Scheduler Tests** (`test_lr_schedulers.py`)
- Scheduler creation: cosine, plateau, onecycle, warmup, warmup_cosine
- Fast-forward functionality for resume
- LR decay behavior

### **Callback Tests** (`test_callbacks.py`)
- Base Callback class
- CallbackList management
- ProgressCallback and GradientMonitorCallback
- Custom callback implementation

### **Checkpointing Tests** (`test_checkpointing.py`)
- CheckpointManager save/load operations
- Checkpoint rotation
- State restoration
- Error handling

### **Trainer Core Tests** (`test_trainer_core.py`)
- VAETrainer initialization
- Optimizer creation
- Batch unpacking
- Training loop
- Checkpoint integration
- Callback integration

## How to Run Tests

### **Option 1: Run All Tests with Pytest**

```bash
# Install pytest if needed
pip install pytest

# Run all tests
python -m pytest tests/vae/ -v

# Run with coverage (if pytest-cov installed)
pip install pytest-cov
python -m pytest tests/vae/ --cov=trainer --cov-report=html
```

### **Option 2: Run Specific Test File**

```bash
# Test configuration
python -m pytest tests/vae/test_config.py -v

# Test early stopping
python -m pytest tests/vae/test_early_stopping.py -v

# Test beta scheduler
python -m pytest tests/vae/test_beta_scheduler.py -v

# Test trainer core
python -m pytest tests/vae/test_trainer_core.py -v
```

### **Option 3: Run Specific Test Class or Method**

```bash
# Run specific test class
python -m pytest tests/vae/test_config.py::TestTrainerConfig -v

# Run specific test method
python -m pytest tests/vae/test_early_stopping.py::TestEarlyStopping::test_patience_counting -v
```

### **Option 4: Quick Verification**

```bash
# Quick verification of core components
python -c "
import sys
import os
sys.path.insert(0, 'src/genofoundation/models/vae')

from trainer import BetaScheduler, EarlyStopping, TrainerConfig

print('Testing core components...')

# BetaScheduler
bs = BetaScheduler('linear', 0.0, 1.0, 10)
assert bs.step(0) == 0.0 and bs.step(10) == 1.0
print('✓ BetaScheduler OK')

# EarlyStopping
es = EarlyStopping(patience=2, min_delta=0.1)
es(1.0)
es(0.85)  # Improvement > min_delta
assert es.counter == 0
es(0.86)  # No improvement
assert es.counter == 1
print('✓ EarlyStopping OK')

# TrainerConfig
config = TrainerConfig(epochs=50, batch_size=64)
assert config.epochs == 50
print('✓ TrainerConfig OK')

print('All core tests passed!')
"
```

## Test Fixtures

The `conftest.py` file provides common fixtures:

- `temp_dir`: Temporary directory for test artifacts
- `simple_vae`, `small_vae`: VAE models for testing
- `train_dataset`, `val_dataset`: Test datasets
- `train_loader`, `val_loader`: Data loaders
- `basic_config`, `fast_config`: Trainer configurations
- `beta_scheduler`, `early_stopping`: Scheduler instances
- `checkpoint_manager`: Checkpoint management
- `mock_callback`: Mock callback for testing

## Requirements

- Python 3.8+
- PyTorch 2.0+
- pytest (for running tests)
- accelerate (for trainer tests)
- omegaconf (for config tests)

## Notes

- Tests use CPU by default for compatibility
- Integration tests may take longer due to actual training loops
- Some tests require `accelerate` library for distributed training features
