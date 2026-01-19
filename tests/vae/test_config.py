"""
Unit tests for configuration classes.
"""

import pytest
from trainer import TrainerConfig, DataConfig, ExperimentConfig, Config


class TestTrainerConfig:
    """Test cases for TrainerConfig."""

    def test_default_initialization(self):
        """Test TrainerConfig with default values."""
        config = TrainerConfig()

        assert config.epochs == 100
        assert config.batch_size == 128
        assert config.learning_rate == 1e-3
        assert config.weight_decay == 1e-5
        assert config.beta == 1.0
        assert config.optimizer == "adamw"
        assert config.scheduler == "cosine"
        assert config.mixed_precision == "no"
        assert config.grad_clip_norm == 50.0
        assert config.record_grad_norm is True

    def test_custom_initialization(self):
        """Test TrainerConfig with custom values."""
        config = TrainerConfig(
            epochs=50,
            batch_size=64,
            learning_rate=5e-4,
            optimizer="adam",
            scheduler="plateau",
        )

        assert config.epochs == 50
        assert config.batch_size == 64
        assert config.learning_rate == 5e-4
        assert config.optimizer == "adam"
        assert config.scheduler == "plateau"

    def test_beta_schedule_options(self):
        """Test valid beta schedule options."""
        for schedule in ["linear", "cyclical", "constant", "sigmoid"]:
            config = TrainerConfig(beta_schedule=schedule)
            assert config.beta_schedule == schedule

    def test_invalid_beta_schedule(self):
        """Test invalid beta schedule raises error."""
        with pytest.raises(ValueError, match="beta_schedule must be one of"):
            TrainerConfig(beta_schedule="invalid")

    def test_optimizer_options(self):
        """Test valid optimizer options."""
        for optimizer in ["adam", "adamw", "sgd"]:
            config = TrainerConfig(optimizer=optimizer)
            assert config.optimizer == optimizer

    def test_invalid_optimizer(self):
        """Test invalid optimizer raises error."""
        with pytest.raises(ValueError, match="optimizer must be one of"):
            TrainerConfig(optimizer="invalid")

    def test_scheduler_options(self):
        """Test valid scheduler options."""
        for scheduler in ["cosine", "plateau", "onecycle", "warmup_cosine", "warmup", None]:
            config = TrainerConfig(scheduler=scheduler)
            assert config.scheduler == scheduler

    def test_invalid_scheduler(self):
        """Test invalid scheduler raises error."""
        with pytest.raises(ValueError, match="scheduler must be one of"):
            TrainerConfig(scheduler="invalid")

    def test_mixed_precision_options(self):
        """Test valid mixed precision options."""
        for mp in ["no", "fp16", "bf16"]:
            config = TrainerConfig(mixed_precision=mp)
            assert config.mixed_precision == mp

    def test_invalid_mixed_precision(self):
        """Test invalid mixed precision raises error."""
        with pytest.raises(ValueError, match="mixed_precision must be one of"):
            TrainerConfig(mixed_precision="fp32")

    def test_wandb_mode_options(self):
        """Test valid wandb mode options."""
        for mode in ["online", "offline", "disabled"]:
            config = TrainerConfig(wandb_mode=mode)
            assert config.wandb_mode == mode

    def test_invalid_wandb_mode(self):
        """Test invalid wandb mode raises error."""
        with pytest.raises(ValueError, match="wandb_mode must be one of"):
            TrainerConfig(wandb_mode="invalid")

    def test_wandb_config_options(self):
        """Test wandb configuration options."""
        config = TrainerConfig(
            wandb_mode="offline",
            wandb_project="test_project",
            wandb_entity="test_entity",
            wandb_run_name="test_run",
            wandb_tags=["tag1", "tag2"],
            wandb_notes="Test notes",
            wandb_dir="/tmp/wandb",
        )
        assert config.wandb_mode == "offline"
        assert config.wandb_project == "test_project"
        assert config.wandb_entity == "test_entity"
        assert config.wandb_run_name == "test_run"
        assert config.wandb_tags == ["tag1", "tag2"]
        assert config.wandb_notes == "Test notes"
        assert config.wandb_dir == "/tmp/wandb"

    def test_negative_epochs_raises_error(self):
        """Test negative epochs raises error."""
        with pytest.raises(ValueError, match="epochs must be positive"):
            TrainerConfig(epochs=-1)

    def test_zero_epochs_raises_error(self):
        """Test zero epochs raises error."""
        with pytest.raises(ValueError, match="epochs must be positive"):
            TrainerConfig(epochs=0)

    def test_negative_batch_size_raises_error(self):
        """Test negative batch_size raises error."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TrainerConfig(batch_size=-1)

    def test_negative_learning_rate_raises_error(self):
        """Test negative learning_rate raises error."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TrainerConfig(learning_rate=-0.001)

    def test_negative_weight_decay_raises_error(self):
        """Test negative weight_decay raises error."""
        with pytest.raises(ValueError, match="weight_decay must be non-negative"):
            TrainerConfig(weight_decay=-0.01)

    def test_negative_beta_raises_error(self):
        """Test negative beta raises error."""
        with pytest.raises(ValueError, match="beta must be non-negative"):
            TrainerConfig(beta=-1.0)

    def test_invalid_gradient_accumulation_raises_error(self):
        """Test gradient_accumulation_steps < 1 raises error."""
        with pytest.raises(ValueError, match="gradient_accumulation_steps must be >= 1"):
            TrainerConfig(gradient_accumulation_steps=0)

    def test_gradient_settings(self):
        """Test gradient-related configuration options."""
        config = TrainerConfig(
            grad_clip_norm=10.0,
            gradient_accumulation_steps=4,
            record_grad_norm=False,
        )
        assert config.grad_clip_norm == 10.0
        assert config.gradient_accumulation_steps == 4
        assert config.record_grad_norm is False

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = TrainerConfig(epochs=50, batch_size=64)
        d = config.to_dict()

        assert isinstance(d, dict)
        assert d['epochs'] == 50
        assert d['batch_size'] == 64
        assert 'learning_rate' in d
        assert 'optimizer' in d
        assert 'grad_clip_norm' in d
        assert 'record_grad_norm' in d
        assert d['record_grad_norm'] is True

    def test_betas_conversion(self):
        """Test betas list is properly handled."""
        config = TrainerConfig(betas=[0.9, 0.99])
        assert config.betas == [0.9, 0.99]
        assert isinstance(config.betas, list)


class TestDataConfig:
    """Test cases for DataConfig."""

    def test_default_initialization(self):
        """Test DataConfig with default values."""
        config = DataConfig()

        assert config.train_path is None
        assert config.val_path is None
        assert config.num_workers == 4
        assert config.pin_memory is True
        assert config.val_split == 0.1
        assert config.normalize is True

    def test_custom_initialization(self):
        """Test DataConfig with custom values."""
        config = DataConfig(
            train_path="/path/to/train",
            val_path="/path/to/val",
            num_workers=8,
            val_split=0.2,
        )

        assert config.train_path == "/path/to/train"
        assert config.val_path == "/path/to/val"
        assert config.num_workers == 8
        assert config.val_split == 0.2


class TestExperimentConfig:
    """Test cases for ExperimentConfig."""

    def test_default_initialization(self):
        """Test ExperimentConfig with default values."""
        config = ExperimentConfig()

        assert config.name == "vae_training"
        assert config.seed == 42
        assert config.output_dir == "./outputs"

    def test_custom_initialization(self):
        """Test ExperimentConfig with custom values."""
        config = ExperimentConfig(
            name="my_experiment",
            seed=123,
            output_dir="/custom/output",
        )

        assert config.name == "my_experiment"
        assert config.seed == 123
        assert config.output_dir == "/custom/output"


class TestConfig:
    """Test cases for root Config class."""

    def test_default_initialization(self):
        """Test Config with default sub-configs."""
        config = Config()

        assert isinstance(config.experiment, ExperimentConfig)
        assert isinstance(config.trainer, TrainerConfig)
        assert isinstance(config.data, DataConfig)

    def test_custom_sub_configs(self):
        """Test Config with custom sub-configs."""
        experiment = ExperimentConfig(name="test")
        trainer = TrainerConfig(epochs=50)
        data = DataConfig(num_workers=2)

        config = Config(
            experiment=experiment,
            trainer=trainer,
            data=data,
        )

        assert config.experiment.name == "test"
        assert config.trainer.epochs == 50
        assert config.data.num_workers == 2
