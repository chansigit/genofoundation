# VAE Training Experiments

## Quick Start

```bash
cd experiments/vae

# Train on full TMS dataset (341k train, 14k test)
python train_vanillavae.py

# Train on mini TMS dataset (20k train, 14k test)
python train_vanillavae.py 'data.train_path=${hydra:runtime.cwd}/../../data/tms/tms_preprocessed-mini.pt'
```

## Dataset Options

| Dataset | Train Size | Test Size | Path |
|---------|------------|-----------|------|
| Full (default) | 341,704 | 14,509 | `data/tms/tms_preprocessed.pt` |
| Mini | 20,000 | 14,509 | `data/tms/tms_preprocessed-mini.pt` |

## Common Overrides

```bash
# Adjust training parameters
python train_vanillavae.py trainer.epochs=100 trainer.batch_size=512 trainer.learning_rate=0.0005

# Enable wandb logging
python train_vanillavae.py trainer.wandb_mode=online trainer.wandb_project=my-project

# Change model architecture
python train_vanillavae.py model.latent_dim=64

# Change output directory
python train_vanillavae.py experiment.output_dir=../outputs/my_experiment
```

## Hyperparameter Search

```bash
# Full dataset
./run-hparam-scan-vanillavae-tms.sh

# Mini dataset (faster iteration)
./run-hparam-scan-vanillavae-tms.sh mini
```

## Config Structure

```
configs/
└── vanillavae_tms.yaml    # Main config (model, data, trainer settings)
```

## Output

Checkpoints and logs are saved to `outputs/vae_tms/` (configurable via `experiment.output_dir`).
