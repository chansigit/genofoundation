"""
Train Vanilla VAE on TMS dataset.

Usage:
    # Full dataset (341k train, 14k test)
    python train_vanillavae.py

    # Mini dataset (20k train, 14k test)
    python train_vanillavae.py data=tms_mini

    # Override parameters
    python train_vanillavae.py data=tms_mini trainer.epochs=100 trainer.batch_size=128

    # Enable wandb logging
    python train_vanillavae.py trainer.wandb_mode=online
"""

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from genofoundation.models.vae.vanilla_vae import SimpleVAE
from genofoundation.models.vae.trainer import VAETrainer, TrainerConfig


@hydra.main(
    config_path="configs",
    config_name="vanillavae_tms",
    version_base=None,
)
def main(cfg: DictConfig):
    print("=" * 60)
    print("Vanilla VAE Training")
    print("=" * 60)

    # Print config
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))

    # Create output directory
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load preprocessed dataset
    data_path = Path(cfg.data.train_path)
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False)

    X_train = data['X_train']
    X_test = data['X_test']
    n_features = data['n_features']
    n_classes = data['n_classes']

    print(f"\nDataset:")
    print(f"  Train samples: {X_train.shape[0]:,}")
    print(f"  Test samples:  {X_test.shape[0]:,}")
    print(f"  Features:      {n_features:,}")
    print(f"  Classes:       {n_classes}")

    # Create datasets
    train_dataset = TensorDataset(X_train)
    val_dataset = TensorDataset(X_test)

    # Create TrainerConfig from hydra config
    trainer_cfg = TrainerConfig(**OmegaConf.to_container(cfg.trainer, resolve=True))

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=trainer_cfg.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=trainer_cfg.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    # Create model
    model = SimpleVAE(input_dim=n_features, latent_dim=cfg.model.latent_dim)
    print(f"\nModel:")
    print(f"  Type:       SimpleVAE")
    print(f"  Input dim:  {n_features}")
    print(f"  Latent dim: {cfg.model.latent_dim}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create trainer
    trainer = VAETrainer(
        model=model,
        config=trainer_cfg,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=str(output_dir),
        seed=cfg.experiment.seed,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    train_losses, val_losses = trainer.train()

    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Final train loss: {train_losses[-1]:.4f}")
    if val_losses:
        print(f"Final val loss:   {val_losses[-1]:.4f}")
        print(f"Best val loss:    {min(val_losses):.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
