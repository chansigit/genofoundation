"""
Train Conv VAE on TMS dataset.

Usage:
    # Full dataset (341k train, 14k test)
    python train_convvae.py

    # Mini dataset (20k train, 14k test)
    python train_convvae.py 'data.train_path=${hydra:runtime.cwd}/../../data/tms/tms_preprocessed-mini.pt'

    # Override parameters
    python train_convvae.py trainer.epochs=100 trainer.batch_size=128

    # Enable wandb logging
    python train_convvae.py trainer.wandb_mode=online
"""

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from genofoundation.models.vae.conv_vae import ConvVAE
from genofoundation.models.vae.trainer import VAETrainer, TrainerConfig


@hydra.main(
    config_path="configs",
    config_name="convvae_tms",
    version_base=None,
)
def main(cfg: DictConfig):
    print("=" * 60)
    print("Conv VAE Training")
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

    # Load pixel indices
    px_ind_path = Path(cfg.data.px_ind_path)
    print(f"Loading pixel indices from {px_ind_path}...")
    px_ind = torch.load(px_ind_path, weights_only=False)

    print(f"\nDataset:")
    print(f"  Train samples: {X_train.shape[0]:,}")
    print(f"  Test samples:  {X_test.shape[0]:,}")
    print(f"  Features:      {n_features:,}")
    print(f"  Classes:       {n_classes}")
    print(f"  Pixel indices: {px_ind.shape[0]:,}")

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
    model = ConvVAE(
        px_ind=px_ind,
        img_h=cfg.model.img_h,
        img_w=cfg.model.img_w,
        latent_dim=cfg.model.latent_dim,
        base_channels=cfg.model.base_channels,
        num_res_blocks=cfg.model.num_res_blocks,
        max_channels=cfg.model.max_channels,
        groups=cfg.model.groups,
        up_mode=cfg.model.up_mode,
    )

    num_params = sum(p.numel() for p in model.parameters())
    num_params_m = num_params / 1e6

    print(f"\nModel:")
    print(f"  Type:           Conv VAE")
    print(f"  Image size:     {cfg.model.img_h}x{cfg.model.img_w}")
    print(f"  Latent dim:     {cfg.model.latent_dim}")
    print(f"  Base channels:  {cfg.model.base_channels}")
    print(f"  Num res blocks: {cfg.model.num_res_blocks}")
    print(f"  Max channels:   {cfg.model.max_channels}")
    print(f"  Groups:         {cfg.model.groups}")
    print(f"  Up mode:        {cfg.model.up_mode}")
    print(f"  Parameters:     {num_params_m:.2f}M")

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
