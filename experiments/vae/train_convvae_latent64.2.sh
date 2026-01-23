#!/bin/bash
# ConvVAE training with latent_dim=64 and num_workers=16
# Usage: ./train_convvae_latent64.sh [additional overrides...]

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python -m genofoundation.models.vae.train_scripts.train_convvae \
    --config-path="$SCRIPT_DIR/configs" \
    model.latent_dim=64 \
    model.num_res_blocks=4 \
    model.max_channels=512 \
    data.num_workers=16 \
    trainer.learning_rate=1e-3 \
    trainer.wandb_mode=online \
    trainer.mixed_precision=bf16 \
    "$@"

python -m genofoundation.models.vae.train_scripts.train_convvae \
    --config-path="$SCRIPT_DIR/configs" \
    model.latent_dim=64 \
    model.num_res_blocks=12 \
    model.max_channels=512 \
    data.num_workers=16 \
    trainer.learning_rate=1e-3 \
    trainer.wandb_mode=online \
    trainer.mixed_precision=bf16 \
    "$@"


python -m genofoundation.models.vae.train_scripts.train_convvae \
    --config-path="$SCRIPT_DIR/configs" \
    model.latent_dim=64 \
    model.num_res_blocks=20 \
    model.max_channels=512 \
    data.num_workers=16 \
    trainer.learning_rate=1e-3 \
    trainer.wandb_mode=online \
    trainer.mixed_precision=bf16 \
    "$@"
