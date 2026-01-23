#!/bin/bash
# Entry point for FFN VAE training
# Usage: ./train_ffnvae.sh trainer.epochs=100 model.latent_dim=256

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"


python -m genofoundation.models.vae.train_scripts.train_ffnvae \
    --config-path="$SCRIPT_DIR/configs" \
    model.latent_dim=64 \
    data.num_workers=8 \
    trainer.learning_rate=1e-4 \
    trainer.wandb_mode=online \
    trainer.mixed_precision=bf16 \
    trainer.batch_size=256 \
    'model.encoder_hidden_dims=[1024,512]' \
    'model.decoder_hidden_dims=[512,1024]' \
    trainer.wandb_project=ffnvae_tms \
    "$@"

python -m genofoundation.models.vae.train_scripts.train_ffnvae \
    --config-path="$SCRIPT_DIR/configs" \
    model.latent_dim=64 \
    data.num_workers=8 \
    trainer.learning_rate=1e-4 \
    trainer.wandb_mode=online \
    trainer.mixed_precision=bf16 \
    trainer.batch_size=256 \
    'model.encoder_hidden_dims=[512]' \
    'model.decoder_hidden_dims=[512]' \
    trainer.wandb_project=ffnvae_tms \
    "$@"