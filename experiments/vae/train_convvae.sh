#!/bin/bash
# Entry point for ConvVAE training
# Usage: ./train_convvae.sh trainer.epochs=100 model.latent_dim=256

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python -m genofoundation.models.vae.train_scripts.train_convvae \
    --config-path="$SCRIPT_DIR/configs" \
    "$@"
