#!/bin/bash
# Entry point for Vanilla VAE training
# Usage: ./train_vanillavae.sh trainer.epochs=100 model.latent_dim=256

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python -m genofoundation.models.vae.train_scripts.train_vanillavae \
    --config-path="$SCRIPT_DIR/configs" \
    "$@"
