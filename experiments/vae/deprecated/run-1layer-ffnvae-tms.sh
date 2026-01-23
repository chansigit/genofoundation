#!/bin/bash
# =============================================================================
# Training for FFN VAE training on TMS dataset
# =============================================================================
#
# Usage:
#   chmod +x run-hparam-scan-ffnvae-tms.sh
#
#   # Full dataset (341k train)
#   ./run-hparam-scan-ffnvae-tms.sh
#
#   # Mini dataset (20k train) - faster iteration
#   ./run-hparam-scan-ffnvae-tms.sh mini
#
# Learning rate: 2^(-k) for k in [10, 14]
#   k=10: lr ≈ 9.77e-4
#   k=11: lr ≈ 4.88e-4
#   k=12: lr ≈ 2.44e-4
#   k=13: lr ≈ 1.22e-4
#   k=14: lr ≈ 6.10e-5
#
# =============================================================================

set -e

# Dataset selection
if [ "$1" = "mini" ]; then
    DATA_CONFIG='data.train_path=${hydra:runtime.cwd}/../../data/tms/tms_preprocessed-mini.pt'
    echo "Using MINI dataset (20k train)"
else
    DATA_CONFIG=""
    echo "Using FULL dataset (341k train)"
fi

echo "========================================"
echo "Starting FFN VAE training..."
echo "========================================"

python train_ffnvae.py -m \
    ${DATA_CONFIG} \
    trainer.learning_rate=0.0009765625  \
    trainer.batch_size=256 \
    'model.encoder_hidden_dims=[1024]' \
    'model.decoder_hidden_dims=[1024]' \
    trainer.wandb_mode=online \
    trainer.mixed_precision=bf16 \
    trainer.wandb_project=ffnvae-tms \
    data.num_workers=8

echo "========================================"
echo "Training complete!"
echo "========================================"
