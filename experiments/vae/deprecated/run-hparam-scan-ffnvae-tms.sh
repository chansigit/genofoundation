#!/bin/bash
# =============================================================================
# Hyperparameter search for FFN VAE training on TMS dataset
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
# Batch size: 64, 128, 256, 512
#
# Latent dim: 128 (fixed)
#
# Hidden dims (encoder/decoder):
#   small:  [256, 128] / [128, 256]
#   large:  [1024, 512, 256] / [256, 512, 1024]
#
# Total experiments: 5 x 4 x 2 = 40
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
echo "Starting FFN VAE hyperparameter search..."
echo "========================================"

python train_ffnvae.py -m \
    ${DATA_CONFIG} \
    trainer.learning_rate=0.0009765625,0.00048828125,0.000244140625,0.0001220703125,0.00006103515625 \
    trainer.batch_size=64,128,256,512 \
    'model.encoder_hidden_dims=[256,128],[1024,512,256]' \
    'model.decoder_hidden_dims=[128,256],[256,512,1024]' \
    trainer.wandb_mode=online \
    trainer.wandb_project=ffnvae-tms

echo "========================================"
echo "Hyperparameter search complete!"
echo "========================================"
