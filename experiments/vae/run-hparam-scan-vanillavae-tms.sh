#!/bin/bash
# =============================================================================
# Hyperparameter search for VAE training on TMS dataset
# =============================================================================
#
# Usage:
#   chmod +x run-hparam-scan-vanillavae-tms.sh
#
#   # Full dataset (341k train)
#   ./run-hparam-scan-vanillavae-tms.sh
#
#   # Mini dataset (20k train) - faster iteration
#   ./run-hparam-scan-vanillavae-tms.sh mini
#
# Learning rate: 2^(-k) for k in [10, 14]
#   k=10: lr ≈ 9.77e-4
#   k=11: lr ≈ 4.88e-4
#   k=12: lr ≈ 2.44e-4
#   k=13: lr ≈ 1.22e-4
#   k=14: lr ≈ 6.10e-5
#
# Batch size: 32, 64, 128, 256
#
# Total experiments: 5 x 4 = 20
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
echo "Starting hyperparameter search..."
echo "========================================"

python train_vanillavae.py -m \
    ${DATA_CONFIG} \
    trainer.learning_rate=0.0009765625,0.00048828125,0.000244140625,0.0001220703125,0.00006103515625 \
    trainer.batch_size=32,64,128,256 \
    trainer.wandb_mode=online \
    trainer.wandb_project=vanillavae-tms

echo "========================================"
echo "Hyperparameter search complete!"
echo "========================================"
