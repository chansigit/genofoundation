#!/bin/bash
# =============================================================================
# Hyperparameter search for Conv VAE training on TMS dataset
# =============================================================================
#
# Usage:
#   chmod +x run-hparam-scan-convvae-tms.sh
#
#   # Full dataset (341k train)
#   ./run-hparam-scan-convvae-tms.sh
#
#   # Mini dataset (20k train) - faster iteration
#   ./run-hparam-scan-convvae-tms.sh mini
#
# Sweep:
#   num_res_blocks: 1, 2, 3, 4
#
# Total experiments: 4
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
echo "Starting Conv VAE hyperparameter search..."
echo "========================================"

python train_convvae.py -m \
    ${DATA_CONFIG} \
    model.num_res_blocks=2,4,6,8 \
    trainer.wandb_mode=online \
    trainer.wandb_project=convvae-tms

echo "========================================"
echo "Hyperparameter search complete!"
echo "========================================"
