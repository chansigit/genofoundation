#!/bin/bash
# =============================================================================
# Training for Conv VAE training on TMS dataset
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
#
# Total experiments: 1
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
echo "Starting Conv VAE Training..."
echo "========================================"

python train_convvae.py -m \
    ${DATA_CONFIG} \
    model.num_res_blocks=2 \
    model.max_channels=512 \
    trainer.wandb_mode=online \
    trainer.wandb_project=convvae-tms \
    trainer.batch_size=128 \
    trainer.mixed_precision=bf16 \
    data.num_workers=16 \

echo "========================================"
echo "Training complete!"
echo "========================================"
