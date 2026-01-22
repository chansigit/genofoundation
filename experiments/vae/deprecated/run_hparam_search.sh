#!/bin/bash
# =============================================================================
# Hyperparameter search for VAE training using Hydra Multirun
# =============================================================================
#
# Usage:
#   chmod +x run_hparam_search.sh
#   ./run_hparam_search.sh
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

python train_vanillavae_miniTMS.py -m \
    trainer.learning_rate=0.0009765625,0.00048828125,0.000244140625,0.0001220703125,0.00006103515625 \
    trainer.batch_size=32,64,128,256 \
    trainer.wandb_mode=online \
    trainer.wandb_project=vanilla-vae

echo "========================================"
echo "Hyperparameter search complete!"
echo "========================================"
