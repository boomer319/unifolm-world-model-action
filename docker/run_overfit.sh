#!/usr/bin/env bash
# Launch one G1 Dex3 overfit run on a single GPU.
#
# Usage:
#   ./run_overfit.sh <gpu> <run_name> <ckpt_relpath> [seed] [max_steps] [ckpt_every] [config]
#
# Example (Dual base, 28-DoF-initialised checkpoint):
#   ./run_overfit.sh 0 overfit_dual_s20250912 checkpoints/unifolm_wma_dual_28dof_init.ckpt
#
# Notes:
#   * WMA_GPU selects the GPU. Default 0, per the standing .240 rule. A GPU is
#     only attached if you pass it explicitly.
#   * max_steps is counted in BATCHES by pytorch-lightning 1.9.5, and with this
#     config one batch is one weight update (measured: batches_in_window=1), so
#     max_steps is effectively the number of weight updates.
#   * WMA_BASE_CKPT is what the RunRecorder records as the seeded checkpoint.
#   * Checkpoints are ~16 GB each (fp32, no optimizer state) and
#     save_top_k=-1 keeps every one, so budget disk: max_steps/ckpt_every * 16 GB.
set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 <gpu> <run_name> <ckpt_relpath> [seed] [max_steps] [ckpt_every] [config]" >&2
    exit 2
fi

GPU=$1
NAME=$2
CKPT=$3
SEED=${4:-20250912}
STEPS=${5:-10000}
EVERY=${6:-1000}
CONFIG=${7:-configs/train/config_g1_dex3.yaml}

cd "$(dirname "$0")"

export WMA_GPU="$GPU"
export WMA_RUN_NAME="$NAME"
export WMA_BASE_CKPT="/workspace/$CKPT"

echo ">>> $NAME on GPU $GPU"
echo "    checkpoint : $CKPT"
echo "    seed       : $SEED"
echo "    max_steps  : $STEPS (batches == weight updates here)"
echo "    ckpt every : $EVERY  (~$((STEPS / EVERY * 16)) GB)"
echo "    config     : $CONFIG"
echo "    log        : ../docker_data/logs/$NAME.log"

# -e WITHOUT a value forwards the exported host variable into the container.
# Exporting alone is not enough: compose only passes what is listed in the
# service environment or given with -e, so all four runs silently fell back to
# the same runs/tensorboard directory and overwrote each other's CSVs.
exec docker compose run --rm -e WMA_RUN_NAME -e WMA_BASE_CKPT wma-shell \
    python scripts/trainer.py --train \
        --base "$CONFIG" \
        --name "$NAME" \
        --logdir /docker_data/runs \
        --seed "$SEED" \
        --devices 1 --total_gpus=1 \
        lightning.trainer.num_nodes=1 \
        lightning.strategy=auto \
        lightning.trainer.max_steps="$STEPS" \
        lightning.callbacks.model_checkpoint.params.every_n_train_steps="$EVERY" \
        lightning.callbacks.model_checkpoint.params.save_top_k=-1 \
        model.pretrained_checkpoint="$CKPT"