#!/usr/bin/env bash
# Replay-evaluate a list of checkpoints of one run, sequentially, on one GPU.
#
# Why sequential per GPU: each checkpoint is a 16 GB model, so two fit on a card
# only if both skip the VAE work; running one at a time keeps the measurement
# clean and the memory predictable. Parallelism comes from giving each run its
# own GPU (see replay_sweep_all.sh).
#
# Usage: replay_sweep.sh <gpu> <run_name> [step ...]
#   no steps -> every checkpoint present on disk
set -euo pipefail

GPU="$1"; RUN="$2"; shift 2
STEPS=("$@")

REPO=/data/docker-services/world_action_models/unifolm_wma
EXP=/experiments/unifolm_wma
CKPT_DIR="$REPO/docker_data/runs/$RUN/checkpoints/trainstep_checkpoints"

cd "$REPO/docker"

if [ ${#STEPS[@]} -eq 0 ]; then
  # step=1000.ckpt -> 1000, sorted numerically
  mapfile -t STEPS < <(ls "$CKPT_DIR" | sed -n 's/^step=\([0-9]*\)\.ckpt$/\1/p' | sort -n)
fi

echo "=== replay sweep: $RUN on GPU $GPU, ${#STEPS[@]} checkpoints ==="
for STEP in "${STEPS[@]}"; do
  CKPT="$CKPT_DIR/step=$STEP.ckpt"
  [ -f "$CKPT" ] || { echo "  step $STEP: MISSING, skipped"; continue; }
  OUT="$EXP/replay/$RUN/step$STEP"
  if [ -f "$OUT/summary.json" ]; then
    echo "  step $STEP: already done, skipped"
    continue
  fi
  echo "  --- step $STEP ---"
  START=$(date +%s)
  WMA_GPU="$GPU" docker compose run --rm --no-deps \
      -e WMA_GPU="$GPU" \
      wma-shell python docker/replay_eval.py \
      --config configs/train/config_g1_dex3.yaml \
      --ckpt "$CKPT" \
      --anchors 8 --dump-video \
      --out "$OUT" 2>&1 | grep -aE "anchor |MAE|delta|correlation|video PSNR|verdict" \
    || echo "  step $STEP: FAILED (see log)"
  echo "  step $STEP took $(( $(date +%s) - START ))s"
done
echo "=== sweep done: $RUN ==="