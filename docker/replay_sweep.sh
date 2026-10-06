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
# This script runs on the host, but /experiments is the CONTAINER mount point.
# Paths given to docker compose (--out) use /experiments/...; anything bash
# itself touches - log redirection, reading results - must use the real host path.
EXP=/experiments/unifolm_wma
EXP_HOST=/data/docker-services/world_action_models/experiments
LOG_DIR="$EXP_HOST/logs"
CKPT_DIR="$REPO/docker_data/runs/$RUN/checkpoints"

cd "$REPO/docker"

if [ ${#STEPS[@]} -eq 0 ]; then
  # PL names these epoch=N-step=M.ckpt; we want the step, sorted numerically.
  # (The sibling trainstep_checkpoints/ dir is empty in our runs, which is why
  # a first attempt that looked there reported "0 checkpoints".)
  mapfile -t STEPS < <(ls "$CKPT_DIR" 2>/dev/null \
    | sed -n 's/^epoch=[0-9]*-step=\([0-9]*\)\.ckpt$/\1/p' | sort -n)
fi

echo "=== replay sweep: $RUN on GPU $GPU, ${#STEPS[@]} checkpoints ==="
for STEP in "${STEPS[@]}"; do
  CKPT=$(ls "$CKPT_DIR"/epoch=*-step="$STEP".ckpt 2>/dev/null | head -1)
  [ -n "$CKPT" ] || { echo "  step $STEP: MISSING, skipped"; continue; }
  # This script runs on the host but the container sees the repo root as
  # /workspace, so the checkpoint has to be handed over as a container path. A
  # host path fails with FileNotFoundError from inside torch.load - which is how
  # an entire sweep silently "completed" while evaluating nothing.
  CKPT_IN_CONTAINER=/workspace/${CKPT#"$REPO"/}
  OUT="$EXP/replay/$RUN/step$STEP"
  if [ -f "$OUT/summary.json" ]; then
    echo "  step $STEP: already done, skipped"
    continue
  fi
  echo "  --- step $STEP ---"
  START=$(date +%s)
  STEP_LOG="$LOG_DIR/replay_${RUN}_step${STEP}.log"
  # Cap per-process threads. Without this each PyTorch process grabs all 192
  # cores for intra-op parallelism; five concurrent sweeps drove the load
  # average to 400 and per-checkpoint time from 141s to 240s, almost all of it
  # spent thrashing during the 16 GB checkpoint load rather than computing.
  if WMA_GPU="$GPU" OMP_NUM_THREADS="${THREADS:-8}" \
     MKL_NUM_THREADS="${THREADS:-8}" \
     docker compose run --rm --no-deps \
      -e WMA_GPU="$GPU" \
      -e OMP_NUM_THREADS="${THREADS:-8}" \
      -e MKL_NUM_THREADS="${THREADS:-8}" \
      wma-shell python docker/replay_eval.py \
      --config configs/train/config_g1_dex3.yaml \
      --ckpt "$CKPT_IN_CONTAINER" \
      --anchors 8 --dump-video \
      --out "$OUT" > "$STEP_LOG" 2>&1; then
    grep -aE "MAE  |per-step|correlation|video PSNR|verdict" "$STEP_LOG" | sed "s/^/  /"
  else
    # Print the tail of the real log. An earlier version piped everything through
    # grep, so a run that failed on step 1 reported "FAILED (see log)" while the
    # log itself did not exist - the actual traceback was thrown away.
    echo "  step $STEP: FAILED - tail of $STEP_LOG"
    tail -6 "$STEP_LOG" | sed "s/^/      /"
  fi
  echo "  step $STEP took $(( $(date +%s) - START ))s"
done
echo "=== sweep done: $RUN ==="