#!/usr/bin/env bash
# Reorganise the experiment tree around the question, not the model.
#
# Why: the thesis asks one question - can a world-action model memorise a single
# robot episode? - and answers it with two models. The 1-episode overfit is the
# SAME experiment on both: DreamZero's agibot_lora_1ep_4k / 12k runs and WMA's
# overfit_* arms tackle the identical episode with identical metrics. So the
# experiment should be the top-level unit, with the models side by side inside
# it, and the comparison becomes legible without archaeology.
#
# "replay" also goes: it names the mechanism rather than the measurement, and it
# collides with DreamZero's separate use of the word. "eval" is neutral.
#
# Usage:
#   restructure.sh            dry run, prints every move
#   restructure.sh --apply    actually do it
set -euo pipefail

E=/data/docker-services/world_action_models/experiments
APPLY=0
[ "${1:-}" = "--apply" ] && APPLY=1

# Refuse to run while evaluations are in flight: their output paths are computed
# at launch, so moving the tree mid-sweep would split one run's results across
# two directories without any error.
if pgrep -f "replay_eval|replay_sweep" > /dev/null; then
  echo "REFUSING: a replay sweep is still running."
  echo "  Output paths are resolved when each step launches, so renaming now"
  echo "  would scatter a run's results over two directories."
  echo "  Wait for the sweeps to finish, then re-run."
  exit 1
fi

move() {  # move <src> <dst>
  if [ ! -e "$1" ]; then echo "  skip (absent): ${1#$E/}"; return; fi
  echo "  ${1#$E/}  ->  ${2#$E/}"
  [ "$APPLY" = 1 ] && { mkdir -p "$(dirname "$2")"; mv "$1" "$2"; }
  return 0
}

echo "=== 1. headline experiment: 1-episode memorisation, both models ==="
M=$E/memorisation_1ep
mkdir -p "$M/unifolm_wma" 2>/dev/null || true
[ "$APPLY" = 1 ] && mkdir -p "$M/unifolm_wma"

# WMA: training records and per-checkpoint evaluations
move "$E/unifolm_wma/runs"        "$M/unifolm_wma/training"
move "$E/unifolm_wma/eval"        "$M/unifolm_wma/eval"      # after replay->eval
move "$E/unifolm_wma/replay"      "$M/unifolm_wma/eval"
move "$E/unifolm_wma/figures"     "$M/unifolm_wma/figures"
move "$E/unifolm_wma/configs"     "$M/unifolm_wma/training/configs"

# DreamZero's matching 1-episode runs come along so the two models sit together
for r in agibot_lora_1ep_4k agibot_lora_12k agibot_lora_12k_teacherforced; do
  move "$E/dreamzero/replay/$r" "$M/dreamzero/$r"
done

echo "=== 2. WMA data-diversity arm (10 episodes, WMA only) ==="
M10=$E/memorisation_10ep
[ "$APPLY" = 1 ] && mkdir -p "$M10/unifolm_wma"
if [ -d "$E/unifolm_wma/replay/overfit_dual_10ep_s20250912" ]; then
  move "$E/unifolm_wma/replay/overfit_dual_10ep_s20250912" \
       "$M10/unifolm_wma/eval/overfit_dual_10ep_s20250912"
fi
if [ -d "$E/unifolm_wma/runs/overfit_dual_10ep_s20250912" ]; then
  move "$E/unifolm_wma/runs/overfit_dual_10ep_s20250912" \
       "$M10/unifolm_wma/training/overfit_dual_10ep_s20250912"
fi

echo "=== 3. DreamZero side-quests (teacher forcing, 3ep, AllMerged) ==="
[ "$APPLY" = 1 ] && mkdir -p "$E/dreamzero_explorations"
if [ -d "$E/dreamzero/replay" ]; then
  for d in "$E/dreamzero/replay"/*; do
    [ -e "$d" ] || continue
    base=$(basename "$d")
    case "$base" in
      agibot_lora_1ep_4k|agibot_lora_12k|agibot_lora_12k_teacherforced)
        ;;  # already moved in step 1
      *) move "$d" "$E/dreamzero_explorations/$base" ;;
    esac
  done
fi

echo "=== 4. drop the emptied model-level dirs ==="
for d in "$E/unifolm_wma/replay" "$E/unifolm_wma/runs" "$E/unifolm_wma/figures" \
         "$E/dreamzero/replay" "$E/dreamzero" "$E/unifolm_wma"; do
  if [ -d "$d" ] && [ -z "$(ls -A "$d" 2>/dev/null)" ]; then
    echo "  rmdir ${d#$E/}"
    [ "$APPLY" = 1 ] && rmdir "$d"
  fi
done

echo
if [ "$APPLY" = 1 ]; then
  echo "=== done. new layout: ==="
  find "$E" -maxdepth 3 -type d | sed "s|$E|experiments|"
else
  echo "(dry run - nothing moved. re-run with --apply)"
fi