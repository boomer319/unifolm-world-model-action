# Experiments

Central record for every experiment run on `.240`, across models. One tree so a
thesis chapter can be assembled from it without archaeology.

## Layout

    experiments/
      unifolm_wma/          this workstream
        replay/             memorisation test, one dir per (run, checkpoint)
        runs/               training records: manifest, metrics, timing, configs
        figures/            plots and generated videos for the write-up
      dreamzero/            the earlier DreamZero / G1 Dex3 experiments
      shared/               anything both models need
      notes/                

## Why replay, and not the training loss

A diffusion loss is a denoising objective, not a tracking metric. DreamZero
reached an action loss of 0.005 on a single episode while its actions were
**2.15x worse than standing still**, and its per-step joint deltas were **42x
larger than ground truth** with a consistent sign across 28 of 28 joints — the
mechanical "arms went up step by step" staircase seen in simulation.

So the loss curve cannot answer the question we care about. What can:

| metric | meaning | why it is in the table |
|---|---|---|
| `mae` | mean abs error of the predicted chunk vs ground truth, rad | the direct tracking error |
| `mae_no_motion` | same, for "repeat the anchor state" | the honest reference; a model must beat it |
| `ratio_vs_baseline` | `mae / mae_no_motion` | **< 1 means better than doing nothing** |
| `delta_pred` / `delta_gt` | mean per-step abs joint delta, predicted vs ground truth | the decisive number (see below) |
| `delta_ratio` | the ratio of those two | ground truth is 0.0027–0.0057 rad/step |
| `corr` | Pearson correlation of the flattened chunk | **positive for the wrong reason**: both prediction and ground truth hover near the anchor pose, so this flatters a model that has learnt nothing |
| `video_psnr` | generated vs ground-truth world-model frames | an independent read, from the jointly trained video branch |

### How to read `delta_ratio`

Ground truth in this episode moves 0.0027–0.0057 rad per action step (small,
because 15 Hz is fast relative to smooth motion — but the *accumulated* path is
2.2–13.1 rad per joint, so the arm does move a great deal).

A model that has **memorised the trajectory** emits deltas of roughly that size,
so `delta_ratio` lands near 1.

Two ways to be wrong, both seen:
- **far too large** — DreamZero at ~42x, the un-finetuned 28-DoF head at ~125x.
  In closed loop a consistent per-joint offset integrates into a ratchet.
- **far too small** — the model has collapsed onto the mean pose and stopped.

The un-finetuned baseline measured here sits at `ratio_vs_baseline` ~19.7 and
`delta_ratio` ~125, i.e. far worse than standing still. That is the "before" row
every trained checkpoint has to beat.

## Design

Fine-tuning only — no LoRA, because the architecture is Diffusion Policy and no
LoRA support is exposed. 28 DoF, Unitree G1 with Dex3 hands, absolute joint
targets, min/max normalisation fitted per dataset.

**1-episode arm**, four runs, 10 000 weight updates each, batch 1 (the CSV has
one row per episode per view, so a single episode forces batch size 1):

| run | base | GPU | why |
|---|---|---|---|
| `overfit_dual_s20250912` | Dual init (28-DoF) | 0 | warm action+state heads |
| `overfit_base_s20250912` | Base (no policy head) | 2 | the README's own recommended pretrained checkpoint |
| `overfit_dual_s42` | Dual init, seed 42 | 3 | seed variance |
| `overfit_base_s42` | Base, seed 42 | 4 | seed variance |

Base and Dual differ in exactly one respect: Base is the Step-1 world model with
no action head at all, Dual additionally has action and state heads post-trained
on Unitree data. Crossing that with two seeds is what makes a difference
readable as signal rather than noise.

**10-episode arm**, `overfit_dual_10ep_s20250912` on GPU 5: same checkpoint, same
seed, same 10 000 updates, but ten episodes instead of one. Episodes 0-9 of
GraspSquare, which has 301 episodes but **exactly one task string**.

It is therefore a **trajectory-density** ablation, not a diversity one: it raises
the window count from 1 173 to ~11 730 at fixed compute, so each episode gets
~1 000 passes instead of 10 000. What varies is initial condition, object
placement jitter and which trajectory sample of the behaviour; the task, the
objects, the scene, the instruction and the camera are identical.

Testing genuine diversity needs the other tasks in AllMerged (BlockStacking,
ObjectPlacement, CameraPackaging) — different scenes and instructions — which is
the axis that probes generalisation rather than sample count.

## Reproducing

    # one arm on one GPU, all its checkpoints
    bash docker/replay_sweep.sh 0 overfit_dual_s20250912

    # the comparison table
    python docker/replay_table.py --md /experiments/unifolm_wma/figures/replay_table.md

Each replay directory holds `summary.json` (aggregate + per-anchor),
`replay.npz` (predictions, ground truth, anchors) and per-anchor
`video_anchor*.mp4` / `cmp_anchor*.mp4`. `cmp_` clips are ground truth on the
left, prediction on the right.

## Provenance

`runs/<run>/manifest.json` records the git SHA and dirty state, a hash of the
config, the base checkpoint path and size, dataset hashes, parameter counts, and
the resolved host/GPU/library versions. `metrics.csv` has one row per weight
update, `timing.csv` one row per batch.