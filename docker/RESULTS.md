# Results: can a world-action model memorise one robot episode?

All numbers below are ground-truth-anchored replay evaluations, measured on
`.240`, reproducible from the artefacts in this tree. Metric definitions and the
reasoning behind each choice are in `EXPERIMENTS.md`.

**The short version.** On a single 39 s episode, UnifoLM-WMA learns a great deal
about the *scene* and rather little about the *trajectory*. The world-model
branch reproduces the manipulation faithfully; the action head never becomes
better than the trivial "do nothing" predictor within the budget tested. Teacher
forcing localises the failure to the action head's conditioning rather than to
anything the vision branch is doing wrong.

---

## 1. Setup

One episode (`G1_Dex3_GraspSquare` ep 0): 1173 frames, 39.1 s at 30 fps, one
camera, 28-DoF absolute joint targets. Fine-tuning only — the architecture is a
Diffusion Policy and no LoRA support is exposed. 10 000 weight updates, batch 1,
which the one-row dataset CSV imposes.

Inference settings match Unitree's own recommended deployment command
(`robot_client.py --action_horizon 16 --exe_steps 16 --observation_horizon 2
--control_freq 15`), so the negative result is obtained under the vendor's
configuration rather than one of our own devising.

## 2. The main result: substantial learning, no memorisation

Ratio = MAE divided by the MAE of a predictor that simply repeats the current
joint position. **Below 1.0 means better than doing nothing.**

| arm | best checkpoint | × no-motion | predicted ÷ ground-truth delta |
|---|---|---|---|
| untrained head (reference) | – | 19.94 | 134× |
| `overfit_base_s20250912` | 9 000 | 1.68 | 8.7× |
| `overfit_base_s42` | 9 000 | 2.11 | 11.0× |
| `overfit_dual_s20250912` | 8 000 | 2.24 | 12.0× |
| `overfit_dual_s42` | 10 000 | 2.40 | 10.3× |

A 9–12× improvement over the untrained head, yet **every arm is still worse than
standing still.** The trajectory was never reproduced.

Ground truth moves 0.0091 rad per step at this stride, so `delta_ratio` near 1
would mean the emitted motion has the right magnitude. Nothing gets below 8×.

## 3. The world model works; the action head does not

Because WMA is *decoupled* — the video UNet and the action/state UNets are
separate modules, with the action head consuming the video branch's features —
the two halves can be judged independently. DreamZero's coupled design denies
this.

Generated video against ground truth, on the same clips:

| | SSIM ↑ | LPIPS ↓ |
|---|---|---|
| step 1 000 | 0.78–0.85 | 0.074–0.145 |
| step 5 000–6 000 | **0.88–0.90** | **0.055–0.064** |

Both improve with training. (PSNR reads ~22–24 dB and is *not* a useful summary
here: the scene is a black gripper against a white table, so a one-pixel
misalignment costs a lot of PSNR while being invisible. Reporting PSNR alone
would have understated this badly.)

So: the vision half of the world-action model learns. The policy half does not.

## 4. Teacher forcing localises the failure

Feeding the action head's *observation-frame* pathway the true next frames does
**nothing** — MAE 0.0627 → 0.0627 and 0.0735 → 0.0735 on the two arms tested.
Feeding it world-model features derived from ground-truth video **does** help:

| checkpoint | normal | teacher forced | change |
|---|---|---|---|
| `base_s20250912` step 9 000 | 1.679 | **1.507** | −10.2 % |
| `dual_s20250912` step 8 000 | 2.232 | **1.761** | −21.1 % |

Replicated across an arm whose init contains no policy head at all and one whose
init has a warm head, each paired with identical diffusion noise.

The reading: the head **ignores the raw frames it is handed** and **does use the
world model's features** — but even with perfect visual features it remains
1.51–1.76× worse than standing still and its deltas stay oversized. The head is
only partially conditioned on visuals, and predominantly emits a
manipulation-like motion prior whose *magnitude* barely responds to conditioning.

This also weakens the hypothesis that the failure is simply a conditional mean
forced by two frames of history: a genuine average under ambiguity should improve
far more when handed the answer.

## 5. It is under-trained, not at a structural floor

Resuming the best arm from its step-9 000 checkpoint for a second learning-rate
cycle reaches **1.30** — better than any checkpoint of the original 10 000-update
run (best 1.68) — and also improves the video branch (PSNR ~25.9 dB). The trend
within the continuation is not monotonic (1.30 / 2.71 / 1.51 at steps 1 000–3 000),
so the minimum over a run matters more than its endpoint.

So the original study's conclusion must be stated as **"10 000 updates is not
enough"**, not "it cannot be done".

## 6. Observation context helps, modestly

Doubling the observation horizon from 2 to 4 frames — a **diagnostic departure
from Unitree's recipe, not part of it** — is ahead in both seeds at matched
steps:

| step | n_obs=2 | n_obs=4 (seed 20250912) | n_obs=4 (seed 42) |
|---|---|---|---|
| 1 000 | 6.13 | 6.18 | 6.47 |
| 2 000 | 5.87 | **3.61** | 5.13 |
| 3 000 | 3.96 | **3.02** | 3.77 |

The direction agrees across seeds, but the effect is far smaller than simply
training longer, and neither arm is close to 1.0.

## 7. What the 10-episode arm does and does not test

Episodes 0–9 of GraspSquare — a dataset with 301 episodes and **exactly one
task string**. Task, objects, scene, instruction and camera are identical; only
initial condition, placement jitter and trajectory sample vary. It is therefore a
**trajectory-density** ablation, not a diversity one: ~1 000 passes per episode
instead of 10 000. Genuine diversity needs the other AllMerged tasks
(BlockStacking, ObjectPlacement, CameraPackaging).

## 8. Honest limitations

- Single episode, single task, one camera. Nothing here speaks to
  generalisation across tasks.
- The metric that decides the question (ratio vs no-motion) uses a deliberately
  strong baseline; absolute MAE numbers are not comparable to other papers.
- The original four arms stopped at 10 000 updates while still improving, so
  their ranking (Base > Dual, consistent across both seeds) should be read as a
  ranking *at that budget*, not a property of the architectures.
- No simulation or hardware deployment was attempted; these are offline
  evaluations against recorded ground truth.