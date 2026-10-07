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

## 2. The main result: substantial learning, memorisation only just reachable

Ratio = MAE divided by the MAE of a predictor that simply repeats the current
joint position. **Below 1.0 means better than doing nothing.**

| arm | best checkpoint | × no-motion | predicted ÷ ground-truth delta |
|---|---|---|---|
| untrained head (reference) | – | 19.94 | 134× |
| `overfit_base_s20250912` | 9 000 | 1.68 | 8.7× |
| `overfit_base_s42` | 9 000 | 2.11 | 11.0× |
| `overfit_dual_s20250912` | 8 000 | 2.24 | 12.0× |
| `overfit_dual_s42` | 10 000 | 2.40 | 10.3× |

A 9–12× improvement over the untrained head, yet at this budget **every arm is
still worse than standing still.** Ground truth moves 0.0091 rad per step at this
stride, so `delta_ratio` near 1 would mean the emitted motion has the right
magnitude; nothing here gets below 8×. (A later second learning-rate cycle does
cross the baseline — see §5.)

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

Resuming the best arms from their step-9 000 checkpoints for a second
learning-rate cycle (≈19 000 total updates) reaches **1.10** and **1.11** in the
two seeds — agreement that makes it a real effect rather than the seed noise we
measured at ~0.6 in this ratio. It also improves the video branch (PSNR 24–27 dB
versus 23–24 for the original arms).

The trend *within* a continuation is not monotonic, so the minimum over a run
matters more than its endpoint.

So the original study's conclusion must be stated as **"10 000 updates is not
enough"**, not "it cannot be done".

At 22 000 total updates the first seed **does** cross the line: ratio **0.979**
(MAE 0.0360 against a no-motion baseline of 0.0550), correlation +0.990, emitted
motion still 5.2× ground truth. Taken alone that is the first evidence of
single-episode memorisation in this study — but it should be read with three
qualifications, and it would be wrong to announce it as a clean success:

- it is **marginal** — 0.979 is a 2 % margin, not a decisive separation;
- it is **not replicated** — the second continuation seed bottoms out at 1.11;
- it is **uneven across the episode** — 5 of 8 anchors individually beat the
  do-nothing baseline, with a per-anchor median of 0.996 but a maximum of 2.06.

So the accurate claim is that **single-episode memorisation becomes reachable with
roughly twice the compute we first gave it, in one of two seeds**. Whether it is
reliably reachable is exactly what a longer run with periodic evaluation and an
early-stopping criterion would settle.

## 6. Observation context helps too — separably from training longer

Doubling the observation horizon from 2 to 4 frames is a **diagnostic departure
from Unitree's recipe, not part of it**. At *matched* 10 000 updates:

| | seed 20250912 | seed 42 |
|---|---|---|
| n_obs = 2 (original, 10 k) | 1.95 | 2.18 |
| n_obs = 4 (10 k) | **1.36** | **1.55** |
| n_obs = 4 (best, 12 k) | 1.36 | **1.25** |

So roughly a 30 % improvement from observation context alone, in both seeds. The
early steps showed a much larger apparent lead (3.61 vs 5.87 at step 2 000) that
narrowed as the n_obs = 2 arm caught up — a reminder not to read a trend off
early checkpoints, which is the same mistake that produced the incorrect
"all arms peak at step 9000" claim.

Both levers are real and independent: **training longer** (10 k → 19 k: 1.95 →
1.11) is the larger of the two, **more observation context** (2 → 4 frames at
10 k: 1.95 → 1.36) is a genuine but smaller gain.

## 6b. And more trajectory data does nothing

The 10-episode arm is indistinguishable from the 1-episode arm at matched steps
(2.96 vs 3.11 at step 4 000; 2.53 vs 2.69 at step 6 000). Ten times the windows,
each seen ten times fewer, buys nothing at fixed compute.

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
- Arms are compared at different total update counts (10 k original, 12 k for
  obs4, 19 k for the continuations). Only the within-budget comparisons in §6 are
  matched.
- The best ratio anywhere in the study is 1.10. "Better than standing still" was
  not reached; the honest claim is that it came close.
- Both levers show an overfitting regime: the continuation arms peak mid-run and
  then degrade (1.11 at 19 000 total, then 1.34 by 21 000), and obs4 does the same
  (1.36 at 10 000, then 1.99 at 12 000). There is a real optimum within roughly
  10 000–20 000 updates on a single episode, and simply running longer is not a
  monotone route to better tracking.
- No simulation or hardware deployment was attempted; these are offline
  evaluations against recorded ground truth.