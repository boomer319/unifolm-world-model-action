#!/usr/bin/env python3
"""Launch the decision-making server with a shape trace on the de-normaliser.

Why this exists
---------------
The stock server finishes the whole 16-frame DDIM rollout and then fails on its
last statement:

    real_eval_server.py:421-422
      pred_action = pred_action[..., action_mask[0] == 1.0][0].cpu()
      pred_action = ...unnormalizer({'action': pred_action})['action']

with normolize.py:227 raising "The size of tensor a (2) must match the size of
tensor b (28) at non-singleton dimension 1" - i.e. the tensor handed to the
de-normaliser has 2 elements where the statistics have 28. The client only sees a
truncated `desc`, so the offending shape is invisible from outside.

Rather than patch upstream's server, this wraps it: it monkey-patches
Unnormalize.forward to print the shapes it is given (and the statistics it
applies them to), then runs the server unchanged via runpy. The trace tells us
the exact layout of the returned action chunk, which is what a simulator provider
needs to know anyway.

Usage:
    python docker/run_dm_server.py --ckpt_path ... --config ... [all server args]
"""
import runpy
import sys

import unifolm_wma.data.normolize as N

_orig_forward = N.Unnormalize.forward
_orig_init = N.Unnormalize.__init__


def _init(self, *a, **k):
    _orig_init(self, *a, **k)
    shapes = {}
    for key in ("action", "observation.state"):
        buf = getattr(self, f"buffer_{key.replace('.', '_')}", None)
        if isinstance(buf, dict):
            shapes[key] = {k: tuple(v.shape) for k, v in buf.items()}
    print(f"[UNNORM] statistics shapes: {shapes}", flush=True)


def _forward(self, batch):
    print(f"[UNNORM] forward called with "
          f"{ {k: tuple(v.shape) for k, v in batch.items()} }", flush=True)
    return _orig_forward(self, batch)


N.Unnormalize.__init__ = _init
N.Unnormalize.forward = _forward

# Also report the dimensions the constructed model believes it has, since the
# DDIM sampler builds its action noise as (b, 16, model.agent_action_dim) and
# the returned chunk came back with 2 values per step instead of 28.
import unifolm_wma.models.ddpms as _D

_orig_lm_init = _D.LatentVisualDiffusion.__init__


def _lm_init(self, *a, **k):
    _orig_lm_init(self, *a, **k)
    for attr in ("agent_state_dim", "agent_action_dim", "n_obs_steps_acting",
                 "n_obs_steps_imagen", "decision_making_only"):
        print(f"[MODEL] self.{attr:22s} = {getattr(self, attr, 'MISSING')}", flush=True)
    try:
        dm = self.model.diffusion_model
        print(f"[MODEL] inner .model type = {type(self.model).__name__}", flush=True)
        print(f"[MODEL] diffusion_model    = {type(dm).__name__}", flush=True)
        for name in ("action_unet", "state_unet"):
            head = getattr(dm, name, None)
            print(f"[MODEL] {name}: horizon={getattr(head, 'horizon', '?')} "
                  f"input_dim={getattr(head, 'input_dim', '?')} "
                  f"n_obs_steps={getattr(head, 'n_obs_steps', '?')}", flush=True)
    except Exception as e:
        print("[MODEL] head introspection failed:", type(e).__name__, e, flush=True)


_D.LatentVisualDiffusion.__init__ = _lm_init

if __name__ == "__main__":
    print("[DEBUG] Unnormalize traced; running scripts/evaluation/real_eval_server.py",
          flush=True)
    runpy.run_path("scripts/evaluation/real_eval_server.py", run_name="__main__")
