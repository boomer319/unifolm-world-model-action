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

if __name__ == "__main__":
    print("[DEBUG] Unnormalize traced; running scripts/evaluation/real_eval_server.py",
          flush=True)
    runpy.run_path("scripts/evaluation/real_eval_server.py", run_name="__main__")
