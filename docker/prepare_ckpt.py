#!/usr/bin/env python3
"""Rewrite a UnifoLM-WMA checkpoint so it loads into a different-DoF model.

Why this exists
---------------
Post-training a new embodiment means loading a 16-DoF released checkpoint into
our 28-DoF G1 Dex3 model. Upstream's loader,
src/unifolm_wma/utils/train.py:load_checkpoints, calls

    model.load_state_dict(pl_sd["state_dict"], strict=False)

strict=False ignores missing and unexpected keys but still RAISES on a size
mismatch, and the function's fallback path then retries with strict=True, which
raises too. So a 16 -> 28 DoF transition crashes the trainer even though
everything else is fine. docker/load_test.py reports exactly which tensors are
incompatible; this script produces a checkpoint with those tensors removed, so
training can proceed against a self-consistent file and upstream code stays
untouched (same philosophy as prepare_data/prepare_g1_dex3.py, which wraps the
stock converter instead of patching it).

What gets dropped is only the DoF-dependent input/output projections of the
action and state heads plus their EMA copies - measured on the released Dual
checkpoint: 68 of 3646 tensors, 8.43% of all parameters, while the entire video
world model, VAE, text/image encoders and head trunk transfer unchanged.

Usage
-----
    python docker/prepare_ckpt.py \
        --ckpt checkpoints/unifolm_wma_dual.ckpt \
        --out  checkpoints/unifolm_wma_dual_28dof_init.ckpt \
        --config configs/train/config_g1_dex3.yaml

The output is a plain {"state_dict": ...} dict, i.e. the same format the stock
loader expects, so it can be pointed at directly:

    lightning.model.pretrained_checkpoint=checkpoints/unifolm_wma_dual_28dof_init.ckpt
"""
import argparse
import os
import sys
from collections import OrderedDict

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="source checkpoint")
    p.add_argument("--out", required=True, help="destination checkpoint")
    p.add_argument("--config", required=True,
                   help="training config describing the TARGET model (e.g. 28 DoF)")
    p.add_argument("--precision", type=int, default=32, choices=[16, 32],
                   help="32 keeps the file fp32; 16 halves its size but loses "
                        "precision in the frozen world model, so 32 is the default")
    p.add_argument("--force", action="store_true", help="overwrite --out if present")
    return p.parse_args()


def main():
    args = parse_args()
    if os.path.exists(args.out) and not args.force:
        print(f"ERROR: {args.out} exists; pass --force to overwrite")
        sys.exit(2)

    from omegaconf import OmegaConf
    from unifolm_wma.utils.utils import instantiate_from_config

    print("=" * 70)
    print(" target model from config")
    print("=" * 70)
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    print(f"  config                : {args.config}")
    print(f"  agent_state_dim       : {cfg.model.params.agent_state_dim}")
    print(f"  agent_action_dim      : {cfg.model.params.agent_action_dim}")
    model = instantiate_from_config(cfg.model)
    msd = model.state_dict()
    del model  # we only need the shapes; free the memory before loading the file
    print(f"  instantiated, {len(msd)} target tensors")

    print()
    print("=" * 70)
    print(" read source checkpoint")
    print("=" * 70)
    if not os.path.exists(args.ckpt):
        print(f"ERROR: {args.ckpt} not found")
        sys.exit(2)
    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        sd, how = obj["state_dict"], "state_dict"
    elif isinstance(obj, dict) and "module" in obj:
        sd = OrderedDict((k[16:], v) for k, v in obj["module"].items())
        how = "deepspeed module"
    else:
        sd, how = obj, "raw state dict"
    print(f"  {args.ckpt}: {how}, {len(sd)} tensors, "
          f"{os.path.getsize(args.ckpt)/2**30:.2f} GiB")

    print()
    print("=" * 70)
    print(" diff")
    print("=" * 70)
    ck, mk = set(sd), set(msd)
    missing = sorted(mk - ck)
    unexpected = sorted(ck - mk)
    mismatched = []
    for k in sorted(ck & mk):
        a, b = sd[k], msd[k]
        if hasattr(a, "shape") and hasattr(b, "shape") and tuple(a.shape) != tuple(b.shape):
            mismatched.append((k, tuple(a.shape), tuple(b.shape)))

    print(f"  missing (target tensor with no ckpt entry, starts random): {len(missing)}")
    print(f"  unexpected (ckpt entry with no target tensor)            : {len(unexpected)}")
    print(f"  shape-mismatched (would RAISE even with strict=False)    : {len(mismatched)}")
    for k, a, b in mismatched[:12]:
        print(f"    {k}: {a} -> {b}")
    if len(mismatched) > 12:
        print(f"    ... and {len(mismatched) - 12} more")

    dropped_params = 0
    out = OrderedDict()
    for k, v in sd.items():
        if k in msd and hasattr(v, "shape") and tuple(v.shape) != tuple(msd[k].shape):
            n = 1
            for s in msd[k].shape:
                n *= s
            dropped_params += n
            continue
        out[k] = v.half() if (args.precision == 16 and v.is_floating_point()) else v

    total_in = sum(v.numel() for v in sd.values())
    print()
    print(f"  kept     : {len(out)} tensors, {sum(v.numel() for v in out.values())/1e9:.3f} B params")
    print(f"  dropped  : {len(mismatched)} tensors, {dropped_params/1e6:.1f} M params "
          f"({100*dropped_params/max(total_in,1):.2f}% of the source)")
    if missing:
        print(f"  still missing after the rewrite: {len(missing)} -> these start "
              f"random (expected when the target has modules the source lacks)")

    print()
    print("=" * 70)
    print(" write")
    print("=" * 70)
    torch.save({"state_dict": out}, args.out)
    size = os.path.getsize(args.out) / 2**30
    print(f"  wrote {args.out}: {size:.2f} GiB (fp{args.precision})")

    # Re-open and prove it loads cleanly with the stock loader's call.
    print()
    print("=" * 70)
    print(" verify: the stock load_checkpoints call must now succeed")
    print("=" * 70)
    from unifolm_wma.utils.utils import instantiate_from_config as ifc
    m2 = ifc(cfg.model)
    res = m2.load_state_dict(out, strict=False)
    print(f"  load_state_dict(strict=False) -> missing={len(res.missing_keys)} "
          f"unexpected={len(res.unexpected_keys)}")
    print("  upstream utils/train.py:load_checkpoints will now succeed.")
    print()
    print("=" * 70)
    print(" PREPARE CKPT: DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()