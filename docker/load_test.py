#!/usr/bin/env python3
"""Load-test for UnifoLM-WMA on the G1 Dex3 28-DoF configuration.

Why this script exists
----------------------
Upstream's own loader (src/unifolm_wma/utils/train.py:153-179) calls
``model.load_state_dict(pl_sd["state_dict"], strict=False)``. strict=False
silently ignores *missing* and *unexpected* keys, so a partially loaded model
looks like a successful load. It does NOT ignore size mismatches - those raise.
That matters for us because the released Base checkpoint is a 16-DoF model: at
28 DoF the action/state UNet input projections have a different shape, so the
load would hard-fail.

This script therefore:
  1. instantiates the model from a config,
  2. diffs the checkpoint against the model's state_dict BEFORE loading,
  3. reports every missing / unexpected / shape-mismatched key,
  4. loads with the shape-incompatible keys dropped (the sane behaviour for
     post-training a 16-DoF checkpoint onto a 28-DoF model: keep the world
     model, re-learn the head), or fails loudly with --strict,
  5. reports parameter counts and GPU memory.

Usage:
    python docker/load_test.py --config configs/train/config_g1_dex3.yaml \
                               --ckpt checkpoints/unifolm_wma_base.ckpt
"""
import argparse
import json
import os
import sys

import torch
from omegaconf import OmegaConf


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--precision", type=int, default=32, choices=[16, 32],
                   help="16 loads weights in fp16 (halves the 27 GB ckpt to 13.5 GB)")
    p.add_argument("--strict", action="store_true",
                   help="fail on any missing/unexpected/shape-mismatched key")
    p.add_argument("--max-list", type=int, default=25,
                   help="how many offending keys to print per category")
    p.add_argument("--report", default=None, help="write a JSON report here")
    return p.parse_args()


def load_sd(path):
    """Reproduce utils/train.py:load_checkpoints state-dict extraction."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], "state_dict"
    if isinstance(obj, dict) and "module" in obj:
        from collections import OrderedDict
        sd = OrderedDict()
        for k, v in obj["module"].items():
            sd[k[16:]] = v
        return sd, "deepspeed module (keys stripped by 16 chars)"
    if isinstance(obj, dict):
        return obj, "raw state dict"
    raise TypeError(f"unrecognised checkpoint structure: {type(obj)}")


def human(n):
    return f"{n/1e9:.2f} B" if n >= 1e9 else f"{n/1e6:.2f} M"


def main():
    args = parse_args()
    from unifolm_wma.utils.utils import instantiate_from_config

    print("=" * 70)
    print(" config")
    print("=" * 70)
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    m = cfg.model
    print(f"  target                 : {m.target}")
    print(f"  agent_state_dim        : {m.params.agent_state_dim}")
    print(f"  agent_action_dim       : {m.params.agent_action_dim}")
    print(f"  decision_making_only   : {m.params.decision_making_only}")
    print(f"  temporal_length        : {m.params.wma_config.params.temporal_length}")
    print(f"  image_size             : {list(m.params.image_size)}")
    print(f"  base_learning_rate     : {m.base_learning_rate}")
    print(f"  data_dir               : {cfg.data.params.train.params.data_dir}")
    print(f"  dataset_and_weights    : {dict(cfg.data.params.dataset_and_weights)}")
    print(f"  batch_size             : {cfg.data.params.batch_size}")
    print(f"  precision (lightning)  : {cfg.lightning.get('precision')}")
    print(f"  device                 : {args.device}  precision: fp{args.precision}")

    print()
    print("=" * 70)
    print(" instantiate model  (downloads CLIP/T5 weights on first run)")
    print("=" * 70)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
    model = instantiate_from_config(m)
    model.eval()
    msd = model.state_dict()
    print(f"  instantiated OK, {len(msd)} tensors in state_dict")

    print()
    print("=" * 70)
    print(" read checkpoint")
    print("=" * 70)
    if not os.path.exists(args.ckpt):
        print(f"  ERROR: checkpoint not found: {args.ckpt}")
        print("  Download it first (see docker/README.md, step 'checkpoints').")
        sys.exit(2)
    sd, how = load_sd(args.ckpt)
    print(f"  {args.ckpt}")
    print(f"  format: {how}, {len(sd)} tensors, "
          f"{os.path.getsize(args.ckpt)/2**30:.2f} GiB on disk")
    if args.precision == 16:
        sd = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
        print("  converted checkpoint tensors to fp16")

    print()
    print("=" * 70)
    print(" diff: checkpoint vs model")
    print("=" * 70)
    ckpt_keys, model_keys = set(sd), set(msd)
    missing = sorted(model_keys - ckpt_keys)          # randomly initialised
    unexpected = sorted(ckpt_keys - model_keys)       # ignored by strict=False
    mismatched = []
    for k in sorted(ckpt_keys & model_keys):
        a, b = sd[k], msd[k]
        if hasattr(a, "shape") and hasattr(b, "shape") and tuple(a.shape) != tuple(b.shape):
            mismatched.append((k, tuple(a.shape), tuple(b.shape)))

    def show(title, items, fmt=None):
        print(f"\n  {title}: {len(items)}")
        for it in items[:args.max_list]:
            print("    " + (fmt(it) if fmt else it))
        if len(items) > args.max_list:
            print(f"    ... and {len(items) - args.max_list} more")

    show("missing (model tensor with NO checkpoint entry -> random init)",
         missing)
    show("unexpected (checkpoint entry with no model tensor -> dropped)",
         unexpected)
    show("shape-mismatched (same name, different shape -> raises even with strict=False)",
         mismatched,
         fmt=lambda t: f"{t[0]}: ckpt{t[1]} vs model{t[2]}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print()
    print(f"  model tensors   : {len(msd)}")
    print(f"  total params    : {human(total)}")
    print(f"  trainable params: {human(trainable)} "
          f"({100*trainable/max(total,1):.1f}%)")

    print()
    print("=" * 70)
    print(" load")
    print("=" * 70)
    load_sd_filtered = dict(sd)
    dropped = []
    if mismatched and not args.strict:
        for k, a_shape, m_shape in mismatched:
            load_sd_filtered.pop(k, None)
            dropped.append(k)
        print(f"  dropping {len(dropped)} shape-incompatible entries so the "
              f"16-DoF checkpoint can seed a 28-DoF model:")
        for k, a_shape, m_shape in mismatched[:args.max_list]:
            print(f"    {k}: ckpt{a_shape} -> model{m_shape} (random init)")
        if len(mismatched) > args.max_list:
            print(f"    ... and {len(mismatched) - args.max_list} more")
    elif mismatched:
        print("  --strict: refusing to load, shape mismatches present")
        sys.exit(3)

    result = model.load_state_dict(load_sd_filtered, strict=False)
    still_missing = list(result.missing_keys)
    still_unexpected = list(result.unexpected_keys)
    print(f"  load_state_dict -> missing={len(still_missing)} "
          f"unexpected={len(still_unexpected)}")
    if still_missing:
        random_init = [k for k in still_missing
                       if k not in {m[0] for m in mismatched}]
        print(f"  of which {len(random_init)} have no checkpoint counterpart at all "
              f"(these start random - expected for a DoF change)")

    if args.device == "cuda":
        model = model.to(args.device)
        if args.precision == 16:
            model = model.half()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated()
        print(f"  GPU {torch.cuda.get_device_name(0)}")
        print(f"  weights on GPU  : {(peak - before)/2**30:.2f} GiB")
        print(f"  peak allocated  : {peak/2**30:.2f} GiB")
        print(f"  GPU total       : "
              f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")

    if args.report:
        rep = {
            "config": args.config,
            "checkpoint": args.ckpt,
            "ckpt_format": how,
            "agent_state_dim": int(m.params.agent_state_dim),
            "agent_action_dim": int(m.params.agent_action_dim),
            "model_tensors": len(msd),
            "ckpt_tensors": len(sd),
            "total_params": total,
            "trainable_params": trainable,
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
            "shape_mismatch_count": len(mismatched),
            "shape_mismatched": [
                {"key": k, "ckpt_shape": list(a), "model_shape": list(b)}
                for k, a, b in mismatched
            ],
            "missing": missing,
            "unexpected": unexpected,
        }
        with open(args.report, "w") as f:
            json.dump(rep, f, indent=2)
        print(f"\n  report written to {args.report}")

    print()
    print("=" * 70)
    print(" LOAD TEST: DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()