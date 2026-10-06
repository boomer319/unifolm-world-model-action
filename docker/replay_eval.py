#!/usr/bin/env python3
"""Replay evaluation: does the trained policy reproduce the ground-truth actions?

Why this exists
---------------
The training loss is a diffusion loss, not a tracking metric. DreamZero reached
0.005 while its actions were 2.15x worse than standing still, so a low loss
proves nothing about whether the trajectory was memorised. This harness is the
decisive test: it feeds ground-truth observations at several anchors along the
episode and compares the predicted action chunk against the ground truth.

What it reports, per checkpoint
-------------------------------
  mae              mean |pred - gt| over the 16-step chunk, radians
  mae_no_motion    the same, for the trivial predictor that repeats the anchor
                   state for all 16 steps. A model only beats this if it is
                   predicting motion rather than a pose.
  delta_pred       mean |diff| along the predicted chunk. THIS is the number that
                   decides memorisation: the ground truth averages 0.0027-0.0057
                   rad per step, so a model that has learned the trajectory
                   approaches that, and one that has learned a mean pose emits
                   deltas that are far too small (or, as DreamZero did, far too
                   large).
  delta_gt         the same for the ground truth, for reference
  corr             Pearson correlation of the flattened chunk
  endpoint         |pred[-1] - gt[-1]|
  ratio_vs_baseline  mae / mae_no_motion, i.e. < 1 means better than doing
                   nothing

Usage
-----
    python docker/replay_eval.py --config configs/train/config_g1_dex3.yaml \
        --ckpt <checkpoint.ckpt> --anchors 8 --out <dir>
"""
import argparse
import importlib.util
import json
import os
import time

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", default="/data_wma")
    p.add_argument("--dataset", default="g1_dex3_graspsquare_1ep")
    p.add_argument("--view", default="observation.images.cam_left_high")
    p.add_argument("--anchors", type=int, default=8,
                   help="number of GT-anchored evaluation points")
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--ddim-steps", type=int, default=16)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", required=True)
    p.add_argument("--dump-video", action="store_true",
                   help="write the world's generated video per anchor as mp4")
    p.add_argument("--frame-stride", type=int, default=2,
                   help="source-frame stride between generated frames, matching "
                        "the dataset's sampling so video metrics align")
    return p.parse_args()


def load_image_guided_synthesis():
    """Import image_guided_synthesis from the stock evaluation script.

    Imported rather than copied, so the evaluation uses exactly the code path the
    deployed server uses - otherwise we would be measuring a different model.
    """
    spec = importlib.util.spec_from_file_location(
        "real_eval_server",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                     "scripts", "evaluation", "real_eval_server.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.image_guided_synthesis, mod.load_model_checkpoint


def metrics(pred, gt, anchor_state):
    """pred, gt: (H, DoF) in radians. anchor_state: (DoF,)."""
    d_pred = np.abs(np.diff(pred, axis=0)).mean()
    d_gt = np.abs(np.diff(gt, axis=0)).mean()
    no_motion = np.repeat(anchor_state[None, :], pred.shape[0], axis=0)
    mae = float(np.abs(pred - gt).mean())
    mae_nm = float(np.abs(no_motion - gt).mean())
    pf, gf = pred.ravel() - pred.mean(), gt.ravel() - gt.mean()
    denom = (np.linalg.norm(pf) * np.linalg.norm(gf))
    corr = float(pf @ gf / denom) if denom > 1e-9 else float("nan")
    return {
        "mae": mae,
        "mae_no_motion": mae_nm,
        "ratio_vs_baseline": mae / mae_nm if mae_nm > 1e-9 else float("nan"),
        "delta_pred": float(d_pred),
        "delta_gt": float(d_gt),
        "delta_ratio": float(d_pred / d_gt) if d_gt > 1e-9 else float("nan"),
        "corr": corr,
        "endpoint": float(np.abs(pred[-1] - gt[-1]).mean()),
    }


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)

    from omegaconf import OmegaConf
    from unifolm_wma.utils.data import DataModuleFromConfig
    from unifolm_wma.utils.utils import instantiate_from_config

    igs, load_ckpt = load_image_guided_synthesis()

    print(f"=== config ===", flush=True)
    cfg = OmegaConf.load(a.config)
    OmegaConf.resolve(cfg)
    cfg["model"]["params"]["wma_config"]["params"]["use_checkpoint"] = False

    print("=== load model + checkpoint ===", flush=True)
    t0 = time.time()
    model = instantiate_from_config(cfg.model)
    model.perframe_ae = True
    model = load_ckpt(model, a.ckpt)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    print(f"  loaded in {time.time()-t0:.0f}s on {device}", flush=True)

    # The test dataset gives us the exact preprocessing the server uses, plus the
    # min/max statistics that were fitted on THIS data.
    data = instantiate_from_config(cfg.data)
    data.setup()
    # Our config declares only a train split - the evaluation script upstream
    # builds its own test split - so take whichever split exists. It is the same
    # WMAData with the same parameters; what we need from it is the spatial
    # transform and the min/max statistics fitted on this data.
    dset = None
    for split in ("test", "validation", "train"):
        datasets = getattr(data, f"{split}_datasets", None)
        if datasets:
            ds_name = a.dataset if a.dataset in datasets else sorted(datasets)[0]
            dset = datasets[ds_name]
            print(f"  dataset: {ds_name} (split={split})", flush=True)
            break
    if dset is None:
        raise RuntimeError("no dataset split available from the data module")

    # Ground truth from the converted h5 (absolute joint targets, radians).
    import h5py
    h5 = os.path.join(a.data_dir, "transitions", a.dataset, "0.h5")
    with h5py.File(h5, "r") as f:
        gt_actions = np.array(f["action"][:], dtype=np.float32)
        gt_states = np.array(f["observation.state"][:], dtype=np.float32)
    T = len(gt_actions)
    print(f"  episode: {T} frames", flush=True)

    from decord import VideoReader, cpu
    vr = VideoReader(os.path.join(a.data_dir, "videos", a.dataset, a.view, "0.mp4"),
                     ctx=cpu(0))

    # Anchor points: spread over the episode, leaving room for the horizon and
    # for the 2-frame observation history.
    lo, hi = 2, T - a.horizon - 1
    anchors = np.linspace(lo, hi, a.anchors).astype(int).tolist()

    h, w = a.height // 8, a.width // 8
    channels = model.model.diffusion_model.out_channels
    noise_shape = [1, channels, a.horizon, h, w]

    preds, gts, anchors_state = [], [], []
    per_anchor = []
    for t in anchors:
        t0 = time.time()
        frames = vr.get_batch([t - 1, t]).asnumpy()          # (2,H,W,C)
        img = torch.tensor(np.transpose(frames, (0, 3, 1, 2)))  # (T,C,H,W)
        img = dset.spatial_transform(img).unsqueeze(0).to(device)
        img = (img / 255 - 0.5) * 2                          # server's normalize_image

        st = torch.tensor(np.stack([gt_states[t - 1], gt_states[t]]))
        st = dset.normalizer({'observation.state': st})['observation.state']
        st, _ = dset._map_to_uni_state(st, "joint position")
        st = st.unsqueeze(0).to(device)

        ph = torch.zeros((a.horizon, gt_states.shape[-1]), dtype=torch.float32)
        ph, mask = dset._map_to_uni_action(ph, "joint position")
        ph = ph.unsqueeze(0).to(device)

        observation = {'observation.images.top': img,
                       'observation.state': st,
                       'action': ph}

        torch.manual_seed(a.seed + t)      # reproducible diffusion noise per anchor
        with torch.no_grad():
            vid, act, _ = igs(model, "placeholder", observation, noise_shape,
                              ddim_steps=a.ddim_steps, ddim_eta=1.0,
                              unconditional_guidance_scale=1.0,
                              fs=30 / 2, timestep_spacing="uniform_trailing",
                              guidance_rescale=0.7)
        act = act[..., mask[0] == 1.0][0].cpu()
        act = dset.unnormalizer({'action': act})['action'].numpy().astype(np.float32)

        gt = gt_actions[t:t + a.horizon]

        # The video branch is trained jointly with the action branch, so it is a
        # second, independent read on whether anything was memorised. Compare the
        # generated frames against ground truth at the dataset's own stride.
        vm = {}
        gt_frames = vr.get_batch([t + a.frame_stride * i
                                  for i in range(a.horizon)]).asnumpy()
        v = vid[0].detach().cpu().float().clamp(-1, 1)          # (C,T,H,W)
        v = ((v + 1) / 2 * 255).permute(1, 2, 3, 0).numpy()      # (T,H,W,C) uint8-ish
        gt_im = np.transpose(gt_frames, (0, 2, 3, 1)).astype(np.float32)
        v = np.clip(v, 0, 255)
        mse = float(((v - gt_im) ** 2).mean())
        vm = {"video_psnr": float(10 * np.log10(255.0 ** 2 / max(mse, 1e-9))),
              "video_mae_px": float(np.abs(v - gt_im).mean()),
              "gt_frame_mae_px": float(np.abs(gt_im - gt_im.mean()).mean())}
        if a.dump_video:
            import imageio
            imageio.mimsave(os.path.join(a.out, f"video_anchor{t:05d}.mp4"),
                            [x.astype(np.uint8) for x in v], fps=15)
            half = [np.concatenate([x.astype(np.uint8),
                                    np.clip(y, 0, 255).astype(np.uint8)], axis=1)
                    for x, y in zip(gt_im, v)]
            imageio.mimsave(os.path.join(a.out, f"cmp_anchor{t:05d}.mp4"),
                            half, fps=15)

        m = metrics(act, gt, gt_states[t])
        m.update(vm)
        m["anchor"] = int(t)
        m["seconds"] = round(time.time() - t0, 2)
        per_anchor.append(m)
        preds.append(act)
        gts.append(gt)
        anchors_state.append(gt_states[t])
        print(f"  anchor {t:5d}: mae {m['mae']:.4f} (baseline {m['mae_no_motion']:.4f})"
              f"  delta {m['delta_pred']:.4f} vs gt {m['delta_gt']:.4f}"
              f"  corr {m['corr']:+.3f}  [{m['seconds']:.1f}s]", flush=True)

    agg = {}
    for k in ("mae", "mae_no_motion", "ratio_vs_baseline", "delta_pred",
              "delta_gt", "delta_ratio", "corr", "endpoint",
              "video_psnr", "video_mae_px"):
        vals = [m[k] for m in per_anchor if not np.isnan(m[k])]
        agg[k] = float(np.mean(vals)) if vals else float("nan")
        agg[k + "_std"] = float(np.std(vals)) if vals else float("nan")

    summary = {
        "checkpoint": a.ckpt,
        "config": a.config,
        "dataset": a.dataset,
        "anchors": a.horizon and len(anchors),
        "horizon": a.horizon,
        "ddim_steps": a.ddim_steps,
        "seed": a.seed,
        "aggregate": agg,
        "per_anchor": per_anchor,
        "verdict": {
            "beats_no_motion_baseline": agg["ratio_vs_baseline"] < 1.0,
            "delta_within_5x_of_gt": 0.2 < agg["delta_ratio"] < 5.0,
        },
    }
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        os.path.join(a.out, "replay.npz"),
        pred=np.stack(preds), gt=np.stack(gts),
        anchors=np.array([m["anchor"] for m in per_anchor]),
        anchor_states=np.stack(anchors_state))

    print()
    print(f"  MAE              {agg['mae']:.4f}  (no-motion baseline {agg['mae_no_motion']:.4f},"
          f" ratio {agg['ratio_vs_baseline']:.3f})")
    print(f"  per-step |delta| {agg['delta_pred']:.4f}  vs GT {agg['delta_gt']:.4f}"
          f"  (ratio {agg['delta_ratio']:.1f}x)")
    print(f"  correlation      {agg['corr']:+.4f}")
    print(f"  video PSNR       {agg['video_psnr']:.2f} dB"
          f"  (pixel MAE {agg['video_mae_px']:.2f})")
    print(f"  verdict          beats_no_motion={summary['verdict']['beats_no_motion_baseline']}"
          f"  delta_within_5x_of_gt={summary['verdict']['delta_within_5x_of_gt']}")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()