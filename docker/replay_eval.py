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
    p.add_argument("--teacher-force-z", action="store_true",
                   help="complete teacher forcing: substitute the video branch's "
                        "noisy latent with ground truth, so the world-model "
                        "features the action head consumes also come from truth")
    p.add_argument("--teacher-force", action="store_true",
                   help="feed the action head the true next frames in place of "
                        "the last observed ones, paired against the normal path")
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

    # ---------------------------------------------------------------------
    # Teacher forcing.
    #
    # The action head has TWO visual pathways (ConditionalUnet1D.forward takes
    # imagen_cond, the world model's UNet features, AND cond["image"], the raw
    # observation frames). Wrapping action_unet.forward and substituting only
    # cond[0] gives the head the true next frames while leaving the world model,
    # its sampling loop and the KV/cache path completely untouched. That is a
    # far cleaner instrument than the equivalent test on DreamZero, where
    # video and action share one DiT and teacher forcing had to be cut into the
    # denoising loop.
    #
    # If actions improve markedly, the head can map visuals to actions and the
    # limit is what it can observe at inference. If they do not, the head never
    # learned the mapping.
    # ---------------------------------------------------------------------
    _orig_head_forward = None
    if a.teacher_force:
        # Verified by walking the live module tree rather than by reading it:
        #   model                             LatentVisualDiffusion
        #    .model                           DiffusionWrapper
        #     .diffusion_model                WMAModel
        #      .action_unet                   ConditionalUnet1D   <- the head
        # Guessing this twice cost two runs; introspect the tree instead.
        head = model.model.diffusion_model.action_unet
        for _obj, _attr, _where in ((model, "n_obs_steps_acting", "model"),
                                    (head, "forward", "action_unet")):
            if not hasattr(_obj, _attr):
                raise AttributeError(
                    f"teacher forcing needs {_where}.{_attr}, which does not "
                    f"exist - check the module tree before running")
        _orig_head_forward = head.forward

        def _tf_forward(sample, timestep, imagen_cond=None, cond=None, **kw):
            if cond is not None and _TF_FRAMES.get("x") is not None:
                cond = [_TF_FRAMES["x"], cond[1]]      # images -> GT, state kept
            return _orig_head_forward(sample, timestep, imagen_cond, cond, **kw)

        head.forward = _tf_forward
        print("  teacher forcing: action head will see the true next frames",
              flush=True)

    _TF_FRAMES = {"x": None}

    # ---------------------------------------------------------------------
    # Complete teacher forcing.
    #
    # The action head also receives imagen_cond - the world model's multi-scale
    # UNet features - so substituting only cond["image"] leaves the second, and
    # arguably the richer, pathway untouched. Substituting the video branch's
    # noisy latent instead makes every feature derive from ground truth, and
    # because the latent is re-noised at each step with the model's own
    # q_sample, the noise level still matches the step being evaluated.
    # ---------------------------------------------------------------------
    _TFZ = {"z": None}
    if a.teacher_force_z:
        # Class hierarchy, verified rather than assumed:
        #   DDPM -> LatentDiffusion (encode_first_stage) -> LatentVisualDiffusion
        # so the outer model carries both encode_first_stage and q_sample;
        # only the WMAModel that owns action_unet lives under DiffusionWrapper.
        _wm = model.model.diffusion_model
        _orig_wm_forward = _wm.forward

        def _wm_forward_tf(x, *args, **kw):
            if _TFZ.get("z") is not None:
                # WMAModel.forward is (x, x_action, x_state, timesteps, ...);
                # the sampler passes timesteps positionally, but accept the
                # keyword form too rather than assume.
                ts = args[2] if len(args) > 2 else kw.get("timesteps")
                if ts is None:
                    raise RuntimeError("teacher forcing could not find the "
                                       "timestep argument")
                # DDIM hands the timestep over as a float tensor; q_sample
                # gathers the schedule with it, which needs an integer index.
                if not torch.is_tensor(ts):
                    ts = torch.tensor([ts])
                ts = ts.long()
                z0 = _TFZ["z"]
                if z0.shape[-3] != x.shape[-3]:
                    z0 = torch.nn.functional.interpolate(
                        z0, size=x.shape[-3:], mode="nearest")
                eps = torch.randn_like(x)
                z0 = z0.expand_as(x) if z0.shape[1] == 1 else z0
                # q_sample's signature is (x_start, t, noise) - x_start FIRST.
                # Passing (ts, z0, eps) put the float latent where the schedule
                # index goes, and gather() rejected it with "Expected dtype
                # int64 for index", which sent me chasing a dtype problem that
                # was really an argument-order one. Keywords, so the order is
                # not something to remember. Upstream calls it q_sample(x0, ts).
                x = model.q_sample(x_start=z0, t=ts, noise=eps)
                _TFZ["fired"] = _TFZ.get("fired", 0) + 1
            return _orig_wm_forward(x, *args, **kw)

        _wm.forward = _wm_forward_tf
        print("  teacher forcing (complete): video branch latent comes from GT",
              flush=True)

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

    # Anchor points: spread over the episode. The limit is set by the VIDEO
    # comparison, not the actions: at frame_stride 2 the last compared ground
    # truth frame is t + 2*(horizon-1), so an anchor placed for the action chunk
    # alone can run past the end of the episode and decord raises.
    lo, hi = 2, T - a.frame_stride * (a.horizon - 1) - 1
    if hi <= lo:
        raise RuntimeError(
            f"episode too short for {a.horizon} frames at stride "
            f"{a.frame_stride}: T={T}")
    anchors = np.linspace(lo, hi, a.anchors).astype(int).tolist()

    h, w = a.height // 8, a.width // 8
    channels = model.model.diffusion_model.out_channels
    noise_shape = [1, channels, a.horizon, h, w]

    preds, preds_tf, gts, anchors_state = [], [], [], []
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

        def _run(tag=""):
            torch.manual_seed(a.seed + t)   # same noise for both paths
            with torch.no_grad():
                v, ac, _ = igs(model, "placeholder", observation, noise_shape,
                               ddim_steps=a.ddim_steps, ddim_eta=1.0,
                               unconditional_guidance_scale=1.0,
                               fs=30 / 2, timestep_spacing="uniform_trailing",
                               guidance_rescale=0.7)
            ac = ac[..., mask[0] == 1.0][0].cpu()
            return v, dset.unnormalizer({'action': ac})['action'].numpy().astype(np.float32)

        gt_for_tf = None
        if a.teacher_force:
            # The next two true frames, in the head's (B, C, T, H, W) layout.
            n_act = model.n_obs_steps_acting   # on the LatentVisualDiffusion
            fut = vr.get_batch([t + a.frame_stride * i for i in range(n_act)]
                              ).asnumpy()
            ft = dset.spatial_transform(
                torch.tensor(np.transpose(fut, (0, 3, 1, 2)))).to(device)
            ft = (ft / 255 - 0.5) * 2
            gt_for_tf = ft.permute(1, 0, 2, 3).unsqueeze(0)   # (1,C,T,H,W)

        z_gt = None
        if a.teacher_force_z:
            gt_t = dset.spatial_transform(
                torch.tensor(np.transpose(
                    vr.get_batch([t + a.frame_stride * i
                                  for i in range(a.horizon)]).asnumpy(),
                    (0, 3, 1, 2)))).to(device)
            gt_t = (gt_t / 255 - 0.5) * 2
            with torch.no_grad():
                # gt_t is (T, C, H, W); the VAE reads dim 1 as channels, so it
                # needs (B, C, T, H, W).
                z_gt = model.encode_first_stage(
                    gt_t.permute(1, 0, 2, 3).unsqueeze(0))

        _TFZ["z"] = z_gt
        vid, act = _run("normal")
        _TFZ["z"] = None
        act_tfz = None
        if a.teacher_force_z:
            _TFZ["z"] = z_gt
            _, act_tfz = _run("tfz")
            _TFZ["z"] = None
        act_tf = None
        if a.teacher_force:
            _TF_FRAMES["x"] = gt_for_tf
            _, act_tf = _run("tf")
            _TF_FRAMES["x"] = None
        # Ground truth on the STRIDE the model was trained on. WMAData samples
        # its targets as action[start_idx + frame_stride*i], so the chunk we
        # predict ends at t + stride*(horizon-1) and index i pairs with
        # t + stride*i - not with t + i. Scoring against contiguous actions
        # understates the model: it reports ratio 2.20 where 1.68 is correct.
        gt = gt_actions[t + a.frame_stride * np.arange(a.horizon)]

        # The video branch is trained jointly with the action branch, so it is a
        # second, independent read on whether anything was memorised. Compare the
        # generated frames against ground truth at the dataset's own stride.
        vm = {}
        # decord returns (T,H,W,C) in the source resolution (640x480); the model
        # works at 320x512 after resize_center_crop. Put ground truth through the
        # SAME transform the observations go through, so the comparison is like
        # for like instead of against a differently sized frame.
        gt_frames = vr.get_batch([t + a.frame_stride * i
                                  for i in range(a.horizon)]).asnumpy()
        gt_t = torch.tensor(np.transpose(gt_frames, (0, 3, 1, 2)))   # (T,C,H,W)
        gt_im = dset.spatial_transform(gt_t).permute(0, 2, 3, 1).numpy().astype(np.float32)

        v = vid[0].detach().cpu().float().clamp(-1, 1)          # (C,T,H,W)
        v = ((v + 1) / 2 * 255).permute(1, 2, 3, 0).numpy()      # (T,H,W,C)
        v = np.clip(v, 0, 255)
        mse = float(((v - gt_im) ** 2).mean())
        vm = {"video_psnr": float(10 * np.log10(255.0 ** 2 / max(mse, 1e-9))),
              "video_mae_px": float(np.abs(v - gt_im).mean()),
              "gt_frame_mae_px": float(np.abs(gt_im - gt_im.mean()).mean())}
        if a.dump_video:
            # Upstream's own writer, unifolm_wma.utils.save_video.tensor_to_mp4.
            # It takes (b, c, t, h, w) in -1..1 and does the permute to the
            # (T, H, W, C) layout torchvision.io.write_video wants. Hand-rolling
            # that permute got it wrong twice (imageio could not infer a codec,
            # then write_video rejected a (3, 320, 512) frame), so the packaged
            # utility is both shorter and correct.
            from unifolm_wma.utils.save_video import tensor_to_mp4
            tensor_to_mp4(vid.detach().cpu(), os.path.join(
                a.out, f"video_anchor{t:05d}.mp4"), fps=15)
            # Ground truth alongside, built the same way so both clips share a
            # writer and a resolution: ref on the left, prediction on the right.
            # gt_im is still 0..255 (the spatial transform preserved uint8), and
            # tensor_to_mp4 rescales from -1..1, so the conversion has to be
            # explicit. Feeding it 0..255 directly and only doing *2-1 mapped most
            # values past +1, which the writer clamps to white - the ground-truth
            # half of the comparison clips came out white with green speckle.
            ref = (torch.from_numpy(gt_im).float().permute(3, 0, 1, 2)
                   / 255.0 * 2.0 - 1.0)                              # (C,T,H,W)
            gen = vid[0].detach().cpu()
            side = torch.cat([ref, gen], dim=3).unsqueeze(0)           # (1,C,T,2H,W)
            tensor_to_mp4(side, os.path.join(a.out, f"cmp_anchor{t:05d}.mp4"),
                          fps=15)

        m = metrics(act, gt, gt_states[t])
        m.update(vm)
        if act_tf is not None:
            mtf = metrics(act_tf, gt, gt_states[t])
            m["tf_mae"] = mtf["mae"]
            m["tf_delta_pred"] = mtf["delta_pred"]
            m["tf_ratio_vs_baseline"] = mtf["ratio_vs_baseline"]
            m["tf_corr"] = mtf["corr"]
            preds_tf.append(act_tf)
        if act_tfz is not None:
            mz = metrics(act_tfz, gt, gt_states[t])
            m["tfz_mae"] = mz["mae"]
            m["tfz_ratio_vs_baseline"] = mz["ratio_vs_baseline"]
            m["tfz_delta_pred"] = mz["delta_pred"]
            m["tfz_corr"] = mz["corr"]
            preds_tf.append(act_tfz)
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
              "video_psnr", "video_mae_px",
              "tf_mae", "tf_ratio_vs_baseline", "tf_delta_pred", "tf_corr",
              "tfz_mae", "tfz_ratio_vs_baseline", "tfz_delta_pred", "tfz_corr"):
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
        pred=np.stack(preds),
        pred_tf=(np.stack(preds_tf) if preds_tf else np.zeros(0)),
        gt=np.stack(gts),
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
    if preds_tf and "tfz_mae" in agg and agg["tfz_mae"] == agg["tfz_mae"]:
        print(f"  TF (complete)    MAE {agg['tfz_mae']:.4f}"
              f"  ratio {agg['tfz_ratio_vs_baseline']:.3f}"
              f"  delta {agg['tfz_delta_pred']:.4f}"
              f"  corr {agg['tfz_corr']:+.4f}")
    if preds_tf and "tf_mae" in agg and agg["tf_mae"] == agg["tf_mae"]:
        print(f"  TEACHER FORCED   MAE {agg['tf_mae']:.4f}"
              f" (baseline {agg['mae_no_motion']:.4f},"
              f" ratio {agg['tf_ratio_vs_baseline']:.3f})"
              f"  delta {agg['tf_delta_pred']:.4f}"
              f"  corr {agg['tf_corr']:+.4f}")
    print(f"  verdict          beats_no_motion={summary['verdict']['beats_no_motion_baseline']}"
          f"  delta_within_5x_of_gt={summary['verdict']['delta_within_5x_of_gt']}")
    print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()