#!/usr/bin/env python3
"""Re-score saved replay predictions against correctly-aligned ground truth.

Why this exists
---------------
The action head is trained on stride-sampled targets. WMAData.__getitem__ builds

    frame_indices = [start_idx + frame_stride * i for i in range(video_length)]
    actions       = transition_dict['action'][frame_indices, :]

with frame_stride = 2, so the model's output at position i corresponds to source
action index t + 2*i, NOT t + i. The first version of the harness compared
against contiguous actions, which is the wrong alignment.

This matters because the stride also changes the ground truth the error is
measured against: per-step joint deltas roughly double (0.0050 -> 0.0091 rad),
so both the no-motion baseline and the delta ratio move.

It re-scores from the replay.npz files already on disk, so no inference is
re-run - the predictions are unaffected by the bug, only the comparison was.

Usage:
    python docker/rescore_replay.py [--root /experiments/unifolm_wma/replay]
                                   [--stride 2] [--apply]
"""
import argparse
import glob
import json
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/experiments/unifolm_wma/replay")
    p.add_argument("--h5", default="/data_wma/transitions/g1_dex3_graspsquare_1ep/0.h5")
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--horizon", type=int, default=16)
    p.add_argument("--apply", action="store_true",
                   help="rewrite summary.json (default: report only)")
    return p.parse_args()


def score(pred, gt, anchor_state):
    """pred, gt: (H, DoF) radians. anchor_state: (DoF,)."""
    d_pred = float(np.abs(np.diff(pred, axis=0)).mean())
    d_gt = float(np.abs(np.diff(gt, axis=0)).mean())
    no_motion = np.repeat(anchor_state[None, :], pred.shape[0], axis=0)
    mae = float(np.abs(pred - gt).mean())
    mae_nm = float(np.abs(no_motion - gt).mean())
    pf, gf = pred.ravel() - pred.mean(), gt.ravel() - gt.mean()
    den = np.linalg.norm(pf) * np.linalg.norm(gf)
    return {
        "mae": mae,
        "mae_no_motion": mae_nm,
        "ratio_vs_baseline": mae / mae_nm if mae_nm > 1e-9 else float("nan"),
        "delta_pred": d_pred,
        "delta_gt": d_gt,
        "delta_ratio": d_pred / d_gt if d_gt > 1e-9 else float("nan"),
        "corr": float(pf @ gf / den) if den > 1e-9 else float("nan"),
        "endpoint": float(np.abs(pred[-1] - gt[-1]).mean()),
    }


def main():
    a = parse_args()
    import h5py
    with h5py.File(a.h5, "r") as f:
        A = np.array(f["action"][:], dtype=np.float32)
        S = np.array(f["observation.state"][:], dtype=np.float32)

    changed = 0
    for path in sorted(glob.glob(os.path.join(a.root, "*", "*", "summary.json"))):
        npz = os.path.join(os.path.dirname(path), "replay.npz")
        if not os.path.exists(npz):
            continue
        d = np.load(npz)
        pred = d["pred"]
        if "gt" in d:
            gt_old = d["gt"]
        else:
            continue
        anchors = d["anchors"]
        # Two separate teacher-forced arrays: the image-pathway pass and the
        # latent-pathway pass. They used to share one list, which made the
        # second half mis-paired against anchors.
        pred_tf = d["pred_tf"] if "pred_tf" in d and d["pred_tf"].size else None
        pred_tfz = d["pred_tfz"] if "pred_tfz" in d and d["pred_tfz"].size else None

        per_anchor, per_anchor_tf, per_anchor_tfz = [], [], []
        for k, t in enumerate(anchors):
            gi = t + a.stride * np.arange(a.horizon)
            if gi.max() >= len(A):
                continue
            gt = A[gi]
            m = score(pred[k], gt, S[t])
            m["anchor"] = int(t)
            per_anchor.append(m)
            if pred_tf is not None and k < len(pred_tf):
                mtf = score(pred_tf[k], gt, S[t])
                m["tf_mae"] = mtf["mae"]
                m["tf_ratio_vs_baseline"] = mtf["ratio_vs_baseline"]
                m["tf_delta_pred"] = mtf["delta_pred"]
                m["tf_corr"] = mtf["corr"]
                per_anchor_tf.append(mtf)
            if pred_tfz is not None and k < len(pred_tfz):
                mz = score(pred_tfz[k], gt, S[t])
                m["tfz_mae"] = mz["mae"]
                m["tfz_ratio_vs_baseline"] = mz["ratio_vs_baseline"]
                m["tfz_delta_pred"] = mz["delta_pred"]
                m["tfz_corr"] = mz["corr"]
                per_anchor_tfz.append(mz)

        if not per_anchor:
            continue
        keys = list(per_anchor[0].keys())
        agg = {k: float(np.mean([m[k] for m in per_anchor if k in m
                                 and not np.isnan(m[k])])) for k in keys}
        for k in list(agg):
            agg[k + "_std"] = float(np.std([m[k] for m in per_anchor if k in m
                                            and not np.isnan(m[k])]))
        if per_anchor_tf:
            for k in ("mae", "ratio_vs_baseline", "delta_pred", "corr"):
                agg["tf_" + k] = float(np.mean([m[k] for m in per_anchor_tf]))
        if per_anchor_tfz:
            for k in ("mae", "ratio_vs_baseline", "delta_pred", "corr"):
                agg["tfz_" + k] = float(np.mean([m[k] for m in per_anchor_tfz]))

        summary = json.load(open(path))
        old_ratio = summary["aggregate"].get("ratio_vs_baseline")
        summary["aggregate"] = agg
        summary["per_anchor"] = per_anchor
        summary["rescored"] = {
            "stride": a.stride,
            "reason": "action targets are stride-sampled during training, so "
                      "predicted[i] corresponds to source action t + stride*i",
            "previous_ratio_vs_baseline_contiguous": old_ratio,
        }
        changed += 1
        rel = os.path.relpath(os.path.dirname(path), a.root)
        print(f"  {rel:<48} ratio {old_ratio:.2f} -> {agg['ratio_vs_baseline']:.2f}"
              f"   delta x{agg['delta_ratio']:.1f}")
        if a.apply:
            json.dump(summary, open(path, "w"), indent=2)

    print(f"\n{changed} evaluations rescored"
          f"{'' if a.apply else ' (report only - re-run with --apply to write)'}")


if __name__ == "__main__":
    main()