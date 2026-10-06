#!/usr/bin/env python3
"""Perceptual video metrics, computed from the comparison clips already on disk.

Why this exists
---------------
The replay harness originally reported video PSNR as "an independent read on
memorisation". That was a poor choice for this content and it understated the
result. The scene is a black gripper against a white table - an extremely
high-contrast edge - so a one or two pixel misalignment produces a large pixel
error while being invisible to the eye. Generated frames at 22 dB PSNR look
remarkably faithful to a human.

So PSNR is reported alongside SSIM (structural similarity, which tolerates small
misalignment) and LPIPS (a learned perceptual distance) when its weights are
available. The decision on which to trust is then made on evidence rather than on
a single misleading number.

No re-inference is needed: each cmp_anchor*.mp4 holds ground truth on the left
and the prediction on the right, so both are recoverable from the clip.

Usage:
    python docker/video_metrics.py [--root /experiments/unifolm_wma/replay]
                                   [--csv out.csv]
"""
import argparse
import csv
import glob
import os
import re

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/experiments/unifolm_wma/replay")
    p.add_argument("--csv")
    p.add_argument("--lpips", action="store_true",
                   help="also compute LPIPS if its weights can be loaded")
    return p.parse_args()


def get_ssim():
    from torchmetrics.image import StructuralSimilarityIndexMeasure
    return StructuralSimilarityIndexMeasure(data_range=1.0)


def get_lpips():
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
    return LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)


def read_clip(path):
    import decord
    vr = decord.VideoReader(path)
    a = vr.get_batch(list(range(len(vr)))).asnumpy().astype(np.float32) / 255.0
    half = a.shape[2] // 2
    return a[:, :, :half], a[:, :, half:]


def main():
    args = parse_args()
    ssim = get_ssim()
    lpips = get_lpips() if args.lpips else None
    if lpips is not None:
        try:
            lpips.eval()
        except Exception as e:                                   # noqa: BLE001
            print(f"  LPIPS unavailable ({type(e).__name__}: {e}) - continuing without it")
            lpips = None

    rows = []
    for clip in sorted(glob.glob(os.path.join(args.root, "*", "step*", "cmp_anchor*.mp4"))):
        run = os.path.basename(os.path.dirname(os.path.dirname(clip)))
        step_dir = os.path.basename(os.path.dirname(clip))
        anchor = int(re.search(r"(\d+)", os.path.basename(clip)).group(1))
        try:
            gt, pred = read_clip(clip)
        except Exception as e:                                   # noqa: BLE001
            print(f"!! {run}/{step_dir}/{os.path.basename(clip)}: {e}")
            continue

        g = torch.from_numpy(gt).permute(0, 3, 1, 2)              # (T,C,H,W)
        p = torch.from_numpy(pred).permute(0, 3, 1, 2)
        mse = float(((p - g) ** 2).mean())
        row = {
            "run": run, "ckpt": step_dir, "anchor": anchor,
            "psnr": float(10 * np.log10(1.0 / max(mse, 1e-12))),
            "ssim": float(ssim(g.clamp(0, 1), p.clamp(0, 1))),
        }
        if lpips is not None:
            row["lpips"] = float(lpips(g.clamp(0, 1), p.clamp(0, 1)))
        rows.append(row)

    if not rows:
        print("no comparison clips found")
        return

    # Aggregate per checkpoint so each cell is comparable to the replay table.
    agg = {}
    for r in rows:
        k = (r["run"], r["ckpt"])
        agg.setdefault(k, []).append(r)

    keys = ["psnr", "ssim"] + (["lpips"] if lpips is not None else [])
    print(f"Video similarity to ground truth, per checkpoint "
          f"({len(rows)//max(len(agg),1)} anchors each)\n")
    runs = sorted(set(k[0] for k in agg))
    steps = sorted(set(int(re.search(r"(\d+)", k[1]).group(1)) for k in agg))
    for metric in keys:
        better_high = metric != "lpips"
        print(f"  {metric.upper()}  (higher is better: {better_high})")
        print(f"    {'run':<26}" + "".join(f"{s:>8}" for s in steps))
        for run in runs:
            cells = []
            for s in steps:
                v = [np.mean([x[metric] for x in agg[k]])
                     for k in agg if k[0] == run
                     and int(re.search(r"(\d+)", k[1]).group(1)) == s]
                cells.append(f"{v[0]:>8.3f}" if v else f"{'-':>8}")
            print(f"    {run:<26}" + "".join(cells))
        print()

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}")


if __name__ == "__main__":
    main()