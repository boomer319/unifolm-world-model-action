#!/usr/bin/env python3
"""Collect every replay evaluation into one table.

Reads all summary.json files under the replay tree and emits a markdown table
ordered by run and checkpoint, plus a per-run summary line. This is the artifact
that goes into the thesis: one table comparing Base against Dual, two seeds each,
and the 10-episode diversity arm against the 1-episode arm.

Usage:
    python docker/replay_table.py [--root /experiments/unifolm_wma/replay]
                                  [--csv out.csv] [--md out.md]
"""
import argparse
import glob
import json
import os
import re

# The number that decides memorisation. Ground truth averages 0.0027-0.0057 rad
# per step, so a model that has learnt the trajectory approaches 1.0 here.
COLS = [
    ("mae", "MAE", "%.4f"),
    ("mae_no_motion", "no-motion", "%.4f"),
    ("ratio_vs_baseline", "x baseline", "%.2f"),
    ("delta_pred", "pred |d|", "%.4f"),
    ("delta_gt", "GT |d|", "%.4f"),
    ("delta_ratio", "x GT", "%.1f"),
    ("corr", "corr", "%+.3f"),
    ("endpoint", "endpoint", "%.4f"),
    ("video_psnr", "vid PSNR", "%.2f"),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/experiments/unifolm_wma/replay")
    p.add_argument("--csv")
    p.add_argument("--md")
    return p.parse_args()


def main():
    a = parse_args()
    rows = []
    for f in sorted(glob.glob(os.path.join(a.root, "*", "*", "summary.json"))):
        run = os.path.basename(os.path.dirname(os.path.dirname(f)))
        ckpt = os.path.basename(os.path.dirname(f))
        try:
            d = json.load(open(f))
        except Exception as e:                                  # noqa: BLE001
            print(f"!! {run}/{ckpt}: unreadable ({e})")
            continue
        agg = d.get("aggregate", {})
        m = re.search(r"step(\d+)", ckpt)
        rows.append({
            "run": run,
            "ckpt": ckpt,
            "step": int(m.group(1)) if m else -1,
            **{k: agg.get(k, float("nan")) for k, _, _ in COLS},
        })

    if not rows:
        print("no replay summaries found - has a sweep run yet?")
        return

    def key(r):
        # keep the un-finetuned baseline first, then runs and steps in order
        return (0 if r["step"] < 0 else 1, r["run"], r["step"])

    rows.sort(key=key)

    header = ["run", "ckpt"] + [lbl for _, lbl, _ in COLS]
    md = ["| " + " | ".join(header) + " |",
          "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        cells = [r["run"], r["ckpt"]]
        for k, _, fmt in COLS:
            v = r[k]
            cells.append("-" if v != v else (fmt % v))          # NaN -> '-'
        md.append("| " + " | ".join(cells) + " |")

    table = "\n".join(md)
    print(table)
    print()

    # Per-run roll-up, so each arm gets one line of verdict.
    byrun = {}
    for r in rows:
        byrun.setdefault(r["run"], []).append(r)
    print("per run (best checkpoint by ratio_vs_baseline):")
    for run, rs in sorted(byrun.items()):
        best = min(rs, key=lambda r: r["ratio_vs_baseline"])
        last = max(rs, key=lambda r: r["step"])
        print(f"  {run:<30} n={len(rs):>2}  best {best['ckpt']} "
              f"ratio {best['ratio_vs_baseline']:.2f} delta x{best['delta_ratio']:.1f}"
              f"  |  last {last['ckpt']} ratio {last['ratio_vs_baseline']:.2f} "
              f"delta x{last['delta_ratio']:.1f}")

    if a.md:
        open(a.md, "w").write(table + "\n")
        print(f"\nwrote {a.md}")
    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=header)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.csv}")


if __name__ == "__main__":
    main()