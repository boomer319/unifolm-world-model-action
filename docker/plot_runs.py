#!/usr/bin/env python3
"""Plot loss curves for the recorded runs, and optionally print a live table.

TensorBoard is unnecessary here: the RunRecorder already writes every logged
metric to metrics.csv (one row per weight update), which is both plainer and
richer than the tfevents copy Lightning also keeps. This reads the CSVs and
renders one PNG with all runs overlaid, plus a text progress table.

Usage:
    python docker/plot_runs.py --runs-dir /docker_data/runs \
        --runs overfit_dual_s20250912 overfit_base_s20250912 \
        --out /docker_data/runs/curves.png
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(runs_dir, run):
    path = os.path.join(runs_dir, run, "metrics.csv")
    if not os.path.exists(path):
        return None, None
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return None, None
    series = []
    for r in rows:
        m = {}
        for k, v in r.items():
            if k.startswith("m_") and v not in ("", None):
                name = k[2:]
                for pre in ("train/", "val/", "test/"):
                    if name.startswith(pre):
                        name = name[len(pre):]
                m[name] = float(v)
        series.append((int(r.get("update") or 0), float(r["wall_s"]), m))
    summ = None
    sp = os.path.join(runs_dir, run, "summary.json")
    if os.path.exists(sp):
        summ = json.load(open(sp))
    return series, summ


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="/docker_data/runs")
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--out", default="/docker_data/runs/curves.png")
    p.add_argument("--keys", nargs="*",
                   default=["loss_action_step", "loss_state_step", "loss_step"])
    p.add_argument("--title", default="UnifoLM-WMA 1-episode G1 Dex3 overfit")
    p.add_argument("--smooth", type=int, default=25,
                   help="rolling-mean window for readability (0 = raw)")
    p.add_argument("--table", action="store_true", help="also print a text table")
    p.add_argument("--linear", action="store_true",
                   help="linear y-axis. The default is log, because the loss "
                        "drops ~20x in the first 50 updates and a linear axis "
                        "then compresses everything afterwards into a flat line "
                        "- which reads as 'plateaued' when it is still falling")
    p.add_argument("--skip", type=int, default=0,
                   help="skip the first N updates (e.g. 50 to hide warmup)")
    return p.parse_args()


def smooth(xs, ys, n):
    if n <= 1 or len(ys) < n:
        return xs, ys
    out, acc = [], []
    s = 0.0
    for i, y in enumerate(ys):
        s += y
        acc.append(s)
        if i >= n:
            s -= ys[i - n]
        out.append((acc[-1] - (acc[-n - 1] if len(acc) > n else 0.0)) /
                   (n if len(acc) > n else len(acc)))
    return xs, out


def main():
    a = parse_args()
    data = {}
    for r in a.runs:
        s, summ = load(a.runs_dir, r)
        if s:
            data[r] = (s, summ)
        else:
            print(f"!! {r}: no metrics.csv yet")

    if not data:
        print("nothing to plot")
        return

    # ------------------------------------------------------------ text table
    if a.table:
        print(f"\n{'run':26} {'upd':>7} {'wall_h':>7} {'s/upd':>7} "
              f"{'act':>8} {'state':>8} {'video':>8}")
        print("-" * 76)
        for r, (s, _) in data.items():
            w = s[-1][1]
            d = sorted(s[i + 1][1] - s[i][1] for i in range(3, len(s) - 1))
            spu = d[len(d) // 2] if d else float("nan")
            last = s[-1][2]
            def g(k):
                return last.get(k, float("nan"))
            print(f"{r:26} {s[-1][0]:7d} {w/3600:7.2f} {spu:7.2f} "
                  f"{g('loss_action_step'):8.4f} {g('loss_state_step'):8.4f} "
                  f"{g('loss_step'):8.4f}")
        print()

    # ----------------------------------------------------------------- plots
    n = len(a.keys)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.0), squeeze=False)
    colours = plt.cm.tab10.colors
    for j, k in enumerate(a.keys):
        ax = axes[0][j]
        for i, (r, (s, _)) in enumerate(sorted(data.items())):
            xs = [row[0] for row in s if row[0] >= a.skip]
            ys = [row[2].get(k, float("nan")) for row in s if row[0] >= a.skip]
            xs2, ys2 = smooth(xs, ys, a.smooth)
            ax.plot(xs2, ys2, color=colours[i % 10], lw=1.6,
                    label=f"{r}  (final {ys[-1]:.4f})")
        ax.set_title(k.replace("loss_", "").replace("_step", ""))
        ax.set_xlabel("weight update")
        ax.grid(alpha=0.25)
        if j == 0:
            ax.set_ylabel("loss")
        ax.legend(fontsize=7.5, frameon=False)
    ttl = a.title + (f"   ({a.smooth}-step moving average)" if a.smooth > 1 else "")
    ttl += "   [log y]" if not a.linear else "   [linear y]"
    if a.skip:
        ttl += f"   [first {a.skip} updates hidden]"
    fig.suptitle(ttl)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()