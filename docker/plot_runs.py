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
    p.add_argument("--bin", type=int, default=200,
                   help="bin size in updates; the per-update loss is far too "
                        "noisy to plot raw or with a short moving average")
    p.add_argument("--table", action="store_true", help="also print a text table")
    p.add_argument("--logy", action="store_true", help="log y-axis")
    p.add_argument("--skip", type=int, default=0,
                   help="skip the first N updates (e.g. 50 to hide warmup)")
    return p.parse_args()


def binned(xs, ys, n):
    """Mean per bin of n updates.

    Every update draws a random diffusion timestep AND random noise, so the
    per-update loss has enormous variance: a 25-step moving average still swings
    between -0.2 and +0.2 and can even cross zero. Binning ~200 samples averages
    that down and shows the actual trend. Returns (bin_centres, bin_means).
    """
    pairs = [(x, y) for x, y in zip(xs, ys) if y == y]
    if not pairs:
        return [], []
    pairs.sort()
    out_x, out_y = [], []
    cur, acc = [], []
    for x, y in pairs:
        cur.append(x)
        acc.append(y)
        if len(cur) == n:
            out_x.append(sum(cur) / n)
            out_y.append(sum(acc) / len(acc))
            cur, acc = [], []
    if cur:
        out_x.append(sum(cur) / len(cur))
        out_y.append(sum(acc) / len(acc))
    return out_x, out_y


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
            xs2, ys2 = binned(xs, ys, a.bin)
            ax.plot(xs2, ys2, color=colours[i % 10], lw=1.8, marker="o",
                    ms=2.5, label=f"{r}  (last bin {ys2[-1]:.4f})")
        if a.logy:
            ax.set_yscale("log")
        ax.set_title(k.replace("loss_", "").replace("_step", ""))
        ax.set_xlabel("weight update")
        ax.grid(alpha=0.25)
        if j == 0:
            ax.set_ylabel("loss")
        ax.legend(fontsize=7.5, frameon=False)
    ttl = a.title + f"   (mean per {a.bin} updates)"
    if a.skip:
        ttl += f"   [first {a.skip} updates hidden]"
    fig.suptitle(ttl)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=110)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()