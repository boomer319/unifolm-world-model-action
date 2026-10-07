#!/usr/bin/env python3
"""Thesis figures: the memorisation study as a set of plots and one summary table.

Four panels, chosen because each answers a different question the thesis asks:

  1. action loss by arm          - did anything learn?
  2. ratio vs no-motion baseline - does it track better than standing still?
                                  1.0 is the line that matters; everything above
                                  it means the model is worse than doing nothing.
  3. per-step |delta| vs ground truth
                                - the decisive number. Ground truth is
                                  ~0.0091 rad/step at this stride; a memorised
                                  trajectory approaches it.
  4. teacher forcing            - how much of the error is the world's own
                                  video versus the action head.

Everything is read from the artefacts already on disk: the replay summaries,
the video metrics CSV and the training metrics. No inference is re-run.

Usage:
    python docker/make_figures.py [--exp /experiments/unifolm_wma] [--out ...]
"""
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Arms are grouped by what they vary, so the legend carries the comparison and
# not just a list of names.
GROUPS = {
    "n_obs=2, Base (no policy head)": ("#1f77b4", ["overfit_base_s20250912",
                                                    "overfit_base_s42"]),
    "n_obs=2, Dual (warm head)": ("#d62728", ["overfit_dual_s20250912",
                                              "overfit_dual_s42"]),
    "continue from step 9000": ("#2ca02c", ["cont_base_s20250912",
                                             "cont_base_s42"]),
    "n_obs=4 (observability test)": ("#9467bd", ["obs4_base_s20250912",
                                                 "obs4_base_s42"]),
    "10 episodes (density)": ("#ff7f0e", ["overfit_dual_10ep_s20250912"]),
}
BASELINE_COLOUR = "#7f7f7f"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp", default="/experiments/unifolm_wma")
    p.add_argument("--runs", default="/docker_data/runs")
    p.add_argument("--out", default=None)
    return p.parse_args()


def load_replay(exp):
    """(run, step) -> aggregate, for every rescored evaluation."""
    out = {}
    for f in glob.glob(os.path.join(exp, "replay", "*", "step*", "summary.json")):
        run = os.path.basename(os.path.dirname(os.path.dirname(f)))
        m = re.search(r"step(\d+)", os.path.basename(os.path.dirname(f)))
        if not m:
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        out[(run, int(m.group(1)))] = d.get("aggregate", {})
    return out


def load_losses(runs):
    """run -> (steps, binned action loss). Binned, not smoothed: the per-update
    diffusion loss is stochastic and a moving average of 25 swings by +/-0.2,
    which made an early reading of this data wrong twice."""
    out = {}
    for run in glob.glob(os.path.join(runs, "*")):
        p = os.path.join(run, "metrics.csv")
        if not os.path.exists(p):
            continue
        vals, steps = [], []
        with open(p) as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                v = row.get("m_train/loss_action")
                if v not in (None, ""):
                    vals.append(float(v))
                    steps.append(i)
        if not vals:
            continue
        n = len(vals)
        nb = 20
        xs = [steps[min(n - 1, j * n // nb)] for j in range(nb)]
        ys = [float(np.mean(vals[j * n // nb:(j + 1) * n // nb]))
              for j in range(nb)]
        out[os.path.basename(run)] = (xs, ys)
    return out


def series(replay, runs):
    """run -> (steps, values) for one metric, in step order."""
    d = defaultdict(list)
    for (run, step), agg in replay.items():
        if run in runs:
            d[run].append((step, agg))
    res = {}
    for run, items in d.items():
        items.sort()
        res[run] = ([i[0] for i in items], [i[1] for i in items])
    return res


def main():
    a = parse_args()
    outdir = a.out or os.path.join(a.exp, "figures")
    os.makedirs(outdir, exist_ok=True)
    replay = load_replay(a.exp)
    losses = load_losses(a.runs)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax_loss, ax_ratio, ax_delta, ax_tf = axes.ravel()

    # ---- 1. action loss ------------------------------------------------
    for label, (colour, runs) in GROUPS.items():
        for run in runs:
            if run in losses:
                x, y = losses[run]
                ax_loss.plot(x, y, color=colour, alpha=0.85, lw=1.4,
                             label=label if run == runs[0] else None)
    ax_loss.set_xlabel("weight update")
    ax_loss.set_ylabel("action loss (binned mean)")
    ax_loss.set_title("1. Action loss falls - but see panel 2")
    ax_loss.legend(fontsize=7)
    ax_loss.grid(alpha=0.3)

    # ---- 2. ratio vs no-motion ----------------------------------------
    for label, (colour, runs) in GROUPS.items():
        s = series(replay, runs)
        if not s:
            continue
        run = list(s)[0]
        x = s[run][0]
        y = [v.get("ratio_vs_baseline", np.nan) for v in s[run][1]]
        ax_ratio.plot(x, y, "o-", color=colour, label=label)
    ax_ratio.axhline(1.0, color="k", ls="--", lw=1.2)
    ax_ratio.text(0.02, 1.05, "1.0 = as good as standing still", fontsize=8,
                  transform=ax_ratio.get_yaxis_transform())
    ax_ratio.set_yscale("log")
    ax_ratio.set_xlabel("checkpoint (weight update)")
    ax_ratio.set_ylabel("MAE / no-motion MAE")
    ax_ratio.set_title("2. Tracking vs the do-nothing baseline")
    ax_ratio.legend(fontsize=7)
    ax_ratio.grid(alpha=0.3)

    # ---- 3. per-step delta vs ground truth ---------------------------
    for label, (colour, runs) in GROUPS.items():
        s = series(replay, runs)
        if not s:
            continue
        run = list(s)[0]
        x = s[run][0]
        pred = [v.get("delta_pred", np.nan) for v in s[run][1]]
        gt = [v.get("delta_gt", np.nan) for v in s[run][1]]
        ax_delta.plot(x, pred, "o-", color=colour, label=f"{label}: predicted")
        if not all(np.isnan(gt)):
            ax_delta.plot(x, gt, ":", color=colour, alpha=0.6, lw=1.2)
    bl = replay.get(("baseline_unfinetuned_dual28dofinit", -1))
    ax_delta.set_xlabel("checkpoint (weight update)")
    ax_delta.set_ylabel("mean per-step |delta| (rad)")
    ax_delta.set_title("3. Predicted (solid) vs ground-truth (dotted) joint motion")
    ax_delta.legend(fontsize=6)
    ax_delta.grid(alpha=0.3)
    ax_delta.annotate("a memorised trajectory\nwould reach the dotted line",
                      xy=(0.55, 0.86), xycoords="axes fraction", fontsize=8)

    # ---- 4. teacher forcing -------------------------------------------
    tf_runs = sorted(glob.glob(os.path.join(a.exp, "replay", "_tf_*", "summary.json")))
    if tf_runs:
        labels, ratios, tf_ratios = [], [], []
        for f in tf_runs:
            ag = json.load(open(f))["aggregate"]
            name = os.path.basename(os.path.dirname(f)).lstrip("_tf_")
            labels.append(name)
            ratios.append(ag.get("ratio_vs_baseline", np.nan))
            tf_ratios.append(ag.get("tf_ratio_vs_baseline", np.nan))
        i = np.arange(len(labels))
        ax_tf.bar(i - 0.2, ratios, 0.4, label="normal", color=BASELINE_COLOUR)
        ax_tf.bar(i + 0.2, tf_ratios, 0.4, label="teacher forced",
                  color="#2ca02c")
        ax_tf.axhline(1.0, color="k", ls="--", lw=1.2)
        ax_tf.set_xticks(i)
        ax_tf.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        ax_tf.set_ylabel("MAE / no-motion MAE")
        ax_tf.set_title("4. Teacher forcing (lower is better)")
        ax_tf.legend(fontsize=8)
        ax_tf.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "memorisation_study.png"), dpi=140)
    print(f"wrote {os.path.join(outdir, 'memorisation_study.png')}")

    # ---- one summary table --------------------------------------------
    lines = ["| run | best ckpt | x no-motion | delta x GT | last ckpt | x no-motion |",
             "|---|---|---|---|---|---|"]
    for label, (_, runs) in GROUPS.items():
        for run in runs:
            s = series(replay, runs)
            if run not in s:
                continue
            x, items = s[run]
            best = min(range(len(items)),
                       key=lambda i: items[i].get("ratio_vs_baseline", 9e9))
            lines.append(
                f"| {run} | {x[best]} | {items[best]['ratio_vs_baseline']:.2f} | "
                f"{items[best]['delta_ratio']:.1f}x | {x[-1]} | "
                f"{items[-1]['ratio_vs_baseline']:.2f} |")
    tbl = "\n".join(lines)
    with open(os.path.join(outdir, "memorisation_table.md"), "w") as fh:
        fh.write("# Memorisation study: 1-episode overfit, ground-truth anchored\n\n")
        fh.write("`x no-motion` = MAE divided by the do-nothing baseline; "
                 "below 1.0 beats standing still.\n\n")
        fh.write(tbl + "\n")
    print(tbl)
    print(f"\nwrote {os.path.join(outdir, 'memorisation_table.md')}")


if __name__ == "__main__":
    main()