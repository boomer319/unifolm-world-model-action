#!/usr/bin/env python3
"""Compare UnifoLM-WMA runs recorded by the RunRecorder callback.

Reads <runs_dir>/<run>/metrics.csv (written by
src/unifolm_wma_ext/callbacks.py) and prints a loss-curve comparison plus the
provenance needed to interpret it, so a results table for the thesis can be
produced straight from the run directory with no manual transcription.

Usage:
    python docker/compare_runs.py --runs-dir /docker_data/runs \
        --runs verify_dual verify_base [--step 10]
"""
import argparse
import csv
import json
import os
import statistics


def load(runs_dir, run):
    d = os.path.join(runs_dir, run)
    with open(os.path.join(d, "metrics.csv")) as f:
        rows = list(csv.DictReader(f))
    series = []
    for r in rows:
        # Column names are m_<pl metric name>, and PL prefixes training metrics
        # with "train/". Strip both so --keys can be written as loss_step.
        m = {}
        for k, v in r.items():
            if not k.startswith("m_") or v in ("", None):
                continue
            name = k[2:]
            for prefix in ("train/", "val/", "test/"):
                if name.startswith(prefix):
                    name = name[len(prefix):]
            m[name] = float(v)
        series.append({"wall_s": float(r["wall_s"]),
                       "batch": int(r["batch"]),
                       "global_step": int(r["global_step"]),
                       "peak_GiB": float(r["peak_GiB"]) if r["peak_GiB"] else None,
                       **m})
    manifest = summary = None
    for name, setter in (("manifest.json", "manifest"), ("summary.json", "summary")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            with open(p) as f:
                if setter == "manifest":
                    manifest = json.load(f)
                else:
                    summary = json.load(f)
    return series, timing, manifest, summary


def mean(xs):
    return statistics.mean(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="/docker_data/runs")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--step", type=int, default=10, help="print every Nth batch")
    ap.add_argument("--keys", nargs="*",
                    default=["loss_step", "loss_action_step", "loss_state_step"])
    args = ap.parse_args()

    data = {}
    for run in args.runs:
        try:
            data[run] = load(args.runs_dir, run)
        except (FileNotFoundError, KeyError) as e:
            print(f"!! {run}: {e}")

    # ---------------------------------------------------------- provenance
    print("=" * 78)
    print(" provenance")
    print("=" * 78)
    for run, (_, _, man, _) in data.items():
        if not man:
            continue
        bc = man["base_checkpoint"]
        print(f"\n  {run}")
        print(f"    git           {man['git']['sha'][:12]} on {man['git']['branch']}"
              f"{' (DIRTY)' if man['git']['dirty'] else ''}")
        print(f"    config sha256 {str(man['config']['sha256'])[:16]}...")
        print(f"    checkpoint    {os.path.basename(str(bc['path']))}")
        print(f"    ckpt bytes    {bc['size_bytes']}  exists={bc['exists']}")
        print(f"    dataset       {man['dataset']['name']} "
              f"h5={str(man['dataset'].get('h5_sha256'))[:16]}...")
        print(f"    model         {man['model']['total_params']:,} total / "
              f"{man['model']['trainable_params']:,} trainable "
              f"({100*man['model']['trainable_fraction']:.1f}%)")
        print(f"    trainer       max_steps={man['trainer']['max_steps']} "
              f"accum={man['trainer']['accumulate_grad_batches']} "
              f"precision={man['trainer']['precision']} "
              f"devices={man['trainer'].get('num_devices')}")

    # --------------------------------------------------------------- timing
    print()
    print("=" * 78)
    print(" cost")
    print("=" * 78)
    print(f"  {'run':22s} {'batches':>8} {'wall_s':>8} {'s/batch':>8} "
          f"{'peak_GiB':>9} {'ended_by':>12}")
    for run, (series, _, _, summ) in data.items():
        if not series:
            continue
        walls = [s["wall_s"] for s in series]
        deltas = sorted(walls[i + 1] - walls[i] for i in range(3, len(walls) - 1))
        med = deltas[len(deltas) // 2] if deltas else float("nan")
        peaks = [s["peak_GiB"] for s in series if s["peak_GiB"] is not None]
        peak = max(peaks)
        # torch's max_memory_allocated is a high-water mark and includes a
        # startup transient; the steady figure is the last recorded value.
        steady = series[-1]["peak_GiB"]
        ended = (summ or {}).get("ended_by", "?")
        print(f"  {run:22s} {len(series):>8} {walls[-1]:>8.1f} {med:>8.2f} "
              f"{peak:>9.2f} {steady:>11.2f} {ended[:12]:>12}")
    print("\n  NOTE: pytorch-lightning 1.9.5 counts max_steps/global_step in")
    print("        BATCHES, not optimizer updates, so batches / accumulate =")
    print("        the number of weight updates.")

    # --------------------------------------------------------------- curves
    n = min(len(s) for s, _, _ in data.values()) if data else 0
    idx = list(range(0, n, args.step)) + ([n - 1] if n and (n - 1) % args.step else [])
    for k in args.keys:
        print()
        print("=" * 78)
        print(f" {k}")
        print("=" * 78)
        head = f"  {'update':>7} |" + "".join(f" {r[:18]:>18}" for r in data)
        print(head)
        for i in idx:
            row = f"  {i:>7} |"
            for run, (series, _, _, _) in data.items():
                v = series[i].get(k) if i < len(series) else None
                row += f" {v:>18.4f}" if isinstance(v, float) else f" {'-':>18}"
            print(row)

    # -------------------------------------------------------- first vs last
    print()
    print("=" * 78)
    print(f" {'metric':22s} {'run':22s} {'first20':>10} {'last20':>10} "
          f"{'delta%':>9} {'min':>10}")
    print("=" * 78)
    for k in args.keys:
        for run, (series, _, _, _) in data.items():
            v = [s[k] for s in series if isinstance(s.get(k), float)]
            if not v:
                continue
            w = min(20, max(1, len(v) // 5))
            f20, l20 = mean(v[:w]), mean(v[-w:])
            delta = 100 * (l20 - f20) / f20 if f20 else float("nan")
            print(f" {k:22s} {run:22s} {f20:>10.4f} {l20:>10.4f} "
                  f"{delta:>8.1f}% {min(v):>10.4f}")


if __name__ == "__main__":
    main()