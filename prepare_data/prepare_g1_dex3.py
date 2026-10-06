#!/usr/bin/env python3
"""Convert a LeRobot v2 dataset into UnifoLM-WMA format for G1 Dex3 training.

Why a wrapper instead of patching prepare_training_data.py
----------------------------------------------------------
The stock converter (prepare_data/prepare_training_data.py) has two properties we
cannot use as-is:

1. It always processes *all* episodes listed in meta/info.json's total_episodes
   and *all* camera views it finds under videos/chunk-000/. For every
   (episode, view) pair it appends one CSV row, so a 301-episode 4-view dataset
   yields 1204 rows across 4 views. WMAData consumes exactly ONE view (README:
   "remove other views from the CSV"), and for a 1-episode overfit study we must
   not convert 301 episodes.

2. It has no subset option.

So this wrapper builds a temporary "shadow" LeRobot tree that contains only the
requested episodes and only the requested view (symlinks for the heavy files, a
patched info.json for total_episodes), then calls the stock converter unmodified
via subprocess. Upstream code stays untouched; the subsetting lives here.

Because the shadow tree contains a single view, the CSV it produces is already
single-view - no post-filtering needed.

Example
-------
    python prepare_data/prepare_g1_dex3.py \
        --source_dir /datasets \
        --dataset_name G1_Dex3_GraspSquare_Dataset_GEAR \
        --target_dir /data_wma \
        --output_name g1_dex3_graspsquare_1ep \
        --view observation.images.cam_left_high \
        --robot_name 'Unitree G1 Robot with Dex3 Hands' \
        --num_episodes 1
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CONVERTER = Path(__file__).resolve().parent / "prepare_training_data.py"
DEFAULT_VIEW = "observation.images.cam_left_high"
DEFAULT_ROBOT = "Unitree G1 Robot with Dex3 Hands"
ACTION_DIM = 28  # G1 Dex3: 7 left arm + 7 right arm + 7 left hand + 7 right hand


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source_dir", required=True,
                   help="directory CONTAINING the LeRobot v2 dataset dir")
    p.add_argument("--dataset_name", required=True,
                   help="LeRobot v2 dataset dir name, e.g. G1_Dex3_GraspSquare_Dataset_GEAR")
    p.add_argument("--target_dir", required=True,
                   help="root for the WMA-format dataset (videos/, transitions/, <name>.csv)")
    p.add_argument("--output_name", required=True,
                   help="dataset name inside the WMA format; also the CSV stem "
                        "and must match dataset_and_weights in the training config")
    p.add_argument("--view", default=DEFAULT_VIEW,
                   help=f"single camera view to convert (default {DEFAULT_VIEW})")
    p.add_argument("--robot_name", default=DEFAULT_ROBOT,
                   help="written to the h5 robot_type attr and the CSV embodiment column")
    p.add_argument("--num_episodes", type=int, default=1,
                   help="episodes to convert, 0-based from episode_000000")
    p.add_argument("--start_episode", type=int, default=0)
    p.add_argument("--shadow_dir", default=None,
                   help="where to build the temporary shadow LeRobot tree; "
                        "defaults to target_dir. It must be writable and must NOT "
                        "be inside source_dir, which is normally a read-only mount.")
    p.add_argument("--keep_shadow", action="store_true",
                   help="keep the temporary shadow tree for inspection")
    p.add_argument("--no_clean", action="store_true",
                   help="do not remove a previous output for this --output_name; "
                        "the stock converter's ffmpeg call has no -y, so it will "
                        "abort on an existing video file")
    p.add_argument("--skip_verify", action="store_true")
    return p.parse_args()


def clean_previous_output(target_dir: Path, out_name: str) -> None:
    """Make conversion idempotent for one dataset name.

    prepare_training_data.py's convert_to_h264 calls ffmpeg without -y, so a
    leftover file from an aborted run makes the converter exit non-zero. A
    partially converted dataset is worse than none (CSV rows, h5 files and stats
    must agree), so the previous output for this exact name is removed first.
    """
    victims = [target_dir / "videos" / out_name,
               target_dir / "transitions" / out_name,
               target_dir / f"{out_name}.csv"]
    found = [v for v in victims if v.exists()]
    if not found:
        return
    print(f">>> removing previous output for '{out_name}':")
    for v in found:
        print(f"    {v}")
        shutil.rmtree(v) if v.is_dir() else v.unlink()


def build_shadow(source_root: Path, shadow_root: Path, dataset_name: str,
                 out_name: str, view: str, num: int, start: int) -> Path:
    """Create a minimal LeRobot v2 tree with `num` episodes and one view.

    shadow_root must be writable and separate from source_root: the source is
    normally mounted read-only, so the tree cannot be staged next to it.
    """
    src = source_root / dataset_name
    for rel in ("data/chunk-000", "meta", f"videos/chunk-000/{view}"):
        if not (src / rel).exists():
            raise FileNotFoundError(f"missing {src / rel}")

    shadow = shadow_root / f".shadow_{out_name}"
    if shadow.exists():
        shutil.rmtree(shadow)
    (shadow / out_name / "data" / "chunk-000").mkdir(parents=True)
    (shadow / out_name / "videos" / "chunk-000" / view).mkdir(parents=True)
    (shadow / out_name / "meta").mkdir(parents=True)

    linked = 0
    for i in range(start, start + num):
        parquet = src / "data" / "chunk-000" / f"episode_{i:06d}.parquet"
        video = src / "videos" / "chunk-000" / view / f"episode_{i:06d}.mp4"
        if not parquet.exists():
            raise FileNotFoundError(f"missing {parquet}")
        if not video.exists():
            raise FileNotFoundError(f"missing {video}")
        os.symlink(parquet.resolve(),
                   shadow / out_name / "data" / "chunk-000" / f"episode_{i:06d}.parquet")
        os.symlink(video.resolve(),
                   shadow / out_name / "videos" / "chunk-000" / view / f"episode_{i:06d}.mp4")
        linked += 1

    # tasks.jsonl supplies the instruction string (converter uses tasks[0]['task']).
    os.symlink((src / "meta" / "tasks.jsonl").resolve(),
               shadow / out_name / "meta" / "tasks.jsonl")

    with open(src / "meta" / "info.json") as f:
        info = json.load(f)
    info["total_episodes"] = num
    with open(shadow / out_name / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f">>> shadow tree: {shadow}")
    print(f"    episodes {start}..{start + num - 1} ({linked} linked), view {view}")
    print(f"    info.json total_episodes -> {num}")
    return shadow


def run_converter(shadow: Path, out_name: str, target_dir: Path, robot: str) -> None:
    cmd = [sys.executable, str(CONVERTER),
           "--source_dir", str(shadow),
           "--dataset_name", out_name,
           "--target_dir", str(target_dir),
           "--robot_name", robot]
    print(">>> running stock converter:")
    print("    " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def verify(target_dir: Path, out_name: str, view: str) -> None:
    import h5py
    import pandas as pd
    import torch
    from safetensors.torch import load_file

    print()
    print("=" * 70)
    print(" verify converted dataset")
    print("=" * 70)

    csv_path = target_dir / f"{out_name}.csv"
    df = pd.read_csv(csv_path, dtype=str)
    print(f"\n  CSV {csv_path.name}: {len(df)} rows, columns={list(df.columns)}")
    print(f"    first row: {dict(df.iloc[0])}")
    assert "instruction" in df.columns and "embodiment" in df.columns
    assert "data_dir" in df.columns and "videoid" in df.columns
    views_in_csv = {d.split("/")[-1] for d in df["data_dir"]}
    assert views_in_csv == {view}, f"expected only {view}, got {views_in_csv}"
    print(f"    views present: {views_in_csv}  (single view OK)")

    h5_path = target_dir / "transitions" / out_name / "0.h5"
    with h5py.File(h5_path, "r") as f:
        print(f"\n  H5 {h5_path}:")
        for k in f.keys():
            print(f"    {k:20s} shape={f[k].shape} dtype={f[k].dtype}")
        print(f"    attrs: {dict(f.attrs)}")
        state = torch.tensor(f["observation.state"][()])
        action = torch.tensor(f["action"][()])
        assert state.shape[1] == ACTION_DIM, f"state dim {state.shape[1]} != {ACTION_DIM}"
        assert action.shape[1] == ACTION_DIM, f"action dim {action.shape[1]} != {ACTION_DIM}"
        print(f"    frames={state.shape[0]}, dim={state.shape[1]} (== {ACTION_DIM} OK)")

    stats_path = target_dir / "transitions" / out_name / "meta_data" / "stats.safetensors"
    stats = load_file(str(stats_path))
    print(f"\n  stats.safetensors: {len(stats)} tensors")
    for k in sorted(stats):
        print(f"    {k:34s} shape={tuple(stats[k].shape)}")

    mp4 = target_dir / "videos" / out_name / view / "0.mp4"
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=codec_name,width,height,nb_read_frames",
         "-of", "default=noprint_wrappers=1", str(mp4)],
        capture_output=True, text=True, check=True)
    print(f"\n  video {mp4.name}:")
    for line in probe.stdout.strip().splitlines():
        print(f"    {line}")


def verify_sample(target_dir: Path, out_name: str) -> None:
    """Instantiate WMAData exactly as the training config does and pull one item."""
    from unifolm_wma.data.wma_data import WMAData

    print()
    print("=" * 70)
    print(" verify WMAData sample (params mirrored from configs/train/config_g1_dex3.yaml)")
    print("=" * 70)
    ds = WMAData(
        meta_path=str(target_dir / f"{out_name}.csv"),
        data_dir=str(target_dir),
        transition_dir=str(target_dir / "transitions"),
        dataset_name=out_name,
        video_length=16,
        frame_stride=2,
        load_raw_resolution=True,
        resolution=[320, 512],
        spatial_transform="resize_center_crop",
        crop_resolution=[320, 512],
        random_fs=False,
        cond_robot_label_prob=0.0,
        normalization_mode="min_max",
        individual_normalization=True,
        n_obs_steps=2,
        max_action_dim=ACTION_DIM,
        max_state_dim=ACTION_DIM,
    )
    print(f"\n  dataset length (CSV rows) = {len(ds)}")
    print("  NOTE: DataModuleFromConfig uses a WeightedRandomSampler with "
          "drop_last=True,\n        so batch_size <= number of CSV rows. One episode "
          "=> one row => batch_size must be 1.")
    item = ds[0]
    print("\n  sample keys and shapes:")
    for k, v in item.items():
        if hasattr(v, "shape"):
            print(f"    {k:24s} {tuple(v.shape)}  {v.dtype}")
        else:
            print(f"    {k:24s} {v!r}"[:150])
    print(f"\n    action_mask sum   = {int(item['action_mask'][0].sum())} "
          f"(expect {ACTION_DIM} for a fully-observed Dex3 action)")
    print(f"    state_mask sum    = {int(item['state_mask'][0].sum())}")
    print(f"    instruction       = {item['instruction']!r}")
    print(f"    fps (post-stride) = {item['fps']}")


def main():
    args = parse_args()
    source_root = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    src = source_root / args.dataset_name
    with open(src / "meta" / "info.json") as f:
        info = json.load(f)
    avail = info["total_episodes"]
    print(f">>> source: {src}")
    print(f"    total_episodes={avail}, fps={info.get('fps')}, "
          f"robot_type={info.get('robot_type')}")
    if args.start_episode + args.num_episodes > avail:
        raise SystemExit(f"ERROR: requested episodes "
                         f"{args.start_episode}..{args.start_episode + args.num_episodes - 1} "
                         f"but only {avail} exist")

    shadow_root = Path(args.shadow_dir) if args.shadow_dir else target_dir
    shadow_root.mkdir(parents=True, exist_ok=True)
    print(f">>> shadow root: {shadow_root}")
    if not args.no_clean:
        clean_previous_output(target_dir, args.output_name)
    shadow = build_shadow(source_root, shadow_root, args.dataset_name,
                          args.output_name, args.view, args.num_episodes,
                          args.start_episode)
    try:
        run_converter(shadow, args.output_name, target_dir, args.robot_name)
    finally:
        if args.keep_shadow:
            print(f">>> keeping shadow tree {shadow}")
        else:
            shutil.rmtree(shadow, ignore_errors=True)

    if not args.skip_verify:
        verify(target_dir, args.output_name, args.view)
        verify_sample(target_dir, args.output_name)

    print()
    print("=" * 70)
    print(" CONVERT: DONE")
    print(f" dataset_and_weights entry for the config: {{'{args.output_name}': 1.0}}")
    print("=" * 70)


if __name__ == "__main__":
    main()