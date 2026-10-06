# UnifoLM-WMA on G1 Dex3 - Docker environment

Containerised run of Unitree's [UnifoLM-WMA](https://github.com/unitreerobotics/unifolm-world-model-action)
on the G1 Dex3 (28-DoF upper body), targeting `.240`.

Mirrors the working `dreamzero_docker_env` pattern: raw `/dev/nvidia*` device
nodes plus host `libcuda` bind-mounts, because `.240` has **no
nvidia-container-toolkit** (no nvidia runtime, no CDI specs).

## Layout on .240

```
/data/docker-services/world_action_models/
├── unifolm_wma/                 # this repo (git clone of boomer319/unifolm-world-model-action, branch g1-dex3)
│   ├── checkpoints/             # unifolm_wma_base.ckpt (27.26 GB, HF-gated) - gitignored
│   ├── data/                    # converted WMA-format data (videos/, transitions/, *.csv)
│   └── docker_data/             # caches (HF, torch), logs, lightning workspace - gitignored
└── dreamzero/datasets/          # existing LeRobot v2 datasets, mounted read-only at /datasets
```

## One-time setup

```bash
git clone --recurse-submodules -b g1-dex3 \
    https://github.com/boomer319/unifolm-world-model-action.git
cd unifolm-world-model-action/docker && ./build.sh
```

`build.sh` writes `.env` with the host uid/gid/username (the container user must
match the owner of the bind-mounted files), builds the image, and runs the
environment verification. The base image
`nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04` is already cached on
`.240`, so no registry pull is needed for it.

## Services

| Service | What it does |
|---|---|
| `wma-env` | Verifies interpreter, every import in the training/serving dependency closure, ffmpeg, CUDA, xformers kernels. Fails loudly. |
| `wma-shell` | Interactive shell (GPU 0 attached). |
| `wma-convert` | LeRobot v2 -> UnifoLM-WMA format, 1 episode, single view. |
| `wma-load` | Instantiates the 28-DoF config, loads the Base checkpoint, reports **every** weight that did not load. |
| `wma-smoke` | 100 trainer steps: s/step and VRAM measurement. |
| `wma-train` | Real fine-tuning run (Phase 1: 1-episode overfit). |
| `wma-server` | Decision-making eval server, FastAPI/uvicorn on port 8000. |

```bash
docker compose run --rm wma-env
docker compose run --rm wma-convert
docker compose run --rm wma-load
docker compose run --rm wma-smoke
docker compose up -d wma-shell
```

## Checkpoints

| Model | Size | Gate | Use |
|---|---|---|---|
| `unitreerobotics/UnifoLM-WMA-0-Base` | 27.26 GB | **gated** - accept terms on the model page first | Open-X finetune; the right starting point for a new embodiment |
| `unitreerobotics/UnifoLM-WMA-0-Dual` | 16.84 GB | not gated | Z1 + G1 Dex1 finetune; fallback while the Base gate is pending |

```bash
docker compose run --rm wma-shell
# inside:
hf download unitreerobotics/UnifoLM-WMA-0-Base \
    unifolm_wma_base.ckpt --local-dir checkpoints
```

## Deviations from upstream, and why

**1. Base image is `nvidia/cuda:12.4.1`, not `nvcr.io/nvidia/pytorch`.**
The NGC pytorch image is python 3.12 (upstream pins `==3.10.18`) and ships a
global `pip.conf` pointing at `pypi.ngc.nvidia.com`, which is dead on the lab
network.

**2. Dependencies are baked into the image, in a venv at `/opt/wma`.**
dreamzero instead creates its venv at runtime inside the bind-mounted home
(`dreamzero_docker_env/build.sh:20-24`), which means the image alone cannot run
and every consumer must run `build.sh` first. A self-contained image lets
`docker compose run <svc>` work immediately. `/opt` rather than `/home/$USER`
because compose does not bind-mount over the home directory.

**3. `torch==2.3.1` + `torchvision==0.18.1` + `xformers==0.0.27` - upstream's own
pins, unchanged.** They are consistent: `xformers 0.0.27` declares
`torch==2.3.1` and `torchvision 0.18.1` declares `torch==2.3.1`. (This file
initially "fixed" the triple to torch 2.4.1 on the assumption that xformers
0.0.27 tracked 2.4.x; pip rejected it with `ResolutionImpossible`, which is what
settled the question.) PyPI's default linux wheels bundle CUDA 12.1, which the
12.4 driver supports. xformers is not optional: without it upstream reports
~200 s/iteration (GitHub issue #41).

**4. `pytorch-lightning==1.9.5` instead of `1.9.3`.** Hard requirement, not a
preference: `scripts/trainer.py:178` calls `Trainer.from_argparse_args`, which
pytorch-lightning 2.x removed, so the install must stay on the 1.x line, and
1.9.5 is its final release. `docker/env_smoke.sh` asserts the attribute exists so
the constraint stays self-verifying.

**5. Dependencies upstream's `pyproject.toml` omits but the code imports:**
`h5py` (wma_data.py:5), `safetensors` (data/utils.py:6),
`sentencepiece` (condition.py:75 uses the slow `T5Tokenizer`),
`fastapi` + `uvicorn` + `matplotlib` (real_eval_server.py), `tensorboard`
(`utils/train.py:130` defaults to `TensorBoardLogger` and the training config
sets no logger), and `pyarrow` (`prepare_training_data.py:118` reads the
LeRobot v2 parquet through pandas; upstream only gets pyarrow transitively via
`datasets`, which is itself unused).

**6. Dependencies dropped** because nothing under `src/`, `scripts/` or
`prepare_data/` imports them: `gradio`, `tensorflow-metadata`,
`tensorflow-graphics`, `protobuf`, `datasets`, `fairscale`, `draccus`,
`scikit-learn`, `moviepy`, `accelerate`, `pinocchio`.
`external/dlimp` is checked out but **not installed** - a scan of all 35 python
files under `src/`, `scripts/` and `prepare_data/` found zero `dlimp` imports;
the README's `pip install -e external/dlimp` is vestigial (the code is
robomimic-derived). If a future code path needs it, patch it at build time from
`docker/patches/dlimp/` rather than forking `kvablack/dlimp`.

**7. `pip install -e . --no-deps --ignore-requires-python`.**
All deps are pinned explicitly above; `--ignore-requires-python` is needed because
upstream pins `==3.10.18` while Ubuntu 22.04 ships python 3.10.12 (patch-level
difference only). The editable install targets `/workspace`, whose path is
identical at build and run time, so the bind-mounted working tree is imported.

## Landmines this setup already handles

| Trap | Where | Handling |
|---|---|---|
| `int(os.environ.get('LOCAL_RANK'))` crashes when unset | trainer.py:79-81 | `LOCAL_RANK/RANK/WORLD_SIZE` set in the compose environment |
| `trainer_config.devices` / `num_nodes` dereferenced unconditionally | trainer.py:123-124 | set in `config_g1_dex3.yaml` (upstream passes them on the command line only) |
| Default strategy is `DDPShardedStrategy`, needs >1 GPU | utils/train.py:139-151 | `lightning.strategy: auto` in the config |
| `configs/train/meta.json` hardcoded via `os.getcwd()` | multi_image_obs_encoder.py:44-45 | `shape_meta_path` passed via config (`meta_g1_dex3.json`, 28 dims) |
| `strict=False` hides missing/unexpected weights | utils/train.py:165 | `docker/load_test.py` diffs and reports all three categories |
| 28 DoF changes action/state UNet input shapes, which raises even with `strict=False` | - | `load_test.py` drops and lists exactly those tensors (world model is reused, heads re-learn) |
| `drop_last=True` + one CSV row per (episode, view) | utils/data.py:165-168 | `batch_size: 1`, `accumulate_grad_batches: 4` |
| Converter emits one CSV row per **view** and skips non-AV1 videos | prepare_training_data.py:95-147 | shadow tree contains only the chosen view, so the CSV is single-view by construction |
| Converter has no episode-subset option | prepare_training_data.py:87 | `prepare_data/prepare_g1_dex3.py` builds a shadow LeRobot tree of symlinks + patched `info.json` and calls the stock converter unmodified |
| YAML merge keys are shallow - a service-level `environment:` replaces the base one | - | no per-service `environment:` in this compose file |
| ~27 GB per checkpoint at ~27 GB each in fp32 | our config | 500-step interval keeping all; 10k steps ~ 540 GB, budget accordingly |

## License

Code in this repository is under **CC BY-NC-SA 4.0** (the 20,930-byte `LICENSE`
file), which contradicts the `BSD-3-Clause` string in `pyproject.toml`. The
released weights are also CC BY-NC-SA 4.0. Non-commercial use only - fine for a
thesis, not for a product.