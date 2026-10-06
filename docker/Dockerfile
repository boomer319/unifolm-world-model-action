# syntax=docker/dockerfile:1
# UnifoLM-WMA (Unitree) - G1 Dex3 adaptation.
# Build context is the REPO ROOT:  docker compose build  (context: ..)
#
# Base image nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 is already
# cached on .240, so the build needs no registry pull for the base layer.
# Deliberately NOT the NGC pytorch image: it is python 3.12 (upstream pins
# ==3.10.18) and ships a global pip.conf pointing at the dead
# pypi.ngc.nvidia.com index.
FROM nvcr.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG tmp_UID=1000
ARG tmp_GID=1000
ARG USERNAME=user

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/wma \
    PATH="/opt/wma/bin:$PATH" \
    LD_LIBRARY_PATH=/nvidia-host

# ffmpeg is mandatory: prepare_training_data.py probes codecs with ffprobe and
# transcodes AV1 -> H.264 with ffmpeg.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -g "${tmp_GID}" "${USERNAME}" \
    && useradd -m -u "${tmp_UID}" -g "${tmp_GID}" -s /bin/bash "${USERNAME}"

# The venv lives in /opt, NOT under /home/${USERNAME}: docker-compose.yml does
# not bind-mount over the home directory, so the image stays self-contained and
# `docker compose run <svc>` works without a prior setup step. (The dreamzero
# image instead creates its venv at runtime inside the mounted home - see
# dreamzero_docker_env/build.sh:20-24 - which means the image alone cannot run.)
RUN python3 -m venv "${VIRTUAL_ENV}" \
    && pip install --upgrade "pip<25" wheel

# --- torch stack ------------------------------------------------------------
# Upstream's own pins, verified consistent by pip's resolver:
# xformers 0.0.27 declares `torch==2.3.1` and torchvision 0.18.1 declares
# `torch==2.3.1`. (An earlier draft of this file assumed 0.0.27 tracked torch
# 2.4.1 and "fixed" the triple to 2.4.1; pip rejected that with
# ResolutionImpossible, which is what proved the upstream combination correct.)
# PyPI's default linux wheels bundle CUDA 12.1, which the 12.4 driver supports.
# xformers is not optional: without it upstream reports ~200 s/iteration
# (GitHub issue #41).
RUN pip install \
    torch==2.3.1 \
    torchvision==0.18.1 \
    xformers==0.0.27

# --- remaining runtime deps -------------------------------------------------
# Upstream pins, minus every dependency with zero imports under src/, scripts/
# and prepare_data/ (verified by grep): gradio, tensorflow-metadata,
# tensorflow-graphics, protobuf, datasets, fairscale, draccus, scikit-learn,
# moviepy, accelerate, pinocchio.
RUN pip install \
    av==12.3.0 \
    decord==0.6.0 \
    diffusers==0.30.2 \
    einops==0.8.0 \
    imageio==2.35.1 \
    kornia==0.7.3 \
    numpy==1.24.2 \
    omegaconf==2.3.0 \
    open-clip-torch==2.22.0 \
    opencv-python==4.10.0.84 \
    pandas==2.0.0 \
    pillow==9.5.0 \
    pyyaml==6.0 \
    timm==0.9.10 \
    tqdm==4.66.5 \
    transformers==4.40.1

# pytorch-lightning 1.9.x, with setuptools pinned alongside it. Two constraints:
#   * 1.9.5 rather than 2.x because scripts/trainer.py:178 calls
#     Trainer.from_argparse_args, which pytorch-lightning 2.x removed; 1.9.5 is
#     its last 1.x release.
#   * setuptools<81 because PL 1.9.x imports pkg_resources, which recent
#     setuptools releases no longer ship. Without this pin, importing
#     pytorch_lightning fails with "No module named 'pkg_resources'".
RUN pip install "setuptools<81" pytorch-lightning==1.9.5

# --- dependencies upstream OMITS from pyproject.toml but the code imports ----
#   h5py         src/unifolm_wma/data/wma_data.py:5, prepare_data/*.py
#   safetensors  src/unifolm_wma/data/utils.py:6 (load_stats), prepare_data
#   sentencepiece modules/encoders/condition.py:75 uses T5Tokenizer (slow,
#                sentencepiece-backed, not the HF-fast variant)
#   fastapi/uvicorn/matplotlib  scripts/evaluation/real_eval_server.py:6,10,17
#   tensorboard   utils/train.py:130 defaults the logger to TensorBoardLogger
#                 and configs/train/config.yaml sets no logger
RUN pip install \
    fastapi \
    h5py \
    matplotlib \
    safetensors \
    sentencepiece \
    tensorboard \
    uvicorn

# Install the project itself without dependency resolution (all deps are
# pinned above). --ignore-requires-python because upstream pins ==3.10.18 while
# this image ships Ubuntu 22.04's python 3.10.12 (patch-level difference only).
# Editable install into /workspace, whose path is identical at build and run
# time, so the bind-mounted working tree is what gets imported.
COPY pyproject.toml /workspace/pyproject.toml
COPY src /workspace/src
RUN pip install --no-deps --ignore-requires-python -e /workspace

RUN mkdir -p /workspace /data_wma /docker_data \
    && chown -R "${tmp_UID}:${tmp_GID}" /workspace /data_wma /docker_data

USER ${USERNAME}
WORKDIR /workspace

CMD ["/bin/bash"]