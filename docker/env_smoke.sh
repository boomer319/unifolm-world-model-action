#!/usr/bin/env bash
# Environment verification for the UnifoLM-WMA image.
#
# The meaningful check is the last one: importing the repo's own model class
# pulls in the entire dependency closure of the training and serving paths, so a
# clean import there proves the dependency set is complete (upstream's
# pyproject.toml is NOT complete - it omits h5py, safetensors, sentencepiece,
# fastapi, uvicorn, matplotlib and tensorboard, all of which are imported).
set -uo pipefail
cd /workspace

fail=0

echo "=============================================================="
echo " interpreter"
echo "=============================================================="
python --version
python - <<'PY'
import sys
print("sys.executable:", sys.executable)
if sys.version_info[:2] != (3, 10):
    print("WARN: expected python 3.10.x (upstream pins ==3.10.18)")
PY

echo
echo "=============================================================="
echo " ffmpeg / ffprobe  (required by prepare_training_data.py)"
echo "=============================================================="
for b in ffmpeg ffprobe; do
    p=$(command -v "$b" || true)
    if [ -n "$p" ]; then echo "OK   $b -> $p"; else echo "FAIL $b not found"; fail=1; fi
done

echo
echo "=============================================================="
echo " core packages"
echo "=============================================================="
for m in torch torchvision xformers numpy pandas decord einops \
         pytorch_lightning omegaconf transformers diffusers open_clip \
         kornia timm cv2 av imageio h5py safetensors matplotlib \
         fastapi uvicorn sentencepiece tensorboard; do
    out=$(python -c "import $m" 2>&1)
    if [ -z "$out" ]; then
        ver=$(python -c "import $m; print(getattr($m,'__version__','?'))" 2>/dev/null)
        printf 'OK   %-18s %s\n' "$m" "$ver"
    else
        printf 'FAIL %-18s %s\n' "$m" "$(echo "$out" | tail -1)"
        fail=1
    fi
done

echo
echo "=============================================================="
echo " torch / CUDA"
echo "=============================================================="
python - <<'PY'
import torch
print("torch          :", torch.__version__)
print("cuda available :", torch.cuda.is_available())
print("torch cuda ver :", torch.version.cuda)
if torch.cuda.is_available():
    print("device count   :", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  gpu{i}: {p.name} sm_{p.major}{p.minor} "
              f"{p.total_memory/2**30:.1f} GiB")
try:
    import xformers, xformers.ops
    print("xformers       :", xformers.__version__)
except Exception as e:
    print("xformers.ops FAIL:", e)
PY

echo
echo "=============================================================="
echo " repo import closure (the real test)"
echo "=============================================================="
python - <<'PY'
import importlib, sys, traceback

# pytorch-lightning must stay on the 1.x line: scripts/trainer.py:178 calls
# Trainer.from_argparse_args, which 2.x removed.
try:
    import pytorch_lightning as pl
    assert hasattr(pl.Trainer, "from_argparse_args"), "Trainer.from_argparse_args missing"
    print(f"OK   pytorch_lightning {pl.__version__} keeps Trainer.from_argparse_args (1.x API)")
except Exception:
    print("FAIL pytorch_lightning 1.x API requirement")
    traceback.print_exc(limit=2)
    sys.exit(1)

targets = [
    ("unifolm_wma",                        "the package itself"),
    ("unifolm_wma.data.wma_data",          "WMAData, the dataset we train on"),
    ("unifolm_wma.data.utils",             "load_stats (safetensors)"),
    ("unifolm_wma.data.normolize",         "min_max Normalize/Unnormalize"),
    ("unifolm_wma.utils.data",             "DataModuleFromConfig"),
    ("unifolm_wma.utils.train",            "trainer helpers"),
    ("unifolm_wma.utils.callbacks",        "ImageLogger / CUDACallback"),
    ("unifolm_wma.modules.networks.wma_model", "WMAModel (video UNet + action/state UNets)"),
    ("unifolm_wma.modules.encoders.condition", "FrozenOpenCLIP / T5 / CLIP encoders"),
    ("unifolm_wma.modules.vision.base_vision", "timm-based vision encoder"),
    ("unifolm_wma.models.autoencoder",      "AutoencoderKL"),
    ("unifolm_wma.models.samplers.ddim",    "DDIMSampler"),
    ("unifolm_wma.models.diffusion_head.conditional_unet1d", "ConditionalUnet1D"),
    ("unifolm_wma.models.diffusion_head.vision.multi_image_obs_encoder",
     "MultiImageObsEncoder"),
    ("unifolm_wma.models.ddpms",            "LatentVisualDiffusion (full closure)"),
]

bad = 0
for mod, why in targets:
    try:
        importlib.import_module(mod)
        print(f"OK   {mod:60s} <- {why}")
    except Exception:
        bad += 1
        print(f"FAIL {mod:60s} <- {why}")
        traceback.print_exc(limit=3)
sys.exit(1 if bad else 0)
PY
rc=$?
[ $rc -ne 0 ] && fail=1

echo
echo "=============================================================="
echo " deliberately absent (verified unused upstream)"
echo "=============================================================="
for m in dlimp gradio tensorflow_metadata tensorflow_graphics fairscale \
         draccus sklearn moviepy pinocchio; do
    if python -c "import $m" 2>/dev/null; then
        echo "NOTE $m is importable; upstream code never imports it"
    else
        echo "OK   $m absent (no upstream import)"
    fi
done
if [ -d external/dlimp ]; then
    echo "NOTE external/dlimp submodule is checked out but NOT pip-installed;"
    echo "     nothing under src/ scripts/ prepare_data/ imports it."
fi

echo
echo "=============================================================="
if [ $fail -eq 0 ]; then
    echo " ENV SMOKE: PASS"
else
    echo " ENV SMOKE: FAIL"
fi
echo "=============================================================="
exit $fail