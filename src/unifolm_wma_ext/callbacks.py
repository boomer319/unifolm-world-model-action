"""RunRecorder: a LightningCallback that captures everything a thesis needs.

Why
---
Upstream's only persistence is ModelCheckpoint plus a TensorBoardLogger
(utils/train.py:112-136). That is enough to resume, but not enough to make a run
auditable months later: there is no record of which commit produced the numbers,
which checkpoint seeded it, what the resolved config actually was, how much
memory the step peaked at, or the exact loss curve at optimizer-step resolution
(the progress bar only shows the latest value).

This callback writes, per run, into <out_dir>/<run_name>/:
  manifest.json   provenance: git SHA, resolved config + its SHA256, dataset
                  identity and file hashes, base checkpoint identity, host/GPU,
                  library versions, parameter counts
  metrics.csv     one row per training batch: global_step, epoch, every logged
                  metric, wall-clock and peak GPU memory
  summary.json    written at fit end: final metrics, timing, peak memory, and
                  whether the run ended cleanly or by exception

Rows are flushed per batch, so a killed run still leaves usable data.
"""
import csv
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone

import pytorch_lightning as pl
import torch


def _sha256(path, limit=None):
    """SHA256 of a file, or of its first `limit` bytes if given."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        n = 0
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
            n += len(chunk)
            if limit and n >= limit:
                break
    return h.hexdigest()


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return None


class RunRecorder(pl.Callback):
    def __init__(self, out_dir="/docker_data/runs", config=None, data_dir=None,
                 base_ckpt=None, dataset_name=None, tag=""):
        super().__init__()
        self.out_dir = out_dir
        self.config = config
        self.data_dir = data_dir
        self.base_ckpt = base_ckpt
        self.dataset_name = dataset_name
        self.tag = tag
        self._rows = []
        self._keys = None
        self._fh = None
        self._writer = None
        self._t0 = None

    # ------------------------------------------------------------------ setup
    def _base_ckpt(self, trainer):
        """Which checkpoint seeded this run.

        WMA_BASE_CKPT wins because a run may override
        model.pretrained_checkpoint on the command line (wma-verify-dual does);
        otherwise fall back to the config value, then to the model.yaml that
        init_workspace saved next to the run.
        """
        env = os.environ.get("WMA_BASE_CKPT")
        if env:
            return env
        if self.base_ckpt:
            return self.base_ckpt
        cfgdir = getattr(trainer, "logdir", "")
        for cand in (os.path.join(os.path.dirname(str(cfgdir)), "configs", "model.yaml"),):
            if cand and os.path.exists(cand):
                try:
                    import yaml
                    with open(cand) as f:
                        v = (yaml.safe_load(f) or {}).get("pretrained_checkpoint")
                    if v:
                        return v
                except Exception:
                    pass
        return None

    def _resolve(self, trainer):
        """Work out a unique, human-meaningful directory for this run.

        PL's trainer.logdir is the LOGGER's directory, which for the default
        TensorBoardLogger is <workdir>/tensorboard - using it directly would
        make every run write to the same "tensorboard" folder. init_workspace
        (utils/train.py:13-17) puts the run name one level up, so prefer that,
        then an explicit WMA_RUN_NAME override.
        """
        logdir = str(getattr(trainer, "logdir", "") or "").rstrip("/")
        candidates = [
            os.environ.get("WMA_RUN_NAME"),
            os.path.basename(os.path.dirname(logdir)) if logdir else None,
            os.path.basename(logdir) if logdir else None,
            getattr(trainer, "default_hp", {}).get("run_name") if hasattr(trainer, "default_hp") else None,
            "run",
        ]
        name = next((c for c in candidates if c), "run")
        d = os.path.join(self.out_dir, name)
        os.makedirs(d, exist_ok=True)
        return d, name

    def on_fit_start(self, trainer, pl_module):
        import unifolm_wma_ext  # noqa: F401  (import check)
        self.run_dir, self.run_name = self._resolve(trainer)
        self._t0 = time.time()

        total = sum(p.numel() for p in pl_module.parameters())
        trainable = sum(p.numel() for p in pl_module.parameters() if p.requires_grad)
        manifest = {
            "run_name": self.run_name,
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "git": {
                "sha": _git("rev-parse", "HEAD"),
                "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(_git("status", "--porcelain")),
                "remote": _git("remote", "get-url", "origin"),
            },
            "config": {
                "path": self.config,
                "sha256": _sha256(self.config) if self.config and
                os.path.exists(self.config) else None,
            },
            "base_checkpoint": {
                "path": self._base_ckpt(trainer),
                "exists": bool(self.base_ckpt and os.path.exists(self.base_ckpt)),
                "sha256_first_1MiB": _sha256(self._base_ckpt(trainer), 1 << 20)
                if self._base_ckpt(trainer) and
                os.path.exists(self._base_ckpt(trainer)) else None,
                "size_bytes": os.path.getsize(self._base_ckpt(trainer))
                if self._base_ckpt(trainer) and
                os.path.exists(self._base_ckpt(trainer)) else None,
            },
            "dataset": {"name": self.dataset_name, "data_dir": self.data_dir},
            "model": {
                "total_params": total,
                "trainable_params": trainable,
                "trainable_fraction": trainable / max(total, 1),
                "agent_state_dim": int(pl_module.agent_state_dim)
                if hasattr(pl_module, "agent_state_dim") else None,
                "agent_action_dim": int(pl_module.agent_action_dim)
                if hasattr(pl_module, "agent_action_dim") else None,
            },
            "environment": {
                "host": platform.node(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "lightning": pl.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "gpu_total_GiB": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)
                if torch.cuda.is_available() else None,
            },
            "trainer": {
                "max_steps": trainer.max_steps,
                "accumulate_grad_batches": trainer.accumulate_grad_batches,
                "precision": trainer.precision,
                "strategy": str(trainer.strategy),
                "devices": str(trainer._accelerator_connector.devices)
                if hasattr(trainer, "_accelerator_connector") else None,
            },
        }
        # Dataset identity: hash the CSV and the h5 so a thesis claim can be tied
        # to exact bytes on disk.
        if self.data_dir and self.dataset_name:
            csv_path = os.path.join(self.data_dir, f"{self.dataset_name}.csv")
            h5 = os.path.join(self.data_dir, "transitions", self.dataset_name, "0.h5")
            manifest["dataset"].update({
                "csv_sha256": _sha256(csv_path) if os.path.exists(csv_path) else None,
                "h5_sha256": _sha256(h5) if os.path.exists(h5) else None,
                "h5_bytes": os.path.getsize(h5) if os.path.exists(h5) else None,
            })
        with open(os.path.join(self.run_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        self._fh = open(os.path.join(self.run_dir, "metrics.csv"), "w", newline="")
        self._writer = csv.writer(self._fh)
        self._fh.flush()
        print(f">>> RunRecorder: writing to {self.run_dir}")

    # ------------------------------------------------------------- per batch
    @staticmethod
    def _numeric(metrics):
        return {k: float(v) for k, v in metrics.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)}

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._writer is None:
            return
        m = self._numeric(trainer.callback_metrics)
        # The first batch fixes the column order from whatever PL exposes then;
        # later batches reuse it so the CSV stays rectangular.
        if self._keys is None:
            self._keys = sorted(m)
            self._writer.writerow(["batch", "global_step", "epoch", "wall_s",
                                   "peak_GiB"] + ["m_" + k for k in self._keys])
        row = [batch_idx, trainer.global_step, trainer.current_epoch,
               round(time.time() - self._t0, 2),
               round(torch.cuda.max_memory_allocated() / 2**30, 3)
               if torch.cuda.is_available() else ""]
        row += [m.get(k, "") for k in self._keys]
        self._writer.writerow(row)
        self._fh.flush()

    # ----------------------------------------------------------------- finish
    def _summary(self, trainer, ended_by):
        if not hasattr(self, "run_dir"):
            return
        m = trainer.callback_metrics
        summary = {
            "run_name": self.run_name,
            "ended_utc": datetime.now(timezone.utc).isoformat(),
            "ended_by": ended_by,
            "wall_seconds": round(time.time() - self._t0, 2) if self._t0 else None,
            "global_step": trainer.global_step,
            "peak_gpu_GiB": round(torch.cuda.max_memory_allocated() / 2**30, 3)
            if torch.cuda.is_available() else None,
            "seconds_per_optimizer_step": (
                round((time.time() - self._t0) / trainer.global_step, 3)
                if trainer.global_step else None),
            "final_metrics": {k: float(v) for k, v in m.items()
                              if isinstance(v, (int, float)) and not isinstance(v, bool)},
        }
        with open(os.path.join(self.run_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        if self._fh:
            self._fh.close()

    def on_fit_end(self, trainer, pl_module):
        self._summary(trainer, "fit_end")

    def on_exception(self, trainer, pl_module, exception):
        self._summary(trainer, f"exception: {type(exception).__name__}: {exception}")