"""Kaggle GPU entry point: the repository from the al-seg-voc-bundle dataset + Pascal VOC 2012 from Kaggle.

Runs the shell commands in COMMANDS (filled in by kaggle/push.sh) from the repo root, one after another.
Results end up in /kaggle/working/repo/outputs and are downloaded with `kaggle kernels output`.
Secrets (attach them to the kernel once in the Kaggle editor: Add-ons -> Secrets):
  HF_TOKEN       required for DINOv3 (gated on Hugging Face)
  WANDB_API_KEY  optional; without it W&B logs offline into repo/wandb/
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

COMMANDS = []

INPUT = Path("/kaggle/input")


def find_bundle():
    """Located by content: input layouts differ. os.walk with followlinks, because inputs are mounted via symlinks
    and Path.rglob stopped following them in Python 3.13."""
    for dirpath, _, files in os.walk(INPUT, followlinks=True):
        root = Path(dirpath).parent  # Kaggle unpacks repo.zip either into repo/ or straight into the dataset root
        if "al_loop.py" in files and Path(dirpath).name == "src" and (root / "requirements.txt").exists():
            return root
    for dirpath, dirs, files in os.walk(INPUT, followlinks=True):  # diagnostics for the log
        if dirpath.count(os.sep) - str(INPUT).count(os.sep) < 4:
            print(dirpath, sorted(dirs)[:10], sorted(files)[:10])
    raise FileNotFoundError("al-seg-voc-bundle not found under /kaggle/input")


def find_hf_weights():
    """Hugging Face model folders attached as datasets (config.json + model.safetensors), by model_type.
    Secrets do not reach API-pushed kernels, so gated weights (DINOv3) come as a private dataset instead of HF_TOKEN."""
    found = {}
    for dirpath, _, files in os.walk(INPUT, followlinks=True):
        if "config.json" in files and "model.safetensors" in files:
            model_type = json.loads((Path(dirpath) / "config.json").read_text()).get("model_type", "")
            found.setdefault(model_type, dirpath)
    return found


BUNDLE = find_bundle()
REPO = Path("/kaggle/working/repo")
print(f"[kaggle] code bundle: {BUNDLE}", flush=True)
shutil.copytree(BUNDLE, REPO, dirs_exist_ok=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements.txt")], check=True)
env = dict(os.environ)
weights = find_hf_weights()
print(f"[kaggle] attached HF weights: {weights}", flush=True)
for model_type, path in weights.items():  # e.g. DINOV3_DIR for encoders.dinov3.name=$DINOV3_DIR
    env[f"{model_type.split('_')[0].upper()}_DIR"] = path
try:
    from kaggle_secrets import UserSecretsClient

    secrets = UserSecretsClient()
    for name in ("HF_TOKEN", "WANDB_API_KEY"):
        try:
            env[name] = secrets.get_secret(name)
        except Exception as e:
            print(f"secret {name} not attached: {e}")
except ImportError:
    pass
if "WANDB_API_KEY" not in env:
    env["WANDB_MODE"] = "offline"
subprocess.run(["nvidia-smi"], check=False)
for cmd in COMMANDS:
    done = subprocess.run(["bash", "-c", cmd], cwd=REPO, env=env)
    print(f"[kaggle] {cmd}: exit code {done.returncode}", flush=True)
