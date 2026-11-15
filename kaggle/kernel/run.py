"""Kaggle GPU entry point: the repository from the al-seg-voc-bundle dataset + Pascal VOC 2012 from Kaggle.

Runs the shell commands in COMMANDS (filled in by kaggle/push.sh) from the repo root, one after another.
Results end up in /kaggle/working/repo/outputs and are downloaded with `kaggle kernels output`.
Secrets (attach them to the kernel once in the Kaggle editor: Add-ons -> Secrets):
  HF_TOKEN       required for DINOv3 (gated on Hugging Face)
  WANDB_API_KEY  optional; without it W&B logs offline into repo/wandb/
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

COMMANDS = []

INPUT = Path("/kaggle/input")
BUNDLE = next(INPUT.rglob("repo/src/al_loop.py")).parents[2]  # located by content: input layouts differ
REPO = Path("/kaggle/working/repo")
shutil.copytree(BUNDLE / "repo", REPO, dirs_exist_ok=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(REPO / "requirements.txt")], check=True)
env = dict(os.environ)
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
