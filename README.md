# a-cv-2026-lab-2 — Active learning for semantic segmentation (Pascal VOC 2012)

Frozen encoders (DINOv3 ViT-B/16, MAE ViT-B/16) + a small segmentation head trained from scratch every round.
Plan and protocol: [docs/assignment2_plan.md](docs/assignment2_plan.md).

## Layout
```
configs/        base.yaml (all defaults) + per-experiment overrides
scripts/        part1.sh (full-data upper bound), part2.sh (Uncertainty top-B AL: RIPU, BALD)
src/
  data.py       VOC ids, masks, dev split            oracle.py   Oracle + post-hoc cost proxies
  features.py   frozen encoders, feature caching (entrypoint)
  head.py       segmentation head                    metrics.py  mIoU, per-class IoU, ECE
  train.py      LightningModule, datasets, W&B weight histograms, train_head / evaluate
  selection/    herding.py (MaxHerding), uncertainty.py (RIPU, BALD), registry SELECTORS
  al_loop.py    AL rounds (entrypoint)               run_full.py full-data upper bound (entrypoint)
  tracking.py   results CSV, W&B run setup, head checkpoints
tests/          MaxHerding toy example + sanity checks, Oracle, metrics
```

Any run is reproducible from a config plus `key=value` overrides (dotted keys allowed):
```
python -m src.features encoder=dinov3
python -m src.run_full --config configs/full.yaml encoder=mae seed=1
python -m src.al_loop --config configs/uncertainty.yaml encoder=dinov3 al.scorer=bald
python -m src.al_loop --config configs/maxherding.yaml encoder=dinov3
```
Results are appended to `outputs/results.csv`; selections, cold-start batch, plots and head weights go to `outputs/`.

## Running on Kaggle (T4)
Attach the Pascal VOC 2012 dataset (the VOC root is found automatically under `/kaggle/input`, or pass `voc_root=...`).
Add Kaggle secrets `WANDB_API_KEY` and `HF_TOKEN` (DINOv3 is gated on Hugging Face; accept its license first).

```python
import os
from kaggle_secrets import UserSecretsClient
s = UserSecretsClient()
os.environ["WANDB_API_KEY"] = s.get_secret("WANDB_API_KEY")
os.environ["HF_TOKEN"] = s.get_secret("HF_TOKEN")
```
```
!git clone <repo-url> /kaggle/working/lab2
!pip install -q -r /kaggle/working/lab2/requirements.txt
!bash /kaggle/working/lab2/scripts/part1.sh     # session 1: tests + full-data upper bound, 2 encoders x 3 seeds
!bash /kaggle/working/lab2/scripts/part2.sh     # session 2: Uncertainty top-B (RIPU, BALD), 6 rounds, 2 encoders
```
Extra overrides are forwarded, e.g. `bash scripts/part2.sh wandb.mode=offline`.
Features are cached in `/tmp/al_cache` (outside the 20 GB `/kaggle/working` limit) and deleted after each encoder.
