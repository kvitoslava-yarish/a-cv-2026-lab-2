# a-cv-2026-lab-2 — Active learning for semantic segmentation (Pascal VOC 2012)

Frozen encoders (DINOv3 ViT-B/16, MAE ViT-B/16) + a small segmentation head trained from scratch every round.
Plan and protocol: [docs/assignment2_plan.md](docs/assignment2_plan.md).

## Layout
```
configs/        base.yaml (all defaults) + per-experiment overrides
scripts/        part1.sh (full-data upper bound), part2.sh (Uncertainty top-B AL: RIPU, BALD),
                part3.sh (MaxHerding, ProbCover, MaxHerding sigma ablation)
src/
  data.py       VOC ids, masks, dev split            oracle.py   Oracle + post-hoc cost proxies
  features.py   frozen encoders, feature caching (entrypoint)
  head.py       segmentation head                    metrics.py  mIoU, per-class IoU, ECE
  train.py      LightningModule, datasets, W&B weight histograms, train_head / evaluate
  selection/    herding.py (MaxHerding), probcover.py (ProbCover, bonus), uncertainty.py (RIPU, BALD),
                registry SELECTORS
  al_loop.py    AL rounds (entrypoint)               run_full.py full-data upper bound (entrypoint)
  tracking.py   results CSV, W&B run setup, head checkpoints
  report.py     learning curves, summary table (AUBC, gain over Random), cost and coverage curves (entrypoint)
kaggle/         bundle.sh (code -> private Kaggle dataset), push.sh + kernel/run.py (run commands on a Kaggle T4)
tests/          MaxHerding toy example + sanity checks, ProbCover checks, Oracle, metrics
```

Any run is reproducible from a config plus `key=value` overrides (dotted keys allowed):
```
python -m src.features encoder=dinov3
python -m src.run_full --config configs/full.yaml encoder=mae seed=1
python -m src.al_loop --config configs/uncertainty.yaml encoder=dinov3 al.scorer=bald
python -m src.al_loop --config configs/maxherding.yaml encoder=dinov3
python -m src.al_loop --config configs/maxherding.yaml encoder=dinov3 al.sigma_scale=0.5   # sigma ablation
python -m src.al_loop --config configs/probcover.yaml encoder=dinov3
python -m src.report                       # -> outputs/report/: summary.{csv,md}, learning/cost/coverage curves
```
Results are appended to `outputs/results.csv` (selector-side stats such as ProbCover delta, coverage and the number
of greedy picks go to `outputs/selector_stats.csv`); selections, cold-start batch, plots and head weights go to `outputs/`.

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
!bash /kaggle/working/lab2/scripts/part3.sh     # session 3: MaxHerding, ProbCover, sigma ablation (ABLATION=0 skips it)
```
Extra overrides are forwarded, e.g. `bash scripts/part2.sh wandb.mode=offline`.
Features are cached in `/tmp/al_cache` (outside the 20 GB `/kaggle/working` limit) and deleted after each encoder.

## Running on Kaggle from the command line
Same as assignment 1: the local working tree is uploaded as a private dataset and a script kernel runs on a T4,
no GitHub push needed. Needs `~/.kaggle/kaggle.json`. Attach the `HF_TOKEN` (and optionally `WANDB_API_KEY`) secret
to the `al-seg-voc` kernel once in the Kaggle editor (Add-ons -> Secrets); without a W&B key it logs offline.
```
bash kaggle/bundle.sh && uvx --with six kaggle datasets create -p build/kaggle_bundle --dir-mode zip     # first time
bash kaggle/bundle.sh && uvx --with six kaggle datasets version -p build/kaggle_bundle -m update --dir-mode zip
bash kaggle/push.sh "bash scripts/part3.sh" "python -m src.report"
KERNEL=al-seg-voc-baselines bash kaggle/push.sh "ENCODERS=mae bash scripts/part1.sh"   # 2nd kernel in parallel
uvx --with six kaggle kernels status <user>/al-seg-voc
uvx --with six kaggle kernels output <user>/al-seg-voc -p build/kaggle_out
```
Kaggle secrets do not reach API-pushed kernels, so DINOv3 (gated) comes as a private dataset of its HF weights,
downloaded once locally with your own token (`build/dinov3_weights/`, with its LICENSE), and the kernel passes the
folder it finds as `encoders.dinov3.name`:
```
uvx --with six kaggle datasets create -p build/dinov3_weights --dir-mode zip
EXTRA_DATASETS=dinov3-vitb16-weights KERNEL=al-seg-voc-dinov3 bash kaggle/push.sh \
  'ENCODERS=dinov3 bash scripts/part3.sh encoders.dinov3.name=${DINOV3_DIR:?}' "python -m src.report"
```
`selector_stats.csv` also holds two label-free coverage metrics for every method, with the same median sigma and
ProbCover delta for all: generalized (Gaussian) coverage and P(C(L, delta)).

## Cold start
Model-free methods (MaxHerding, ProbCover) pick their own first batch of 21 images. Model-based methods
(Uncertainty top-B, UHerding) have no task model at round 0 and all use the same MaxHerding first batch.

## ProbCover (bonus)
Hard-ball version of MaxHerding on the same L2-normalized image vectors: the kernel is `[d(x, s) < delta]`, and the
greedy step is the same `herding_core`. `delta` is label-free: k-means (k = 21) pseudo-labels on the pool vectors,
then the largest `delta` whose balls are pure for at least 95% of the pool (closed form: an order statistic of the
distance to the nearest point from another cluster). Once the pool is fully covered every gain is 0, and the rest of
the batch is random, as in the paper.
