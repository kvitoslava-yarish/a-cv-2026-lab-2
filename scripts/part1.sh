#!/usr/bin/env bash
# Part 1: tests, feature extraction, full-data upper bound (2 encoders x 3 seeds).
set -euo pipefail
cd "$(dirname "$0")/.."
CACHE_DIR=${CACHE_DIR:-/tmp/al_cache}

python -m pytest -q tests
for enc in dinov3 mae; do
  python -m src.features encoder=$enc cache_dir=$CACHE_DIR "$@"
  for seed in 0 1 2; do
    python -m src.run_full --config configs/full.yaml encoder=$enc seed=$seed cache_dir=$CACHE_DIR "$@"
  done
  rm -f "$CACHE_DIR"/feats_${enc}_*.npy "$CACHE_DIR"/vec_${enc}.npy "$CACHE_DIR"/${enc}.done
done
