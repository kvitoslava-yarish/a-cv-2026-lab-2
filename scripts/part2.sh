#!/usr/bin/env bash
# Part 2: reduced-budget AL (21 -> 672) with Uncertainty top-B, RIPU and BALD scorers.
set -euo pipefail
cd "$(dirname "$0")/.."
CACHE_DIR=${CACHE_DIR:-/tmp/al_cache}

for enc in dinov3 mae; do
  python -m src.features encoder=$enc cache_dir=$CACHE_DIR "$@"
  for scorer in ripu bald; do
    python -m src.al_loop --config configs/uncertainty.yaml encoder=$enc al.scorer=$scorer cache_dir=$CACHE_DIR "$@"
  done
  rm -f "$CACHE_DIR"/feats_${enc}_*.npy "$CACHE_DIR"/vec_${enc}.npy "$CACHE_DIR"/${enc}.done
done
