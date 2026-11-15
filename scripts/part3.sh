#!/usr/bin/env bash
# Part 3: coverage-based AL (21 -> 672): MaxHerding, ProbCover (bonus), MaxHerding sigma ablation; 2 encoders.
# Skip the ablation with ABLATION=0; pick encoders with ENCODERS="mae" (default: dinov3 mae).
set -euo pipefail
cd "$(dirname "$0")/.."
CACHE_DIR=${CACHE_DIR:-/tmp/al_cache}
ABLATION=${ABLATION:-1}
ENCODERS=${ENCODERS:-dinov3 mae}

python -m pytest -q tests/test_herding.py tests/test_probcover.py
for enc in $ENCODERS; do
  python -m src.features encoder=$enc cache_dir=$CACHE_DIR "$@"
  python -m src.al_loop --config configs/maxherding.yaml encoder=$enc cache_dir=$CACHE_DIR "$@"
  python -m src.al_loop --config configs/probcover.yaml encoder=$enc cache_dir=$CACHE_DIR "$@"
  if [ "$ABLATION" = "1" ]; then
    for s in 0.5 2; do
      python -m src.al_loop --config configs/maxherding.yaml encoder=$enc al.sigma_scale=$s cache_dir=$CACHE_DIR "$@"
    done
  fi
  rm -f "$CACHE_DIR"/feats_${enc}_*.npy "$CACHE_DIR"/vec_${enc}.npy "$CACHE_DIR"/${enc}.done
done
