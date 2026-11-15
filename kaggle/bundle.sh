#!/usr/bin/env bash
# Build build/kaggle_bundle/: the local code (committed or not), uploaded as one private Kaggle dataset, so the GPU
# kernel runs exactly this working tree without pushing to GitHub.
#   bash kaggle/bundle.sh && uvx --with six kaggle datasets create -p build/kaggle_bundle --dir-mode zip            (first time)
#   bash kaggle/bundle.sh && uvx --with six kaggle datasets version -p build/kaggle_bundle -m "update" --dir-mode zip
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=build/kaggle_bundle
rm -rf "$OUT" && mkdir -p "$OUT/repo"
cp -r requirements.txt src configs scripts tests "$OUT/repo/"
find "$OUT" -name __pycache__ -prune -exec rm -rf {} +
USER=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
cat > "$OUT/dataset-metadata.json" <<JSON
{"title": "al-seg-voc-bundle", "id": "$USER/al-seg-voc-bundle", "licenses": [{"name": "other"}]}
JSON
du -sh "$OUT"
