#!/usr/bin/env bash
# Push the GPU kernel that runs the given shell commands from the repo root, e.g.
#   bash kaggle/push.sh "bash scripts/part3.sh"
#   bash kaggle/push.sh "ABLATION=0 bash scripts/part3.sh" "python -m src.report"
# KERNEL=al-seg-voc-baselines bash kaggle/push.sh ... runs a second kernel in parallel (default slug: al-seg-voc).
# Update the code bundle first (kaggle/bundle.sh). When the kernel has finished:
#   uvx --with six kaggle kernels status <user>/al-seg-voc
#   uvx --with six kaggle kernels output <user>/al-seg-voc -p build/kaggle_out
set -euo pipefail
cd "$(dirname "$0")/.."
USER=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
KERNEL=${KERNEL:-al-seg-voc}
EXTRA_DATASETS=${EXTRA_DATASETS:-}  # e.g. EXTRA_DATASETS=dinov3-vitb16-weights
OUT=build/kaggle_kernel_$KERNEL
rm -rf "$OUT" && mkdir -p "$OUT"
python3 - "$OUT/run.py" "$@" <<'PY'
import json, sys
src = open("kaggle/kernel/run.py").read()
out, cmds = sys.argv[1], sys.argv[2:]
assert cmds, "give at least one command, e.g. 'bash scripts/part3.sh'"
open(out, "w").write(src.replace("COMMANDS = []", f"COMMANDS = {json.dumps(cmds)}", 1))
PY
EXTRA=$(for d in $EXTRA_DATASETS; do printf ', "%s/%s"' "$USER" "$d"; done)
cat > "$OUT/kernel-metadata.json" <<JSON
{"id": "$USER/$KERNEL", "title": "$KERNEL", "code_file": "run.py", "language": "python",
 "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true,
 "dataset_sources": ["$USER/al-seg-voc-bundle", "gopalbhattrai/pascal-voc-2012-dataset"$EXTRA]}
JSON
uvx --with six kaggle kernels push -p "$OUT" --accelerator NvidiaTeslaT4
