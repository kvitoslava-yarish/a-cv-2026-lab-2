#!/usr/bin/env bash
# Push the GPU kernel that runs the given shell commands from the repo root, e.g.
#   bash kaggle/push.sh "bash scripts/part3.sh"
#   bash kaggle/push.sh "ABLATION=0 bash scripts/part3.sh" "python -m src.report"
# Update the code bundle first (kaggle/bundle.sh). When the kernel has finished:
#   uvx kaggle kernels status <user>/al-seg-voc
#   uvx kaggle kernels output <user>/al-seg-voc -p build/kaggle_out
set -euo pipefail
cd "$(dirname "$0")/.."
USER=$(python3 -c "import json,os; print(json.load(open(os.path.expanduser('~/.kaggle/kaggle.json')))['username'])")
OUT=build/kaggle_kernel
rm -rf "$OUT" && mkdir -p "$OUT"
python3 - "$OUT/run.py" "$@" <<'PY'
import json, sys
src = open("kaggle/kernel/run.py").read()
out, cmds = sys.argv[1], sys.argv[2:]
assert cmds, "give at least one command, e.g. 'bash scripts/part3.sh'"
open(out, "w").write(src.replace("COMMANDS = []", f"COMMANDS = {json.dumps(cmds)}", 1))
PY
cat > "$OUT/kernel-metadata.json" <<JSON
{"id": "$USER/al-seg-voc", "title": "al-seg-voc", "code_file": "run.py", "language": "python",
 "kernel_type": "script", "is_private": true, "enable_gpu": true, "enable_internet": true,
 "dataset_sources": ["$USER/al-seg-voc-bundle", "gopalbhattrai/pascal-voc-2012-dataset"]}
JSON
uvx kaggle kernels push -p "$OUT" --accelerator NvidiaTeslaT4
