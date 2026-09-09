#!/usr/bin/env bash
# Usage: bash scripts/maple/few_shot.sh data corn 4 [CLIP]
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${4:-CLIP}" != "CLIP" ]]; then
  echo "MaPLe currently supports CLIP ViT-B/16 and ViT-B/32" >&2
  exit 2
fi
exec "${PYTHON_BIN:-python}" -u "$SCRIPT_DIR/run_fewshot.py" --data "${1:?data root}" --datasets "${2:?dataset}" --shots "${3:?shots}"
