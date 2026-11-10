#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHONPATH_VALUE="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
DATASET_ROOT="${1:-${INTENTIR_DATASET_ROOT:-}}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Validate the real CUDA/TileLang multi-frontend Dataset manifest.

Usage:
  bash rundemo/02_validate_multifrontend_dataset.sh [DATASET_ROOT]

DATASET_ROOT is required.  Pass it as an argument or set
INTENTIR_DATASET_ROOT to an existing Dataset root.
EOF
    exit 0
fi

if [[ -z "$DATASET_ROOT" ]]; then
    echo "Dataset root is required." >&2
    echo "Pass DATASET_ROOT as an argument or set INTENTIR_DATASET_ROOT." >&2
    exit 2
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "Dataset root does not exist: $DATASET_ROOT" >&2
    exit 2
fi

exec env PYTHONPATH="$PYTHONPATH_VALUE" \
    python "$REPO_ROOT/tools/frontend_dataset.py" \
    --dataset-root "$DATASET_ROOT" \
    --validate
