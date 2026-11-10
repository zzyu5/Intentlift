#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHONPATH_VALUE="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
DATASET_ROOT="${INTENTIR_DATASET_ROOT:-}"
WORKERS="${INTENTIR_DATASET_WORKERS:-1}"
TIMEOUT="${INTENTIR_DATASET_TIMEOUT:-120}"
CUDA_ARCH="${INTENTIR_CUDA_ARCH:-120}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Run the real native CUDA Dataset compiler probe.

Usage:
  bash rundemo/04_probe_cuda_dataset.sh

Environment overrides:
  INTENTIR_DATASET_ROOT   required existing multi-frontend Dataset root
  INTENTIR_DATASET_WORKERS compiler workers, default 1
  INTENTIR_DATASET_TIMEOUT per-case timeout in seconds, default 120
  INTENTIR_CUDA_ARCH      target CUDA architecture, default 120

This uses CudaCompiler and the Dataset's actual CUDA source/ABI.  It is not a
Triton/TTIR fallback and it returns nonzero when the native toolchain is
unavailable or a real Dataset case fails.
EOF
    exit 0
fi

if [[ $# -ne 0 ]]; then
    echo "unexpected argument: $1" >&2
    exit 2
fi

if [[ -z "$DATASET_ROOT" ]]; then
    echo "Dataset root is required." >&2
    echo "Set INTENTIR_DATASET_ROOT to an existing Dataset root." >&2
    exit 2
fi

if [[ ! -d "$DATASET_ROOT" ]]; then
    echo "Dataset root does not exist: $DATASET_ROOT" >&2
    exit 2
fi

exec env PYTHONPATH="$PYTHONPATH_VALUE" \
    python "$REPO_ROOT/tools/frontend_dataset.py" \
    --dataset-root "$DATASET_ROOT" \
    --frontend cuda \
    --probe-cuda \
    --workers "$WORKERS" \
    --timeout "$TIMEOUT" \
    --cuda-arch "$CUDA_ARCH"
