#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHONPATH_VALUE="$REPO_ROOT/python${PYTHONPATH:+:$PYTHONPATH}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'EOF'
Validate the active real Triton/TTIR SCRC catalog.

This is a catalog/source-resolution prerequisite, not a Gate1/Gate2 closure.
It does not call an LLM provider and does not create evidence under the repo.

Usage:
  bash rundemo/01_validate_scrc_catalog.sh
EOF
    exit 0
fi

if [[ $# -ne 0 ]]; then
    echo "unexpected argument: $1" >&2
    exit 2
fi

exec env PYTHONPATH="$PYTHONPATH_VALUE" \
    python "$REPO_ROOT/tools/scrc_batch.py" \
    --validate-manifest \
    --json
