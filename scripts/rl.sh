#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${ROOT}/.venv/bin/python}"

if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi
if ! [[ "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "找不到项目 Python：${PYTHON}" >&2
  exit 2
fi

exec "${PYTHON}" "${ROOT}/scripts/train_rl.py" "$@"
