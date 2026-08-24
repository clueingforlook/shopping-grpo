#!/usr/bin/env bash
set -euo pipefail

WLX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WLX_PYTHON="${WLX_PYTHON:-${WLX_ROOT}/.venv/bin/python}"

if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi
if ! [[ "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -x "${WLX_PYTHON}" ]]; then
  echo "找不到项目 Python：${WLX_PYTHON}" >&2
  exit 2
fi

exec "${WLX_PYTHON}" "${WLX_ROOT}/scripts/wlx_train_rl.py" "$@"
