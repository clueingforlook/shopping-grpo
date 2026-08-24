#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_MODEL=""
LORA_ADAPTER=""
SERVED_MODEL_NAME="shopping-agent"
PORT="8000"

# FlashInfer 0.6.x 会把 Blackwell SM 12.0 误判为低于 sm75，并在 vLLM
# warmup 的 top-k/top-p sampler 中终止。只关闭该 sampler；注意力后端与
# 模型计算仍由 vLLM 按当前 GPU 正常选择。
export VLLM_USE_FLASHINFER_SAMPLER=0

# Some server images export these as 0. vLLM passes OMP_NUM_THREADS directly
# to torch.set_num_threads(), which requires a positive integer.
if ! [[ "${OMP_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export OMP_NUM_THREADS=1
fi
if ! [[ "${MKL_NUM_THREADS:-}" =~ ^[1-9][0-9]*$ ]]; then
  export MKL_NUM_THREADS="$OMP_NUM_THREADS"
fi

usage() {
  echo "用法：$0 --base-model PATH [--lora-adapter PATH] [--served-model NAME] [--port PORT]" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-model)
      BASE_MODEL="${2:-}"
      shift 2
      ;;
    --lora-adapter)
      LORA_ADAPTER="${2:-}"
      shift 2
      ;;
    --served-model)
      SERVED_MODEL_NAME="${2:-}"
      shift 2
      ;;
    --port)
      PORT="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      echo "未知参数：$1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$BASE_MODEL" ]]; then
  usage
  exit 2
fi
if [[ ! -e "$BASE_MODEL" ]]; then
  echo "Base 模型不存在：$BASE_MODEL" >&2
  exit 2
fi
if [[ -n "$LORA_ADAPTER" && ! -e "$LORA_ADAPTER" ]]; then
  echo "LoRA adapter 不存在：$LORA_ADAPTER" >&2
  exit 2
fi
if [[ -z "$LORA_ADAPTER" && -d "$BASE_MODEL/lora_adapter" ]]; then
  echo "检测到导出目录中的 LoRA，但没有传入 --lora-adapter；已拒绝只评底座模型。" >&2
  exit 2
fi
if [[ ! -x "$ROOT/.venv/bin/vllm" ]]; then
  echo "当前项目虚拟环境没有 vLLM：$ROOT/.venv/bin/vllm" >&2
  exit 2
fi

COMMON_ARGS=(
  serve "$BASE_MODEL"
  --port "$PORT"
  --max-model-len 24576
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
)

if [[ -n "$LORA_ADAPTER" ]]; then
  exec "$ROOT/.venv/bin/vllm" "${COMMON_ARGS[@]}" \
    --served-model-name "wlx-base-backbone" \
    --enable-lora \
    --lora-modules "$SERVED_MODEL_NAME=$LORA_ADAPTER"
fi

exec "$ROOT/.venv/bin/vllm" "${COMMON_ARGS[@]}" \
  --served-model-name "$SERVED_MODEL_NAME"
