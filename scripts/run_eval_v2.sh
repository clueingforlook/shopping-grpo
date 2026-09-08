#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_dir"

python_bin="${PYTHON:-.venv/bin/python}"
model="${EVAL_V2_MODEL:-deepseek-v4-pro}"
base_url="${DEEPSEEK_BASE_URL:-https://api.deepseek.com}"
concurrency="${EVAL_V2_CONCURRENCY:-2}"
api_key_file="${DEEPSEEK_API_KEY_FILE:-}"

if [[ ! -x "$python_bin" ]]; then
  echo "找不到 Python：$python_bin" >&2
  exit 1
fi
if [[ -z "${DEEPSEEK_API_KEY:-}" && -z "$api_key_file" ]]; then
  echo "请设置 DEEPSEEK_API_KEY，或设置 DEEPSEEK_API_KEY_FILE。" >&2
  exit 1
fi

api_key_args=()
if [[ -n "$api_key_file" ]]; then
  api_key_args=(--api-key-file "$api_key_file")
fi

rubrics="outputs/evaluation/eval200-rubrics-v1/rubrics.jsonl"
base_trajectories="outputs/evaluation/base-final200-v2/trajectories.jsonl"
sft_trajectories="outputs/evaluation/sft-final200-v2/trajectories.jsonl"
gold_manifest="outputs/evaluation/judge-gold-v4/judge-gold-manifest.json"
gold_labels="outputs/evaluation/judge-gold-v6-eval-v2/judge-gold-labels-frozen.jsonl"

eval_root="outputs/evaluation/eval-v2"
calibration_dir="$eval_root/judge-calibration-v4-pro-prompt-v2"
calibration_manifest="$calibration_dir/judge-calibration-manifest-gold-v6.json"
production_dir="$eval_root/judge-production-v4-pro-prompt-v2"
base_eval_dir="$eval_root/base"
sft_eval_dir="$eval_root/sft"
comparison_dir="$eval_root/base-vs-sft"

echo "[1/5] 用 $model 运行 24 条 Gold 校准轨迹"
"$python_bin" scripts/run_eval_judge.py \
  --trajectory-run "base=$base_trajectories" \
  --trajectory-run "sft=$sft_trajectories" \
  --rubrics "$rubrics" \
  --selected-cases "$gold_manifest" \
  --mode calibration \
  --model "$model" \
  --base-url "$base_url" \
  --max-tokens 8192 \
  --concurrency "$concurrency" \
  --output-dir "$calibration_dir" \
  "${api_key_args[@]}"

if [[ ! -f "$calibration_dir/judge-results.jsonl" ]]; then
  echo "校准 Judge 仍有失败或未完成项；直接重跑本命令会从断点继续。" >&2
  exit 3
fi

if [[ ! -f "$calibration_manifest" ]]; then
  echo "[2/5] 对照人工 Gold 生成冻结校准门禁"
  "$python_bin" scripts/calibrate_eval_judge.py \
    --gold-labels "$gold_labels" \
    --judge-results "$calibration_dir/judge-results.jsonl" \
    --model "$model" \
    --output "$calibration_manifest"
else
  echo "[2/5] 已存在校准门禁，复用：$calibration_manifest"
fi

calibration_status="$($python_bin -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["status"])' "$calibration_manifest")"
if [[ "$calibration_status" != "frozen" ]]; then
  echo "V4 Pro 未通过 Gold 校准，已停止生产评测。请先查看：$calibration_manifest" >&2
  exit 2
fi

echo "[3/5] 用通过校准的 $model 评价 Base/SFT 共 400 条轨迹"
"$python_bin" scripts/run_eval_judge.py \
  --trajectory-run "base=$base_trajectories" \
  --trajectory-run "sft=$sft_trajectories" \
  --rubrics "$rubrics" \
  --mode production \
  --calibration-manifest "$calibration_manifest" \
  --model "$model" \
  --base-url "$base_url" \
  --max-tokens 8192 \
  --concurrency "$concurrency" \
  --output-dir "$production_dir" \
  "${api_key_args[@]}"

if [[ ! -f "$production_dir/judge-results.jsonl" ]]; then
  echo "生产 Judge 仍有失败或未完成项；直接重跑本命令会从断点继续。" >&2
  exit 4
fi

echo "[4/5] 合并客观指标、全部 Rubric 判断、过程分和失败归因"
if [[ ! -f "$base_eval_dir/eval-manifest.json" ]]; then
  "$python_bin" scripts/run_offline_evaluation.py \
    --trajectories "$base_trajectories" \
    --rubrics "$rubrics" \
    --judge-results "$production_dir/judge-results.jsonl" \
    --judge-case-prefix base \
    --output-dir "$base_eval_dir"
fi
if [[ ! -f "$sft_eval_dir/eval-manifest.json" ]]; then
  "$python_bin" scripts/run_offline_evaluation.py \
    --trajectories "$sft_trajectories" \
    --rubrics "$rubrics" \
    --judge-results "$production_dir/judge-results.jsonl" \
    --judge-case-prefix sft \
    --output-dir "$sft_eval_dir"
fi

echo "[5/5] 生成 Base/SFT 成对比较"
if [[ ! -f "$comparison_dir/comparison-summary.json" ]]; then
  "$python_bin" scripts/compare_evaluation_runs.py \
    --baseline "$base_eval_dir/trajectory-evaluations.jsonl" \
    --candidate "$sft_eval_dir/trajectory-evaluations.jsonl" \
    --output-dir "$comparison_dir"
fi

echo "Eval v2 完成。"
echo "Base 报告：$base_eval_dir/run-summary.md"
echo "SFT 报告：$sft_eval_dir/run-summary.md"
echo "对比报告：$comparison_dir/comparison-summary.md"
