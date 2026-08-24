# Shopping GRPO

基于 ShopSimulator 的购物智能体后训练与轨迹评测项目。项目覆盖 SFT 数据构建、LoRA SFT、GRPO 强化学习、模型评测，以及可复现的实验结果记录。

## 主要功能

- 使用结构化工具与 ShopSimulator 购物环境交互
- 收集并构建无数据泄漏的 SFT 数据集
- 使用 Transformers + PEFT 进行 LoRA SFT
- 基于 veRL 运行 GRPO 训练
- 对 Base、SFT 和 GRPO 模型进行轨迹评测与结果汇总
- 提供 CPU 冒烟测试和完整的 pytest 测试集

## 项目结构

```text
configs/                 训练、Agent Loop 与工具配置
data/                    SFT、GRPO 和评测数据
environments/            内置 ShopSimulator 环境
scripts/                 数据处理、训练和评测入口
src/shopping_grpo/       Python 核心包
tests/                   项目测试
wlx-docs/                训练与评测的详细说明
wlx-results/             已保存的实验结果
```

## 环境要求

- Python 3.10 及以上
- [uv](https://docs.astral.sh/uv/)
- Bash（建议使用 Linux 或 WSL）
- SFT/GRPO 训练需要支持 CUDA 的 GPU 环境

## 快速开始

仅安装基础包和开发依赖：

```bash
uv sync --extra dev
```

运行不依赖模型和 ShopSimulator 的 CPU 冒烟测试：

```bash
uv run shopping-grpo smoke
```

运行测试：

```bash
uv run pytest
```

安装完整训练环境并启动 ShopSimulator：

```bash
bash scripts/setup.sh
bash scripts/start_environment.sh
```

环境默认运行在 `http://127.0.0.1:5700`。可在另一个终端验证接口：

```bash
uv run python scripts/smoke_shop_env.py
```

## 训练与评测

LoRA SFT 示例：

```bash
uv run python scripts/train_lora_sft.py \
  --model Qwen/Qwen3.5-2B \
  --train data/sft/train.jsonl \
  --validation data/sft/validation.jsonl \
  --output outputs/models/sft-lora
```

GRPO 训练入口：

```bash
uv run python scripts/train_grpo.py --dry-run
uv run python scripts/train_grpo.py
```

更多配置和完整流程请查看 [`wlx-docs/`](wlx-docs/)。生成的模型、检查点与日志默认保存在 `outputs/` 等已被 `.gitignore` 忽略的目录中。

## 数据说明

仓库包含教程使用的 SFT、GRPO 和评测数据，详细规模和划分见 [`data/README.md`](data/README.md)。请勿将 API Key、模型权重、检查点或本地 `.env` 文件提交到公开仓库。
