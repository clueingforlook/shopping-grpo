# WLX 实验结果

该目录保存可直接查看和复核的评测、训练指标与说明文档：

- `docs/`：SFT Eval 与 Eval v2 的说明、结果和典型案例。
- `eval-v2/`：Base/SFT 的 LLM Judge 评测结果及配对比较。
- `eval-v3/`：Base、SFT、GRPO step 250/500 的 ShopSimulator 结构化评测结果。
- `grpo-training/`：完整 500 步训练日志、250/500 步曲线及对应 CSV 指标。

为控制仓库体积，本目录不包含原始轨迹、采集事件、中间 Judge 请求或 review queue。曾经未正确加载 LoRA 的 `wlx-grpo-step250` 和 `wlx-grpo-step300` 结果也已排除。

GRPO step 250/500 LoRA 权重见 [GitHub Release](https://github.com/clueingforlook/shopping-grpo-longhorizon/releases/tag/wlx-grpo-lora-v1)。
