# WLX Harness Core 使用说明

当前实现是旁路新增版本：不删除、不替换现有 SFT、GRPO 或 Evaluation 入口。

## 1. 运行前路径

仓库根目录执行：

```bash
export PYTHONPATH="$PWD/src:$PWD/wlx-harness-core:${PYTHONPATH:-}"
```

首版没有修改 `pyproject.toml`，因此目前必须保留这一步。

## 2. 运行一条消息级 Episode

```python
import asyncio

from shopping_grpo.evaluation.rollout import client_from_env
from wlx_harness_core import (
    EpisodeRequest,
    EpisodeRunner,
    HarnessConfig,
    LegacyChatPolicyAdapter,
)

client = client_from_env(model="deepseek-v4-flash")
policy = LegacyChatPolicyAdapter(client)

trajectory = asyncio.run(
    EpisodeRunner().run(
        EpisodeRequest(task_id=1001),
        policy,
        HarnessConfig(max_steps=35),
    )
)
```

`EpisodeRunner` 负责租约、串行工具、Guard、轨迹、Reward v3 校验和最终释放。模型 endpoint、API Key 等仍由现有客户端和环境变量管理。

如果启用 Context/Observation，Harness 和客户端参数必须一致：

```python
from wlx_harness_core import ContextPolicy, ObservationPolicy

client = client_from_env(
    model="deepseek-v4-flash",
    max_tokens=512,
    context_window=24576,
    context_safety_margin=512,
    context_compaction_enable=True,
    observation_token_budget=1536,
    observation_detail_token_budget=4096,
    observation_generic_token_budget=768,
    observation_search_top_k=20,
)

config = HarnessConfig(
    max_steps=35,
    context=ContextPolicy(
        window_tokens=24576,
        generation_reserve_tokens=512,
        safety_margin_tokens=512,
        input_budget_tokens=16384,
        compaction_enabled=True,
    ),
    observation=ObservationPolicy(token_budget=1536),
)
```

Legacy Adapter 会在调用模型前执行更严格的 `input_budget_tokens`，并在任何配置不一致时 fail-fast。

## 3. SFT 阶段

单条验收：

```python
from wlx_harness_core import SFTStageAdapter

sft = SFTStageAdapter()
output = sft.prepare(trajectory)
if output.accepted:
    training_row = output.training_row
else:
    print(output.rejection_reasons)
```

批量构造：

```python
summary = sft.build_artifacts(
    raw_path="outputs/sft-collection/raw.jsonl",
    output_dir="outputs/sft-collection",
    validation_ratio=0.1,
    seed=42,
)
```

Adapter 默认且强制读取 `data/evaluation/tasks.jsonl`；无需自行传 held-out IDs，也不能通过省略参数关闭防泄漏。

## 4. Evaluation 阶段

```python
from wlx_harness_core import EvaluationStageAdapter

evaluation_adapter = EvaluationStageAdapter()
output = evaluation_adapter.prepare(trajectory)
normalized = output.normalized_trajectory
metrics = output.deterministic_metrics

summary = evaluation_adapter.summarize([1001], [trajectory])
```

`Trajectory` 对象、`trajectory.to_dict()` 和 legacy dict 均可输入。默认结果不会把 hidden goal、raw observation 或私有 reasoning 送给 Judge。

## 5. GRPO 阶段

GRPO 必须使用 `parallel_tool_call_policy="reject"`：

```python
from wlx_harness_core import HarnessConfig, VerlHarnessBridge

bridge = VerlHarnessBridge(
    HarnessConfig(
        max_steps=35,
        max_assistant_turns=40,
        parallel_tool_call_policy="reject",
    )
)

agent_loop_config = bridge.to_agent_loop_config()
tool_config = bridge.to_verl_tool_config()
```

新增的 canonical 工具配置应在训练前做一致性检查：

```python
import json
from pathlib import Path

tools_path = Path("wlx-harness-core/wlx-tools.json")
bridge.assert_tool_config_compatible(
    json.loads(tools_path.read_text(encoding="utf-8"))
)

verl_overrides = bridge.to_verl_overrides(
    agent_loop_config_path="configs/agent_loop.yaml",
    tool_config_path=str(tools_path),
)
```

现有 `configs/tools.json` 保留给旧路径，但其描述已与注册表漂移，新 Harness 不应继续使用。若修改 GRPO 参数，应从 `to_agent_loop_config()` / `to_verl_tool_config()` 重新生成 `wlx-` 前缀配置，不要手工维护第二份 Tool Schema。

veRL rollout 返回后，把公共诊断接入统一 metadata：

```python
stage_metadata = bridge.to_stage_metadata(agent_loop_output)
```

Bridge 只映射和校验配置、工具与 `extra_fields["shopping"]`；不会启动训练，也不会替代 veRL 的 token/logprob 循环。

## 6. 审计与公开数据

```python
audit_json = trajectory.to_dict()
legacy_json = trajectory.to_legacy_dict()
```

- `to_dict()` 是完整审计格式，可能包含原始环境返回，只应存入受控位置。
- `to_legacy_dict()` 用于现有 SFT/Evaluation。
- 默认 `RunObserver` 收到的是公开事件，不含 raw result、私有 reasoning 和未验证 reward。

## 7. 离线测试

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=src:wlx-harness-core \
.venv/bin/python -m unittest discover \
  -s wlx-harness-core/wlx-tests \
  -p 'wlx_test*.py' -v
```

这些测试使用 Fake Environment/Fake Policy，不连接模型、不启动 ShopSimulator、不运行训练。

## 8. 真正运行时你需要配置的内容

仅在连接真实服务时需要：

- ShopSimulator 地址：`HarnessConfig.environment_base_url`；
- 模型服务：现有 `OPENAI_BASE_URL`、`OPENAI_API_KEY`、模型名；
- GRPO 运行时：veRL 环境和最终 agent-loop/tool config 路径。

本次实现与离线验证不需要填写这些值。
