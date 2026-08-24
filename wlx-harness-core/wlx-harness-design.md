# WLX 统一 Harness 设计

> 状态：v1 已实现。它是旁路新增架构，不删除或替换项目现有 SFT、GRPO、Evaluation 入口。

## 1. 目标与边界

WLX Harness 把购物 Agent 各阶段共有的运行约束放进统一底座，各阶段只保留自己的数据或训练逻辑。

```text
SFT Collection             Evaluation                GRPO / veRL
过滤、清洗、切分、落盘      指标、汇总、Judge          token、logprob、优化
       │                        │                         │
       ▼                        ▼                         ▼
wlx_stage_sft.py       wlx_stage_evaluation.py    wlx_stage_grpo.py
       └───────────────┬───────────────┬─────────────────┘
                       ▼
                 WLX Harness Core
 contracts / config / tools / environment / policy / reward / runner
                       │
                       ▼
          现有 shopping_grpo.environment + ShopSimulator
```

Core 负责：

- 一条 Episode 的 `start -> model -> guard -> tool -> done -> close` 生命周期；
- Environment v2.1、Observation v2、Reward v3 和 Tool Schema v2 契约；
- 上下文预算、Observation 投影、串行工具调用与错误归类；
- 统一 Trajectory、审计数据和公开 observer 事件。

Core 不负责 SFT 训练、GRPO 优化器、评测指标定义，也不替代 veRL 的 token 级循环。

## 2. 代码组织

| 文件 | 责任 |
|---|---|
| `wlx_contracts.py` | 请求、工具调用、Step、Trajectory、错误和终止类型 |
| `wlx_config.py` | Core、上下文、Observation 配置及参数校验 |
| `wlx_tools.py` | 单一 Tool Schema 注册表、Guard、环境动作转换、指纹校验 |
| `wlx_environment.py` | `ShopAgentEnv` 的异步生命周期与版本校验 |
| `wlx_policy.py` | 现有 `complete(messages, tools)` 客户端适配 |
| `wlx_reward.py` | Reward v3 结构和语义校验 |
| `wlx_runner.py` | SFT/Evaluation 共用的消息级 Episode 状态机 |
| `wlx_serialization.py` | Core JSON 与现有 legacy trajectory 的显式转换 |
| `wlx_stage_sft.py` | SFT 验收、held-out 防泄漏和 artifact 构造入口 |
| `wlx_stage_evaluation.py` | 归一化、确定性指标与汇总入口 |
| `wlx_stage_grpo.py` | HarnessConfig 到 veRL AgentLoop/Tool/metadata 的桥接 |

Python 包使用 `wlx_` 前缀，因为 `-` 不能用于 Python import；外层目录和文档继续使用 `wlx-`。

## 3. 核心接口

```python
class EnvironmentAdapter(Protocol):
    async def start(self, request: EpisodeRequest) -> EnvironmentResult: ...
    async def execute(self, action: str) -> EnvironmentResult: ...
    async def close(self) -> None: ...

class PolicyAdapter(Protocol):
    async def generate(self, request: ModelRequest) -> AssistantTurn: ...

class EpisodeRunner:
    async def run(
        self,
        request: EpisodeRequest | Mapping,
        policy: PolicyAdapter,
        config: HarnessConfig,
        observers: Sequence[RunObserver] = (),
    ) -> Trajectory: ...
```

`EpisodeRequest` 的主要参数：

| 参数 | 含义 |
|---|---|
| `task_id` | ShopSimulator 任务 ID，必填 |
| `instruction` | 可选外部指令；环境指令优先 |
| `prompt` | 可选预置消息 |
| `attempt_index` | 同任务第几次采样，默认 `0` |
| `metadata` | 阶段私有元数据，不进入模型凭据 |

## 4. 配置

```python
HarnessConfig(
    max_steps=35,
    environment_base_url="http://127.0.0.1:5700",
    environment_timeout_s=60,
    required_environment_version="shopsimulator-environment-v2.1",
    required_reward_version="shopsimulator-reward-v3",
    max_guard_rejections=3,
    max_assistant_turns=None,
    parallel_tool_call_policy="truncate",  # 或 reject
    context=None,
    observation=None,
    validate_terminal_reward=True,
)
```

`ContextPolicy`：

| 参数 | 含义 |
|---|---|
| `window_tokens` | 模型总上下文窗口 |
| `generation_reserve_tokens` | 单回合生成预留，默认 `512` |
| `safety_margin_tokens` | 安全余量，默认 `512` |
| `input_budget_tokens` | 可选的更严格输入预算 |
| `compaction_enabled` | 超预算时是否按完整 tool group 压缩 |
| `preserve_recent_groups` | 至少保留的最近交互组，默认 `1` |

`ObservationPolicy`：

| 参数 | 含义 |
|---|---|
| `token_budget` | Observation 主预算 |
| `detail_token_budget` | 商品详情页预算，默认 `4096` |
| `generic_token_budget` | 一般信息页预算，默认 `768` |
| `search_top_k` | 搜索页最大保留商品数，默认 `20` |

显式启用 Context/Observation 后，Policy 必须实现 `validate_harness_config()`；旧客户端适配器会逐项校验并实际执行预算。配置不一致时在模型和环境启动前失败，禁止“参数写了但没生效”。

## 5. 关键安全约束

- 用户购物指令与页面 Observation 分开保存：指令送给模型，页面状态供 Guard 判断。
- Tool Schema 只取自 `SHOPPING_TOOL_REGISTRY`；GRPO 工具配置由它生成并校验指纹。
- 每个 assistant 回合最多执行一个工具；`truncate` 和 `reject` 必须显式选择。
- 只有正常终局且 Reward v3 校验通过，`final_reward` 才能成为学习信号。
- 无法验证的 Reward 设置 `sampling_invalid=true`；结构错误或基础设施失败将 `final_reward` 清零。
- `Trajectory.to_dict()` 是完整审计格式，可能含 raw result；默认 observer 事件不含 raw observation、终局原始结果、私有 reasoning 或 audit reward。
- SFT 永久排除 `data/evaluation/tasks.jsonl`，调用方不能通过省略参数绕过。
- ShopSimulator 当前没有 lease token；release 网络调用最多一次。结果不确定时记录错误并禁止复用该 Adapter，避免按旧 `env_idx` 误释放新租约。

## 6. Trajectory 与兼容格式

统一 Trajectory 主要字段：

```text
schema_version, trajectory_id, task_id, attempt_index,
status, termination_category, termination_reason,
messages, steps, blocked_tool_calls, context_compactions,
initial_result, terminal_result, final_reward,
reward_valid, sampling_invalid, infrastructure_invalid,
error, release_error, stage_metadata, created_at, finished_at
```

- `trajectory.to_dict()`：Core v1 审计 JSON。
- `trajectory.to_legacy_dict()`：现有 SFT/Evaluation 可直接消费的格式。
- `trajectory_to_legacy(mapping)`：落盘后把 Core JSON 显式转成 legacy；未知 WLX schema 会 fail-fast。

## 7. 各阶段调用

### SFT Collection

```python
sft = SFTStageAdapter()
decision = sft.prepare(trajectory)

summary = sft.build_artifacts(
    raw_path="outputs/sft-collection/raw.jsonl",
    output_dir="outputs/sft-collection",
    validation_ratio=0.1,
    seed=42,
)
```

阶段层继续复用现有 `collection/sft.py`。它只接受完整的 Reward v3 `gold_purchase`，清除私有 reasoning/终局 Reward 文本，并强制 held-out 隔离。

### Evaluation

```python
evaluation = EvaluationStageAdapter().prepare(trajectory)
normalized = evaluation.normalized_trajectory
metrics = evaluation.deterministic_metrics

summary = EvaluationStageAdapter().summarize(task_ids, trajectories)
```

默认归一化只保留 Actor 可见事件；raw observation 只能通过显式 audit 参数开启。

### GRPO / veRL

```python
grpo_harness = HarnessConfig(
    max_steps=35,
    max_assistant_turns=40,
    parallel_tool_call_policy="reject",
)
bridge = VerlHarnessBridge(grpo_harness)

agent_loop_config = bridge.to_agent_loop_config()
tool_config = bridge.to_verl_tool_config()
verl_overrides = bridge.to_verl_overrides(
    agent_loop_config_path="configs/agent_loop.yaml",
    tool_config_path="wlx-harness-core/wlx-tools.json",
)
```

Bridge 还提供：

- `assert_tool_config_compatible()`：训练前检查工具配置漂移；
- `make_agent_loop()` / `make_session()`：延迟解析运行时类，不主动运行；
- `extract_extra_fields()`：校验 `AgentLoopOutput.extra_fields["shopping"]`；
- `to_stage_metadata()`：写入统一 `Trajectory.stage_metadata["shopping"]`。

GRPO 固定要求 Reward v3、Guard 上限 `3`、并行工具策略 `reject` 和终局 Reward 校验。veRL 仍负责生成、并发、logprob 和参数更新。

## 8. 错误分类

| 分类 | 示例 | 是否可作为训练信号 |
|---|---|---:|
| `model` | 未知工具、参数 JSON 错误 | 否，但不伪装成基础设施故障 |
| `policy` | 上下文硬超限 | 否 |
| `protocol` | Observation/Reward/环境返回结构错误 | 否 |
| `infrastructure` | HTTP、超时、ShopSimulator 服务错误 | 否 |
| `release` | 环境释放失败或结果不确定 | 否 |
| `internal` | 未分类的 Harness 内部异常 | 否 |

## 9. 接入方式

首版保持旁路使用，不修改项目打包配置：

```bash
export PYTHONPATH="$PWD/src:$PWD/wlx-harness-core:${PYTHONPATH:-}"
```

先让新旧入口并行存在；固定输入的动作、终止、Reward、held-out 隔离和资源释放全部等价后，再决定是否迁移旧入口。
