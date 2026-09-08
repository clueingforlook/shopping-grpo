# Package layout

The package follows the tutorial pipeline:

```text
environment/       connect to ShopSimulator and enforce the action contract
collection/        reference SFT collection workflow
harness/           shared episode runner, SFT data pipeline and Eval v2 adapters
training/sft/      build assistant-only SFT examples
training/grpo/     connect the shopping AgentLoop and reward to veRL
evaluation/        hard checks, Rubric curation, trajectory Judge and aggregation
cli.py             small installed-package commands
smoke.py           CPU-only public smoke path
```

User-facing commands remain in the repository-level `scripts/` directory.
Those launchers call these modules; they are not a second implementation.

The extended training path uses `training/grpo/reward_v4.py` and
`training/grpo/step_grpo.py`; deterministic final-state scoring lives in
`evaluation/eval_v3.py`. The original training path is retained for comparison.

The harness is included in the installed `shopping-grpo` package. Repository
datasets are kept outside the wheel; when using an installed package away from
the checkout, pass an explicit `held_out_tasks_path` to the SFT stage adapter.
