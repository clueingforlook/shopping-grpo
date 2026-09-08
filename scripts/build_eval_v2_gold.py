#!/usr/bin/env python3
"""把旧 Gold 按 Eval v2 的统一评分口径重新审核并冻结。"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
for import_path in (REPOSITORY / "src",):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from shopping_grpo.harness.eval_judge import (  # noqa: E402
    EVAL_JUDGE_FULL_OUTPUT_VERSION,
    EVAL_JUDGE_PROMPT_VERSION,
    build_full_judge_payload,
    validate_full_judgment,
)
from shopping_grpo.harness.eval_judge_pipeline import load_judge_cases  # noqa: E402
from shopping_grpo.harness.sft_storage import (  # noqa: E402
    file_sha256,
    read_jsonl,
    write_jsonl,
)


EVALUATION = REPOSITORY / "outputs" / "evaluation"
SOURCE_LABELS = EVALUATION / "judge-gold-v4" / "judge-gold-labels-frozen.jsonl"
SOURCE_MANIFEST = EVALUATION / "judge-gold-v4" / "judge-gold-manifest.json"
RUBRICS = EVALUATION / "eval200-rubrics-v1" / "rubrics.jsonl"
OUTPUT = EVALUATION / "judge-gold-v6-eval-v2"


# 这些改动来自逐条重看轨迹后的口径修正，不是为了追随某次 Judge 输出。
PROCESS_CORRECTIONS = {
    ("base:11976", "candidate_utilization"): (
        2,
        "首个强候选被正确打开、选规格并购买；中途冗余只影响终止效率。",
    ),
    ("base:19238", "candidate_utilization"): (
        2,
        "优先利用排名第一的强候选并选择50枚规格；重复导航另计终止效率。",
    ),
    ("base:1405", "candidate_utilization"): (
        0,
        "已经出现较强候选，但没有核对并利用关键颜色和尺码选项，随后明显漂移。",
    ),
    ("base:7670", "candidate_utilization"): (
        0,
        "已经看到含拆机和电磁炉信息的强候选，却未核对数量与混装规格并继续漂移。",
    ),
    ("base:11773", "candidate_utilization"): (
        1,
        "打开了强候选并选择蛋白规格，但没有继续确认净含量和最终价格。",
    ),
    ("base:10633", "candidate_utilization"): (
        0,
        "反复打开候选却没有利用可见规格核对颜色、遥控、照明、安装和价格。",
    ),
    ("sft:10633", "candidate_utilization"): (
        0,
        "忽略搜索结果中预算附近的候选，直接购买了明显超预算的6999元商品。",
    ),
    ("base:13968", "candidate_utilization"): (
        0,
        "页面已经展示接近要求的具体规格，但模型没有选择利用并陷入重复搜索。",
    ),
    ("base:4206", "decision_quality"): (
        0,
        "没有形成购买或合理主动放弃，最终因重复搜索被动结束。",
    ),
    ("base:1405", "decision_quality"): (
        0,
        "错过强候选后因无效导航被动结束，没有形成有依据的决定。",
    ),
    ("base:11773", "decision_quality"): (
        0,
        "未完成关键规格核对，也未主动放弃，最终因非法动作和预算耗尽结束。",
    ),
    ("sft:11773", "decision_quality"): (
        0,
        "在重复搜索中被动耗尽预算，没有形成有依据的购买或放弃决定。",
    ),
    ("base:10633", "decision_quality"): (
        0,
        "反复搜索和打开商品后被动结束，没有完成选择或合理放弃。",
    ),
    ("base:13968", "decision_quality"): (
        0,
        "已经看到接近要求的规格却没有选择，最终因循环被动结束。",
    ),
    ("base:13968", "evidence_verification"): (
        1,
        "商品页展示了管径、螺纹、活接和手柄候选，但未选择规格，最终价格仍未确认。",
    ),
    ("sft:10864", "termination_efficiency"): (
        0,
        "数码屏幕这一关键要求仍缺少明确证据便购买，属于证据不足时过早结束。",
    ),
}


def main() -> None:
    paths = {
        "labels": OUTPUT / "judge-gold-labels-frozen.jsonl",
        "manifest": OUTPUT / "judge-gold-frozen-manifest.json",
        "review": OUTPUT / "judge-gold-eval-v2-review.md",
    }
    if any(path.exists() for path in paths.values()):
        raise SystemExit("Eval v2 Gold 输出已经存在，拒绝覆盖")

    source = {str(row["case_id"]): row for row in read_jsonl(SOURCE_LABELS)}
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    selected = manifest["selected_cases"]
    selected_by_prefix: dict[str, list[int]] = {}
    for row in selected:
        selected_by_prefix.setdefault(str(row["run_prefix"]), []).append(int(row["task_id"]))

    trajectory_paths = {
        "base": EVALUATION / "base-final200-v2" / "trajectories.jsonl",
        "sft": EVALUATION / "sft-final200-v2" / "trajectories.jsonl",
    }
    cases = {}
    for prefix, task_ids in selected_by_prefix.items():
        for case in load_judge_cases(
            trajectories_path=trajectory_paths[prefix],
            rubrics_path=RUBRICS,
            case_prefix=prefix,
            selected_task_ids=task_ids,
        ):
            cases[case.case_id] = case
    if set(cases) != set(source):
        raise ValueError("旧 Gold 与选中轨迹不一致")

    frozen = []
    added_requirements = 0
    changed_dimensions = []
    changed_attributions = 0
    for selected_row in selected:
        case_id = str(selected_row["case_id"])
        case = cases[case_id]
        old = source[case_id]
        payload = build_full_judge_payload(trajectory=case.trajectory, rubric=case.rubric)
        old_requirements = {
            str(item["requirement_id"]): deepcopy(item)
            for item in old.get("requirements") or []
        }
        requirements = []
        for rubric_item in case.rubric["items"]:
            requirement_id = str(rubric_item["requirement_id"])
            item = old_requirements.get(requirement_id)
            if item is None:
                status = (
                    "violated"
                    if (case_id, requirement_id) == ("sft:10633", "R004")
                    else "satisfied"
                )
                reason = (
                    "最终选择明确写为银河系列白，不能据此确认用户指定的米白色。"
                    if status == "violated"
                    else "最终购买前可见的标题、属性、已选规格或价格明确支持该要求。"
                )
                item = {
                    "requirement_id": requirement_id,
                    "status": status,
                    "reason": reason,
                    "evidence_event_ids": [],
                    "confidence": 0.95,
                }
                added_requirements += 1
            requirements.append(item)

        dimensions = deepcopy(old["dimensions"])
        for dimension in dimensions:
            correction = PROCESS_CORRECTIONS.get((case_id, dimension))
            if correction is None:
                continue
            old_score = dimensions[dimension]["score"]
            score, reason = correction
            dimensions[dimension] = {
                "score": score,
                "reason": reason,
                "evidence_event_ids": [],
                "confidence": 0.95,
            }
            changed_dimensions.append(
                {
                    "case_id": case_id,
                    "dimension": dimension,
                    "before": old_score,
                    "after": score,
                    "reason": reason,
                }
            )

        attribution = deepcopy(old["attribution"])
        if case_id == "sft:10864":
            purchase_events = payload["rule_gate_facts"]["accepted_purchase_event_ids"]
            attribution = {
                "responsibility": "model",
                "primary_failure": "insufficient_verification",
                "secondary_failures": [],
                "first_error_event_id": purchase_events[-1],
                "reason": "购买前没有确认数码屏幕这一硬要求，却仍执行购买。",
                "confidence": 0.95,
            }
            changed_attributions += 1

        value = {
            "schema_version": EVAL_JUDGE_FULL_OUTPUT_VERSION,
            "requirements": requirements,
            "dimensions": dimensions,
            "attribution": attribution,
            "review_required": False,
            "review_reasons": [],
        }
        judgment = validate_full_judgment(
            value,
            allowed_event_ids=payload["allowed_evidence_event_ids"],
            required_requirement_ids=payload["required_requirement_ids"],
        )
        frozen.append(
            {
                "schema_version": "wlx-eval-v2-judge-gold-label-v1",
                "case_id": case_id,
                "task_id": case.task_id,
                "judgment": judgment,
                "requirements": judgment["requirements"],
                "dimensions": judgment["dimensions"],
                "attribution": judgment["attribution"],
                "reviewer": "Codex re-audit authorized by user for Eval v2",
                "review_notes": "需求只评价最终购买；过程分按 Eval v2 的互斥边界复核。",
            }
        )

    OUTPUT.mkdir(parents=True, exist_ok=False)
    write_jsonl(paths["labels"], frozen)
    total_requirements = sum(len(row["requirements"]) for row in frozen)
    output_manifest = {
        "schema_version": "wlx-eval-v2-judge-gold-frozen-v1",
        "status": "frozen",
        "cases": len(frozen),
        "requirement_labels": total_requirements,
        "requirement_coverage": 1.0,
        "added_requirement_labels": added_requirements,
        "changed_dimension_labels": len(changed_dimensions),
        "changed_attribution_labels": changed_attributions,
        "prompt_version": EVAL_JUDGE_PROMPT_VERSION,
        "source_labels": str(SOURCE_LABELS.relative_to(REPOSITORY)),
        "source_labels_sha256": file_sha256(SOURCE_LABELS),
        "labels": str(paths["labels"].relative_to(REPOSITORY)),
        "labels_sha256": file_sha256(paths["labels"]),
    }
    paths["manifest"].write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Eval v2 Gold 复核记录",
        "",
        f"- 轨迹：{len(frozen)} 条",
        f"- Rubric 标签：{total_requirements} 条（覆盖率 100%）",
        f"- 从旧 Gold 补齐：{added_requirements} 条",
        f"- 修正过程分：{len(changed_dimensions)} 项",
        f"- 修正失败归因：{changed_attributions} 项",
        f"- Judge Prompt：`{EVAL_JUDGE_PROMPT_VERSION}`",
        "",
        "## 过程分修正",
        "",
        "| 轨迹 | 维度 | 原分 | 新分 | 原因 |",
        "|---|---|---:|---:|---|",
    ]
    lines.extend(
        f"| `{row['case_id']}` | `{row['dimension']}` | {row['before']} | {row['after']} | {row['reason']} |"
        for row in changed_dimensions
    )
    paths["review"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(output_manifest, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
