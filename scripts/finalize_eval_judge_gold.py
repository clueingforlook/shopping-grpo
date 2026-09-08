#!/usr/bin/env python3
"""校验人工确认标签并冻结 Judge Gold Set。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
for import_path in (REPOSITORY / "src",):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from shopping_grpo.harness.eval_judge import (  # noqa: E402
    EVAL_JUDGE_FULL_OUTPUT_VERSION,
    build_full_judge_payload,
    validate_full_judgment,
)
from shopping_grpo.harness.eval_judge_pipeline import load_judge_cases  # noqa: E402
from shopping_grpo.harness.sft_storage import (  # noqa: E402
    file_sha256,
    read_jsonl,
    write_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-manifest", type=Path, required=True)
    parser.add_argument("--confirmed-labels", type=Path, required=True)
    parser.add_argument("--output-labels", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        result = finalize(args)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


def finalize(args: argparse.Namespace) -> dict:
    if args.output_labels.exists() or args.output_manifest.exists():
        raise FileExistsError("refusing to overwrite frozen Gold outputs")
    manifest = json.loads(args.gold_manifest.read_text(encoding="utf-8"))
    sources = manifest["sources"]
    selected = manifest["selected_cases"]
    selected_by_prefix: dict[str, list[int]] = {}
    for row in selected:
        selected_by_prefix.setdefault(row["run_prefix"], []).append(int(row["task_id"]))
    source_by_prefix = {
        "base": sources["baseline"]["path"],
        "sft": sources["candidate"]["path"],
    }
    cases = {}
    for prefix, task_ids in selected_by_prefix.items():
        for case in load_judge_cases(
            trajectories_path=source_by_prefix[prefix],
            rubrics_path=sources["rubrics"]["path"],
            case_prefix=prefix,
            selected_task_ids=task_ids,
        ):
            cases[case.case_id] = case
    labels = {str(row["case_id"]): row for row in read_jsonl(args.confirmed_labels)}
    if set(labels) != set(cases):
        raise ValueError("confirmed labels do not exactly cover selected Gold cases")
    frozen = []
    for case_id in [row["case_id"] for row in selected]:
        label = labels[case_id]
        if label.get("confirmation_status") != "confirmed" or not str(
            label.get("reviewer") or ""
        ).strip():
            raise ValueError(f"case {case_id} has not been human-confirmed")
        encoded_label = json.dumps(label, ensure_ascii=False).casefold()
        if "draft" in encoded_label or "placeholder" in encoded_label or any(
            float(item.get("confidence", 0.0)) <= 0
            for item in [
                *(label.get("requirements") or []),
                *((label.get("dimensions") or {}).values()),
                label.get("attribution") or {},
            ]
            if isinstance(item, dict)
        ):
            raise ValueError(
                f"case {case_id} still contains draft reasons or zero confidence"
            )
        case = cases[case_id]
        payload = build_full_judge_payload(
            trajectory=case.trajectory,
            rubric=case.rubric,
        )
        value = {
            "schema_version": EVAL_JUDGE_FULL_OUTPUT_VERSION,
            "requirements": label.get("requirements") or [],
            "dimensions": label.get("dimensions"),
            "attribution": label.get("attribution"),
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
                "schema_version": "wlx-eval-judge-gold-label-v1",
                "case_id": case_id,
                "task_id": case.task_id,
                "judgment": judgment,
                "requirements": judgment["requirements"],
                "dimensions": judgment["dimensions"],
                "attribution": judgment["attribution"],
                "reviewer": label["reviewer"],
                "review_notes": str(label.get("review_notes") or ""),
            }
        )
    write_jsonl(args.output_labels, frozen)
    frozen_manifest = {
        "schema_version": "wlx-eval-judge-gold-frozen-v1",
        "status": "frozen",
        "cases": len(frozen),
        "source_manifest": str(args.gold_manifest),
        "source_manifest_sha256": file_sha256(args.gold_manifest),
        "labels": str(args.output_labels),
        "labels_sha256": file_sha256(args.output_labels),
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        json.dumps(frozen_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return frozen_manifest


if __name__ == "__main__":
    main()
