#!/usr/bin/env python3
"""Parse a WLX GRPO log and render dependency-free SVG training curves."""

from __future__ import annotations

import argparse
import csv
from html import escape
import json
import math
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
TRAIN_FIELDS = {
    "orm": "wlx_reward/orm_mean",
    "gold": "wlx_reward/gold_rate",
    "kl": "actor/ppo_kl",
    "grad_norm": "actor/grad_norm",
    "effective_group_ratio": "group/effective_ratio",
    "response_length": "response_length/mean",
    "step_seconds": "timing_s/step",
    "memory_gib": "actor/perf/max_memory_allocated_gb",
    "learning_rate": "actor/lr",
    "optimizer_updated": "training/optimizer_updated",
}
VAL_FIELDS = {
    "reward": "val-core/shopsimulator/reward/mean@1",
    "gold": "val-aux/shopsimulator/wlx_gold/mean@1",
    "orm": "val-aux/shopsimulator/wlx_orm/mean@1",
    "turns": "val-aux/num_turns/mean",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rolling-window", type=int, default=10)
    parser.add_argument("--max-step", type=int)
    return parser


def _number(text: str, key: str) -> float | None:
    match = re.search(
        re.escape(key) + r":(?:np\.[^( ]+\()?" + NUMBER,
        text,
    )
    if match is None:
        return None
    value = float(match.group(1))
    return value if math.isfinite(value) else None


def parse_log(path: Path) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Keep the latest occurrence of a step, which handles checkpoint resumes."""

    training: dict[int, dict[str, float]] = {}
    validation: dict[int, dict[str, float]] = {}
    with path.open(encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = ANSI_RE.sub("", raw_line)
            global_step = _number(line, "training/global_step")
            if global_step is not None:
                step = int(global_step)
                row: dict[str, float] = {"step": float(step)}
                for output_name, log_name in TRAIN_FIELDS.items():
                    value = _number(line, log_name)
                    if value is not None:
                        row[output_name] = value
                if len(row) > 1:
                    training[step] = row

            if "val-core/shopsimulator/reward/mean@1" not in line:
                continue
            step_match = re.search(r"(?:^|\s)step:(\d+)(?:\s|$)", line)
            if step_match is None:
                continue
            step = int(step_match.group(1))
            row = {"step": float(step)}
            for output_name, log_name in VAL_FIELDS.items():
                value = _number(line, log_name)
                if value is not None:
                    row[output_name] = value
            if len(row) > 1:
                validation[step] = row
    return (
        [training[step] for step in sorted(training)],
        [validation[step] for step in sorted(validation)],
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, float]]) -> None:
    fields = ["step"]
    fields.extend(
        sorted({key for row in rows for key in row if key != "step"})
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _rolling(
    rows: Sequence[Mapping[str, float]], key: str, window: int
) -> list[tuple[float, float]]:
    result = []
    history: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        history.append(float(value))
        result.append((float(row["step"]), sum(history[-window:]) / len(history[-window:])))
    return result


def _points(
    rows: Sequence[Mapping[str, float]], key: str
) -> list[tuple[float, float]]:
    return [
        (float(row["step"]), float(row[key]))
        for row in rows
        if row.get(key) is not None
    ]


def _bounds(
    series: Sequence[Sequence[tuple[float, float]]],
    *,
    fixed: tuple[float, float] | None = None,
    include_zero: bool = False,
) -> tuple[float, float]:
    if fixed is not None:
        return fixed
    values = [value for points in series for _, value in points]
    if not values:
        return 0.0, 1.0
    lower, upper = min(values), max(values)
    if include_zero:
        lower = min(0.0, lower)
    if lower == upper:
        padding = max(abs(lower) * 0.1, 0.1)
    else:
        padding = (upper - lower) * 0.1
    return lower - padding, upper + padding


def _polyline(
    points: Sequence[tuple[float, float]],
    *,
    x0: float,
    y0: float,
    width: float,
    height: float,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
) -> str:
    xmin, xmax = x_bounds
    ymin, ymax = y_bounds
    xspan = max(xmax - xmin, 1.0)
    yspan = max(ymax - ymin, 1e-12)
    return " ".join(
        f"{x0 + (x - xmin) / xspan * width:.2f},{y0 + height - (y - ymin) / yspan * height:.2f}"
        for x, y in points
    )


def _panel(
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    raw: Sequence[tuple[float, float]],
    smooth: Sequence[tuple[float, float]],
    validation: Sequence[tuple[float, float]] = (),
    y_fixed: tuple[float, float] | None = None,
    include_zero: bool = False,
) -> list[str]:
    margin_left, margin_right, margin_top, margin_bottom = 58, 18, 38, 42
    plot_x = x + margin_left
    plot_y = y + margin_top
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    all_series = [raw, smooth, validation]
    all_x = [value for points in all_series for value, _ in points]
    x_bounds = (min(all_x), max(all_x)) if all_x else (0.0, 1.0)
    y_bounds = _bounds(all_series, fixed=y_fixed, include_zero=include_zero)
    ymin, ymax = y_bounds
    elements = [
        f'<g><rect x="{x}" y="{y}" width="{width}" height="{height}" rx="8" fill="#ffffff" stroke="#d8dee9"/>',
        f'<text x="{x + 16}" y="{y + 25}" class="panel-title">{escape(title)}</text>',
    ]
    for index in range(5):
        fraction = index / 4
        grid_y = plot_y + plot_h * fraction
        tick_value = ymax - (ymax - ymin) * fraction
        elements.append(
            f'<line x1="{plot_x}" y1="{grid_y:.2f}" x2="{plot_x + plot_w}" y2="{grid_y:.2f}" class="grid"/>'
        )
        elements.append(
            f'<text x="{plot_x - 7}" y="{grid_y + 4:.2f}" text-anchor="end" class="tick">{tick_value:.3g}</text>'
        )
    xmin, xmax = x_bounds
    for index in range(5):
        fraction = index / 4
        grid_x = plot_x + plot_w * fraction
        tick_value = xmin + (xmax - xmin) * fraction
        elements.append(
            f'<text x="{grid_x:.2f}" y="{plot_y + plot_h + 20}" text-anchor="middle" class="tick">{tick_value:.0f}</text>'
        )
    elements.append(
        f'<text x="{plot_x + plot_w / 2:.2f}" y="{y + height - 8}" text-anchor="middle" class="axis">step</text>'
    )
    if raw:
        points = _polyline(
            raw,
            x0=plot_x,
            y0=plot_y,
            width=plot_w,
            height=plot_h,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
        elements.append(f'<polyline points="{points}" class="raw"/>')
    if smooth:
        points = _polyline(
            smooth,
            x0=plot_x,
            y0=plot_y,
            width=plot_w,
            height=plot_h,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
        elements.append(f'<polyline points="{points}" class="smooth"/>')
    if validation:
        xmin, xmax = x_bounds
        xspan = max(xmax - xmin, 1.0)
        yspan = max(ymax - ymin, 1e-12)
        for point_x, point_y in validation:
            cx = plot_x + (point_x - xmin) / xspan * plot_w
            cy = plot_y + plot_h - (point_y - ymin) / yspan * plot_h
            elements.append(
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="4" class="validation"><title>eval step {point_x:.0f}: {point_y:.5g}</title></circle>'
            )
    elements.append("</g>")
    return elements


def render_svg(
    training: Sequence[Mapping[str, float]],
    validation: Sequence[Mapping[str, float]],
    *,
    rolling_window: int,
    source_name: str,
) -> str:
    width, height = 1400, 1060
    panel_w, panel_h = 660, 285
    panels = (
        ("Train ORM / Eval ORM", "orm", "orm", None, False),
        ("Train Gold / Eval Gold", "gold", "gold", (0.0, 1.0), False),
        ("Approximate KL", "kl", None, None, True),
        ("Gradient norm", "grad_norm", None, None, True),
        ("Effective group ratio", "effective_group_ratio", None, (0.0, 1.0), False),
        ("Mean response length (tokens)", "response_length", None, None, True),
    )
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:DejaVu Sans,Arial,sans-serif;fill:#263238}.title{font-size:24px;font-weight:700}.subtitle{font-size:13px;fill:#607d8b}.panel-title{font-size:16px;font-weight:700}.tick{font-size:11px;fill:#607d8b}.axis{font-size:12px;fill:#455a64}.grid{stroke:#e8edf2;stroke-width:1}.raw{fill:none;stroke:#90caf9;stroke-width:1;opacity:.65}.smooth{fill:none;stroke:#1565c0;stroke-width:2.5}.validation{fill:#e53935;stroke:#fff;stroke-width:1.5}</style>",
        '<rect width="100%" height="100%" fill="#f5f7fa"/>',
        '<text x="40" y="38" class="title">WLX GRPO training curves</text>',
        f'<text x="40" y="61" class="subtitle">source: {escape(source_name)} · blue: raw/rolling mean ({rolling_window}) · red: fixed validation</text>',
    ]
    for index, (title, train_key, val_key, fixed, include_zero) in enumerate(panels):
        column, row = index % 2, index // 2
        elements.extend(
            _panel(
                x=35 + column * 690,
                y=85 + row * 315,
                width=panel_w,
                height=panel_h,
                title=title,
                raw=_points(training, train_key),
                smooth=_rolling(training, train_key, rolling_window),
                validation=_points(validation, val_key) if val_key else (),
                y_fixed=fixed,
                include_zero=include_zero,
            )
        )
    elements.append("</svg>\n")
    return "\n".join(elements)


def _last(rows: Sequence[Mapping[str, float]]) -> Mapping[str, float]:
    return rows[-1] if rows else {}


def main() -> None:
    args = build_parser().parse_args()
    if args.rolling_window < 1:
        raise SystemExit("--rolling-window must be positive")
    if args.max_step is not None and args.max_step < 1:
        raise SystemExit("--max-step must be positive")
    if not args.log.is_file():
        raise SystemExit(f"log does not exist: {args.log}")
    output_dir = args.output_dir or args.log.parent / "wlx-training-plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    training, validation = parse_log(args.log)
    if args.max_step is not None:
        training = [row for row in training if row["step"] <= args.max_step]
        validation = [row for row in validation if row["step"] <= args.max_step]
    if not training:
        raise SystemExit("no GRPO training step metrics found")
    suffix = f"-step{args.max_step}" if args.max_step is not None else ""
    metrics_path = output_dir / f"wlx-training-metrics{suffix}.csv"
    validation_path = output_dir / f"wlx-validation-metrics{suffix}.csv"
    curves_path = output_dir / f"wlx-training-curves{suffix}.svg"
    summary_path = output_dir / f"wlx-training-curve-summary{suffix}.json"
    _write_csv(metrics_path, training)
    _write_csv(validation_path, validation)
    curves_path.write_text(
        render_svg(
            training,
            validation,
            rolling_window=args.rolling_window,
            source_name=(
                f"{args.log} through step {args.max_step}"
                if args.max_step is not None
                else str(args.log)
            ),
        ),
        encoding="utf-8",
    )
    summary = {
        "source_log": str(args.log),
        "training_steps": len(training),
        "validation_points": len(validation),
        "last_training": dict(_last(training)),
        "last_validation": dict(_last(validation)),
        "rolling_window": args.rolling_window,
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(curves_path)
    print(metrics_path)
    print(validation_path)


if __name__ == "__main__":
    main()
