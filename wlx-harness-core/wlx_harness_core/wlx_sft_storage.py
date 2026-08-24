"""负责 WLX SFT 原始轨迹的追加保存、读取和断点续采。"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Mapping

from wlx_harness_core.wlx_sft_contracts import AttemptEnvelope, SftTask


_PROCESS_WRITE_LOCK = threading.Lock()


class JsonlFormatError(ValueError):
    """表示 JSONL 某一行已经损坏，继续构建数据可能得到错误结果。"""

    pass


class RawTrajectoryStore:
    """把每次尝试追加写入 raw.jsonl，并提供简单可靠的续采信息。"""

    def __init__(self, path: str | Path, *, durable: bool = True) -> None:
        """保存原始文件位置；durable 开启时每写一行都会同步到磁盘。"""

        self.path = Path(path)
        self.durable = bool(durable)

    def append(self, envelope: AttemptEnvelope) -> None:
        """把一条完整尝试作为单独一行追加，绝不覆盖前面已经采到的数据。"""

        if not isinstance(envelope, AttemptEnvelope):
            raise TypeError("envelope 必须是 AttemptEnvelope")
        append_jsonl_row(self.path, envelope.to_dict(), durable=self.durable)

    def rows(self) -> Iterator[dict[str, Any]]:
        """按原始顺序逐行读取尝试，文件还不存在时返回空迭代。"""

        yield from read_jsonl(self.path)

    def completed_valid_attempts(self) -> set[tuple[int, int]]:
        """只把有效尝试算作完成；技术故障不会吃掉三次有效采样名额。"""

        completed: set[tuple[int, int]] = set()
        for row in self.rows():
            if row.get("attempt_valid") is not True:
                continue
            completed.add((int(row["task_id"]), int(row.get("attempt_index", 0))))
        return completed

    def technical_retry_count(self, task_id: int, attempt_index: int) -> int:
        """统计某个逻辑尝试已经发生多少次无效技术运行，供断点续采继续编号。"""

        count = 0
        for row in self.rows():
            if int(row.get("task_id", -1)) != int(task_id):
                continue
            if int(row.get("attempt_index", -1)) != int(attempt_index):
                continue
            if row.get("attempt_valid") is not True:
                count += 1
        return count

    def decisions_for_task(self, task_id: int) -> list[dict[str, Any]]:
        """读取一个任务已有的全部流水线判断，方便正式模式决定是否继续。"""

        return [
            row
            for row in self.rows()
            if int(row.get("task_id", -1)) == int(task_id)
        ]


def append_jsonl_row(
    path: str | Path,
    row: Mapping[str, Any],
    *,
    durable: bool = True,
) -> None:
    """在进程锁和文件锁保护下追加一行，避免并发任务把 JSON 写到一起。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    with _PROCESS_WRITE_LOCK:
        with path.open("ab") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                handle.write(encoded)
                handle.flush()
                if durable:
                    os.fsync(handle.fileno())
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """逐行读取 JSONL；一旦发现坏行就带着行号报错，避免悄悄丢数据。"""

    path = Path(path)
    if not path.exists():
        return
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JsonlFormatError(f"{path} 第 {line_number} 行不是完整 JSON") from exc
            if not isinstance(value, dict):
                raise JsonlFormatError(f"{path} 第 {line_number} 行必须是 JSON 对象")
            yield value


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> int:
    """重新生成一个派生 JSONL 文件，并返回实际写入的行数。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.wlx_tmp")
    count = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    return count


def load_sft_tasks(path: str | Path, *, expected_split: str = "train") -> list[SftTask]:
    """读取任务表、拒绝重复编号，并只构造不含隐藏答案的公开任务对象。"""

    tasks: list[SftTask] = []
    seen: set[int] = set()
    for row in read_jsonl(path):
        task = SftTask.from_mapping(row)
        if task.task_id in seen:
            raise ValueError(f"任务表中 task_id={task.task_id} 重复")
        if expected_split and task.official_split != expected_split:
            raise ValueError(
                f"task_id={task.task_id} 的 official_split={task.official_split!r}，"
                f"预期为 {expected_split!r}"
            )
        seen.add(task.task_id)
        tasks.append(task)
    return tasks


def load_held_out_task_ids(path: str | Path) -> set[int]:
    """读取官方评测任务编号；文件缺失或为空时直接失败，防止误把评测题拿去训练。"""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到官方评测任务表：{path}")
    task_ids = {int(row["task_id"]) for row in read_jsonl(path)}
    if not task_ids:
        raise ValueError(f"官方评测任务表为空：{path}")
    return task_ids


def assert_no_held_out_tasks(tasks: Iterable[SftTask], held_out_ids: set[int]) -> None:
    """在采样前检查训练任务与官方评测任务没有 task_id 重叠。"""

    leaked = sorted({task.task_id for task in tasks}.intersection(held_out_ids))
    if leaked:
        preview = ", ".join(str(item) for item in leaked[:10])
        raise ValueError(f"训练任务与评测任务重叠，共 {len(leaked)} 个：{preview}")


def file_sha256(path: str | Path) -> str:
    """分块计算文件 SHA-256，供冻结数据集时核对内容是否发生变化。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def jsonl_line_count(path: str | Path) -> int:
    """统计 JSONL 中非空行数量，用于 metadata 的文件清单。"""

    with Path(path).open("rb") as handle:
        return sum(1 for line in handle if line.strip())


__all__ = [
    "JsonlFormatError",
    "RawTrajectoryStore",
    "append_jsonl_row",
    "assert_no_held_out_tasks",
    "file_sha256",
    "jsonl_line_count",
    "load_held_out_task_ids",
    "load_sft_tasks",
    "read_jsonl",
    "write_jsonl",
]
