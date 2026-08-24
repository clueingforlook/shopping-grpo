#!/usr/bin/env python3
"""为 GitHub Release 生成去除机器路径的 WLX Raw/SFT 附件。"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import wlx_sft_organize_data as organizer


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPOSITORY / "outputs" / "wlx-data-v1"
DEFAULT_OUTPUT = REPOSITORY / "outputs" / "wlx-github-release-v1"
RELEASE_BUNDLE_VERSION = "wlx-github-release-bundle-v1"
RAW_ASSET = "wlx-raw-v1.tar.gz"
SFT_ASSET = "wlx-sft-v1.tar.gz"
RELEASE_MANIFEST = "wlx-release-manifest.json"
CHECKSUMS = "wlx-SHA256SUMS.txt"
TEXT_SUFFIXES = {".json", ".jsonl", ".log", ".md", ".txt"}
SECRET_PATTERNS = (
    re.compile(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"(?i)bearer\s+[A-Za-z0-9._-]{20,}"),
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_release_bundle(
        source_dir=_absolute(args.source_dir),
        output_dir=_absolute(args.output_dir),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_release_bundle(*, source_dir: Path, output_dir: Path) -> dict[str, Any]:
    """复制、净化、重算 manifest，再生成两个确定性压缩附件。"""

    if output_dir.exists():
        raise FileExistsError(f"目标已存在，不会覆盖：{output_dir}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"找不到整理后的数据：{source_dir}")
    organizer.verify_archive(source_dir, verify_sources=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    staging_parent = Path(
        tempfile.mkdtemp(prefix="wlx-release-staging-", dir=output_dir.parent)
    )
    temporary_output = Path(
        tempfile.mkdtemp(prefix="wlx-release-output-", dir=output_dir.parent)
    )
    staging = staging_parent / "wlx-data-v1"
    try:
        shutil.copytree(source_dir, staging)
        replacement_counts = _sanitize_machine_paths(staging)
        _refresh_manifests(staging, replacement_counts)
        verification = organizer.verify_archive(
            staging,
            verify_sources=False,
            logical_output_dir=source_dir,
        )
        _assert_release_safe(staging)

        raw_path = temporary_output / RAW_ASSET
        sft_path = temporary_output / SFT_ASSET
        _write_asset(staging, "wlx-raw", raw_path)
        _write_asset(staging, "wlx-sft", sft_path)
        assets = [_asset_report(raw_path), _asset_report(sft_path)]
        release_manifest = {
            "release_bundle_version": RELEASE_BUNDLE_VERSION,
            "dataset_archive_version": verification["archive_version"],
            "description": "GitHub Release 附件；两个压缩包解压后合并为 wlx-data-v1",
            "sanitization": {
                "operation": "只替换机器绝对路径，轨迹结果和错误原因保持不变",
                "replacement_tokens": ["<repo>", "<home>"],
                "replacements": replacement_counts,
            },
            "verification": verification,
            "source_archive_manifest_sha256": organizer.file_sha256(
                source_dir / "wlx-manifest.json"
            ),
            "assets": assets,
        }
        manifest_path = temporary_output / RELEASE_MANIFEST
        _write_json(manifest_path, release_manifest)
        checksum_path = temporary_output / CHECKSUMS
        checksum_rows = assets + [_asset_report(manifest_path)]
        checksum_path.write_text(
            "".join(f'{row["sha256"]}  {row["name"]}\n' for row in checksum_rows),
            encoding="utf-8",
        )
        _verify_assets(temporary_output, assets, manifest_path, checksum_path)
        temporary_output.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary_output, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)

    return {
        "release_bundle_version": RELEASE_BUNDLE_VERSION,
        "output_dir": _portable(output_dir),
        "assets": assets,
        "manifest": _portable(output_dir / RELEASE_MANIFEST),
        "checksums": _portable(output_dir / CHECKSUMS),
        "machine_path_replacements": sum(replacement_counts.values()),
    }


def _sanitize_machine_paths(root: Path) -> dict[str, int]:
    repository_prefix = str(REPOSITORY).encode("utf-8")
    home_prefix = str(REPOSITORY.parent).encode("utf-8")
    counts = {"repo_path": 0, "home_path": 0}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        value = path.read_bytes()
        repo_count = value.count(repository_prefix)
        if repo_count:
            value = value.replace(repository_prefix, b"<repo>")
            counts["repo_path"] += repo_count
        home_count = value.count(home_prefix)
        if home_count:
            value = value.replace(home_prefix, b"<home>")
            counts["home_path"] += home_count
        if repo_count or home_count:
            path.write_bytes(value)
    return counts


def _refresh_manifests(root: Path, replacements: dict[str, int]) -> None:
    logical_root = Path("outputs/wlx-data-v1")
    groups = (
        (root / "wlx-raw" / "wlx-calibration-200", logical_root / "wlx-raw" / "wlx-calibration-200"),
        (root / "wlx-raw" / "wlx-formal", logical_root / "wlx-raw" / "wlx-formal"),
        (root / "wlx-raw", logical_root / "wlx-raw"),
        (root / "wlx-sft", logical_root / "wlx-sft"),
    )
    release_note = {
        "version": RELEASE_BUNDLE_VERSION,
        "machine_path_replacements": dict(replacements),
    }
    for actual_root, logical_group in groups:
        manifest_path = actual_root / "wlx-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["release_sanitization"] = release_note
        manifest["files"] = organizer._tree_reports(
            actual_root=actual_root,
            logical_root=logical_group,
            exclude_names={"wlx-manifest.json"},
        )
        organizer._write_json(manifest_path, manifest)

    root_manifest_path = root / "wlx-manifest.json"
    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    root_manifest["release_sanitization"] = release_note
    raw_report = organizer._file_report(
        root / "wlx-raw" / "wlx-manifest.json",
        logical_root / "wlx-raw" / "wlx-manifest.json",
    )
    raw_report["archive_relative_path"] = "wlx-raw/wlx-manifest.json"
    sft_report = organizer._file_report(
        root / "wlx-sft" / "wlx-manifest.json",
        logical_root / "wlx-sft" / "wlx-manifest.json",
    )
    sft_report["archive_relative_path"] = "wlx-sft/wlx-manifest.json"
    root_manifest["manifests"] = [raw_report, sft_report]
    organizer._write_json(root_manifest_path, root_manifest)


def _assert_release_safe(root: Path) -> None:
    forbidden_paths = (str(REPOSITORY).encode(), str(REPOSITORY.parent).encode(), b"/home/")
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        value = path.read_bytes()
        if any(marker in value for marker in forbidden_paths):
            raise ValueError(f"发布副本仍含机器绝对路径：{path.relative_to(root)}")
        if any(pattern.search(value) for pattern in SECRET_PATTERNS):
            raise ValueError(f"发布副本疑似含凭据：{path.relative_to(root)}")


def _write_asset(root: Path, group: str, output: Path) -> None:
    with output.open("wb") as raw_handle:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            compresslevel=1,
            fileobj=raw_handle,
            mtime=0,
        ) as gzip_handle:
            with tarfile.open(fileobj=gzip_handle, mode="w", format=tarfile.PAX_FORMAT) as archive:
                _add_tar_path(archive, root, root, recursive=False)
                _add_tar_path(archive, root / "wlx-manifest.json", root, recursive=False)
                _add_tar_path(archive, root / group, root, recursive=False)
                for path in sorted((root / group).rglob("*")):
                    _add_tar_path(archive, path, root, recursive=False)


def _add_tar_path(
    archive: tarfile.TarFile,
    path: Path,
    root: Path,
    *,
    recursive: bool,
) -> None:
    arcname = Path("wlx-data-v1") / path.relative_to(root)
    info = archive.gettarinfo(str(path), arcname.as_posix())
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    info.mode = 0o755 if path.is_dir() else 0o644
    if path.is_file():
        with path.open("rb") as handle:
            archive.addfile(info, handle)
    else:
        archive.addfile(info)


def _verify_assets(
    directory: Path,
    assets: list[dict[str, Any]],
    manifest_path: Path,
    checksum_path: Path,
) -> None:
    for report in assets:
        path = directory / report["name"]
        if path.stat().st_size != report["bytes"] or _sha256(path) != report["sha256"]:
            raise ValueError(f"附件校验失败：{path.name}")
        with tarfile.open(path, "r:gz") as archive:
            names = archive.getnames()
            if "wlx-data-v1/wlx-manifest.json" not in names:
                raise ValueError(f"附件缺少根 manifest：{path.name}")
    if not manifest_path.is_file() or not checksum_path.is_file():
        raise ValueError("发布 manifest 或 SHA256SUMS 缺失")


def _asset_report(path: Path) -> dict[str, Any]:
    return {"name": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else REPOSITORY / path


def _portable(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPOSITORY.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
