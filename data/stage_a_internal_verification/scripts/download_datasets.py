#!/usr/bin/env python3
"""Download immutable Stage A source snapshots into the suite's raw directory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import requests
from _dataset_common import (
    DATASET_NAMES,
    DEFAULT_CONFIG,
    DEFAULT_MANIFEST_DIR,
    DEFAULT_RAW_DIR,
    atomic_write_json,
    atomic_write_jsonl,
    count_jsonl,
    load_json,
    relative_artifact,
    sha256,
    validate_manifest,
)

RAW_MANIFEST_SCHEMA = "stage-a-raw-manifest-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("JLENSV_HF_CACHE", "/tmp/jlensv_stage_a_hf")),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_NAMES,
        default=list(DATASET_NAMES),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def request_bytes(url: str) -> bytes:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.content


def write_bytes(path: Path, content: bytes, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def download_huggingface(
    name: str,
    spec: dict[str, Any],
    raw_dir: Path,
    cache_dir: Path,
    *,
    overwrite: bool,
) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "Hugging Face downloads require `pip install -r requirements.txt`"
        ) from exc

    artifacts = []
    for split in spec["splits"]:
        path = raw_dir / name / f"{split}.jsonl"
        if path.exists() and not overwrite:
            records = count_jsonl(path)
        else:
            source = load_dataset(
                spec["repo_id"],
                split=split,
                revision=spec["revision"],
                cache_dir=str(cache_dir),
            )

            def rows(source_rows: Any = source) -> Iterable[dict[str, Any]]:
                for source_index, row in enumerate(source_rows):
                    yield {"_source_index": source_index, **dict(row)}

            records = atomic_write_jsonl(path, rows())
        artifacts.append(
            relative_artifact(
                path,
                raw_dir.parent,
                records=records,
                dataset=name,
                source_repo=spec["repo_id"],
                source_revision=spec["revision"],
                source_split=split,
                license=spec["license"],
            )
        )
    return artifacts


def download_prontoqa(
    spec: dict[str, Any], raw_dir: Path, *, overwrite: bool
) -> list[dict[str, Any]]:
    filename = spec["configuration"]
    path = raw_dir / "prontoqa" / filename
    url = (
        f"https://huggingface.co/datasets/{spec['repo_id']}/resolve/"
        f"{spec['revision']}/{filename}"
    )
    write_bytes(path, request_bytes(url) if overwrite or not path.exists() else b"", overwrite=overwrite)
    # Parsing here makes a truncated HTTP response fail before it enters a manifest.
    source = json.loads(path.read_text(encoding="utf-8"))
    return [
        relative_artifact(
            path,
            raw_dir.parent,
            records=None,
            source_objects=len(source),
            dataset="prontoqa",
            source_repo=spec["repo_id"],
            source_revision=spec["revision"],
            source_split="test",
            license=spec["license"],
        )
    ]


def download_bbeh(
    spec: dict[str, Any], raw_dir: Path, *, overwrite: bool
) -> list[dict[str, Any]]:
    revision = spec["revision"]
    base = f"https://raw.githubusercontent.com/{spec['repo_id']}/{revision}"
    targets = [("mini", "bbeh/mini/data.json")]
    targets.extend(
        (task, f"bbeh/benchmark_tasks/bbeh_{task}/task.json")
        for task in spec["tasks"]
    )
    artifacts = []
    for label, remote_path in targets:
        local_name = "mini.json" if label == "mini" else f"task_{label}.json"
        path = raw_dir / "bbeh" / local_name
        write_bytes(
            path,
            request_bytes(f"{base}/{remote_path}")
            if overwrite or not path.exists()
            else b"",
            overwrite=overwrite,
        )
        source = json.loads(path.read_text(encoding="utf-8"))
        records = len(source["examples"])
        artifacts.append(
            relative_artifact(
                path,
                raw_dir.parent,
                records=None,
                source_objects=records,
                dataset="bbeh",
                source_repo=spec["repo_id"],
                source_revision=revision,
                source_split=label,
                license=spec["license"],
            )
        )
    return artifacts


def merge_artifacts(
    manifest_path: Path,
    new_artifacts: list[dict[str, Any]],
    selected: set[str],
) -> list[dict[str, Any]]:
    old_artifacts: list[dict[str, Any]] = []
    if manifest_path.exists():
        previous = load_json(manifest_path)
        if previous.get("schema_version") != RAW_MANIFEST_SCHEMA:
            raise ValueError(f"{manifest_path}: incompatible existing manifest")
        old_artifacts = [
            item
            for item in previous.get("artifacts", [])
            if item.get("dataset") not in selected
        ]
    return sorted(old_artifacts + new_artifacts, key=lambda item: item["path"])


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    selected = set(dict.fromkeys(args.datasets))
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for name in DATASET_NAMES:
        if name not in selected:
            continue
        print(f"Downloading {name}...", flush=True)
        spec = config["datasets"][name]
        if spec["kind"] == "huggingface":
            artifacts.extend(
                download_huggingface(
                    name,
                    spec,
                    args.raw_dir,
                    args.cache_dir,
                    overwrite=args.overwrite,
                )
            )
        elif name == "prontoqa":
            artifacts.extend(
                download_prontoqa(spec, args.raw_dir, overwrite=args.overwrite)
            )
        elif name == "bbeh":
            artifacts.extend(
                download_bbeh(spec, args.raw_dir, overwrite=args.overwrite)
            )
        else:
            raise AssertionError(f"Unsupported downloader for {name}")

    manifest_path = args.manifest_dir / "raw_manifest.json"
    artifacts = merge_artifacts(manifest_path, artifacts, selected)
    atomic_write_json(
        manifest_path,
        {
            "schema_version": RAW_MANIFEST_SCHEMA,
            "suite": config["suite"],
            "source_config": "config/sources.json",
            "source_config_sha256": sha256(args.config),
            "artifacts": artifacts,
        },
    )
    artifact_count, jsonl_rows = validate_manifest(
        manifest_path,
        processed=False,
        expected_schema=RAW_MANIFEST_SCHEMA,
    )
    print(
        f"Verified {artifact_count} raw artifacts "
        f"({jsonl_rows} JSONL rows); manifest: {manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
