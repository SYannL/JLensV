"""Shared, dependency-light utilities for the Stage A data pipeline."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "internal-verification-v1"
SUITE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = SUITE_ROOT / "config" / "sources.json"
DEFAULT_RAW_DIR = SUITE_ROOT / "raw"
DEFAULT_PROCESSED_DIR = SUITE_ROOT / "processed"
DEFAULT_MANIFEST_DIR = SUITE_ROOT / "manifests"
DATASET_NAMES = ("processbench", "math500", "prontoqa", "stepgame", "bbeh")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def to_jsonable(value: Any) -> Any:
    """Convert common dataset scalar/container values to JSON-native values."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        return to_jsonable(value.item())
    if hasattr(value, "tolist"):
        return to_jsonable(value.tolist())
    raise TypeError(f"Cannot serialize {type(value).__name__} as JSON")


def atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(to_jsonable(record), ensure_ascii=False, sort_keys=True)
                    + "\n"
                )
                count += 1
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_jsonl(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            count += 1
    return count


def relative_artifact(path: Path, root: Path, **metadata: Any) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "records": metadata.pop("records", None),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        **metadata,
    }


def base_record(
    *,
    example_id: str,
    dataset: str,
    source_split: str,
    task_family: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "id": example_id,
        "dataset": dataset,
        "source_split": source_split,
        "task_family": task_family,
        "input": {"context": "", "question": "", "choices": []},
        "gold": {
            "answer": None,
            "reasoning_steps": [],
            "first_error_step": None,
            "process_correct": None,
            "final_answer_correct": None,
            "state_trace": [],
        },
        "candidate": {
            "answer": None,
            "reasoning_steps": [],
            "generator": None,
        },
        "capabilities": {
            "outcome_verification": False,
            "process_verification": False,
            "first_error_localization": False,
            "gold_reasoning": False,
            "exact_intermediate_state": False,
            "structured_constraints": False,
        },
        "metadata": {},
    }


def validate_record(record: dict[str, Any], *, location: str = "<record>") -> None:
    top_level = {
        "schema_version",
        "id",
        "dataset",
        "source_split",
        "task_family",
        "input",
        "gold",
        "candidate",
        "capabilities",
        "metadata",
    }
    missing = top_level - record.keys()
    extra = record.keys() - top_level
    if missing or extra:
        raise ValueError(
            f"{location}: invalid top-level fields; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{location}: unexpected schema_version")
    if not isinstance(record["id"], str) or not record["id"]:
        raise ValueError(f"{location}: id must be a non-empty string")
    if record["dataset"] not in DATASET_NAMES:
        raise ValueError(f"{location}: unsupported dataset {record['dataset']!r}")
    for field in ("context", "question"):
        if not isinstance(record["input"].get(field), str):
            raise ValueError(f"{location}: input.{field} must be a string")
    for field in ("choices",):
        if not isinstance(record["input"].get(field), list):
            raise ValueError(f"{location}: input.{field} must be a list")
    for object_name, field in (
        ("gold", "reasoning_steps"),
        ("gold", "state_trace"),
        ("candidate", "reasoning_steps"),
    ):
        if not isinstance(record[object_name].get(field), list):
            raise ValueError(f"{location}: {object_name}.{field} must be a list")
    first_error = record["gold"].get("first_error_step")
    if first_error is not None and (
        isinstance(first_error, bool)
        or not isinstance(first_error, int)
        or first_error < 0
    ):
        raise ValueError(f"{location}: invalid first_error_step={first_error!r}")
    if set(record["capabilities"]) != {
        "outcome_verification",
        "process_verification",
        "first_error_localization",
        "gold_reasoning",
        "exact_intermediate_state",
        "structured_constraints",
    }:
        raise ValueError(f"{location}: invalid capability fields")
    if not all(isinstance(value, bool) for value in record["capabilities"].values()):
        raise ValueError(f"{location}: capabilities must be boolean")


def validate_processed_jsonl(path: Path) -> int:
    seen: set[str] = set()
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            location = f"{path}:{line_number}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{location}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{location}: record must be an object")
            validate_record(record, location=location)
            if record["id"] in seen:
                raise ValueError(f"{location}: duplicate id {record['id']!r}")
            seen.add(record["id"])
            count += 1
    if not count:
        raise ValueError(f"{path}: contains no records")
    return count


def validate_manifest(
    manifest_path: Path,
    *,
    processed: bool,
    expected_schema: str,
) -> tuple[int, int]:
    manifest = load_json(manifest_path)
    if manifest.get("schema_version") != expected_schema:
        raise ValueError(f"{manifest_path}: unexpected manifest schema")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError(f"{manifest_path}: no artifacts")
    total_records = 0
    for artifact in artifacts:
        path = manifest_path.parent.parent / artifact["path"]
        if not path.is_file():
            raise FileNotFoundError(f"missing artifact: {path}")
        if path.stat().st_size != artifact["bytes"]:
            raise ValueError(f"{path}: byte-size mismatch")
        if sha256(path) != artifact["sha256"]:
            raise ValueError(f"{path}: SHA-256 mismatch")
        expected_records = artifact.get("records")
        if expected_records is not None:
            actual_records = (
                validate_processed_jsonl(path) if processed else count_jsonl(path)
            )
            if actual_records != expected_records:
                raise ValueError(
                    f"{path}: expected {expected_records} records, "
                    f"found {actual_records}"
                )
            total_records += actual_records
        elif not processed and path.suffix == ".json":
            load_json(path)
    declared_total = manifest.get("total_records")
    if declared_total is not None and declared_total != total_records:
        raise ValueError(
            f"{manifest_path}: total_records={declared_total}, "
            f"actual={total_records}"
        )
    return len(artifacts), total_records
