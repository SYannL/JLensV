"""Shared utilities for Stage A generation and audit artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

SUITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = SUITE_ROOT.parents[1]
DEFAULT_PROCESSED_DIR = (
    REPO_ROOT / "data" / "stage_a_internal_verification" / "processed"
)
DEFAULT_PROCESSED_MANIFEST = (
    REPO_ROOT
    / "data"
    / "stage_a_internal_verification"
    / "manifests"
    / "processed_manifest.json"
)
DEFAULT_PROFILE = SUITE_ROOT / "config" / "pilot.json"
DEFAULT_SELECTION = SUITE_ROOT / "selections" / "pilot.jsonl"
DEFAULT_SELECTION_MANIFEST = (
    SUITE_ROOT / "selections" / "pilot_manifest.json"
)
DATASET_NAMES = ("processbench", "math500", "prontoqa", "stepgame", "bbeh")
ARTIFACT_SCHEMA = "stage-a-generation-artifact-v1"
SCORER_VERSION = "stage-a-deterministic-scorers-v1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(row)
    return rows


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                )
                count += 1
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return count


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def effective_seed(base_seed: int, example_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}\0{example_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def sample_directory(run_dir: Path, record: dict[str, Any]) -> Path:
    identifier = record["id"]
    digest = hashlib.sha256(identifier.encode()).hexdigest()[:16]
    safe_dataset = re.sub(r"[^a-z0-9_-]+", "-", record["dataset"].lower())
    return run_dir / "samples" / f"{safe_dataset}--{digest}"


def extract_final_answer(text: str) -> tuple[str | None, str]:
    matches = re.findall(r"(?im)^\s*FINAL\s*:\s*(.*?)\s*$", text)
    if matches:
        answer = matches[-1].strip()
        return (answer or None), "final_line"
    nonempty = [line.strip() for line in text.splitlines() if line.strip()]
    if nonempty:
        return nonempty[-1], "last_nonempty_line"
    return None, "missing"


def _strip_boxed(value: str) -> str:
    value = value.strip()
    for command in ("\\boxed", "\\fbox"):
        prefix = f"{command}{{"
        if value.startswith(prefix) and value.endswith("}"):
            return value[len(prefix) : -1].strip()
    return value


def _numeric_value(value: str) -> float | None:
    cleaned = value.replace(",", "").replace("$", "").strip()
    fraction = re.fullmatch(r"([-+]?\d+)\s*/\s*([-+]?\d+)", cleaned)
    if fraction:
        denominator = int(fraction.group(2))
        return int(fraction.group(1)) / denominator if denominator else None
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def normalize_answer(dataset: str, value: Any) -> Any:
    if value is None:
        return None
    text = _strip_boxed(str(value).strip())
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.strip().strip("$").strip()
    if dataset == "processbench":
        folded = text.casefold().rstrip(".").strip("[]() ")
        if folded in {"ok", "correct", "all correct", "-1"}:
            return None
        match = re.fullmatch(
            r"(?:(?:step_id|step)\s*=?\s*)?(\d+)",
            folded,
        )
        return int(match.group(1)) if match else f"invalid:{folded}"
    if dataset == "prontoqa":
        folded = re.sub(r"[^a-z]+", " ", text.casefold()).strip()
        if folded in {"proved", "true", "yes", "entailed", "entails"}:
            return "proved"
        if folded in {"disproved", "false", "no", "not entailed"}:
            return "disproved"
        return folded
    if dataset == "stepgame":
        return re.sub(r"[\s_]+", "-", text.casefold()).strip("-., ")

    numeric = _numeric_value(text)
    if numeric is not None:
        return {"numeric": numeric}
    normalized = re.sub(r"\s+", "", text).casefold().rstrip(".")
    if dataset == "bbeh":
        normalized = normalized.strip()
    return normalized


def grade_answer(
    record: dict[str, Any],
    predicted_raw: str | None,
    parse_method: str,
) -> dict[str, Any]:
    dataset = record["dataset"]
    if dataset == "processbench":
        gold_raw = record["gold"]["first_error_step"]
    else:
        gold_raw = record["gold"]["answer"]
    predicted = normalize_answer(dataset, predicted_raw)
    gold = normalize_answer(dataset, gold_raw)
    parsed = predicted_raw is not None and not (
        isinstance(predicted, str) and predicted.startswith("invalid:")
    )
    correct = bool(parsed and predicted == gold)
    reliability = (
        "provisional_exact_requires_math_verifier_on_mismatch"
        if dataset == "math500"
        else "deterministic_exact"
    )
    return {
        "scorer_version": SCORER_VERSION,
        "dataset": dataset,
        "gold_raw": gold_raw,
        "gold_normalized": gold,
        "predicted_raw": predicted_raw,
        "predicted_normalized": predicted,
        "parse_method": parse_method,
        "parsed": parsed,
        "correct": correct,
        "reliability": reliability,
    }


def artifact_summary(sample_dir: Path) -> dict[str, Any]:
    generation_path = sample_dir / "generation.json"
    metrics_path = sample_dir / "token_metrics.npz"
    error_path = sample_dir / "error.json"
    if not generation_path.exists():
        result: dict[str, Any] = {
            "sample_dir": sample_dir.name,
            "status": "error" if error_path.exists() else "missing",
        }
        if error_path.exists():
            error = load_json(error_path)
            result["id"] = error.get("id")
            result["dataset"] = error.get("dataset")
            result["error"] = error.get("message")
        return result
    artifact = load_json(generation_path)
    quality = artifact["quality"]
    grading = artifact["grading"]
    generation = artifact["generation"]
    return {
        "sample_dir": sample_dir.name,
        "status": artifact["status"],
        "id": artifact["source"]["record"]["id"],
        "dataset": artifact["source"]["record"]["dataset"],
        "source_split": artifact["source"]["record"]["source_split"],
        "track": artifact["prompt"]["track"],
        "n_prompt_tokens": artifact["prompt"]["n_tokens"],
        "n_generated_tokens": generation["n_tokens"],
        "ended_with_eos": quality["ended_with_eos"],
        "hit_max_new_tokens": quality["hit_max_new_tokens"],
        "final_answer_present": quality["final_answer_present"],
        "complete": quality["complete"],
        "parsed": grading["parsed"],
        "correct": grading["correct"],
        "scorer_reliability": grading["reliability"],
        "raw_metrics_complete": quality["raw_metrics_complete"],
        "replay_ready": quality["replay_ready"],
        "metrics_file_present": metrics_path.exists(),
    }
