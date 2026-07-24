#!/usr/bin/env python3
"""Prepare a small, reproducible suite for internal reasoning verification.

The script deliberately normalizes only fields supplied by the source dataset.
It never asks an LLM to invent missing gold steps or intermediate states.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import requests
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi

SCHEMA_VERSION = "internal-verification-v1"
DEFAULT_OUTPUT_DIR = Path("data/internal_verification")
DEFAULT_CACHE_DIR = Path("/tmp/jlensv_hf_datasets")
DATASET_NAMES = ("processbench", "math500", "prontoqa", "stepgame", "bbeh")

HF_SOURCES = {
    "processbench": "Qwen/ProcessBench",
    "math500": "HuggingFaceH4/MATH-500",
    "stepgame": "ZhengyanShi/StepGame",
}

PRONTOQA_REPO = "tasksource/prontoqa"
PRONTOQA_REVISION = "2b99cc18e64b1684e29ba69f966306d9cc563e8d"
PRONTOQA_CONFIG = "1hop_ProofsOnly_5testhops_random_noadj.json"
PRONTOQA_URL = (
    "https://huggingface.co/datasets/"
    f"{PRONTOQA_REPO}/resolve/{PRONTOQA_REVISION}/{PRONTOQA_CONFIG}"
)

BBEH_REPO = "google-deepmind/bbeh"
BBEH_API = f"https://api.github.com/repos/{BBEH_REPO}"
BBEH_TASKS = (
    "boardgame_qa",
    "boolean_expressions",
    "buggy_tables",
    "causal_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "geometric_shapes",
    "hyperbaton",
    "linguini",
    "movie_recommendation",
    "multistep_arithmetic",
    "nycc",
    "object_counting",
    "object_properties",
    "sarc_triples",
    "shuffled_objects",
    "spatial_reasoning",
    "sportqa",
    "temporal_sequence",
    "time_arithmetic",
    "web_of_lies",
    "word_sorting",
    "zebra_puzzles",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_NAMES,
        default=list(DATASET_NAMES),
        help="Datasets to prepare. Defaults to the complete first-stage suite.",
    )
    parser.add_argument(
        "--stepgame-per-hop",
        type=int,
        default=100,
        help="Number of StepGame test examples sampled for each hop 1 through 10.",
    )
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace already prepared dataset files.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate existing JSONL files and manifest without downloading.",
    )
    return parser.parse_args()


def _base_record(
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
        "input": {
            "context": "",
            "question": "",
            "choices": [],
        },
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


def _atomic_write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    count = 0
    with tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            _validate_record(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    os.replace(tmp, path)
    return count


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_record(record: dict[str, Any]) -> None:
    required = {
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
    missing = required - record.keys()
    if missing:
        raise ValueError(f"{record.get('id', '<unknown>')}: missing {sorted(missing)}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"{record['id']}: unexpected schema version")
    if not isinstance(record["id"], str) or not record["id"]:
        raise ValueError("record id must be a non-empty string")
    if not isinstance(record["input"]["choices"], list):
        raise ValueError(f"{record['id']}: input.choices must be a list")
    if not isinstance(record["gold"]["reasoning_steps"], list):
        raise ValueError(f"{record['id']}: gold.reasoning_steps must be a list")
    if not isinstance(record["candidate"]["reasoning_steps"], list):
        raise ValueError(f"{record['id']}: candidate.reasoning_steps must be a list")
    first_error = record["gold"]["first_error_step"]
    if first_error is not None and (not isinstance(first_error, int) or first_error < 0):
        raise ValueError(f"{record['id']}: invalid first_error_step={first_error!r}")


def _validate_jsonl(path: Path) -> int:
    count = 0
    seen: set[str] = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            _validate_record(record)
            if record["id"] in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {record['id']}")
            seen.add(record["id"])
            count += 1
    if count == 0:
        raise ValueError(f"{path}: no records")
    return count


def _split_reference_solution(solution: str) -> list[str]:
    return [part.strip() for part in solution.split("\n\n") if part.strip()]


def _hf_revision(repo_id: str) -> str:
    info = HfApi().dataset_info(repo_id)
    if not info.sha:
        raise RuntimeError(f"Hugging Face did not return a revision for {repo_id}")
    return info.sha


def _load_hf(
    repo_id: str,
    *,
    split: str,
    revision: str,
    cache_dir: Path,
) -> Dataset:
    return load_dataset(
        repo_id,
        split=split,
        revision=revision,
        cache_dir=str(cache_dir),
    )


def _prepare_processbench(
    output_dir: Path,
    cache_dir: Path,
    *,
    overwrite: bool,
) -> list[dict[str, Any]]:
    repo_id = HF_SOURCES["processbench"]
    revision = _hf_revision(repo_id)
    artifacts = []
    # Deliberately exclude the GSM8K split in this expansion.
    for split in ("math", "olympiadbench", "omnimath"):
        path = output_dir / "processbench" / f"{split}.jsonl"
        if path.exists() and not overwrite:
            count = _validate_jsonl(path)
        else:
            source = _load_hf(
                repo_id,
                split=split,
                revision=revision,
                cache_dir=cache_dir,
            )

            def records(
                source_rows: Dataset = source,
                source_split: str = split,
            ) -> Iterable[dict[str, Any]]:
                for source_index, row in enumerate(source_rows):
                    label = int(row["label"])
                    record = _base_record(
                        example_id=f"processbench/{source_split}/{row['id']}",
                        dataset="processbench",
                        source_split=source_split,
                        task_family="mathematical_process_verification",
                    )
                    record["input"]["question"] = row["problem"]
                    record["gold"].update(
                        {
                            "first_error_step": None if label == -1 else label,
                            "process_correct": label == -1,
                            "final_answer_correct": bool(
                                row["final_answer_correct"]
                            ),
                        }
                    )
                    record["candidate"].update(
                        {
                            "reasoning_steps": list(row["steps"]),
                            "generator": row["generator"],
                        }
                    )
                    record["capabilities"].update(
                        {
                            "outcome_verification": True,
                            "process_verification": True,
                            "first_error_localization": True,
                        }
                    )
                    record["metadata"] = {
                        "source_id": row["id"],
                        "source_index": source_index,
                        "official_label_zero_based": label,
                    }
                    yield record

            count = _atomic_write_jsonl(path, records())
        artifacts.append(
            _artifact(
                path,
                count,
                repo_id=repo_id,
                revision=revision,
                license_name="Apache-2.0",
            )
        )
    return artifacts


def _prepare_math500(
    output_dir: Path,
    cache_dir: Path,
    *,
    overwrite: bool,
) -> list[dict[str, Any]]:
    repo_id = HF_SOURCES["math500"]
    revision = _hf_revision(repo_id)
    path = output_dir / "math500" / "test.jsonl"
    if path.exists() and not overwrite:
        count = _validate_jsonl(path)
    else:
        source = _load_hf(
            repo_id,
            split="test",
            revision=revision,
            cache_dir=cache_dir,
        )

        def records() -> Iterable[dict[str, Any]]:
            for source_index, row in enumerate(source):
                record = _base_record(
                    example_id=f"math500/{row['unique_id']}",
                    dataset="math500",
                    source_split="test",
                    task_family="competition_mathematics",
                )
                record["input"]["question"] = row["problem"]
                record["gold"].update(
                    {
                        "answer": row["answer"],
                        "reasoning_steps": _split_reference_solution(
                            row["solution"]
                        ),
                    }
                )
                record["capabilities"].update(
                    {
                        "outcome_verification": True,
                        "gold_reasoning": True,
                    }
                )
                record["metadata"] = {
                    "source_index": source_index,
                    "unique_id": row["unique_id"],
                    "subject": row["subject"],
                    "level": int(row["level"]),
                    "reference_solution": row["solution"],
                    "reasoning_step_note": (
                        "reasoning_steps are blank-line-delimited source "
                        "paragraphs, not independently verified atomic steps"
                    ),
                }
                yield record

        count = _atomic_write_jsonl(path, records())
    return [
        _artifact(
            path,
            count,
            repo_id=repo_id,
            revision=revision,
            license_name="MATH dataset terms; research evaluation",
        )
    ]


def _request_json(url: str) -> Any:
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    return response.json()


def _prepare_prontoqa(
    output_dir: Path,
    *,
    overwrite: bool,
) -> list[dict[str, Any]]:
    path = output_dir / "prontoqa" / "proofsonly_1hop_to_5hop_ood.jsonl"
    if path.exists() and not overwrite:
        count = _validate_jsonl(path)
    else:
        source = _request_json(PRONTOQA_URL)

        def records() -> Iterable[dict[str, Any]]:
            for source_index, (source_id, bundle) in enumerate(source.items()):
                row = bundle["test_example"]
                proof_steps = list(row["chain_of_thought"])
                record = _base_record(
                    example_id=f"prontoqa/{PRONTOQA_CONFIG}/{source_id}",
                    dataset="prontoqa",
                    source_split="test",
                    task_family="formal_deductive_reasoning",
                )
                record["input"].update(
                    {
                        "context": row["question"],
                        "question": row["query"],
                    }
                )
                record["gold"].update(
                    {
                        "answer": "proved",
                        "reasoning_steps": proof_steps,
                    }
                )
                record["capabilities"].update(
                    {
                        "outcome_verification": True,
                        "process_verification": True,
                        "gold_reasoning": True,
                        "exact_intermediate_state": True,
                        "structured_constraints": True,
                    }
                )
                record["metadata"] = {
                    "source_index": source_index,
                    "source_id": source_id,
                    "configuration": PRONTOQA_CONFIG,
                    "train_hops": 1,
                    "test_hops": 5,
                    "proof_target": proof_steps[-1] if proof_steps else None,
                }
                yield record

        count = _atomic_write_jsonl(path, records())
    return [
        _artifact(
            path,
            count,
            repo_id=PRONTOQA_REPO,
            revision=PRONTOQA_REVISION,
            license_name="Apache-2.0",
        )
    ]


def _prepare_stepgame(
    output_dir: Path,
    cache_dir: Path,
    *,
    per_hop: int,
    seed: int,
    overwrite: bool,
) -> list[dict[str, Any]]:
    if per_hop <= 0:
        raise ValueError("--stepgame-per-hop must be positive")
    repo_id = HF_SOURCES["stepgame"]
    revision = _hf_revision(repo_id)
    path = output_dir / "stepgame" / f"test_stratified_{per_hop}_per_hop.jsonl"
    if path.exists() and not overwrite:
        count = _validate_jsonl(path)
    else:
        source = _load_hf(
            repo_id,
            split="test",
            revision=revision,
            cache_dir=cache_dir,
        )
        by_hop: dict[int, list[int]] = defaultdict(list)
        for source_index, hop in enumerate(source["k_hop"]):
            by_hop[int(hop)].append(source_index)
        rng = random.Random(seed)
        selected: list[tuple[int, int]] = []
        for hop in range(1, 11):
            candidates = by_hop.get(hop, [])
            if len(candidates) < per_hop:
                raise ValueError(
                    f"StepGame hop {hop} has {len(candidates)} rows, "
                    f"fewer than requested {per_hop}"
                )
            selected.extend((hop, index) for index in rng.sample(candidates, per_hop))
        selected.sort()

        def records() -> Iterable[dict[str, Any]]:
            for hop, source_index in selected:
                row = source[source_index]
                story = list(row["story"])
                record = _base_record(
                    example_id=f"stepgame/test/{source_index:06d}",
                    dataset="stepgame",
                    source_split="test",
                    task_family="spatial_relation_state_tracking",
                )
                record["input"].update(
                    {
                        "context": "\n".join(story),
                        "question": row["question"],
                    }
                )
                record["gold"]["answer"] = row["label"]
                record["capabilities"].update(
                    {
                        "outcome_verification": True,
                        "structured_constraints": True,
                    }
                )
                record["metadata"] = {
                    "source_index": source_index,
                    "k_hop": hop,
                    "story_facts": story,
                    "sampling_seed": seed,
                    "note": (
                        "The source provides exact constraints and the final "
                        "relation, but not an explicit gold state trajectory."
                    ),
                }
                yield record

        count = _atomic_write_jsonl(path, records())
    return [
        _artifact(
            path,
            count,
            repo_id=repo_id,
            revision=revision,
            license_name="Source repository terms; research use",
        )
    ]


def _github_revision(repo: str) -> str:
    response = requests.get(
        f"https://api.github.com/repos/{repo}/commits/main",
        timeout=120,
    )
    response.raise_for_status()
    revision = response.json().get("sha")
    if not revision:
        raise RuntimeError(f"GitHub did not return a revision for {repo}")
    return revision


def _bbeh_task_lookup(revision: str) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for task_name in BBEH_TASKS:
        url = (
            "https://raw.githubusercontent.com/"
            f"{BBEH_REPO}/{revision}/bbeh/benchmark_tasks/"
            f"bbeh_{task_name}/task.json"
        )
        task_data = _request_json(url)
        for row in task_data["examples"]:
            prompt = row["input"]
            previous = lookup.get(prompt)
            if previous is not None and previous != task_name:
                raise ValueError(
                    "BBEH prompt occurs in multiple tasks: "
                    f"{previous!r} and {task_name!r}"
                )
            lookup[prompt] = task_name
    return lookup


def _prepare_bbeh(
    output_dir: Path,
    *,
    overwrite: bool,
) -> list[dict[str, Any]]:
    revision = _github_revision(BBEH_REPO)
    path = output_dir / "bbeh" / "mini.jsonl"
    if path.exists() and not overwrite:
        count = _validate_jsonl(path)
    else:
        source_url = (
            "https://raw.githubusercontent.com/"
            f"{BBEH_REPO}/{revision}/bbeh/mini/data.json"
        )
        source = _request_json(source_url)
        task_lookup = _bbeh_task_lookup(revision)

        def records() -> Iterable[dict[str, Any]]:
            for source_index, row in enumerate(source["examples"]):
                task_name = task_lookup.get(row["input"])
                if task_name is None:
                    raise ValueError(
                        f"BBEH mini row {source_index} was not found in a full task"
                    )
                record = _base_record(
                    example_id=f"bbeh/mini/{source_index:04d}",
                    dataset="bbeh",
                    source_split="mini",
                    task_family=f"bbeh_{task_name}",
                )
                record["input"]["question"] = row["input"]
                record["gold"]["answer"] = row["target"]
                record["capabilities"]["outcome_verification"] = True
                if task_name == "dyck_languages":
                    record["capabilities"].update(
                        {
                            "process_verification": True,
                            "first_error_localization": True,
                            "exact_intermediate_state": True,
                            "structured_constraints": True,
                        }
                    )
                elif task_name in {
                    "boardgame_qa",
                    "boolean_expressions",
                    "buggy_tables",
                    "multistep_arithmetic",
                    "object_properties",
                    "shuffled_objects",
                    "spatial_reasoning",
                    "temporal_sequence",
                    "time_arithmetic",
                    "web_of_lies",
                    "word_sorting",
                    "zebra_puzzles",
                }:
                    record["capabilities"]["structured_constraints"] = True
                record["metadata"] = {
                    "source_index": source_index,
                    "official_mini": True,
                    "source_task": task_name,
                    "note": (
                        "The official combined mini omits task labels; this "
                        "label was restored by exact prompt matching against "
                        "the pinned official full task files."
                    ),
                }
                yield record

        count = _atomic_write_jsonl(path, records())
    return [
        _artifact(
            path,
            count,
            repo_id=BBEH_REPO,
            revision=revision,
            license_name="CC-BY-4.0 data / Apache-2.0 code",
        )
    ]


def _artifact(
    path: Path,
    count: int,
    *,
    repo_id: str,
    revision: str,
    license_name: str,
) -> dict[str, Any]:
    return {
        "path": str(path),
        "records": count,
        "sha256": _sha256(path),
        "source_repo": repo_id,
        "source_revision": revision,
        "license": license_name,
    }


def _validate_existing(output_dir: Path) -> None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifacts = manifest.get("artifacts", [])
    if not artifacts:
        raise ValueError(f"{manifest_path}: no artifacts")
    total = 0
    for artifact in artifacts:
        path = Path(artifact["path"])
        if not path.is_absolute():
            # Manifest paths are normally repository-relative.
            path = Path.cwd() / path
        count = _validate_jsonl(path)
        checksum = _sha256(path)
        if count != artifact["records"]:
            raise ValueError(
                f"{path}: manifest records={artifact['records']}, actual={count}"
            )
        if checksum != artifact["sha256"]:
            raise ValueError(f"{path}: SHA-256 mismatch")
        total += count
    print(f"Validated {len(artifacts)} artifacts with {total} records.")


def main() -> None:
    args = parse_args()
    if args.validate_only:
        _validate_existing(args.output_dir)
        return

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    selected = list(dict.fromkeys(args.datasets))
    artifacts: list[dict[str, Any]] = []
    for name in selected:
        print(f"Preparing {name}...", flush=True)
        if name == "processbench":
            artifacts.extend(
                _prepare_processbench(
                    args.output_dir,
                    args.cache_dir,
                    overwrite=args.overwrite,
                )
            )
        elif name == "math500":
            artifacts.extend(
                _prepare_math500(
                    args.output_dir,
                    args.cache_dir,
                    overwrite=args.overwrite,
                )
            )
        elif name == "prontoqa":
            artifacts.extend(
                _prepare_prontoqa(
                    args.output_dir,
                    overwrite=args.overwrite,
                )
            )
        elif name == "stepgame":
            artifacts.extend(
                _prepare_stepgame(
                    args.output_dir,
                    args.cache_dir,
                    per_hop=args.stepgame_per_hop,
                    seed=args.seed,
                    overwrite=args.overwrite,
                )
            )
        elif name == "bbeh":
            artifacts.extend(
                _prepare_bbeh(
                    args.output_dir,
                    overwrite=args.overwrite,
                )
            )
        else:  # pragma: no cover - argparse enforces choices.
            raise AssertionError(name)

    manifest_path = args.output_dir / "manifest.json"
    old_artifacts = []
    if manifest_path.exists():
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        old_artifacts = old_manifest.get("artifacts", [])
    new_paths = {item["path"] for item in artifacts}
    artifacts = [
        item for item in old_artifacts if item.get("path") not in new_paths
    ] + artifacts
    artifacts.sort(key=lambda item: item["path"])

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "preparation": {
            "seed": args.seed,
            "stepgame_per_hop": args.stepgame_per_hop,
            "last_selected_datasets": selected,
            "excluded_by_design": {
                "processbench/gsm8k": (
                    "The first expansion intentionally uses only non-GSM splits."
                ),
                "google/reveal": (
                    "Gated evaluation-only dataset. A user must accept its "
                    "access and non-redistribution terms before local download."
                ),
            },
        },
        "artifacts": artifacts,
        "total_records": sum(item["records"] for item in artifacts),
    }
    _atomic_write_json(manifest_path, manifest)
    _validate_existing(args.output_dir)
    print(
        f"Prepared {manifest['total_records']} records in {args.output_dir}.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
