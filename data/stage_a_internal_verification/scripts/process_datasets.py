#!/usr/bin/env python3
"""Normalize downloaded Stage A snapshots into the verification schema."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from _common import (
    DATASET_NAMES,
    DEFAULT_CONFIG,
    DEFAULT_MANIFEST_DIR,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    atomic_write_json,
    atomic_write_jsonl,
    base_record,
    load_json,
    relative_artifact,
    sha256,
    validate_manifest,
    validate_processed_jsonl,
    validate_record,
)

RAW_MANIFEST_SCHEMA = "stage-a-raw-manifest-v1"
PROCESSED_MANIFEST_SCHEMA = "stage-a-processed-manifest-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_NAMES,
        default=list(DATASET_NAMES),
    )
    parser.add_argument(
        "--stepgame-per-hop",
        type=int,
        help="Override the stratified sample size in sources.json.",
    )
    parser.add_argument("--seed", type=int, help="Override the sampling seed.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield row


def emit(
    path: Path,
    records: Iterable[dict[str, Any]],
    *,
    overwrite: bool,
) -> int:
    if path.exists() and not overwrite:
        return validate_processed_jsonl(path)

    def checked() -> Iterable[dict[str, Any]]:
        for record in records:
            validate_record(record, location=record.get("id", "<record>"))
            yield record

    return atomic_write_jsonl(path, checked())


def process_processbench(
    raw_dir: Path,
    processed_dir: Path,
    spec: dict[str, Any],
    *,
    overwrite: bool,
) -> list[dict[str, Any]]:
    artifacts = []
    for split in spec["splits"]:
        source_path = raw_dir / "processbench" / f"{split}.jsonl"
        output_path = processed_dir / "processbench" / f"{split}.jsonl"

        def records(
            source_file: Path = source_path,
            source_split: str = split,
        ) -> Iterable[dict[str, Any]]:
            for row in read_jsonl(source_file):
                label = int(row["label"])
                record = base_record(
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
                        "final_answer_correct": bool(row["final_answer_correct"]),
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
                    "source_index": int(row["_source_index"]),
                    "official_label_zero_based": label,
                }
                yield record

        count = emit(output_path, records(), overwrite=overwrite)
        artifacts.append(
            relative_artifact(
                output_path,
                processed_dir.parent,
                records=count,
                dataset="processbench",
                source_repo=spec["repo_id"],
                source_revision=spec["revision"],
                source_split=split,
                license=spec["license"],
            )
        )
    return artifacts


def process_math500(
    raw_dir: Path,
    processed_dir: Path,
    spec: dict[str, Any],
    *,
    overwrite: bool,
) -> list[dict[str, Any]]:
    source_path = raw_dir / "math500" / "test.jsonl"
    output_path = processed_dir / "math500" / "test.jsonl"

    def records() -> Iterable[dict[str, Any]]:
        for row in read_jsonl(source_path):
            solution = row["solution"]
            record = base_record(
                example_id=f"math500/{row['unique_id']}",
                dataset="math500",
                source_split="test",
                task_family="competition_mathematics",
            )
            record["input"]["question"] = row["problem"]
            record["gold"].update(
                {
                    "answer": row["answer"],
                    "reasoning_steps": [
                        part.strip()
                        for part in solution.split("\n\n")
                        if part.strip()
                    ],
                }
            )
            record["capabilities"].update(
                {"outcome_verification": True, "gold_reasoning": True}
            )
            record["metadata"] = {
                "source_index": int(row["_source_index"]),
                "unique_id": row["unique_id"],
                "subject": row["subject"],
                "level": int(row["level"]),
                "reference_solution": solution,
                "reasoning_step_note": (
                    "Blank-line-delimited source paragraphs; not independently "
                    "verified atomic steps."
                ),
            }
            yield record

    count = emit(output_path, records(), overwrite=overwrite)
    return [
        relative_artifact(
            output_path,
            processed_dir.parent,
            records=count,
            dataset="math500",
            source_repo=spec["repo_id"],
            source_revision=spec["revision"],
            source_split="test",
            license=spec["license"],
        )
    ]


def process_prontoqa(
    raw_dir: Path,
    processed_dir: Path,
    spec: dict[str, Any],
    *,
    overwrite: bool,
) -> list[dict[str, Any]]:
    configuration = spec["configuration"]
    source_path = raw_dir / "prontoqa" / configuration
    output_path = (
        processed_dir / "prontoqa" / "proofsonly_1hop_to_5hop_ood.jsonl"
    )
    source = load_json(source_path)

    def records() -> Iterable[dict[str, Any]]:
        for source_index, (source_id, bundle) in enumerate(source.items()):
            row = bundle["test_example"]
            proof_steps = list(row["chain_of_thought"])
            record = base_record(
                example_id=f"prontoqa/{configuration}/{source_id}",
                dataset="prontoqa",
                source_split="test",
                task_family="formal_deductive_reasoning",
            )
            record["input"].update(
                {"context": row["question"], "question": row["query"]}
            )
            record["gold"].update(
                {"answer": "proved", "reasoning_steps": proof_steps}
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
                "configuration": configuration,
                "train_hops": 1,
                "test_hops": 5,
                "proof_target": proof_steps[-1] if proof_steps else None,
            }
            yield record

    count = emit(output_path, records(), overwrite=overwrite)
    return [
        relative_artifact(
            output_path,
            processed_dir.parent,
            records=count,
            dataset="prontoqa",
            source_repo=spec["repo_id"],
            source_revision=spec["revision"],
            source_split="test",
            license=spec["license"],
        )
    ]


def process_stepgame(
    raw_dir: Path,
    processed_dir: Path,
    spec: dict[str, Any],
    *,
    per_hop: int,
    seed: int,
    overwrite: bool,
) -> list[dict[str, Any]]:
    if per_hop <= 0:
        raise ValueError("--stepgame-per-hop must be positive")
    source_path = raw_dir / "stepgame" / "test.jsonl"
    output_path = (
        processed_dir
        / "stepgame"
        / f"test_stratified_{per_hop}_per_hop.jsonl"
    )
    source = list(read_jsonl(source_path))
    by_hop: dict[int, list[int]] = defaultdict(list)
    for row_index, row in enumerate(source):
        by_hop[int(row["k_hop"])].append(row_index)
    rng = random.Random(seed)
    selected: list[tuple[int, int]] = []
    for hop in range(1, 11):
        candidates = by_hop.get(hop, [])
        if len(candidates) < per_hop:
            raise ValueError(
                f"StepGame hop {hop} has {len(candidates)} rows; "
                f"{per_hop} requested"
            )
        selected.extend(
            (hop, row_index) for row_index in rng.sample(candidates, per_hop)
        )
    selected.sort()

    def records() -> Iterable[dict[str, Any]]:
        for hop, row_index in selected:
            row = source[row_index]
            source_index = int(row["_source_index"])
            story = list(row["story"])
            record = base_record(
                example_id=f"stepgame/test/{source_index:06d}",
                dataset="stepgame",
                source_split="test",
                task_family="spatial_relation_state_tracking",
            )
            record["input"].update(
                {"context": "\n".join(story), "question": row["question"]}
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

    count = emit(output_path, records(), overwrite=overwrite)
    return [
        relative_artifact(
            output_path,
            processed_dir.parent,
            records=count,
            dataset="stepgame",
            source_repo=spec["repo_id"],
            source_revision=spec["revision"],
            source_split="test",
            license=spec["license"],
        )
    ]


def process_bbeh(
    raw_dir: Path,
    processed_dir: Path,
    spec: dict[str, Any],
    *,
    overwrite: bool,
) -> list[dict[str, Any]]:
    source = load_json(raw_dir / "bbeh" / "mini.json")
    task_lookup: dict[str, str] = {}
    for task_name in spec["tasks"]:
        task = load_json(raw_dir / "bbeh" / f"task_{task_name}.json")
        for row in task["examples"]:
            prompt = row["input"]
            previous = task_lookup.get(prompt)
            if previous is not None and previous != task_name:
                raise ValueError(
                    f"BBEH prompt occurs in both {previous!r} and {task_name!r}"
                )
            task_lookup[prompt] = task_name
    output_path = processed_dir / "bbeh" / "mini.jsonl"

    def records() -> Iterable[dict[str, Any]]:
        for source_index, row in enumerate(source["examples"]):
            task_name = task_lookup.get(row["input"])
            if task_name is None:
                raise ValueError(
                    f"BBEH mini row {source_index} does not match a full task"
                )
            record = base_record(
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
                    "The official mini omits task labels; restored by exact "
                    "prompt matching against pinned official full-task files."
                ),
            }
            yield record

    count = emit(output_path, records(), overwrite=overwrite)
    return [
        relative_artifact(
            output_path,
            processed_dir.parent,
            records=count,
            dataset="bbeh",
            source_repo=spec["repo_id"],
            source_revision=spec["revision"],
            source_split="mini",
            license=spec["license"],
        )
    ]


def merge_artifacts(
    manifest_path: Path,
    new_artifacts: list[dict[str, Any]],
    selected: set[str],
) -> list[dict[str, Any]]:
    old_artifacts: list[dict[str, Any]] = []
    if manifest_path.exists():
        previous = load_json(manifest_path)
        if previous.get("schema_version") != PROCESSED_MANIFEST_SCHEMA:
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
    seed = config["seed"] if args.seed is None else args.seed
    per_hop = (
        config["stepgame_per_hop"]
        if args.stepgame_per_hop is None
        else args.stepgame_per_hop
    )
    raw_manifest_path = args.manifest_dir / "raw_manifest.json"
    validate_manifest(
        raw_manifest_path,
        processed=False,
        expected_schema=RAW_MANIFEST_SCHEMA,
    )
    raw_manifest_digest = sha256(raw_manifest_path)
    processed_manifest_path = args.manifest_dir / "processed_manifest.json"
    if processed_manifest_path.exists() and not args.overwrite:
        old_manifest = load_json(processed_manifest_path)
        old_digest = old_manifest.get("raw_manifest_sha256")
        if old_digest != raw_manifest_digest:
            raise ValueError(
                "Raw manifest changed since processing; rerun with --overwrite"
            )

    selected = set(dict.fromkeys(args.datasets))
    args.processed_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for name in DATASET_NAMES:
        if name not in selected:
            continue
        print(f"Processing {name}...", flush=True)
        spec = config["datasets"][name]
        if name == "processbench":
            artifacts.extend(
                process_processbench(
                    args.raw_dir,
                    args.processed_dir,
                    spec,
                    overwrite=args.overwrite,
                )
            )
        elif name == "math500":
            artifacts.extend(
                process_math500(
                    args.raw_dir,
                    args.processed_dir,
                    spec,
                    overwrite=args.overwrite,
                )
            )
        elif name == "prontoqa":
            artifacts.extend(
                process_prontoqa(
                    args.raw_dir,
                    args.processed_dir,
                    spec,
                    overwrite=args.overwrite,
                )
            )
        elif name == "stepgame":
            artifacts.extend(
                process_stepgame(
                    args.raw_dir,
                    args.processed_dir,
                    spec,
                    per_hop=per_hop,
                    seed=seed,
                    overwrite=args.overwrite,
                )
            )
        elif name == "bbeh":
            artifacts.extend(
                process_bbeh(
                    args.raw_dir,
                    args.processed_dir,
                    spec,
                    overwrite=args.overwrite,
                )
            )
    artifacts = merge_artifacts(processed_manifest_path, artifacts, selected)
    manifest = {
        "schema_version": PROCESSED_MANIFEST_SCHEMA,
        "record_schema_version": "internal-verification-v1",
        "suite": config["suite"],
        "source_config_sha256": sha256(args.config),
        "raw_manifest_sha256": raw_manifest_digest,
        "preparation": {
            "seed": seed,
            "stepgame_per_hop": per_hop,
            "excluded": config["excluded"],
        },
        "artifacts": artifacts,
        "total_records": sum(item["records"] for item in artifacts),
    }
    atomic_write_json(processed_manifest_path, manifest)
    artifact_count, record_count = validate_manifest(
        processed_manifest_path,
        processed=True,
        expected_schema=PROCESSED_MANIFEST_SCHEMA,
    )
    print(
        f"Validated {artifact_count} processed artifacts with "
        f"{record_count} records; manifest: {processed_manifest_path}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
