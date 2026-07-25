#!/usr/bin/env python3
"""Create a deterministic, stratified Stage A pilot or full selection."""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from _common import (
    DATASET_NAMES,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_PROCESSED_MANIFEST,
    DEFAULT_PROFILE,
    DEFAULT_SELECTION,
    DEFAULT_SELECTION_MANIFEST,
    atomic_json,
    atomic_jsonl,
    load_json,
    load_jsonl,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--processed-manifest",
        type=Path,
        default=DEFAULT_PROCESSED_MANIFEST,
    )
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_SELECTION_MANIFEST,
    )
    parser.add_argument(
        "--mode",
        choices=("smoke", "pilot", "full"),
        default="pilot",
        help="Smoke selects one record per dataset; full selects every record.",
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def group_key(record: dict[str, Any]) -> tuple[Any, ...]:
    dataset = record["dataset"]
    if dataset == "processbench":
        return (
            record["source_split"],
            record["gold"]["process_correct"],
            record["gold"]["final_answer_correct"],
        )
    if dataset == "math500":
        return (record["metadata"]["subject"], record["metadata"]["level"])
    if dataset == "prontoqa":
        return ("proofsonly_ood",)
    if dataset == "stepgame":
        return (record["metadata"]["k_hop"],)
    if dataset == "bbeh":
        return (record["metadata"]["source_task"],)
    raise ValueError(f"unsupported dataset {dataset}")


def rank_key(seed: int, identifier: str) -> str:
    return hashlib.sha256(f"{seed}\0{identifier}".encode()).hexdigest()


def balanced_selection(
    records: list[dict[str, Any]],
    limit: int,
    seed: int,
) -> list[dict[str, Any]]:
    if limit >= len(records):
        return sorted(records, key=lambda row: row["id"])
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[group_key(record)].append(record)
    for rows in groups.values():
        rows.sort(key=lambda row: rank_key(seed, row["id"]))
    selected = []
    ordered_groups = sorted(groups, key=str)
    offset = 0
    while len(selected) < limit:
        made_progress = False
        for key in ordered_groups:
            rows = groups[key]
            if offset < len(rows):
                selected.append(rows[offset])
                made_progress = True
                if len(selected) == limit:
                    break
        if not made_progress:
            break
        offset += 1
    return sorted(selected, key=lambda row: (row["dataset"], row["id"]))


def processed_paths(manifest: dict[str, Any], processed_dir: Path) -> list[Path]:
    paths = []
    for artifact in manifest["artifacts"]:
        artifact_path = Path(artifact["path"])
        parts = artifact_path.parts
        if not parts or parts[0] != "processed":
            raise ValueError(f"unexpected processed path: {artifact_path}")
        paths.append(processed_dir.joinpath(*parts[1:]))
    return paths


def main() -> None:
    args = parse_args()
    if (args.output.exists() or args.manifest.exists()) and not args.overwrite:
        raise FileExistsError("selection exists; pass --overwrite to replace it")
    manifest = load_json(args.processed_manifest)
    profile = load_json(args.profile)
    seed = profile["seed"] if args.seed is None else args.seed
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in processed_paths(manifest, args.processed_dir):
        for record in load_jsonl(path):
            by_dataset[record["dataset"]].append(record)
    missing = set(DATASET_NAMES) - by_dataset.keys()
    if missing:
        raise ValueError(f"processed suite is missing {sorted(missing)}")

    selected = []
    counts = {}
    for dataset in DATASET_NAMES:
        records = by_dataset[dataset]
        limit = (
            len(records)
            if args.mode == "full"
            else (
                1
                if args.mode == "smoke"
                else int(profile["limits"][dataset])
            )
        )
        chosen = balanced_selection(records, limit, seed)
        counts[dataset] = {
            "available": len(records),
            "selected": len(chosen),
        }
        selected.extend(chosen)
    selected.sort(key=lambda row: (DATASET_NAMES.index(row["dataset"]), row["id"]))
    for order, record in enumerate(selected):
        record["_selection"] = {
            "order": order,
            "profile": args.mode,
            "seed": seed,
            "group": list(group_key(record)),
            "track": (
                "verifier_state"
                if record["dataset"] == "processbench"
                else "solver_state"
            ),
        }
    count = atomic_jsonl(args.output, selected)
    atomic_json(
        args.manifest,
        {
            "schema_version": "stage-a-selection-manifest-v1",
            "mode": args.mode,
            "profile": profile["name"],
            "seed": seed,
            "processed_manifest": str(args.processed_manifest),
            "processed_manifest_sha256": sha256(args.processed_manifest),
            "selection_path": str(args.output),
            "selection_sha256": sha256(args.output),
            "total_records": count,
            "counts": counts,
        },
    )
    print(f"Wrote {count} records to {args.output}")
    for dataset in DATASET_NAMES:
        item = counts[dataset]
        print(f"  {dataset}: {item['selected']}/{item['available']}")
    print(f"Manifest: {args.manifest}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
