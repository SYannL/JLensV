#!/usr/bin/env python3
"""Audit Stage A generation artifacts and build analysis-ready summaries."""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from _common import (
    ARTIFACT_SCHEMA,
    DEFAULT_SELECTION,
    atomic_json,
    atomic_jsonl,
    atomic_text,
    load_json,
    load_jsonl,
    sample_directory,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit nonzero unless every selected artifact is valid and eligible "
            "for internal analysis."
        ),
    )
    return parser.parse_args()


def finite_mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else None


def inspect_sample(
    run_dir: Path,
    record: dict[str, Any],
    selection_digest: str,
) -> dict[str, Any]:
    sample_dir = sample_directory(run_dir, record)
    generation_path = sample_dir / "generation.json"
    metrics_path = sample_dir / "token_metrics.npz"
    error_path = sample_dir / "error.json"
    base = {
        "id": record["id"],
        "dataset": record["dataset"],
        "source_split": record["source_split"],
        "track": record["_selection"]["track"],
        "sample_dir": sample_dir.name,
        "selection_group": record["_selection"]["group"],
    }
    if not generation_path.exists():
        base["artifact_status"] = "error" if error_path.exists() else "missing"
        if error_path.exists():
            error = load_json(error_path)
            base["error_type"] = error.get("type")
            base["error_message"] = error.get("message")
        return base
    try:
        artifact = load_json(generation_path)
        if artifact.get("schema_version") != ARTIFACT_SCHEMA:
            raise ValueError("unexpected generation artifact schema")
        if artifact["source"]["record"]["id"] != record["id"]:
            raise ValueError("source record ID mismatch")
        if artifact["source"]["selection_sha256"] != selection_digest:
            raise ValueError("selection checksum mismatch")
        if not metrics_path.exists():
            raise FileNotFoundError("token_metrics.npz is missing")
        if sha256(metrics_path) != artifact["token_metrics"]["sha256"]:
            raise ValueError("token metrics checksum mismatch")
        with np.load(metrics_path, allow_pickle=False) as arrays:
            required = {
                "generated_token_ids",
                "prediction_positions",
                "raw_chosen_logprobs",
                "raw_chosen_ranks",
                "raw_entropy",
                "raw_max_probability",
                "raw_logsumexp",
                "raw_energy",
                "raw_top_token_ids",
                "raw_top_logits",
                "raw_top_logprobs",
            }
            missing = required - set(arrays.files)
            if missing:
                raise ValueError(f"missing metric arrays: {sorted(missing)}")
            n_tokens = artifact["generation"]["n_tokens"]
            for name in required:
                if len(arrays[name]) != n_tokens:
                    raise ValueError(f"{name} length mismatch")
            generated_ids = arrays["generated_token_ids"]
            if generated_ids.tolist() != artifact["generation"]["token_ids"]:
                raise ValueError("generated token IDs mismatch")
            chosen_logprobs = arrays["raw_chosen_logprobs"]
            mean_logprob = finite_mean(chosen_logprobs)
            mean_nll = -mean_logprob if mean_logprob is not None else None
            mean_entropy = finite_mean(arrays["raw_entropy"])
            mean_max_probability = finite_mean(arrays["raw_max_probability"])
            mean_energy = finite_mean(arrays["raw_energy"])
            top1_rate = float((arrays["raw_chosen_ranks"] == 0).mean())
        quality = artifact["quality"]
        grading = artifact["grading"]
        source = artifact["source"]["record"]
        base.update(
            {
                "artifact_status": "valid",
                "experiment_fingerprint": artifact["experiment_fingerprint"],
                "n_prompt_tokens": artifact["prompt"]["n_tokens"],
                "n_generated_tokens": artifact["generation"]["n_tokens"],
                "ended_with_eos": quality["ended_with_eos"],
                "hit_max_new_tokens": quality["hit_max_new_tokens"],
                "format_valid": quality["format_valid"],
                "complete": quality["complete"],
                "parsed": grading["parsed"],
                "correct": grading["correct"],
                "scorer_reliability": grading["reliability"],
                "raw_metrics_complete": quality["raw_metrics_complete"],
                "replay_ready": quality["replay_ready"],
                "mean_raw_nll": mean_nll,
                "raw_perplexity": (
                    math.exp(min(mean_nll, 700.0))
                    if mean_nll is not None
                    else None
                ),
                "mean_raw_entropy": mean_entropy,
                "mean_raw_max_probability": mean_max_probability,
                "mean_raw_energy": mean_energy,
                "raw_chosen_top1_rate": top1_rate,
                "eligible_for_internal_analysis": bool(
                    quality["complete"]
                    and quality["raw_metrics_complete"]
                    and quality["replay_ready"]
                    and grading["parsed"]
                ),
            }
        )
        if source["dataset"] == "stepgame":
            base["task_group"] = f"hop-{source['metadata']['k_hop']}"
        elif source["dataset"] == "bbeh":
            base["task_group"] = source["metadata"]["source_task"]
        elif source["dataset"] == "math500":
            base["task_group"] = (
                f"{source['metadata']['subject']}/"
                f"level-{source['metadata']['level']}"
            )
        elif source["dataset"] == "processbench":
            base["task_group"] = source["source_split"]
        else:
            base["task_group"] = source["task_family"]
    except Exception as exc:
        base.update(
            {
                "artifact_status": "invalid",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
        )
    return base


def aggregate(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key, "unknown"))].append(row)
    result = {}
    for name, items in sorted(groups.items()):
        valid = [row for row in items if row["artifact_status"] == "valid"]
        scored = [
            row
            for row in valid
            if row.get("complete") and row.get("parsed")
        ]
        correct = [row for row in scored if row.get("correct")]
        eligible = [
            row for row in valid if row.get("eligible_for_internal_analysis")
        ]
        complete = [row for row in valid if row.get("complete")]
        result[name] = {
            "selected": len(items),
            "valid_artifacts": len(valid),
            "generation_complete": len(complete),
            "parsed": len(scored),
            "correct": len(correct),
            "incorrect": len(scored) - len(correct),
            "eligible_for_internal_analysis": len(eligible),
            "coverage": len(valid) / len(items) if items else None,
            "completion_rate": len(complete) / len(valid) if valid else None,
            "accuracy": len(correct) / len(scored) if scored else None,
        }
    return result


def percentage(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Stage A generation audit",
        "",
        f"Generated: {report['timestamp_utc']}",
        "",
        "## Overall readiness",
        "",
        f"- Selected: {report['overall']['selected']}",
        f"- Valid artifacts: {report['overall']['valid_artifacts']}",
        f"- Missing/error/invalid: {report['overall']['invalid_or_missing']}",
        f"- Generation-complete: {report['overall']['generation_complete']}",
        (
            "- Eligible for internal analysis: "
            f"{report['overall']['eligible_for_internal_analysis']}"
        ),
        "",
        "## Dataset summary",
        "",
        "| Dataset | Selected | Valid | Complete | Complete+parsed | Correct | Incorrect | Eligible | Accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset, item in report["by_dataset"].items():
        lines.append(
            f"| {dataset} | {item['selected']} | {item['valid_artifacts']} | "
            f"{item['generation_complete']} | {item['parsed']} | "
            f"{item['correct']} | {item['incorrect']} | "
            f"{item['eligible_for_internal_analysis']} | "
            f"{percentage(item['accuracy'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- MATH-500 mismatches are provisional until checked with a symbolic/official math verifier.",
            "- ProcessBench is the verifier-state track; its candidate steps are not solver-native activations.",
            "- Internal analysis must replay the saved full token IDs and must not resample trajectories.",
            "- JLens should initially run only on complete, parsed, analysis-eligible records and a balanced correct/incorrect subset.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    selection = load_jsonl(args.selection)
    selection_digest = sha256(args.selection)
    rows = [
        inspect_sample(args.run_dir, record, selection_digest)
        for record in selection
    ]
    rows.sort(key=lambda row: (row["dataset"], row["id"]))
    atomic_jsonl(args.run_dir / "audit_summary.jsonl", rows)
    by_dataset = aggregate(rows, "dataset")
    by_track = aggregate(rows, "track")
    valid = [row for row in rows if row["artifact_status"] == "valid"]
    invalid = [row for row in rows if row["artifact_status"] != "valid"]
    complete = [row for row in valid if row.get("complete")]
    eligible = [
        row for row in valid if row.get("eligible_for_internal_analysis")
    ]
    report = {
        "schema_version": "stage-a-generation-audit-v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selection_sha256": selection_digest,
        "overall": {
            "selected": len(rows),
            "valid_artifacts": len(valid),
            "invalid_or_missing": len(invalid),
            "generation_complete": len(complete),
            "eligible_for_internal_analysis": len(eligible),
        },
        "by_dataset": by_dataset,
        "by_track": by_track,
        "invalid_records": [
            {
                "id": row["id"],
                "status": row["artifact_status"],
                "error": row.get("error_message"),
            }
            for row in invalid
        ],
    }
    report_path = args.run_dir / "audit_report.json"
    atomic_json(report_path, report)
    markdown_path = args.run_dir / "audit_report.md"
    atomic_text(markdown_path, markdown_report(report))
    atomic_json(
        args.run_dir / "audit_manifest.json",
        {
            "schema_version": "stage-a-audit-manifest-v1",
            "selection_sha256": selection_digest,
            "summary_sha256": sha256(args.run_dir / "audit_summary.jsonl"),
            "report_json_sha256": sha256(report_path),
            "report_markdown_sha256": sha256(markdown_path),
        },
    )
    print(markdown_report(report))
    not_ready = [
        row
        for row in rows
        if not row.get("eligible_for_internal_analysis", False)
    ]
    if args.strict and not_ready:
        raise RuntimeError(
            f"{len(not_ready)} selected artifacts are not analysis-ready"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
