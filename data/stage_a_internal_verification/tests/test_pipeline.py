from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SUITE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SUITE_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from _dataset_common import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    base_record,
    relative_artifact,
    validate_manifest,
    validate_record,
)
from process_datasets import process_processbench, process_stepgame  # noqa: E402


class PipelineTests(unittest.TestCase):
    def test_record_and_manifest_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "processed" / "sample.jsonl"
            record = base_record(
                example_id="math500/example",
                dataset="math500",
                source_split="test",
                task_family="competition_mathematics",
            )
            record["input"]["question"] = "What is 1 + 1?"
            record["gold"]["answer"] = "2"
            record["capabilities"]["outcome_verification"] = True
            validate_record(record)
            count = atomic_write_jsonl(output, [record])
            artifact = relative_artifact(
                output,
                root,
                records=count,
                dataset="math500",
            )
            manifest = root / "manifests" / "processed_manifest.json"
            atomic_write_json(
                manifest,
                {
                    "schema_version": "stage-a-processed-manifest-v1",
                    "artifacts": [artifact],
                    "total_records": 1,
                },
            )
            self.assertEqual(
                validate_manifest(
                    manifest,
                    processed=True,
                    expected_schema="stage-a-processed-manifest-v1",
                ),
                (1, 1),
            )

    def test_processbench_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            processed = root / "processed"
            source = raw / "processbench" / "math.jsonl"
            atomic_write_jsonl(
                source,
                [
                    {
                        "_source_index": 7,
                        "id": "sample-7",
                        "problem": "Compute 2 + 2.",
                        "steps": ["2 + 2 = 5"],
                        "label": 0,
                        "final_answer_correct": False,
                        "generator": "fixture",
                    }
                ],
            )
            artifacts = process_processbench(
                raw,
                processed,
                {
                    "splits": ["math"],
                    "repo_id": "Qwen/ProcessBench",
                    "revision": "fixture",
                    "license": "Apache-2.0",
                },
                overwrite=False,
            )
            self.assertEqual(artifacts[0]["records"], 1)
            record = json.loads(
                (processed / "processbench" / "math.jsonl")
                .read_text(encoding="utf-8")
                .strip()
            )
            self.assertEqual(record["gold"]["first_error_step"], 0)
            self.assertFalse(record["gold"]["process_correct"])
            self.assertEqual(record["metadata"]["source_index"], 7)

    def test_stepgame_sampling_is_stratified_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "raw"
            source = raw / "stepgame" / "test.jsonl"
            rows = []
            for hop in range(1, 11):
                for duplicate in range(2):
                    rows.append(
                        {
                            "_source_index": hop * 10 + duplicate,
                            "k_hop": hop,
                            "story": [f"A is left of B at hop {hop}."],
                            "question": "Where is A relative to B?",
                            "label": "left",
                        }
                    )
            atomic_write_jsonl(source, rows)
            spec = {
                "repo_id": "ZhengyanShi/StepGame",
                "revision": "fixture",
                "license": "fixture",
            }
            first = root / "processed-first"
            second = root / "processed-second"
            process_stepgame(
                raw,
                first,
                spec,
                per_hop=1,
                seed=42,
                overwrite=False,
            )
            process_stepgame(
                raw,
                second,
                spec,
                per_hop=1,
                seed=42,
                overwrite=False,
            )
            first_bytes = (
                first / "stepgame" / "test_stratified_1_per_hop.jsonl"
            ).read_bytes()
            second_bytes = (
                second / "stepgame" / "test_stratified_1_per_hop.jsonl"
            ).read_bytes()
            self.assertEqual(first_bytes, second_bytes)
            self.assertEqual(len(first_bytes.splitlines()), 10)


if __name__ == "__main__":
    unittest.main()
