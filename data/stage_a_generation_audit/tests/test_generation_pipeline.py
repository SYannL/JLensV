from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

SUITE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUITE_ROOT / "scripts"))

from _common import (  # noqa: E402
    atomic_json,
    atomic_npz,
    effective_seed,
    extract_final_answer,
    grade_answer,
    sample_directory,
    sha256,
)
from audit_generation import inspect_sample  # noqa: E402
from create_selection import balanced_selection  # noqa: E402
from run_generation import (  # noqa: E402
    SYSTEM_PROMPT,
    task_prompt,
    validate_prompt_independence,
)


def record(
    identifier: str,
    dataset: str,
    *,
    group: object = "group",
) -> dict:
    gold_answer: object = "left"
    first_error = None
    if dataset == "processbench":
        gold_answer = None
        first_error = 1
    return {
        "schema_version": "internal-verification-v1",
        "id": identifier,
        "dataset": dataset,
        "source_split": "test",
        "task_family": "fixture",
        "input": {"context": "", "question": "fixture", "choices": []},
        "gold": {
            "answer": gold_answer,
            "reasoning_steps": [],
            "first_error_step": first_error,
            "process_correct": first_error is None,
            "final_answer_correct": False,
            "state_trace": [],
        },
        "candidate": {
            "answer": None,
            "reasoning_steps": ["one", "two"],
            "generator": "fixture",
        },
        "capabilities": {},
        "metadata": {"k_hop": group},
        "_selection": {
            "order": 0,
            "profile": "pilot",
            "seed": 7,
            "group": [group],
            "track": (
                "verifier_state"
                if dataset == "processbench"
                else "solver_state"
            ),
        },
    }


class GenerationPipelineTests(unittest.TestCase):
    def test_final_parser_and_dataset_scorers(self) -> None:
        value, method = extract_final_answer("Reasoning\nFINAL: upper-left\n")
        self.assertEqual((value, method), ("upper-left", "final_line"))
        stepgame = record("stepgame/1", "stepgame")
        stepgame["gold"]["answer"] = "upper-left"
        self.assertTrue(grade_answer(stepgame, value, method)["correct"])
        processbench = record("processbench/1", "processbench")
        self.assertTrue(
            grade_answer(processbench, "1", "final_line")["correct"]
        )
        self.assertTrue(
            grade_answer(
                processbench,
                "[STEP_ID=1]",
                "final_line",
            )["correct"]
        )

    def test_selection_and_seed_are_deterministic(self) -> None:
        rows = [record(f"stepgame/{index}", "stepgame", group=index % 3) for index in range(20)]
        first = [row["id"] for row in balanced_selection(rows, 8, 11)]
        second = [row["id"] for row in balanced_selection(rows, 8, 11)]
        self.assertEqual(first, second)
        self.assertEqual(
            effective_seed(42, "example"),
            effective_seed(42, "example"),
        )

    def test_processbench_prompt_uses_immutable_step_ids(self) -> None:
        source = record("processbench/example", "processbench")
        track, prompt = task_prompt(source)
        self.assertEqual(track, "verifier_state")
        self.assertIn("<STEP_ID=0>", prompt)
        self.assertIn("stop immediately", prompt)
        validate_prompt_independence(source, prompt)

    def test_solver_prompts_are_minimal_and_gold_independent(self) -> None:
        math = record("math500/example", "math500")
        math["input"]["question"] = "Compute 1 + 1."
        _, prompt = task_prompt(math)
        self.assertEqual(prompt, "Compute 1 + 1.")
        validate_prompt_independence(math, prompt)
        self.assertEqual(SYSTEM_PROMPT.count("FINAL"), 1)

    def test_audit_accepts_aligned_metrics(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            source = record("stepgame/example", "stepgame", group=1)
            selection_path = run_dir / "selection.jsonl"
            selection_path.write_text("{}\n", encoding="utf-8")
            selection_digest = sha256(selection_path)
            sample_dir = sample_directory(run_dir, source)
            metrics_path = sample_dir / "token_metrics.npz"
            arrays = {
                "generated_token_ids": np.asarray([10, 11], dtype=np.int32),
                "prediction_positions": np.asarray([3, 4], dtype=np.int32),
                "raw_chosen_logprobs": np.asarray([-0.1, -0.2]),
                "raw_chosen_ranks": np.asarray([0, 1], dtype=np.int32),
                "raw_entropy": np.asarray([0.5, 0.6]),
                "raw_max_probability": np.asarray([0.8, 0.7]),
                "raw_logsumexp": np.asarray([4.0, 4.1]),
                "raw_energy": np.asarray([-4.0, -4.1]),
                "raw_top_token_ids": np.asarray([[10, 1], [1, 11]]),
                "raw_top_logits": np.asarray([[3.0, 2.0], [3.0, 2.0]]),
                "raw_top_logprobs": np.asarray([[-0.1, -1.1], [-0.1, -1.1]]),
            }
            atomic_npz(metrics_path, arrays)
            artifact = {
                "schema_version": "stage-a-generation-artifact-v1",
                "status": "complete",
                "experiment_fingerprint": "fixture",
                "source": {
                    "record": source,
                    "selection_sha256": selection_digest,
                },
                "prompt": {"n_tokens": 4},
                "generation": {"n_tokens": 2, "token_ids": [10, 11]},
                "token_metrics": {"sha256": sha256(metrics_path)},
                "grading": {
                    "parsed": True,
                    "correct": True,
                    "reliability": "deterministic_exact",
                },
                "quality": {
                    "ended_with_eos": True,
                    "hit_max_new_tokens": False,
                    "format_valid": True,
                    "complete": True,
                    "raw_metrics_complete": True,
                    "replay_ready": True,
                },
            }
            atomic_json(sample_dir / "generation.json", artifact)
            result = inspect_sample(run_dir, source, selection_digest)
            self.assertEqual(result["artifact_status"], "valid")
            self.assertTrue(result["eligible_for_internal_analysis"])


if __name__ == "__main__":
    unittest.main()
