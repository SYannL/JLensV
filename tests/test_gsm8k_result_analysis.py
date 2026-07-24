import json

from analyze_gsm8k_lens_results import (
    _align_steps,
    _equation_audit,
    _reasoning_text_and_tokens,
    _split_text_steps,
    _summarize_llm_diagnoses,
)


def test_split_text_steps_preserves_order_and_extracts_math():
    steps = _split_text_steps("First compute 3 + 4 = 7.\nThen divide 14 / 2 = 7.")

    assert [step.index for step in steps] == [0, 1]
    assert steps[0].numbers == ["3", "4", "7"]
    assert steps[0].operations == ["add"]
    assert steps[1].operations == ["divide"]


def test_step_alignment_is_monotonic_and_finds_numeric_matches():
    generated = _split_text_steps(
        "There are 3 groups of 4, making 12.\nHalf of 12 is 6."
    )
    reference = _split_text_steps(
        "Multiply 3 by 4 to obtain 12.\nDivide 12 by 2 to obtain 6."
    )

    alignment = _align_steps(generated, reference)

    assert [(pair["generated_step"], pair["reference_step"]) for pair in alignment["pairs"]] == [
        (0, 0),
        (1, 1),
    ]
    assert alignment["coverage_reference"] == 1.0


def test_equation_audit_detects_valid_and_invalid_arithmetic():
    audit = _equation_audit("We check 3 + 4 = 7. But 8 / 2 = 5 is wrong.")

    assert audit["n_checked"] == 2
    assert audit["n_invalid"] == 1
    assert [item["valid"] for item in audit["equations"]] == [True, False]


def test_equation_audit_handles_latex_multiplication_and_currency():
    audit = _equation_audit(
        r"First: $1000 \times 0.30 = \$300$. "
        r"Then: $1500 \cdot 0.10 = \$150$."
    )

    assert audit["n_checked"] == 2
    assert audit["n_invalid"] == 0


def test_nonthinking_analysis_uses_final_response_instead_of_empty_thinking():
    generation = {
        "generation": {
            "thinking_enabled": False,
            "tokens": ["unused"],
        },
        "model_output": {
            "thinking": {"text": "", "tokens": []},
            "final_response": {
                "text": "Compute 2 + 3 = 5.",
                "tokens": ["Compute", " 2", " +", " 3", " =", " 5", "."],
            },
        },
        "prompt": {"n_tokens": 10},
    }

    text, tokens, base_position, phase = _reasoning_text_and_tokens(generation)

    assert text == "Compute 2 + 3 = 5."
    assert tokens == ["Compute", " 2", " +", " 3", " =", " 5", "."]
    assert base_position == 10
    assert phase == "final_response"


def test_llm_diagnoses_are_joined_to_completion_outcomes(tmp_path):
    diagnosis = {
        "case_id": "wrong-0002",
        "wrong_id": "2",
        "first_problematic_generated_step": 3,
        "first_problematic_reference_step": 1,
        "primary_error_type": "arithmetic",
        "secondary_error_types": ["self_correction_failure"],
        "evidence": ["generated step 3 states 3 + 4 = 8"],
        "jlens_interpretation": "The readout accompanies the textual error.",
        "confidence": 0.9,
        "summary": "An arithmetic error was not corrected.",
    }
    path = tmp_path / "diagnoses.jsonl"
    path.write_text(json.dumps(diagnosis) + "\n", encoding="utf-8")
    cases = [
        {
            "wrong_id": "2",
            "behavior": {"complete": True, "correct": False},
        }
    ]

    summary = _summarize_llm_diagnoses(path, cases)

    assert summary["primary_error_counts"] == {"arithmetic": 1}
    assert summary["primary_errors_by_outcome"] == {
        "complete_incorrect": {"arithmetic": 1}
    }
