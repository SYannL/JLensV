#!/usr/bin/env python3
"""Analyze GSM8K generations and their saved Jacobian-Lens traces.

This is a post-processing script: it never loads the language model or runs
generation. It accepts complete, partial, generation-only, and legacy result
directories. The main artifact is a reasoning-step x layer lexical readout,
aligned monotonically to the official GSM8K rationale.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
WORD_RE = re.compile(r"[A-Za-z]+")
SELF_CORRECTION_RE = re.compile(
    r"\b(?:wait|actually|recheck|double-check|correction|mistake|rather|instead)\b",
    re.IGNORECASE,
)
UNCERTAINTY_RE = re.compile(
    r"\b(?:maybe|perhaps|possibly|likely|unsure|uncertain|seems?)\b",
    re.IGNORECASE,
)
ERROR_TYPES = (
    "comprehension",
    "plan",
    "quantity_tracking",
    "arithmetic",
    "unit",
    "unsupported_inference",
    "self_correction_failure",
    "finalization_format",
    "truncation",
    "no_error",
    "uncertain",
)
LLM_SYSTEM_PROMPT = """Diagnose one GSM8K reasoning trace using only the supplied
case packet. Compare the model steps to the official steps, identify the first
supported divergence, and distinguish truncation/finalization failures from
reasoning failures. J-lens tokens are lexical readouts of representational
availability: report whether they precede, accompany, or follow textual
evidence, but never infer causality from them. Return only JSON matching the
provided schema."""
LLM_OUTPUT_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "case_id",
        "wrong_id",
        "first_problematic_generated_step",
        "first_problematic_reference_step",
        "primary_error_type",
        "secondary_error_types",
        "evidence",
        "jlens_interpretation",
        "confidence",
        "summary",
    ],
    "properties": {
        "case_id": {"type": "string"},
        "wrong_id": {"type": "string"},
        "first_problematic_generated_step": {
            "anyOf": [{"type": "integer"}, {"type": "null"}]
        },
        "first_problematic_reference_step": {
            "anyOf": [{"type": "integer"}, {"type": "null"}]
        },
        "primary_error_type": {"type": "string", "enum": list(ERROR_TYPES)},
        "secondary_error_types": {
            "type": "array",
            "items": {"type": "string", "enum": list(ERROR_TYPES)},
        },
        "evidence": {"type": "array", "items": {"type": "string"}},
        "jlens_interpretation": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "summary": {"type": "string"},
    },
}


@dataclass
class TextStep:
    index: int
    text: str
    start: int
    end: int
    numbers: list[str]
    operations: list[str]
    positions: list[int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Result directory produced by analyze_gsm8k_hard_with_lens.py",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: INPUT_DIR/analysis",
    )
    parser.add_argument(
        "--top-concepts",
        type=int,
        default=5,
        help="J-lens lexical readouts retained per reasoning step and layer",
    )
    parser.add_argument(
        "--max-concept-events",
        type=int,
        default=24,
        help="Detailed position-level events retained per case",
    )
    parser.add_argument(
        "--write-html",
        action="store_true",
        help="Write JLens's interactive position x layer viewer",
    )
    parser.add_argument(
        "--html-limit",
        type=int,
        default=10,
        help="Maximum number of interactive viewers to write",
    )
    parser.add_argument(
        "--wrong-ids",
        default=None,
        help="Optional comma-separated wrong_id subset",
    )
    parser.add_argument(
        "--llm-diagnoses",
        default=None,
        help="Optional completed diagnosis JSONL to validate and summarize",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _canonical_number(value: str) -> str:
    return value.replace("$", "").replace(",", "").strip()


def _numbers(text: str) -> list[str]:
    return [_canonical_number(value) for value in NUMBER_RE.findall(text)]


def _operations(text: str) -> list[str]:
    operations = []
    for symbol, name in (
        ("+", "add"),
        ("-", "subtract"),
        ("*", "multiply"),
        ("×", "multiply"),
        ("/", "divide"),
        ("÷", "divide"),
        ("%", "percent"),
    ):
        if symbol in text:
            operations.append(name)
    lowered = text.lower()
    for word, name in (
        ("each", "rate"),
        ("per", "rate"),
        ("twice", "multiply"),
        ("half", "divide"),
        ("remaining", "subtract"),
        ("left", "subtract"),
        ("total", "aggregate"),
    ):
        if word in lowered:
            operations.append(name)
    return sorted(set(operations))


def _split_text_steps(text: str) -> list[TextStep]:
    """Split text into sentence/line-sized reasoning units with char spans."""
    boundaries = [0]
    for match in re.finditer(r"\n+|(?<=[.!?])\s+(?=[A-Z#])", text):
        boundaries.extend((match.start(), match.end()))
    boundaries.append(len(text))
    spans: list[tuple[int, int]] = []
    for start, end in zip(boundaries[::2], boundaries[1::2], strict=False):
        if start < end:
            spans.append((start, end))
    if not spans and text.strip():
        spans = [(0, len(text))]

    steps = []
    for start, end in spans:
        raw = text[start:end]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        clean_start = start + left
        clean_end = start + right
        value = text[clean_start:clean_end]
        if not value:
            continue
        steps.append(
            TextStep(
                index=len(steps),
                text=value,
                start=clean_start,
                end=clean_end,
                numbers=_numbers(value),
                operations=_operations(value),
                positions=[],
            )
        )
    return steps


def _word_set(text: str) -> set[str]:
    return {word.lower() for word in WORD_RE.findall(text) if len(word) > 2}


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _step_similarity(generated: TextStep, reference: TextStep) -> float:
    numeric = _jaccard(set(generated.numbers), set(reference.numbers))
    lexical = _jaccard(_word_set(generated.text), _word_set(reference.text))
    operation = _jaccard(set(generated.operations), set(reference.operations))
    return 0.55 * numeric + 0.30 * lexical + 0.15 * operation


def _align_steps(
    generated: list[TextStep],
    reference: list[TextStep],
) -> dict[str, Any]:
    """Needleman-Wunsch alignment preserving reasoning-step order."""
    n_generated = len(generated)
    n_reference = len(reference)
    gap = -0.20
    scores = np.zeros((n_generated + 1, n_reference + 1), dtype=np.float64)
    moves = np.zeros((n_generated + 1, n_reference + 1), dtype=np.int8)
    scores[:, 0] = np.arange(n_generated + 1) * gap
    scores[0, :] = np.arange(n_reference + 1) * gap
    moves[1:, 0] = 1
    moves[0, 1:] = 2

    similarities = np.zeros((n_generated, n_reference), dtype=np.float64)
    for i, generated_step in enumerate(generated):
        for j, reference_step in enumerate(reference):
            similarities[i, j] = _step_similarity(generated_step, reference_step)
            match_value = similarities[i, j] - 0.10
            candidates = (
                scores[i, j] + match_value,
                scores[i, j + 1] + gap,
                scores[i + 1, j] + gap,
            )
            move = int(np.argmax(candidates))
            scores[i + 1, j + 1] = candidates[move]
            moves[i + 1, j + 1] = move

    pairs: list[dict[str, Any]] = []
    unmatched_generated: list[int] = []
    unmatched_reference: list[int] = []
    i, j = n_generated, n_reference
    while i or j:
        move = int(moves[i, j])
        if i and j and move == 0:
            similarity = float(similarities[i - 1, j - 1])
            if similarity >= 0.12:
                pairs.append(
                    {
                        "generated_step": i - 1,
                        "reference_step": j - 1,
                        "similarity": round(similarity, 4),
                    }
                )
            else:
                unmatched_generated.append(i - 1)
                unmatched_reference.append(j - 1)
            i -= 1
            j -= 1
        elif i and (not j or move == 1):
            unmatched_generated.append(i - 1)
            i -= 1
        else:
            unmatched_reference.append(j - 1)
            j -= 1

    pairs.reverse()
    unmatched_generated.reverse()
    unmatched_reference.reverse()
    return {
        "pairs": pairs,
        "unmatched_generated_steps": unmatched_generated,
        "unmatched_reference_steps": unmatched_reference,
        "coverage_generated": (
            len(pairs) / n_generated if n_generated else 0.0
        ),
        "coverage_reference": (
            len(pairs) / n_reference if n_reference else 0.0
        ),
        "mean_similarity": (
            float(np.mean([pair["similarity"] for pair in pairs]))
            if pairs
            else 0.0
        ),
    }


def _safe_arithmetic(expression: str) -> float:
    operators = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.USub: lambda a: -a,
        ast.UAdd: lambda a: a,
    }

    def evaluate(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in operators:
            return operators[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in operators:
            return operators[type(node.op)](evaluate(node.operand))
        raise ValueError("unsupported arithmetic")

    return evaluate(ast.parse(expression, mode="eval"))


def _equation_audit(text: str) -> dict[str, Any]:
    checked = []
    text = (
        text.replace("\\times", "*")
        .replace("\\cdot", "*")
        .replace("×", "*")
        .replace("\\$", "")
        .replace("$", "")
    )
    pattern = re.compile(
        r"(?<![<>=])(\(?[-+*/().\d\s]+\d\)?)\s*=\s*"
        r"([-+]?\d+(?:\.\d+)?)(?![=])"
    )
    for match in pattern.finditer(text):
        expression = match.group(1).strip()
        claimed = float(match.group(2))
        if not any(symbol in expression for symbol in "+-*/"):
            continue
        try:
            calculated = _safe_arithmetic(expression)
        except (SyntaxError, ValueError, ZeroDivisionError):
            continue
        checked.append(
            {
                "expression": expression,
                "claimed": claimed,
                "calculated": calculated,
                "valid": math.isclose(calculated, claimed, rel_tol=1e-9, abs_tol=1e-9),
            }
        )
    return {
        "n_checked": len(checked),
        "n_invalid": sum(not item["valid"] for item in checked),
        "equations": checked,
    }


def _repeated_ngram_ratio(text: str, n: int = 4) -> float:
    words = [word.lower() for word in WORD_RE.findall(text)]
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[index : index + n]) for index in range(len(words) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def _repeated_step_signature_ratio(steps: list[TextStep]) -> float:
    signatures = [
        (tuple(step.numbers), tuple(step.operations))
        for step in steps
        if step.numbers or step.operations
    ]
    if not signatures:
        return 0.0
    return 1.0 - len(set(signatures)) / len(signatures)


def _reasoning_text_and_tokens(
    generation: dict[str, Any],
) -> tuple[str, list[str], int, str]:
    generated = generation["generation"]
    model_output = generation.get("model_output", generated)
    prompt_length = generation["prompt"]["n_tokens"]
    thinking = model_output.get("thinking")
    thinking_enabled = bool(generated.get("thinking_enabled", False))
    if thinking_enabled and thinking is not None:
        return (
            thinking["text"],
            thinking["tokens"],
            prompt_length,
            "thinking",
        )
    final_response = model_output.get("final_response")
    if final_response is not None:
        return (
            final_response["text"],
            final_response["tokens"],
            prompt_length,
            "final_response",
        )
    return (
        model_output["text"],
        generated["tokens"],
        prompt_length,
        "response_unsegmented",
    )


def _steps_with_token_positions(
    generation: dict[str, Any],
) -> tuple[list[TextStep], str]:
    text, token_strings, base_position, phase = _reasoning_text_and_tokens(generation)
    reconstructed = "".join(token_strings)
    # Individual-token decoding is preferable because it maps exactly to saved
    # absolute positions. It can retain a special token in legacy artifacts.
    if reconstructed.strip():
        text = reconstructed
    steps = _split_text_steps(text)
    token_spans = []
    cursor = 0
    for offset, token in enumerate(token_strings):
        token_spans.append((cursor, cursor + len(token), base_position + offset))
        cursor += len(token)
    for step in steps:
        step.positions = [
            position
            for start, end, position in token_spans
            if start < step.end and end > step.start
        ]
    return steps, phase


def _reference_steps(generation: dict[str, Any]) -> list[TextStep]:
    rationale = generation["source"]["canonical_gsm8k"]["answer"]
    rationale = rationale.split("####", 1)[0].strip()
    return _split_text_steps(rationale)


def _phase_by_position(generation: dict[str, Any]) -> dict[int, str]:
    prompt_length = generation["prompt"]["n_tokens"]
    phases = {
        position: "prompt"
        for position in range(prompt_length)
    }
    for step in generation["generation"].get("steps", []):
        phases[int(step["token_position"])] = step.get(
            "segment", "response_unsegmented"
        )
    return phases


def _behavior_metrics(
    generation: dict[str, Any],
    generated_steps: list[TextStep],
    reference_steps: list[TextStep],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    generated = generation["generation"]
    model_output = generation.get("model_output", generated)
    reasoning_text, _, _, _ = _reasoning_text_and_tokens(generation)
    final_text = model_output.get("final_response", {}).get("text", "")
    official = generation["source"]["canonical_gsm8k"]["answer"]
    question = generation["source"]["csv_row"]["question"]
    generated_numbers = set(_numbers(reasoning_text))
    reference_numbers = set(_numbers(official))
    question_numbers = set(_numbers(question))
    equation_audit = _equation_audit(reasoning_text)
    quality = generation.get("quality", {})
    terminal_tokens = {"<|endoftext|>", "<|im_end|>"}
    inferred_eos = bool(
        generated.get("tokens")
        and generated["tokens"][-1] in terminal_tokens
    )
    ended_with_eos = generated.get("ended_with_eos", False) or inferred_eos
    hit_max_new_tokens = generated.get(
        "hit_max_new_tokens",
        generated["n_tokens"] >= generated["max_new_tokens"] and not ended_with_eos,
    )
    thinking_enabled = generated.get(
        "thinking_enabled",
        generation["prompt"]["rendered_text"].rstrip().endswith("<think>"),
    )
    generated_token_strings = generated.get("tokens", [])
    inferred_thinking_closed = (
        not thinking_enabled or "</think>" in generated_token_strings
    )
    thinking_closed = quality.get(
        "thinking_closed",
        inferred_thinking_closed,
    )
    valid_final_hash = quality.get(
        "valid_final_hash_answer",
        generation["grading"].get("parse_method") == "hash_delimiter"
        and thinking_closed
        and ended_with_eos,
    )
    complete = bool(ended_with_eos and thinking_closed and valid_final_hash)
    token_steps = generated.get("steps", [])
    chosen_logprobs = [
        float(step["logprob"])
        for step in token_steps
        if step.get("logprob") is not None
        and math.isfinite(float(step["logprob"]))
    ]
    chosen_top1 = [
        step["token_id"] == step["top_token_ids"][0]
        for step in token_steps
        if step.get("top_token_ids")
    ]

    return {
        "generation_tokens": generated["n_tokens"],
        "thinking_tokens": model_output.get("thinking", {}).get("n_tokens"),
        "final_tokens": model_output.get("final_response", {}).get("n_tokens"),
        "ended_with_eos": ended_with_eos,
        "hit_max_new_tokens": hit_max_new_tokens,
        "thinking_closed": thinking_closed,
        "valid_final_hash_answer": valid_final_hash,
        "complete": complete,
        "correct": generation["grading"]["correct"],
        "generated_step_count": len(generated_steps),
        "reference_step_count": len(reference_steps),
        "alignment_generated_coverage": alignment["coverage_generated"],
        "alignment_reference_coverage": alignment["coverage_reference"],
        "alignment_mean_similarity": alignment["mean_similarity"],
        "numeric_reference_recall": (
            len(generated_numbers & reference_numbers) / len(reference_numbers)
            if reference_numbers
            else 0.0
        ),
        "numeric_generated_precision": (
            len(generated_numbers & reference_numbers) / len(generated_numbers)
            if generated_numbers
            else 0.0
        ),
        "novel_number_count": len(
            generated_numbers - reference_numbers - question_numbers
        ),
        "self_correction_markers": len(SELF_CORRECTION_RE.findall(reasoning_text)),
        "uncertainty_markers": len(UNCERTAINTY_RE.findall(reasoning_text)),
        "repeated_4gram_ratio": _repeated_ngram_ratio(reasoning_text),
        "repeated_step_signature_ratio": _repeated_step_signature_ratio(
            generated_steps
        ),
        "mean_chosen_surprisal": (
            float(np.mean([-value for value in chosen_logprobs]))
            if chosen_logprobs
            else None
        ),
        "chosen_top1_rate": (
            float(np.mean(chosen_top1)) if chosen_top1 else None
        ),
        "equations_checked": equation_audit["n_checked"],
        "invalid_equations": equation_audit["n_invalid"],
        "equation_audit": equation_audit["equations"],
        "final_response": final_text,
    }


def _token_decoder(metadata: dict[str, Any]) -> Any:
    vocab = metadata.get("top_vocab", {})
    tracked = metadata.get("tracked_tokens", {})

    def decode(token_id: int) -> str:
        key = str(int(token_id))
        if key in vocab:
            return vocab[key]
        if key in tracked:
            return tracked[key]["token"]
        return f"<id:{token_id}>"

    return decode


def _is_meaningful_readout(token: str) -> bool:
    stripped = token.strip()
    return bool(stripped) and "<|" not in stripped and any(
        character.isalnum() for character in stripped
    )


def _gold_tracked_indices(metadata: dict[str, Any], tracked_ids: np.ndarray) -> list[int]:
    result = []
    tracked_meta = metadata.get("tracked_tokens", {})
    for index, token_id in enumerate(tracked_ids.tolist()):
        token_meta = tracked_meta.get(str(int(token_id)), {})
        provenance = token_meta.get("provenance", [])
        if NUMBER_RE.search(token_meta.get("token", "")) and any(
            item.get("category") == "gold" for item in provenance
        ):
            result.append(index)
    return result


def _mean_topk_entropy(top_logits: np.ndarray) -> np.ndarray:
    values = top_logits.astype(np.float64)
    values -= values.max(axis=-1, keepdims=True)
    probabilities = np.exp(values)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return -(probabilities * np.log(probabilities + 1e-30)).sum(axis=-1).mean(axis=0)


def _lens_metrics(
    metadata: dict[str, Any],
    arrays: Any,
    generation: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    positions = arrays["positions"].astype(np.int64)
    layers = arrays["layers"].astype(np.int64)
    top_ids = arrays["top_token_ids"].astype(np.int64)
    top_logits = arrays["top_logits"]
    tracked_ids = arrays["tracked_token_ids"].astype(np.int64)
    tracked_ranks = arrays["tracked_ranks"].astype(np.int64)
    decoder = _token_decoder(metadata)
    phases = _phase_by_position(generation)
    response_mask = positions >= generation["prompt"]["n_tokens"]
    metric_mask = response_mask if response_mask.any() else np.ones(len(positions), bool)

    top1 = top_ids[:, :, 0]
    final_top1 = top1[:, -1]
    final_agreement = (top1 == final_top1[:, None])[metric_mask].mean(axis=0)
    adjacent_jaccard = np.ones((len(positions), len(layers)), dtype=np.float64)
    for layer_index in range(1, len(layers)):
        left = top_ids[:, layer_index - 1]
        right = top_ids[:, layer_index]
        intersection = (left[:, :, None] == right[:, None, :]).any(axis=2).sum(axis=1)
        union = left.shape[1] * 2 - intersection
        adjacent_jaccard[:, layer_index] = intersection / union

    same_as_final = top1 == final_top1[:, None]
    suffix_same = np.logical_and.accumulate(same_as_final[:, ::-1], axis=1)[:, ::-1]
    convergence_indices = suffix_same.argmax(axis=1)
    turnover = (top1[:, 1:] != top1[:, :-1]).mean(axis=1)
    entropy = _mean_topk_entropy(top_logits[metric_mask])
    gold_indices = _gold_tracked_indices(metadata, tracked_ids)

    layer_rows = []
    for layer_index, layer in enumerate(layers.tolist()):
        top1_strings = [decoder(value) for value in top1[metric_mask, layer_index]]
        numeric_rate = (
            sum(bool(NUMBER_RE.search(value)) for value in top1_strings)
            / len(top1_strings)
            if top1_strings
            else 0.0
        )
        gold_mrr = None
        gold_top100_rate = None
        if gold_indices and metric_mask.any():
            best_rank = tracked_ranks[metric_mask, layer_index][:, gold_indices].min(
                axis=1
            )
            gold_mrr = float(np.mean(1.0 / (best_rank + 1)))
            gold_top100_rate = float(np.mean(best_rank < 100))
        layer_rows.append(
            {
                "layer": int(layer),
                "final_top1_agreement": float(final_agreement[layer_index]),
                "adjacent_topk_jaccard": (
                    float(adjacent_jaccard[metric_mask, layer_index].mean())
                    if layer_index
                    else None
                ),
                "topk_entropy": float(entropy[layer_index]),
                "numeric_top1_rate": numeric_rate,
                "gold_mrr": gold_mrr,
                "gold_top100_rate": gold_top100_rate,
            }
        )

    full_ids = generation["full_token_ids"]
    full_tokens = generation["prompt"]["tokens"] + generation["generation"]["tokens"]
    position_rows = []
    for array_index, position in enumerate(positions.tolist()):
        convergence_index = int(convergence_indices[array_index])
        position_rows.append(
            {
                "array_index": array_index,
                "position": position,
                "token": (
                    full_tokens[position]
                    if position < len(full_tokens)
                    else f"<id:{full_ids[position]}>"
                ),
                "phase": phases.get(position, "response_unsegmented"),
                "predicted_next_token": (
                    full_tokens[position + 1]
                    if position + 1 < len(full_tokens)
                    else None
                ),
                "top1_turnover_rate": float(turnover[array_index]),
                "mean_adjacent_topk_jaccard": float(
                    adjacent_jaccard[array_index, 1:].mean()
                ),
                "final_top1_agreement_rate": float(
                    same_as_final[array_index].mean()
                ),
                "convergence_layer": int(layers[convergence_index]),
            }
        )

    lens_summary = {
        "available": True,
        "n_positions": len(positions),
        "n_layers": len(layers),
        "layers": layers.tolist(),
        "mean_top1_turnover_rate": float(turnover[metric_mask].mean()),
        "mean_final_top1_agreement_rate": float(
            same_as_final[metric_mask].mean()
        ),
        "median_convergence_layer": float(
            np.median(layers[convergence_indices[metric_mask]])
        ),
        "mean_adjacent_topk_jaccard": float(
            adjacent_jaccard[metric_mask, 1:].mean()
        ),
        "gold_tracked_token_ids": [
            int(tracked_ids[index]) for index in gold_indices
        ],
    }
    return lens_summary, layer_rows, position_rows


def _step_layer_concepts(
    steps: list[TextStep],
    alignment: dict[str, Any],
    metadata: dict[str, Any],
    arrays: Any,
    *,
    top_concepts: int,
    phase: str,
) -> list[dict[str, Any]]:
    positions = arrays["positions"].astype(np.int64)
    layers = arrays["layers"].astype(np.int64)
    top_ids = arrays["top_token_ids"].astype(np.int64)
    tracked_ids = arrays["tracked_token_ids"].astype(np.int64)
    tracked_ranks = arrays["tracked_ranks"].astype(np.int64)
    decoder = _token_decoder(metadata)
    position_index = {int(position): index for index, position in enumerate(positions)}
    gold_indices = _gold_tracked_indices(metadata, tracked_ids)
    alignment_by_generated = {
        pair["generated_step"]: pair for pair in alignment["pairs"]
    }
    rows = []
    for step in steps:
        indices = [
            position_index[position]
            for position in step.positions
            if position in position_index
        ]
        per_layer = []
        for layer_index, layer in enumerate(layers.tolist()):
            counter: Counter[int] = Counter()
            for position_array_index in indices:
                for rank, token_id in enumerate(
                    top_ids[position_array_index, layer_index]
                ):
                    if _is_meaningful_readout(decoder(int(token_id))):
                        counter[int(token_id)] += 1.0 / (rank + 1)
            concepts = [
                {
                    "token_id": token_id,
                    "token": decoder(token_id),
                    "score": round(score / max(1, len(indices)), 4),
                }
                for token_id, score in counter.most_common(top_concepts)
            ]
            gold_best_rank = None
            if indices and gold_indices:
                gold_best_rank = int(
                    tracked_ranks[np.ix_(indices, [layer_index], gold_indices)].min()
                )
            per_layer.append(
                {
                    "layer": int(layer),
                    "concepts": concepts,
                    "gold_best_rank": gold_best_rank,
                }
            )
        rows.append(
            {
                "generated_step": step.index,
                "phase": phase,
                "text": step.text,
                "numbers": step.numbers,
                "operations": step.operations,
                "absolute_positions": step.positions,
                "aligned_reference": alignment_by_generated.get(step.index),
                "layers": per_layer,
            }
        )
    return rows


def _concept_events(
    position_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
    arrays: Any,
    *,
    top_concepts: int,
    limit: int,
) -> list[dict[str, Any]]:
    top_ids = arrays["top_token_ids"].astype(np.int64)
    top_logits = arrays["top_logits"].astype(np.float64)
    layers = arrays["layers"].astype(np.int64)
    decoder = _token_decoder(metadata)
    response_rows = [
        row for row in position_rows if row["phase"] != "prompt"
    ]
    numeric = [
        row
        for row in response_rows
        if NUMBER_RE.search(row["token"])
        or NUMBER_RE.search(row["predicted_next_token"] or "")
    ]
    transitions = []
    previous_phase = None
    for row in response_rows:
        if row["phase"] != previous_phase:
            transitions.append(row)
        previous_phase = row["phase"]
    high_turnover = sorted(
        response_rows,
        key=lambda row: row["top1_turnover_rate"],
        reverse=True,
    )
    selected = []
    seen = set()
    for row in numeric + transitions + high_turnover:
        if row["position"] in seen:
            continue
        selected.append(row)
        seen.add(row["position"])
        if len(selected) >= limit:
            break
    selected.sort(key=lambda row: row["position"])

    events = []
    for row in selected:
        array_index = row["array_index"]
        layer_readouts = []
        top1_runs = []
        for layer_index, layer in enumerate(layers.tolist()):
            candidates = [
                {
                    "token_id": int(token_id),
                    "token": decoder(int(token_id)),
                    "logit": round(float(logit), 4),
                }
                for token_id, logit in zip(
                    top_ids[array_index, layer_index],
                    top_logits[array_index, layer_index],
                    strict=True,
                )
            ]
            concepts = [
                item
                for item in candidates
                if _is_meaningful_readout(item["token"])
            ][:top_concepts]
            if not concepts:
                concepts = candidates[:top_concepts]
            layer_readouts.append({"layer": int(layer), "concepts": concepts})
            top1_token = decoder(int(top_ids[array_index, layer_index, 0]))
            if top1_runs and top1_runs[-1]["token"] == top1_token:
                top1_runs[-1]["layer_end"] = int(layer)
            else:
                top1_runs.append(
                    {
                        "layer_start": int(layer),
                        "layer_end": int(layer),
                        "token": top1_token,
                    }
                )
        events.append(
            {
                **{key: value for key, value in row.items() if key != "array_index"},
                "top1_layer_runs": top1_runs,
                "layer_readouts": layer_readouts,
            }
        )
    return events


def _write_interactive_viewer(
    case_output_dir: Path,
    generation: dict[str, Any],
    metadata: dict[str, Any],
    arrays: Any,
) -> None:
    from jlens.vis import SliceData, build_page

    positions = arrays["positions"].astype(np.int64)
    layers = arrays["layers"].astype(np.int64)
    top_ids = arrays["top_token_ids"].astype(np.int32)
    tracked_ids = arrays["tracked_token_ids"].astype(np.int64).tolist()
    tracked_ranks = arrays["tracked_ranks"].astype(np.int32)
    k = top_ids.shape[-1]
    top_ranks = np.broadcast_to(
        np.arange(k, dtype=np.int32),
        top_ids.shape,
    ).copy()
    full_ids = generation["full_token_ids"]
    full_tokens = generation["prompt"]["tokens"] + generation["generation"]["tokens"]
    vocab = {
        int(token_id): token
        for token_id, token in metadata.get("top_vocab", {}).items()
    }
    vocab.update(
        {
            int(token_id): item["token"]
            for token_id, item in metadata.get("tracked_tokens", {}).items()
        }
    )
    vocab.update(
        {
            int(token_id): token
            for token_id, token in zip(full_ids, full_tokens, strict=False)
        }
    )
    pinned = {
        tracked_ids[index]
        for index in _gold_tracked_indices(
            metadata,
            np.asarray(tracked_ids, dtype=np.int64),
        )
    }
    slice_data = SliceData(
        seq_len=len(positions),
        layers=layers.tolist(),
        context_token_ids=full_ids,
        context_token_strs=full_tokens,
        top_ids=top_ids,
        top_ranks=top_ranks,
        tracked_token_ids=tracked_ids,
        rank_tensor=tracked_ranks,
        vocab_fragment=vocab,
        pinned_token_ids=sorted(pinned),
        ctx_offset=int(positions[0]) if len(positions) else 0,
    )
    source = generation["source"]["csv_row"]
    viewer_dir = case_output_dir / "interactive"
    page, _, _ = build_page(
        slice_data,
        source["question"],
        title=f"GSM8K wrong_id={source['wrong_id']}",
        description=(
            "Saved Jacobian-Lens position × layer readout. Bottom row is the "
            "model's actual final-layer output."
        ),
        mode="fetch",
        out_dir=viewer_dir,
    )
    (viewer_dir / "index.html").write_text(page, encoding="utf-8")


def _llm_packet(
    generation: dict[str, Any],
    behavior: dict[str, Any],
    generated_steps: list[TextStep],
    reference_steps: list[TextStep],
    alignment: dict[str, Any],
    lens_summary: dict[str, Any],
    concept_events: list[dict[str, Any]],
) -> dict[str, Any]:
    row = generation["source"]["csv_row"]
    compact_events = []
    for event in concept_events:
        layer_readouts = event["layer_readouts"]
        if layer_readouts:
            landmark_indices = sorted(
                {
                    round(index)
                    for index in np.linspace(
                        0,
                        len(layer_readouts) - 1,
                        min(8, len(layer_readouts)),
                    )
                }
            )
            landmarks = [layer_readouts[index] for index in landmark_indices]
        else:
            landmarks = []
        compact_events.append(
            {
                key: value
                for key, value in event.items()
                if key != "layer_readouts"
            }
            | {"landmark_layer_readouts": landmarks}
        )
    return {
        "case_id": f"wrong-{int(row['wrong_id']):04d}",
        "wrong_id": row["wrong_id"],
        "question": row["question"],
        "official_reasoning_steps": [
            {
                "index": step.index,
                "text": step.text,
                "numbers": step.numbers,
                "operations": step.operations,
            }
            for step in reference_steps
        ],
        "official_answer": generation["grading"]["gold_canonical"],
        "model_thinking_steps": [
            {
                "index": step.index,
                "text": step.text,
                "numbers": step.numbers,
                "operations": step.operations,
            }
            for step in generated_steps
        ],
        "model_final_response": behavior["final_response"],
        "model_answer": generation["grading"]["generated_answer_canonical"],
        "quality_and_behavior": behavior,
        "step_alignment": alignment,
        "lens_summary": lens_summary,
        "selected_jlens_events": compact_events,
        "analysis_contract": {
            "jlens_semantics": (
                "Tokens are lexical readouts of what an activation is disposed "
                "to make the model say, not guaranteed human-level concepts."
            ),
            "requested_output": {
                "case_id": "copy case_id from this packet",
                "wrong_id": "copy wrong_id from this packet",
                "first_problematic_generated_step": "integer or null",
                "first_problematic_reference_step": "integer or null",
                "primary_error_type": (
                    "one of comprehension, plan, quantity_tracking, arithmetic, "
                    "unit, unsupported_inference, self_correction_failure, "
                    "finalization_format, truncation, no_error, uncertain"
                ),
                "secondary_error_types": "list",
                "evidence": "list of direct references to supplied steps/events",
                "jlens_interpretation": (
                    "describe whether readouts precede, accompany, or follow the "
                    "textual error; do not claim causality"
                ),
                "confidence": "number from 0 to 1",
                "summary": "concise case diagnosis",
            },
        },
    }


def _summarize_llm_diagnoses(
    path: Path,
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    diagnoses = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    case_by_id = {str(case["wrong_id"]): case for case in cases}
    primary_counts: Counter[str] = Counter()
    secondary_counts: Counter[str] = Counter()
    confidence_by_type: dict[str, list[float]] = defaultdict(list)
    joined_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    for index, diagnosis in enumerate(diagnoses, start=1):
        missing = set(LLM_OUTPUT_SCHEMA["required"]) - set(diagnosis)
        if missing:
            raise ValueError(
                f"diagnosis line {index} missing fields: {sorted(missing)}"
            )
        primary = diagnosis["primary_error_type"]
        if primary not in ERROR_TYPES:
            raise ValueError(
                f"diagnosis line {index} has invalid primary_error_type={primary!r}"
            )
        confidence = float(diagnosis["confidence"])
        if not 0 <= confidence <= 1:
            raise ValueError(f"diagnosis line {index} confidence is outside [0,1]")
        primary_counts[primary] += 1
        confidence_by_type[primary].append(confidence)
        secondary_counts.update(diagnosis["secondary_error_types"])
        case = case_by_id.get(str(diagnosis["wrong_id"]))
        if case is not None:
            behavior = case["behavior"]
            if not behavior["complete"]:
                outcome = "incomplete"
            elif behavior["correct"]:
                outcome = "complete_correct"
            else:
                outcome = "complete_incorrect"
            joined_outcomes[outcome][primary] += 1
    return {
        "n_diagnoses": len(diagnoses),
        "primary_error_counts": dict(primary_counts.most_common()),
        "secondary_error_counts": dict(secondary_counts.most_common()),
        "mean_confidence_by_primary_type": {
            error_type: float(np.mean(values))
            for error_type, values in sorted(confidence_by_type.items())
        },
        "primary_errors_by_outcome": {
            outcome: dict(counts.most_common())
            for outcome, counts in sorted(joined_outcomes.items())
        },
    }


def _aggregate_layer_rows(
    rows: list[dict[str, Any]],
    *,
    outcome_group: str,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["layer"])].append(row)
    result = []
    for layer, layer_rows in sorted(grouped.items()):
        aggregate: dict[str, Any] = {
            "outcome_group": outcome_group,
            "layer": layer,
            "n_cases": len(layer_rows),
        }
        for field in (
            "final_top1_agreement",
            "adjacent_topk_jaccard",
            "topk_entropy",
            "numeric_top1_rate",
            "gold_mrr",
            "gold_top100_rate",
        ):
            values = [
                float(row[field])
                for row in layer_rows
                if row.get(field) is not None
            ]
            aggregate[f"mean_{field}"] = float(np.mean(values)) if values else None
        result.append(aggregate)
    return result


def _dataset_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    behavior_rows = [case["behavior"] for case in cases]
    with_lens = [case for case in cases if case["lens"]["available"]]

    def count(field: str) -> int:
        return sum(bool(row.get(field)) for row in behavior_rows)

    numeric_fields = (
        "generation_tokens",
        "generated_step_count",
        "alignment_generated_coverage",
        "alignment_reference_coverage",
        "alignment_mean_similarity",
        "numeric_reference_recall",
        "self_correction_markers",
        "repeated_4gram_ratio",
        "repeated_step_signature_ratio",
        "mean_chosen_surprisal",
        "chosen_top1_rate",
        "invalid_equations",
    )
    means = {}
    for field in numeric_fields:
        values = [
            float(row[field])
            for row in behavior_rows
            if row.get(field) is not None
        ]
        means[field] = float(np.mean(values)) if values else None
    groups = {}
    for group_name, predicate in (
        (
            "complete_correct",
            lambda row: row.get("complete") and row.get("correct"),
        ),
        (
            "complete_incorrect",
            lambda row: row.get("complete") and not row.get("correct"),
        ),
        ("incomplete", lambda row: not row.get("complete")),
    ):
        selected = [row for row in behavior_rows if predicate(row)]
        groups[group_name] = {
            "n": len(selected),
            "means": {
                field: (
                    float(
                        np.mean(
                            [
                                float(row[field])
                                for row in selected
                                if row.get(field) is not None
                            ]
                        )
                    )
                    if any(row.get(field) is not None for row in selected)
                    else None
                )
                for field in numeric_fields
            },
        }
    return {
        "n_cases": len(cases),
        "n_with_lens": len(with_lens),
        "n_complete": count("complete"),
        "n_correct": count("correct"),
        "n_ended_with_eos": count("ended_with_eos"),
        "n_thinking_closed": count("thinking_closed"),
        "n_valid_final_hash_answer": count("valid_final_hash_answer"),
        "n_hit_max_new_tokens": count("hit_max_new_tokens"),
        "rates": {
            "lens_available": len(with_lens) / len(cases) if cases else 0.0,
            "complete": count("complete") / len(cases) if cases else 0.0,
            "correct": count("correct") / len(cases) if cases else 0.0,
        },
        "means": means,
        "groups": groups,
    }


def _markdown_report(
    summary: dict[str, Any],
    layer_summary: list[dict[str, Any]],
    llm_summary: dict[str, Any] | None,
) -> str:
    rates = summary["rates"]
    lines = [
        "# GSM8K Jacobian-Lens analysis",
        "",
        "## Coverage and outcome",
        "",
        f"- Cases discovered: {summary['n_cases']}",
        f"- Cases with Lens traces: {summary['n_with_lens']} "
        f"({rates['lens_available']:.1%})",
        f"- Complete generations: {summary['n_complete']} "
        f"({rates['complete']:.1%})",
        f"- Correct generations: {summary['n_correct']} "
        f"({rates['correct']:.1%})",
        f"- Hit max_new_tokens: {summary['n_hit_max_new_tokens']}",
        "",
        "Completion and correctness are reported separately. Behavioral, "
        "alignment, and Lens statistics remain available for incorrect cases; "
        "truncated generations are explicitly marked and should not be treated "
        "as ordinary reasoning failures.",
        "",
        "## Deterministic observations",
        "",
    ]
    for field, value in summary["means"].items():
        if value is not None:
            lines.append(f"- Mean {field}: {value:.4f}")
    lines.extend(["", "## Outcome-independent comparison", ""])
    for group_name, group in summary["groups"].items():
        lines.append(f"- {group_name}: n={group['n']}")
        for field in (
            "alignment_generated_coverage",
            "alignment_reference_coverage",
            "numeric_reference_recall",
            "repeated_4gram_ratio",
            "repeated_step_signature_ratio",
            "invalid_equations",
        ):
            value = group["means"][field]
            if value is not None:
                lines.append(f"  - Mean {field}: {value:.4f}")
    overall_layer_summary = [
        row for row in layer_summary if row["outcome_group"] == "all"
    ]
    if overall_layer_summary:
        best_gold = max(
            (
                row
                for row in overall_layer_summary
                if row["mean_gold_mrr"] is not None
            ),
            key=lambda row: row["mean_gold_mrr"],
            default=None,
        )
        prefinal_rows = overall_layer_summary[:-1] or overall_layer_summary
        most_converged = max(
            prefinal_rows,
            key=lambda row: row["mean_final_top1_agreement"],
        )
        lines.extend(
            [
                "",
                "## Layer-level landmarks",
                "",
                (
                    f"- Highest mean gold-token reciprocal rank: layer "
                    f"{best_gold['layer']} ({best_gold['mean_gold_mrr']:.6f})"
                    if best_gold
                    else "- Gold-token ranks were unavailable."
                ),
                "- Highest agreement with the final-layer top-1 readout: "
                f"layer {most_converged['layer']} "
                f"({most_converged['mean_final_top1_agreement']:.4f})",
            ]
        )
    if llm_summary is not None:
        lines.extend(
            [
                "",
                "## LLM-assisted diagnosis patterns",
                "",
                f"- Diagnoses loaded: {llm_summary['n_diagnoses']}",
            ]
        )
        for error_type, count in llm_summary["primary_error_counts"].items():
            lines.append(f"- {error_type}: {count}")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The recorded tokens are J-lens lexical readouts: what an internal "
            "activation is disposed to make the model say after average-Jacobian "
            "transport. They are evidence about representational availability, "
            "not proof of a discrete human-like concept or a causal computation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.top_concepts <= 0 or args.max_concept_events <= 0:
        raise ValueError("concept limits must be positive")
    input_dir = Path(args.input_dir).resolve()
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else input_dir / "analysis"
    )
    selected_ids = (
        {value.strip() for value in args.wrong_ids.split(",")}
        if args.wrong_ids
        else None
    )
    generation_paths = sorted(input_dir.glob("wrong-*/generation.json"))
    if not generation_paths:
        raise FileNotFoundError(f"no sample generation.json files under {input_dir}")

    cases = []
    all_layer_rows = []
    llm_packets = []
    html_written = 0
    for generation_path in generation_paths:
        sample_dir = generation_path.parent
        generation = _read_json(generation_path)
        row = generation["source"]["csv_row"]
        if selected_ids is not None and row["wrong_id"] not in selected_ids:
            continue
        generated_steps, phase = _steps_with_token_positions(generation)
        reference_steps = _reference_steps(generation)
        alignment = _align_steps(generated_steps, reference_steps)
        behavior = _behavior_metrics(
            generation,
            generated_steps,
            reference_steps,
            alignment,
        )
        lens_metadata_path = sample_dir / "lens_metadata.json"
        lens_trace_path = sample_dir / "lens_trace.npz"
        lens_summary: dict[str, Any] = {
            "available": lens_metadata_path.exists() and lens_trace_path.exists()
        }
        layer_rows: list[dict[str, Any]] = []
        position_rows: list[dict[str, Any]] = []
        step_concepts: list[dict[str, Any]] = []
        concept_events: list[dict[str, Any]] = []
        if lens_summary["available"]:
            metadata = _read_json(lens_metadata_path)
            with np.load(lens_trace_path) as arrays:
                lens_summary, layer_rows, position_rows = _lens_metrics(
                    metadata,
                    arrays,
                    generation,
                )
                step_concepts = _step_layer_concepts(
                    generated_steps,
                    alignment,
                    metadata,
                    arrays,
                    top_concepts=args.top_concepts,
                    phase=phase,
                )
                concept_events = _concept_events(
                    position_rows,
                    metadata,
                    arrays,
                    top_concepts=args.top_concepts,
                    limit=args.max_concept_events,
                )
                if args.write_html and html_written < args.html_limit:
                    _write_interactive_viewer(
                        output_dir / sample_dir.name,
                        generation,
                        metadata,
                        arrays,
                    )
                    html_written += 1
        case = {
            "sample_dir": sample_dir.name,
            "wrong_id": row["wrong_id"],
            "split": row["split"],
            "index": row["index"],
            "behavior": behavior,
            "generated_reasoning_steps": [
                {
                    "index": step.index,
                    "text": step.text,
                    "numbers": step.numbers,
                    "operations": step.operations,
                    "absolute_positions": step.positions,
                }
                for step in generated_steps
            ],
            "official_reasoning_steps": [
                {
                    "index": step.index,
                    "text": step.text,
                    "numbers": step.numbers,
                    "operations": step.operations,
                }
                for step in reference_steps
            ],
            "alignment": alignment,
            "lens": lens_summary,
        }
        case_output_dir = output_dir / sample_dir.name
        _write_json(case_output_dir / "case_analysis.json", case)
        _write_jsonl(case_output_dir / "step_layer_concepts.jsonl", step_concepts)
        _write_jsonl(case_output_dir / "concept_events.jsonl", concept_events)
        _write_jsonl(case_output_dir / "layer_metrics.jsonl", layer_rows)
        _write_jsonl(case_output_dir / "position_metrics.jsonl", position_rows)
        packet = _llm_packet(
            generation,
            behavior,
            generated_steps,
            reference_steps,
            alignment,
            lens_summary,
            concept_events,
        )
        cases.append(case)
        llm_packets.append(packet)
        if behavior["complete"] and behavior["correct"]:
            outcome_group = "complete_correct"
        elif behavior["complete"]:
            outcome_group = "complete_incorrect"
        else:
            outcome_group = "incomplete"
        for layer_row in layer_rows:
            layer_row["wrong_id"] = row["wrong_id"]
            layer_row["outcome_group"] = outcome_group
        all_layer_rows.extend(layer_rows)
        print(
            f"analyzed wrong_id={row['wrong_id']} "
            f"complete={behavior['complete']} correct={behavior['correct']} "
            f"lens={lens_summary['available']}",
            flush=True,
        )

    summary = _dataset_summary(cases)
    layer_summary = _aggregate_layer_rows(
        all_layer_rows,
        outcome_group="all",
    )
    for outcome_group in (
        "complete_correct",
        "complete_incorrect",
        "incomplete",
    ):
        layer_summary.extend(
            _aggregate_layer_rows(
                [
                    row
                    for row in all_layer_rows
                    if row["outcome_group"] == outcome_group
                ],
                outcome_group=outcome_group,
            )
        )
    _write_json(output_dir / "dataset_summary.json", summary)
    _write_jsonl(output_dir / "cases.jsonl", cases)
    _write_jsonl(output_dir / "layer_summary.jsonl", layer_summary)
    _write_jsonl(output_dir / "llm_case_packets.jsonl", llm_packets)
    (output_dir / "llm_system_prompt.txt").write_text(
        LLM_SYSTEM_PROMPT + "\n",
        encoding="utf-8",
    )
    _write_json(output_dir / "llm_output_schema.json", LLM_OUTPUT_SCHEMA)
    llm_summary = None
    if args.llm_diagnoses:
        llm_summary = _summarize_llm_diagnoses(
            Path(args.llm_diagnoses).resolve(),
            cases,
        )
        _write_json(output_dir / "llm_diagnosis_summary.json", llm_summary)
    (output_dir / "report.md").write_text(
        _markdown_report(summary, layer_summary, llm_summary),
        encoding="utf-8",
    )
    print(
        f"Done: cases={summary['n_cases']} lens={summary['n_with_lens']} "
        f"complete={summary['n_complete']} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
