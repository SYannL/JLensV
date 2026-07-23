#!/usr/bin/env python3
"""Generate CoT solutions for hard GSM8K samples and record Jacobian-Lens traces.

The script is deliberately artifact-heavy: every sample gets its original CSV
row, canonical GSM8K record, exact chat prompt and token IDs, token-level
generation scores, parsed answer/correctness, and a position x layer Lens trace.
Runs are resumable at the per-sample generation and Lens-analysis boundaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

# Avoid the Intel-MKL/GNU-OpenMP conflict seen when NumPy and PyTorch coexist in
# the Conda environment. This must be set before importing either package.
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ.pop("MKL_SERVICE_FORCE_INTEL", None)


SYSTEM_PROMPT = """You are a careful mathematical reasoner solving grade-school
word problems. Work through the problem step by step, explicitly checking the
meaning of every quantity and arithmetic operation. Do not skip reasoning steps.
End with a separate final line in exactly this format:
#### <numeric answer>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", default="7", help="Physical CUDA device ID")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--lens",
        default="outputs/qwen35_4b_jacobian_lens.pt",
        help="Fitted JacobianLens file",
    )
    parser.add_argument(
        "--samples-csv",
        default="data/dataset/gms8k/hard_wrong_samples.csv",
    )
    parser.add_argument("--gsm8k-dir", default="data/dataset/gms8k")
    parser.add_argument(
        "--output-dir",
        default="outputs/gsm8k_hard_lens_analysis",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Lens and generation alternatives retained at each position",
    )
    parser.add_argument(
        "--max-tracked-tokens",
        type=int,
        default=32,
        help="Maximum answer/numeric token IDs receiving exact full-vocab ranks",
    )
    parser.add_argument(
        "--rank-position-chunk",
        type=int,
        default=32,
        help="Position chunk used for exact token-rank computation",
    )
    parser.add_argument(
        "--layer-stride",
        type=int,
        default=1,
        help="Analyze every Nth fitted layer; the last fitted/final layers are kept",
    )
    parser.add_argument(
        "--analysis-scope",
        choices=("all", "response"),
        default="all",
        help="Analyze all prompt/response tokens or response plus prediction anchor",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--wrong-ids",
        default=None,
        help="Optional comma-separated wrong_id subset",
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--skip-lens",
        action="store_true",
        help="Only generate and grade CoT responses",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate existing artifacts",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first sample error instead of recording it and continuing",
    )
    return parser.parse_args()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp, path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _normalize_question(text: str) -> str:
    text = text.strip()
    if text.lower().startswith("question:"):
        text = text.split(":", 1)[1].strip()
    return " ".join(text.split())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _canonical_lookup(gsm8k_dir: Path) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for split in ("train", "test"):
        path = gsm8k_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"missing canonical GSM8K split: {path}")
        for index, record in enumerate(_load_jsonl(path)):
            key = _normalize_question(record["question"])
            if key in lookup:
                raise ValueError(f"duplicate normalized GSM8K question: {key[:80]}")
            lookup[key] = {
                "official_split": split,
                "official_index": index,
                "question": record["question"],
                "answer": record["answer"],
            }
    return lookup


def _gold_from_rationale(answer: str) -> str | None:
    matches = re.findall(r"####\s*([^\n]+)", answer)
    return matches[-1].strip() if matches else None


def _extract_generated_answer(text: str) -> tuple[str | None, str]:
    hashes = re.findall(r"####\s*([^\n]+)", text)
    if hashes:
        return hashes[-1].strip(), "hash_delimiter"
    explicit = re.findall(
        r"(?:final\s+answer|answer\s+is)\s*[:=]?\s*"
        r"([-+]?\$?[\d,]+(?:\.\d+)?(?:/\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if explicit:
        return explicit[-1].strip(), "answer_phrase"
    numbers = re.findall(r"[-+]?\$?[\d,]+(?:\.\d+)?(?:/\d+)?", text)
    return (numbers[-1].strip(), "last_number") if numbers else (None, "missing")


def _canonical_number(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip().replace(",", "").replace("$", "")
    cleaned = cleaned.rstrip(". ")
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", cleaned)
    if not match:
        return None
    token = match.group(0)
    try:
        if "/" in token:
            numerator, denominator = token.split("/", 1)
            number = Decimal(numerator) / Decimal(denominator)
        else:
            number = Decimal(token)
    except (InvalidOperation, ZeroDivisionError):
        return None
    normalized = format(number.normalize(), "f")
    return normalized.rstrip("0").rstrip(".") if "." in normalized else normalized


def _decode_tokens(tokenizer: Any, token_ids: list[int]) -> list[str]:
    return [
        tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in token_ids
    ]


def _render_prompt(tokenizer: Any, question: str) -> tuple[list[dict[str, str]], str]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"{question.strip()}\n\n"
                "Solve this problem using complete chain-of-thought reasoning. "
                "Check your arithmetic, then finish with `#### <numeric answer>`."
            ),
        },
    ]
    if getattr(tokenizer, "chat_template", None):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        text = (
            f"System: {SYSTEM_PROMPT}\n\n"
            f"User: {messages[1]['content']}\n\nAssistant:"
        )
    return messages, text


def _generation_details(
    hf_model: Any,
    tokenizer: Any,
    row: dict[str, str],
    canonical: dict[str, Any],
    *,
    max_new_tokens: int,
    top_k: int,
) -> dict[str, Any]:
    import torch

    messages, prompt_text = _render_prompt(tokenizer, row["question"])
    encoded = tokenizer(prompt_text, return_tensors="pt")
    input_ids = encoded.input_ids.to(hf_model.device)
    attention_mask = encoded.attention_mask.to(hf_model.device)
    prompt_len = input_ids.shape[1]
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    with torch.inference_mode():
        generated = hf_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_token_id,
            return_dict_in_generate=True,
            output_scores=True,
        )

    full_ids = generated.sequences[0].tolist()
    generated_ids = full_ids[prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    parsed_answer, parse_method = _extract_generated_answer(generated_text)
    gold = row["ground_truth"] or _gold_from_rationale(canonical["answer"])
    parsed_canonical = _canonical_number(parsed_answer)
    gold_canonical = _canonical_number(gold)

    token_steps: list[dict[str, Any]] = []
    for step, (score, chosen_id) in enumerate(
        zip(generated.scores, generated_ids, strict=True)
    ):
        score = score[0].float()
        k = min(top_k, score.numel())
        top_values, top_ids = score.topk(k)
        log_normalizer = torch.logsumexp(score, dim=-1)
        chosen_logprob = float((score[chosen_id] - log_normalizer).cpu())
        alternative_logprobs = (top_values - log_normalizer).cpu().tolist()
        alternative_ids = top_ids.cpu().tolist()
        token_steps.append(
            {
                "step": step,
                "prediction_position": prompt_len + step - 1,
                "token_position": prompt_len + step,
                "token_id": chosen_id,
                "token": _decode_tokens(tokenizer, [chosen_id])[0],
                "logprob": chosen_logprob,
                "top_token_ids": alternative_ids,
                "top_tokens": _decode_tokens(tokenizer, alternative_ids),
                "top_logprobs": alternative_logprobs,
            }
        )

    return {
        "status": "generated",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "csv_row": row,
            "canonical_gsm8k": canonical,
            "official_gold_from_rationale": _gold_from_rationale(
                canonical["answer"]
            ),
        },
        "prompt": {
            "messages": messages,
            "rendered_text": prompt_text,
            "token_ids": full_ids[:prompt_len],
            "tokens": _decode_tokens(tokenizer, full_ids[:prompt_len]),
            "n_tokens": prompt_len,
        },
        "generation": {
            "mode": "greedy",
            "max_new_tokens": max_new_tokens,
            "token_ids": generated_ids,
            "tokens": _decode_tokens(tokenizer, generated_ids),
            "text": generated_text,
            "n_tokens": len(generated_ids),
            "ended_with_eos": bool(
                generated_ids and generated_ids[-1] == tokenizer.eos_token_id
            ),
            "steps": token_steps,
        },
        "full_token_ids": full_ids,
        "grading": {
            "gold_raw": gold,
            "gold_canonical": gold_canonical,
            "previous_prediction_raw": row.get("parsed_prediction"),
            "previous_prediction_canonical": _canonical_number(
                row.get("parsed_prediction")
            ),
            "generated_answer_raw": parsed_answer,
            "generated_answer_canonical": parsed_canonical,
            "parse_method": parse_method,
            "correct": bool(
                parsed_canonical is not None
                and gold_canonical is not None
                and parsed_canonical == gold_canonical
            ),
        },
    }


def _surface_token_map(
    tokenizer: Any,
    generation: dict[str, Any],
    max_tokens: int,
) -> tuple[list[int], dict[str, Any]]:
    grading = generation["grading"]
    source = generation["source"]
    generated_text = generation["generation"]["text"]
    question = source["csv_row"]["question"]
    rationale = source["canonical_gsm8k"]["answer"]

    surfaces: list[tuple[str, str]] = []
    for category, value in (
        ("gold", grading["gold_raw"]),
        ("generated_answer", grading["generated_answer_raw"]),
        ("previous_wrong_answer", grading["previous_prediction_raw"]),
    ):
        if value:
            surfaces.append((category, str(value)))

    numeric_pattern = r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?"
    for category, text in (
        ("question_number", question),
        ("generated_cot_number", generated_text),
        ("reference_cot_number", rationale),
    ):
        for value in re.findall(numeric_pattern, text):
            surfaces.append((category, value))

    token_ids: list[int] = []
    provenance: dict[int, list[dict[str, str]]] = defaultdict(list)
    seen_surface: set[tuple[str, str]] = set()
    for category, surface in surfaces:
        surface_key = (category, surface)
        if surface_key in seen_surface:
            continue
        seen_surface.add(surface_key)
        for variant_name, variant in (
            ("bare", surface),
            ("leading_space", " " + surface),
            ("after_hashes", "#### " + surface),
        ):
            ids = tokenizer.encode(variant, add_special_tokens=False)
            for token_id in ids:
                entry = {
                    "category": category,
                    "surface": surface,
                    "variant": variant_name,
                }
                if entry not in provenance[token_id]:
                    provenance[token_id].append(entry)
                if token_id not in token_ids and len(token_ids) < max_tokens:
                    token_ids.append(token_id)

    metadata = {
        str(token_id): {
            "token": _decode_tokens(tokenizer, [token_id])[0],
            "provenance": provenance[token_id],
        }
        for token_id in token_ids
    }
    return token_ids, metadata


def _exact_ranks(
    logits: Any,
    tracked_ids: list[int],
    *,
    position_chunk: int,
) -> Any:
    """Exact full-vocabulary rank (0 is best), without a full argsort."""
    import torch

    n_positions = logits.shape[0]
    ranks = torch.empty(
        n_positions,
        len(tracked_ids),
        dtype=torch.int32,
        device="cpu",
    )
    for start in range(0, n_positions, position_chunk):
        chunk = logits[start : start + position_chunk]
        for target_idx, token_id in enumerate(tracked_ids):
            target_value = chunk[:, token_id, None]
            ranks[start : start + len(chunk), target_idx] = (
                (chunk > target_value).sum(dim=1).to(torch.int32).cpu()
            )
    return ranks


def _lens_trace(
    model: Any,
    lens: Any,
    tokenizer: Any,
    generation: dict[str, Any],
    *,
    top_k: int,
    max_tracked_tokens: int,
    rank_position_chunk: int,
    layer_stride: int,
    analysis_scope: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np
    import torch

    from jlens.hooks import ActivationRecorder

    full_ids_list = generation["full_token_ids"]
    prompt_len = generation["prompt"]["n_tokens"]
    full_ids = torch.tensor(
        [full_ids_list],
        dtype=torch.long,
        device=model.input_device,
    )

    fitted = lens.source_layers
    layers = fitted[::layer_stride]
    if fitted[-1] not in layers:
        layers.append(fitted[-1])
    final_layer = model.n_layers - 1
    if final_layer not in layers:
        layers.append(final_layer)
    layers = sorted(set(layers))

    if analysis_scope == "all":
        positions = list(range(len(full_ids_list)))
    else:
        # Include the last prompt token because it predicts the first response token.
        positions = list(range(max(0, prompt_len - 1), len(full_ids_list)))

    tracked_ids, tracked_meta = _surface_token_map(
        tokenizer,
        generation,
        max_tracked_tokens,
    )

    with torch.no_grad(), ActivationRecorder(model.layers, at=layers) as recorder:
        model.forward(full_ids)
        activations = {
            layer: recorder.activations[layer].detach() for layer in layers
        }

    n_positions = len(positions)
    n_layers = len(layers)
    k = min(top_k, model._lm_head.out_features)
    top_ids = np.empty((n_positions, n_layers, k), dtype=np.int32)
    top_logits = np.empty((n_positions, n_layers, k), dtype=np.float16)
    tracked_logits = np.empty(
        (n_positions, n_layers, len(tracked_ids)), dtype=np.float16
    )
    tracked_ranks = np.empty(
        (n_positions, n_layers, len(tracked_ids)), dtype=np.int32
    )

    for layer_idx, layer in enumerate(layers):
        residual = activations[layer][0, positions].float()
        if layer in lens.jacobians:
            residual = lens.transport(residual, layer)
        logits = model.unembed(residual).float()
        values, ids = logits.topk(k, dim=-1)
        top_ids[:, layer_idx] = ids.cpu().numpy().astype(np.int32)
        top_logits[:, layer_idx] = values.cpu().numpy().astype(np.float16)
        if tracked_ids:
            tracked_tensor = torch.tensor(
                tracked_ids, dtype=torch.long, device=logits.device
            )
            tracked_logits[:, layer_idx] = (
                logits[:, tracked_tensor].cpu().numpy().astype(np.float16)
            )
            tracked_ranks[:, layer_idx] = _exact_ranks(
                logits,
                tracked_ids,
                position_chunk=rank_position_chunk,
            ).numpy()
        del logits, residual, values, ids

    token_strings = _decode_tokens(tokenizer, full_ids_list)
    top_vocab_ids = sorted(set(int(x) for x in top_ids.reshape(-1)))
    top_vocab = {
        str(token_id): _decode_tokens(tokenizer, [token_id])[0]
        for token_id in top_vocab_ids
    }
    metadata = {
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "array_layout": "position,layer,token",
        "array_descriptions": {
            "positions": "absolute indices into full_token_ids",
            "layers": "transformer block output indices",
            "top_token_ids": "top-k token IDs from J-lens logits",
            "top_logits": "corresponding post-norm/unembedding logits",
            "tracked_token_ids": "answer/numeric token IDs",
            "tracked_logits": "logits for tracked token IDs",
            "tracked_ranks": "exact full-vocabulary ranks; 0 is top-1",
        },
        "prompt_token_count": prompt_len,
        "response_start_position": prompt_len,
        "full_token_count": len(full_ids_list),
        "analysis_scope": analysis_scope,
        "analyzed_positions": positions,
        "position_tokens": [token_strings[position] for position in positions],
        "layers": layers,
        "final_model_layer": final_layer,
        "top_k": k,
        "tracked_tokens": tracked_meta,
        "top_vocab": top_vocab,
        "interpretation_note": (
            "At position p, logits describe what the residual after token p is "
            "disposed to produce. The generated token at response step s was "
            "predicted from absolute position prompt_len+s-1."
        ),
    }
    arrays = {
        "positions": np.asarray(positions, dtype=np.int32),
        "layers": np.asarray(layers, dtype=np.int16),
        "top_token_ids": top_ids,
        "top_logits": top_logits,
        "tracked_token_ids": np.asarray(tracked_ids, dtype=np.int32),
        "tracked_logits": tracked_logits,
        "tracked_ranks": tracked_ranks,
    }
    return metadata, arrays


def _atomic_npz(path: Path, arrays: dict[str, Any]) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def _sample_dir(output_dir: Path, row: dict[str, str]) -> Path:
    return output_dir / (
        f"wrong-{int(row['wrong_id']):04d}_"
        f"{row['split']}-idx-{int(row['index']):04d}"
    )


def _summary_record(sample_dir: Path) -> dict[str, Any]:
    generation_path = sample_dir / "generation.json"
    lens_path = sample_dir / "lens_metadata.json"
    error_path = sample_dir / "error.json"
    record: dict[str, Any] = {"sample_dir": sample_dir.name}
    if generation_path.exists():
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        row = generation["source"]["csv_row"]
        record |= {
            "wrong_id": row["wrong_id"],
            "split": row["split"],
            "index": row["index"],
            "gold": generation["grading"]["gold_canonical"],
            "previous_prediction": generation["grading"][
                "previous_prediction_canonical"
            ],
            "generated_prediction": generation["grading"][
                "generated_answer_canonical"
            ],
            "generated_correct": generation["grading"]["correct"],
            "generation_tokens": generation["generation"]["n_tokens"],
        }
    record["lens_complete"] = lens_path.exists() and (
        sample_dir / "lens_trace.npz"
    ).exists()
    if error_path.exists():
        record["error"] = json.loads(error_path.read_text(encoding="utf-8"))[
            "message"
        ]
    return record


def main() -> None:
    args = parse_args()
    if args.top_k <= 0 or args.max_tracked_tokens < 0:
        raise ValueError("--top-k must be positive and --max-tracked-tokens nonnegative")
    if args.layer_stride <= 0 or args.rank_position_chunk <= 0:
        raise ValueError("layer/rank chunk arguments must be positive")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    import transformers

    import jlens

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable on selected physical GPU {args.gpu}")

    samples_csv = Path(args.samples_csv).resolve()
    gsm8k_dir = Path(args.gsm8k_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    with samples_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"wrong_id", "split", "index", "question", "ground_truth"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"CSV must contain fields {sorted(required)}")

    selected_ids = None
    if args.wrong_ids:
        selected_ids = {part.strip() for part in args.wrong_ids.split(",")}
        rows = [row for row in rows if row["wrong_id"] in selected_ids]
    rows = rows[args.start : args.end]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        raise ValueError("sample selection is empty")

    canonical_lookup = _canonical_lookup(gsm8k_dir)
    canonical_by_wrong_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _normalize_question(row["question"])
        if key not in canonical_lookup:
            raise ValueError(
                f"wrong_id={row['wrong_id']} does not match canonical GSM8K"
            )
        canonical = canonical_lookup[key]
        csv_gold = _canonical_number(row["ground_truth"])
        official_gold = _canonical_number(_gold_from_rationale(canonical["answer"]))
        if csv_gold != official_gold:
            raise ValueError(
                f"wrong_id={row['wrong_id']} gold mismatch: "
                f"CSV={csv_gold}, official={official_gold}"
            )
        canonical_by_wrong_id[row["wrong_id"]] = canonical

    run_config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "resolved_paths": {
            "samples_csv": str(samples_csv),
            "gsm8k_dir": str(gsm8k_dir),
            "lens": str(Path(args.lens).resolve()),
            "output_dir": str(output_dir),
        },
        "n_selected_samples": len(rows),
        "selected_wrong_ids": [row["wrong_id"] for row in rows],
        "software": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "device": {
            "physical_gpu": str(args.gpu),
            "visible_device": "cuda:0",
            "name": torch.cuda.get_device_name(0),
        },
        "system_prompt": SYSTEM_PROMPT,
    }
    _atomic_json(output_dir / "run_config.json", run_config)

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    print(f"Loading model {args.model} on physical GPU {args.gpu}", flush=True)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
    ).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    model = jlens.from_hf(hf_model, tokenizer)
    lens = None if args.skip_lens else jlens.JacobianLens.load(args.lens)
    if lens is not None and lens.d_model != model.d_model:
        raise ValueError(
            f"lens d_model={lens.d_model} does not match model d_model={model.d_model}"
        )

    for sample_idx, row in enumerate(rows, start=1):
        sample_dir = _sample_dir(output_dir, row)
        sample_dir.mkdir(parents=True, exist_ok=True)
        generation_path = sample_dir / "generation.json"
        lens_metadata_path = sample_dir / "lens_metadata.json"
        lens_trace_path = sample_dir / "lens_trace.npz"
        error_path = sample_dir / "error.json"
        print(
            f"[{sample_idx}/{len(rows)}] wrong_id={row['wrong_id']} "
            f"split={row['split']} index={row['index']}",
            flush=True,
        )
        try:
            if generation_path.exists() and not args.overwrite:
                generation = json.loads(generation_path.read_text(encoding="utf-8"))
                print("  reusing saved generation", flush=True)
            else:
                generation = _generation_details(
                    hf_model,
                    tokenizer,
                    row,
                    canonical_by_wrong_id[row["wrong_id"]],
                    max_new_tokens=args.max_new_tokens,
                    top_k=args.top_k,
                )
                _atomic_json(generation_path, generation)
                print(
                    "  generated answer="
                    f"{generation['grading']['generated_answer_canonical']!r} "
                    f"correct={generation['grading']['correct']}",
                    flush=True,
                )

            if lens is not None:
                if (
                    lens_metadata_path.exists()
                    and lens_trace_path.exists()
                    and not args.overwrite
                ):
                    print("  reusing saved Lens trace", flush=True)
                else:
                    lens_metadata, arrays = _lens_trace(
                        model,
                        lens,
                        tokenizer,
                        generation,
                        top_k=args.top_k,
                        max_tracked_tokens=args.max_tracked_tokens,
                        rank_position_chunk=args.rank_position_chunk,
                        layer_stride=args.layer_stride,
                        analysis_scope=args.analysis_scope,
                    )
                    _atomic_npz(lens_trace_path, arrays)
                    _atomic_json(lens_metadata_path, lens_metadata)
                    print(
                        f"  saved Lens trace: {len(lens_metadata['analyzed_positions'])} "
                        f"positions x {len(lens_metadata['layers'])} layers",
                        flush=True,
                    )
            if error_path.exists():
                error_path.unlink()
            torch.cuda.empty_cache()
        except Exception as exc:
            error = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "wrong_id": row["wrong_id"],
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            _atomic_json(error_path, error)
            print(f"  ERROR: {type(exc).__name__}: {exc}", flush=True)
            if args.fail_fast:
                raise

        completed_dirs = [_sample_dir(output_dir, item) for item in rows]
        _atomic_jsonl(
            output_dir / "summary.jsonl",
            [_summary_record(path) for path in completed_dirs if path.exists()],
        )

    summaries = [_summary_record(_sample_dir(output_dir, row)) for row in rows]
    n_generated = sum("generated_correct" in record for record in summaries)
    n_correct = sum(record.get("generated_correct", False) for record in summaries)
    n_lens = sum(record.get("lens_complete", False) for record in summaries)
    print(
        f"Done: generated={n_generated}/{len(rows)}, correct={n_correct}/{n_generated}, "
        f"lens_complete={n_lens}/{len(rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
