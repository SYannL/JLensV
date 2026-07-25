#!/usr/bin/env python3
"""Run resumable Stage A generation while saving analysis-complete artifacts."""

from __future__ import annotations

import argparse
import copy
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _common import (
    ARTIFACT_SCHEMA,
    DATASET_NAMES,
    DEFAULT_SELECTION,
    artifact_summary,
    atomic_json,
    atomic_jsonl,
    atomic_npz,
    effective_seed,
    extract_final_answer,
    grade_answer,
    load_json,
    load_jsonl,
    sample_directory,
    sha256,
    stable_hash,
)

PROMPT_VERSION = "stage-a-minimal-clean-v4"
SYSTEM_PROMPT = """Give concise reasoning that supports your answer.
End with exactly one line: FINAL: <answer>"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision; recorded together with resolved commit.",
    )
    parser.add_argument("--gpu", default="0", help="Physical CUDA device ID")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--decoding",
        choices=("sample", "greedy"),
        default="sample",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--sampling-top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument(
        "--metric-top-k",
        type=int,
        default=20,
        help="Raw-model top-k retained for each generated token.",
    )
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=DATASET_NAMES,
        default=list(DATASET_NAMES),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.max_new_tokens <= 0 or args.metric_top_k <= 0:
        raise ValueError("token limits must be positive")
    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if not 0 <= args.top_p <= 1 or not 0 <= args.min_p <= 1:
        raise ValueError("--top-p and --min-p must be in [0, 1]")
    if args.sampling_top_k <= 0 or args.repetition_penalty <= 0:
        raise ValueError("top-k and repetition penalty must be positive")
    if args.num_shards <= 0:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, num-shards)")


def task_prompt(record: dict[str, Any]) -> tuple[str, str]:
    dataset = record["dataset"]
    track = record["_selection"]["track"]
    if dataset == "processbench":
        steps = "\n".join(
            f"<STEP_ID={index}> {step}"
            for index, step in enumerate(record["candidate"]["reasoning_steps"])
        )
        user = f"""Problem:
{record['input']['question']}

Candidate reasoning with immutable machine step IDs:
{steps}

Find the earliest incorrect step. Refer to steps only by the literal STEP_ID
shown above; never renumber them as first/second or Step 1/Step 2.
Inspect steps in order and stop immediately after establishing the first error.
Do not independently re-solve the entire problem, revisit the decision, or
analyze later steps. Keep the explanation under 250 words.
If every step is correct, output `FINAL: OK`.
Otherwise output exactly `FINAL: STEP_ID=<n>` using the earliest incorrect
step's displayed integer ID."""
    elif dataset == "math500":
        user = record["input"]["question"]
    elif dataset == "prontoqa":
        user = f"""Use the facts and rules to determine whether the query follows.

Facts and rules:
{record['input']['context']}

Query:
{record['input']['question']}

Answer vocabulary: proved, disproved."""
    elif dataset == "stepgame":
        user = f"""Determine the spatial relation asked for.

Statements:
{record['input']['context']}

Question:
{record['input']['question']}

Answer vocabulary: above, below, left, right, upper-left, upper-right,
lower-left, lower-right, overlap."""
    elif dataset == "bbeh":
        user = record["input"]["question"]
    else:
        raise ValueError(f"unsupported dataset {dataset}")
    return track, user


def validate_prompt_independence(record: dict[str, Any], user_prompt: str) -> None:
    """Ensure task wrappers cannot leak gold labels or analysis metadata."""
    redacted = copy.deepcopy(record)
    redacted["gold"] = {
        key: f"<REDACTED_GOLD_{key}>"
        for key in redacted.get("gold", {})
    }
    redacted["metadata"] = {
        key: f"<REDACTED_METADATA_{key}>"
        for key in redacted.get("metadata", {})
    }
    _, redacted_prompt = task_prompt(redacted)
    if redacted_prompt != user_prompt:
        raise ValueError(
            f"{record['id']}: prompt depends on gold labels or metadata"
        )


def render_prompt(
    tokenizer: Any,
    record: dict[str, Any],
    *,
    thinking: bool,
) -> tuple[str, list[dict[str, str]], str, bool]:
    track, user = task_prompt(record)
    validate_prompt_independence(record, user)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    thinking_kwarg_applied = False
    if getattr(tokenizer, "chat_template", None):
        try:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
            thinking_kwarg_applied = True
        except TypeError:
            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        rendered = "\n\n".join(
            f"{message['role'].title()}: {message['content']}"
            for message in messages
        )
        rendered += "\n\nAssistant:"
    return track, messages, rendered, thinking_kwarg_applied


def decode_tokens(tokenizer: Any, token_ids: list[int]) -> list[str]:
    return [
        tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in token_ids
    ]


def split_response(
    tokenizer: Any,
    generated_ids: list[int],
    *,
    thinking: bool,
) -> dict[str, Any]:
    eos_ids: set[int] = set()
    model_eos = tokenizer.eos_token_id
    if isinstance(model_eos, int):
        eos_ids.add(model_eos)
    elif isinstance(model_eos, list):
        eos_ids.update(model_eos)
    for token in ("<|endoftext|>", "<|im_end|>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id != tokenizer.unk_token_id:
            eos_ids.add(token_id)
    ended_with_eos = bool(generated_ids and generated_ids[-1] in eos_ids)
    content_ids = generated_ids[:-1] if ended_with_eos else generated_ids[:]
    if not thinking:
        return {
            "thinking_ids": [],
            "final_ids": content_ids,
            "thinking_closed": True,
            "ended_with_eos": ended_with_eos,
        }
    end_think_id = tokenizer.convert_tokens_to_ids("</think>")
    if not isinstance(end_think_id, int) or end_think_id == tokenizer.unk_token_id:
        return {
            "thinking_ids": content_ids,
            "final_ids": [],
            "thinking_closed": False,
            "ended_with_eos": ended_with_eos,
        }
    try:
        boundary = content_ids.index(end_think_id)
    except ValueError:
        return {
            "thinking_ids": content_ids,
            "final_ids": [],
            "thinking_closed": False,
            "ended_with_eos": ended_with_eos,
        }
    return {
        "thinking_ids": content_ids[:boundary],
        "final_ids": content_ids[boundary + 1 :],
        "thinking_closed": True,
        "ended_with_eos": ended_with_eos,
    }


def raw_token_metrics(
    raw_logits: tuple[Any, ...],
    generated_ids: list[int],
    *,
    top_k: int,
) -> dict[str, Any]:
    import numpy as np
    import torch

    if len(raw_logits) != len(generated_ids):
        raise ValueError(
            f"raw logits={len(raw_logits)} but generated tokens={len(generated_ids)}"
        )
    n_steps = len(generated_ids)
    k = min(top_k, raw_logits[0].shape[-1]) if n_steps else top_k
    arrays: dict[str, Any] = {
        "generated_token_ids": np.asarray(generated_ids, dtype=np.int32),
        "raw_chosen_logprobs": np.empty(n_steps, dtype=np.float32),
        "raw_chosen_ranks": np.empty(n_steps, dtype=np.int32),
        "raw_entropy": np.empty(n_steps, dtype=np.float32),
        "raw_max_probability": np.empty(n_steps, dtype=np.float32),
        "raw_logsumexp": np.empty(n_steps, dtype=np.float32),
        "raw_energy": np.empty(n_steps, dtype=np.float32),
        "raw_top_token_ids": np.empty((n_steps, k), dtype=np.int32),
        "raw_top_logits": np.empty((n_steps, k), dtype=np.float32),
        "raw_top_logprobs": np.empty((n_steps, k), dtype=np.float32),
    }
    for step, (logits, chosen_id) in enumerate(
        zip(raw_logits, generated_ids, strict=True)
    ):
        vector = logits[0].float()
        logsumexp = torch.logsumexp(vector, dim=-1)
        logprobs = vector - logsumexp
        probabilities = logprobs.exp()
        top_logits, top_ids = vector.topk(k)
        arrays["raw_chosen_logprobs"][step] = float(logprobs[chosen_id].cpu())
        arrays["raw_chosen_ranks"][step] = int((vector > vector[chosen_id]).sum())
        arrays["raw_entropy"][step] = float(
            (-(probabilities * logprobs).sum()).cpu()
        )
        arrays["raw_max_probability"][step] = float(
            probabilities.max().cpu()
        )
        arrays["raw_logsumexp"][step] = float(logsumexp.cpu())
        arrays["raw_energy"][step] = -float(logsumexp.cpu())
        arrays["raw_top_token_ids"][step] = top_ids.cpu().numpy()
        arrays["raw_top_logits"][step] = top_logits.cpu().numpy()
        arrays["raw_top_logprobs"][step] = logprobs[top_ids].cpu().numpy()
    return arrays


def model_provenance(
    hf_model: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    transformers_version: str,
    torch_version: str,
) -> dict[str, Any]:
    config = hf_model.config.to_dict()
    chat_template = getattr(tokenizer, "chat_template", None)
    resolved_revision = getattr(hf_model.config, "_commit_hash", None)
    tokenizer_revision = getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
    return {
        "requested_model": args.model,
        "requested_revision": args.revision,
        "resolved_model_revision": resolved_revision,
        "resolved_tokenizer_revision": tokenizer_revision,
        "model_name_or_path": getattr(hf_model.config, "_name_or_path", None),
        "model_config_sha256": stable_hash(config),
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": len(tokenizer),
        "chat_template_sha256": (
            stable_hash(chat_template) if chat_template is not None else None
        ),
        "dtype": args.dtype,
        "software": {
            "python": sys.version.split()[0],
            "torch": torch_version,
            "transformers": transformers_version,
        },
    }


def generate_one(
    hf_model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    args: argparse.Namespace,
    *,
    experiment_fingerprint: str,
    selection_digest: str,
    provenance: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    track, messages, rendered, thinking_kwarg_applied = render_prompt(
        tokenizer,
        record,
        thinking=args.thinking,
    )
    encoded = tokenizer(rendered, return_tensors="pt")
    input_ids = encoded.input_ids.to(hf_model.device)
    attention_mask = encoded.attention_mask.to(hf_model.device)
    prompt_ids = input_ids[0].tolist()
    prompt_len = len(prompt_ids)
    seed = effective_seed(args.seed, record["id"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    kwargs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.decoding == "sample",
        "pad_token_id": pad_token_id,
        "return_dict_in_generate": True,
        "output_logits": True,
    }
    if args.decoding == "sample":
        kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.sampling_top_k,
                "min_p": args.min_p,
                "repetition_penalty": args.repetition_penalty,
            }
        )
    with torch.inference_mode():
        generated = hf_model.generate(**kwargs)
    full_ids = generated.sequences[0].tolist()
    generated_ids = full_ids[prompt_len:]
    if generated.logits is None:
        raise RuntimeError("Transformers did not return raw generation logits")
    arrays = raw_token_metrics(
        generated.logits,
        generated_ids,
        top_k=args.metric_top_k,
    )
    arrays["prediction_positions"] = (
        torch.arange(
            prompt_len - 1,
            prompt_len - 1 + len(generated_ids),
            dtype=torch.int32,
        )
        .cpu()
        .numpy()
    )
    sections = split_response(
        tokenizer,
        generated_ids,
        thinking=args.thinking,
    )
    thinking_text = tokenizer.decode(
        sections["thinking_ids"], skip_special_tokens=True
    )
    final_text = tokenizer.decode(
        sections["final_ids"], skip_special_tokens=True
    )
    raw_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    parsed_answer, parse_method = extract_final_answer(final_text)
    grading = grade_answer(record, parsed_answer, parse_method)
    hit_max = len(generated_ids) >= args.max_new_tokens and not sections[
        "ended_with_eos"
    ]
    format_valid = parse_method == "final_line"
    raw_metrics_complete = all(
        len(value) == len(generated_ids)
        for key, value in arrays.items()
        if key not in {"raw_top_token_ids", "raw_top_logits", "raw_top_logprobs"}
    )
    artifact = {
        "schema_version": ARTIFACT_SCHEMA,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_fingerprint": experiment_fingerprint,
        "source": {
            "record": record,
            "selection_sha256": selection_digest,
        },
        "model": provenance,
        "prompt": {
            "version": PROMPT_VERSION,
            "track": track,
            "messages": messages,
            "rendered_text": rendered,
            "token_ids": prompt_ids,
            "tokens": decode_tokens(tokenizer, prompt_ids),
            "n_tokens": prompt_len,
            "thinking_requested": args.thinking,
            "thinking_template_kwarg_applied": thinking_kwarg_applied,
            "gold_independence_check": True,
            "content_contract": (
                "input.question + candidate.reasoning_steps"
                if record["dataset"] == "processbench"
                else (
                    "input.context + input.question"
                    if record["dataset"] in {"prontoqa", "stepgame"}
                    else "input.question"
                )
            ),
        },
        "generation": {
            "mode": args.decoding,
            "seed": seed,
            "sampling_parameters": (
                {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.sampling_top_k,
                    "min_p": args.min_p,
                    "repetition_penalty": args.repetition_penalty,
                }
                if args.decoding == "sample"
                else None
            ),
            "max_new_tokens": args.max_new_tokens,
            "token_ids": generated_ids,
            "tokens": decode_tokens(tokenizer, generated_ids),
            "n_tokens": len(generated_ids),
            "full_token_ids": full_ids,
            "prediction_positions": arrays["prediction_positions"].tolist(),
            "raw_text": raw_text,
            "thinking_text": thinking_text,
            "final_text": final_text,
        },
        "token_metrics": {
            "path": "token_metrics.npz",
            "array_layout": {
                "generated_token_ids": "step",
                "prediction_positions": "step; absolute position predicting token",
                "raw_chosen_logprobs": "step; unwarped model distribution",
                "raw_chosen_ranks": "step; zero is top-1",
                "raw_entropy": "step; natural-log entropy",
                "raw_max_probability": "step",
                "raw_logsumexp": "step",
                "raw_energy": "step; negative logsumexp at temperature 1",
                "raw_top_token_ids": "step,rank",
                "raw_top_logits": "step,rank",
                "raw_top_logprobs": "step,rank",
            },
            "metric_top_k": args.metric_top_k,
        },
        "grading": grading,
        "quality": {
            "thinking_closed": sections["thinking_closed"],
            "ended_with_eos": sections["ended_with_eos"],
            "hit_max_new_tokens": hit_max,
            "final_answer_present": parsed_answer is not None,
            "format_valid": format_valid,
            "complete": bool(
                sections["ended_with_eos"]
                and sections["thinking_closed"]
                and format_valid
                and grading["parsed"]
            ),
            "raw_metrics_complete": raw_metrics_complete,
            "replay_ready": bool(
                full_ids
                and provenance["model_config_sha256"]
                and provenance["tokenizer_vocab_size"]
            ),
        },
        "replay_contract": {
            "input": "generation.full_token_ids",
            "prompt_token_count": prompt_len,
            "response_start_position": prompt_len,
            "prediction_alignment": (
                "generated step s is predicted from absolute position "
                "prompt_token_count+s-1"
            ),
            "attention_mask": "all ones; no padding in saved single-example input",
            "purpose": (
                "Replay fixed trajectories for hidden-state probes, CoE, JLens, "
                "SAE, and causal analyses without resampling."
            ),
        },
    }
    return artifact, arrays


def filtered_records(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    selected = set(args.datasets)
    records = [row for row in records if row["dataset"] in selected]
    records = records[args.start : args.end]
    if args.max_samples is not None:
        records = records[: args.max_samples]
    records = [
        row
        for index, row in enumerate(records)
        if index % args.num_shards == args.shard_index
    ]
    if not records:
        raise ValueError("sample selection is empty")
    return records


def main() -> None:
    args = parse_args()
    validate_args(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

    import torch
    import transformers

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA unavailable on physical GPU {args.gpu}")
    records = filtered_records(load_jsonl(args.selection), args)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    print(f"Loading {args.model} on physical GPU {args.gpu}", flush=True)
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    ).cuda()
    hf_model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
    )
    provenance = model_provenance(
        hf_model,
        tokenizer,
        args,
        transformers.__version__,
        torch.__version__,
    )
    selection_digest = sha256(args.selection)
    semantic_config = {
        "selection_sha256": selection_digest,
        "model": provenance,
        "prompt_version": PROMPT_VERSION,
        "thinking": args.thinking,
        "decoding": args.decoding,
        "temperature": args.temperature if args.decoding == "sample" else None,
        "top_p": args.top_p if args.decoding == "sample" else None,
        "sampling_top_k": (
            args.sampling_top_k if args.decoding == "sample" else None
        ),
        "min_p": args.min_p if args.decoding == "sample" else None,
        "repetition_penalty": (
            args.repetition_penalty if args.decoding == "sample" else None
        ),
        "max_new_tokens": args.max_new_tokens,
        "metric_top_k": args.metric_top_k,
        "base_seed": args.seed,
        "code_sha256": {
            "run_generation.py": sha256(Path(__file__)),
            "_common.py": sha256(Path(__file__).with_name("_common.py")),
        },
    }
    fingerprint = stable_hash(semantic_config)
    experiment_path = args.run_dir / "experiment_config.json"
    if experiment_path.exists():
        existing = load_json(experiment_path)
        if existing["experiment_fingerprint"] != fingerprint:
            raise ValueError(
                "run directory contains a different experiment configuration"
            )
    else:
        atomic_json(
            experiment_path,
            {
                "schema_version": "stage-a-experiment-config-v1",
                "experiment_fingerprint": fingerprint,
                "semantic_config": semantic_config,
            },
        )
    shard_config = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_fingerprint": fingerprint,
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "selected_ids": [record["id"] for record in records],
        "device": {
            "physical_gpu": str(args.gpu),
            "visible_device": "cuda:0",
            "name": torch.cuda.get_device_name(0),
        },
    }
    shard_path = (
        args.run_dir
        / f"run_config.shard-{args.shard_index:03d}-of-{args.num_shards:03d}.json"
    )
    atomic_json(shard_path, shard_config)

    for index, record in enumerate(records, start=1):
        sample_dir = sample_directory(args.run_dir, record)
        generation_path = sample_dir / "generation.json"
        metrics_path = sample_dir / "token_metrics.npz"
        error_path = sample_dir / "error.json"
        print(
            f"[{index}/{len(records)}] {record['dataset']} {record['id']}",
            flush=True,
        )
        try:
            if generation_path.exists() and metrics_path.exists() and not args.overwrite:
                existing = load_json(generation_path)
                if existing.get("experiment_fingerprint") != fingerprint:
                    raise ValueError(
                        f"{sample_dir}: artifact fingerprint mismatch"
                    )
                print("  reusing complete artifact", flush=True)
                continue
            artifact, arrays = generate_one(
                hf_model,
                tokenizer,
                record,
                args,
                experiment_fingerprint=fingerprint,
                selection_digest=selection_digest,
                provenance=provenance,
            )
            atomic_npz(metrics_path, arrays)
            artifact["token_metrics"]["sha256"] = sha256(metrics_path)
            artifact["token_metrics"]["bytes"] = metrics_path.stat().st_size
            atomic_json(generation_path, artifact)
            if error_path.exists():
                error_path.unlink()
            print(
                f"  tokens={artifact['generation']['n_tokens']} "
                f"complete={artifact['quality']['complete']} "
                f"correct={artifact['grading']['correct']}",
                flush=True,
            )
        except Exception as exc:
            atomic_json(
                error_path,
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "id": record["id"],
                    "dataset": record["dataset"],
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
            print(f"  ERROR {type(exc).__name__}: {exc}", flush=True)
            if args.fail_fast:
                raise
        finally:
            torch.cuda.empty_cache()

    sample_dirs = sorted((args.run_dir / "samples").glob("*"))
    summaries = [artifact_summary(path) for path in sample_dirs]
    atomic_jsonl(args.run_dir / "summary.jsonl", summaries)
    complete = sum(row.get("complete", False) for row in summaries)
    print(
        f"Shard done. Run currently has {len(summaries)} artifacts; "
        f"{complete} generation-complete.",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
