#!/usr/bin/env python3
"""Fit a Jacobian Lens from a pretraining-like WikiText corpus.

Example:
    python fit_lens.py --gpu 7 --model Qwen/Qwen3.5-4B
    python fit_lens.py --gpus 4,5,6,7 --model Qwen/Qwen3.5-4B

The fitting checkpoint and the finished lens are different files. The former
stores running sums so an interrupted job can resume; the latter is the lens
used by ``JacobianLens.load`` / ``from_pretrained``.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any

# Conda's MKL defaults can select Intel OpenMP while PyTorch wheels load GNU
# OpenMP. Spawned workers then abort as soon as NumPy/MKL and Torch coexist.
# Select one threading runtime before either library is imported, in both the
# parent process and every ``spawn`` child importing this module.
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ.pop("MKL_SERVICE_FORCE_INTEL", None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument(
        "--gpu",
        help="One physical CUDA device ID (default: 7)",
    )
    device_group.add_argument(
        "--gpus",
        help="Comma-separated physical CUDA device IDs for prompt-sharded fitting",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument(
        "--n-prompts",
        type=int,
        default=1000,
        help="Number of WikiText sequences (paper-scale default: 1000)",
    )
    parser.add_argument("--min-chars", type=int, default=600)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument(
        "--dim-batch",
        type=int,
        default=32,
        help="Jacobian output dimensions per backward pass; lower saves VRAM",
    )
    parser.add_argument(
        "--target-layer",
        type=int,
        default=None,
        help="Jacobian target layer; default is the final transformer block",
    )
    parser.add_argument(
        "--source-layers",
        default=None,
        help="Comma-separated layer indices; default is every layer below target",
    )
    parser.add_argument("--checkpoint", default="outputs/jlens_fit_checkpoint.pt")
    parser.add_argument("--output", default="outputs/jacobian_lens.pt")
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile transformer blocks (costly first-time compilation)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore an existing fitting checkpoint",
    )
    return parser.parse_args()


def _parse_gpus(args: argparse.Namespace) -> list[str]:
    raw = args.gpus if args.gpus is not None else (args.gpu or "7")
    gpus = [gpu.strip() for gpu in raw.split(",") if gpu.strip()]
    if not gpus:
        raise ValueError("at least one GPU must be specified")
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"GPU IDs must be unique, got {gpus}")
    return gpus


def _partition_ranges(n_items: int, n_parts: int) -> list[tuple[int, int]]:
    """Contiguous, balanced ``[start, end)`` ranges covering ``n_items``."""
    size, remainder = divmod(n_items, n_parts)
    ranges: list[tuple[int, int]] = []
    start = 0
    for part_idx in range(n_parts):
        end = start + size + (part_idx < remainder)
        ranges.append((start, end))
        start = end
    return ranges


def _part_path(
    path: str, *, part_idx: int, n_parts: int, gpu: str, start: int, end: int
) -> Path:
    base = Path(path).resolve()
    suffix = base.suffix or ".pt"
    stem = base.name[: -len(base.suffix)] if base.suffix else base.name
    name = (
        f"{stem}.part{part_idx + 1:02d}-of-{n_parts:02d}."
        f"prompts{start:04d}-{end:04d}.gpu{gpu}{suffix}"
    )
    return base.with_name(name)


def _load_prompts(n_prompts: int, min_chars: int) -> list[str]:
    """Load once in the parent so workers receive disjoint in-memory slices."""
    from datasets import load_dataset

    dataset = load_dataset(
        "Salesforce/wikitext",
        "wikitext-103-raw-v1",
        split="train",
        streaming=True,
    )
    prompts: list[str] = []
    for record in dataset:
        text = record["text"]
        if len(text.strip()) >= min_chars:
            prompts.append(text)
            if len(prompts) == n_prompts:
                break
    if len(prompts) != n_prompts:
        raise RuntimeError(
            f"requested {n_prompts} prompts but loaded only {len(prompts)}"
        )
    return prompts


def _fit_worker(
    args_dict: dict[str, Any],
    gpu: str,
    prompts: list[str],
    checkpoint_path: str,
    output_path: str,
    part_idx: int,
    n_parts: int,
    global_start: int,
    global_end: int,
) -> None:
    """Fit one prompt shard in a fresh process bound to one physical GPU."""
    # Set visibility before importing torch. Each worker consequently sees its
    # selected physical device as cuda:0 and cannot allocate on another worker's GPU.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    os.environ.pop("MKL_SERVICE_FORCE_INTEL", None)

    import torch
    import transformers

    import jlens

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA is unavailable after selecting physical GPU {gpu}. "
            "Check GPU allocation and PyTorch/CUDA driver compatibility."
        )

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args_dict["dtype"]]
    raw_source_layers = args_dict["source_layers"]
    source_layers = (
        None
        if raw_source_layers is None
        else [int(layer.strip()) for layer in raw_source_layers.split(",")]
    )

    checkpoint = Path(checkpoint_path)
    output = Path(output_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    jlens.configure_logging()
    label = f"worker {part_idx + 1}/{n_parts}, physical GPU {gpu}"
    print(f"[{label}] CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"[{label}] prompts [{global_start}, {global_end}) "
        f"({len(prompts)} sequences)",
        flush=True,
    )
    print(f"[{label}] checkpoint: {checkpoint}", flush=True)

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        args_dict["model"],
        dtype=dtype,
    ).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args_dict["model"])
    model = jlens.from_hf(
        hf_model,
        tokenizer,
        compile=args_dict["compile"],
    )

    lens = jlens.fit(
        model,
        prompts,
        source_layers=source_layers,
        target_layer=args_dict["target_layer"],
        dim_batch=args_dict["dim_batch"],
        max_seq_len=args_dict["max_seq_len"],
        checkpoint_path=str(checkpoint),
        checkpoint_every=args_dict["checkpoint_every"],
        resume=not args_dict["no_resume"],
    )
    lens.save(str(output))
    print(f"[{label}] finished: {lens}", flush=True)
    print(f"[{label}] saved partial lens: {output}", flush=True)


def main() -> None:
    args = parse_args()
    gpus = _parse_gpus(args)

    if args.n_prompts <= 0:
        raise ValueError("--n-prompts must be positive")
    if args.dim_batch <= 0:
        raise ValueError("--dim-batch must be positive")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be positive")
    if args.n_prompts < len(gpus):
        raise ValueError(
            f"--n-prompts={args.n_prompts} cannot be split over {len(gpus)} GPUs"
        )

    checkpoint_path = Path(args.checkpoint).resolve()
    output_path = Path(args.output).resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model: {args.model}")
    print(f"Physical GPUs: {','.join(gpus)}")
    print(f"Fitting prompts: {args.n_prompts}")
    print("Loading WikiText prompts once in the parent process", flush=True)
    prompts = _load_prompts(args.n_prompts, args.min_chars)

    ranges = _partition_ranges(args.n_prompts, len(gpus))
    args_dict = vars(args).copy()
    worker_specs: list[tuple[mp.Process, Path]] = []
    ctx = mp.get_context("spawn")

    for part_idx, (gpu, (start, end)) in enumerate(zip(gpus, ranges, strict=True)):
        if len(gpus) == 1:
            worker_checkpoint = checkpoint_path
            worker_output = output_path
        else:
            worker_checkpoint = _part_path(
                str(checkpoint_path),
                part_idx=part_idx,
                n_parts=len(gpus),
                gpu=gpu,
                start=start,
                end=end,
            )
            worker_output = _part_path(
                str(output_path),
                part_idx=part_idx,
                n_parts=len(gpus),
                gpu=gpu,
                start=start,
                end=end,
            )
        process = ctx.Process(
            target=_fit_worker,
            args=(
                args_dict,
                gpu,
                prompts[start:end],
                str(worker_checkpoint),
                str(worker_output),
                part_idx,
                len(gpus),
                start,
                end,
            ),
            name=f"jlens-gpu-{gpu}",
        )
        process.start()
        worker_specs.append((process, worker_output))

    # Watch all workers concurrently. If any shard fails (for example due to
    # OOM), stop the others immediately instead of letting a partial job run
    # for hours before reporting the failure.
    failures: list[str] = []
    remaining = {process.sentinel: process for process, _ in worker_specs}
    try:
        while remaining and not failures:
            for sentinel in wait(remaining):
                process = remaining.pop(sentinel)
                process.join()
                if process.exitcode != 0:
                    failures.append(
                        f"{process.name} exited with code {process.exitcode}"
                    )
                    break
    except BaseException:
        for process in remaining.values():
            if process.is_alive():
                process.terminate()
        raise
    finally:
        if failures:
            for process in remaining.values():
                if process.is_alive():
                    process.terminate()
        for process in remaining.values():
            process.join()
    if failures:
        raise RuntimeError("one or more fitting workers failed: " + "; ".join(failures))

    if len(gpus) > 1:
        import jlens

        partial_lenses = [
            jlens.JacobianLens.load(str(worker_output))
            for _, worker_output in worker_specs
        ]
        lens = jlens.JacobianLens.merge(partial_lenses)
        if lens.n_prompts != args.n_prompts:
            raise RuntimeError(
                f"merged lens has {lens.n_prompts} prompts, expected {args.n_prompts}"
            )
        lens.save(str(output_path))
        print(f"Merged {len(gpus)} partial lenses: {lens}", flush=True)

    print(f"Saved final lens: {output_path}", flush=True)


if __name__ == "__main__":
    main()
