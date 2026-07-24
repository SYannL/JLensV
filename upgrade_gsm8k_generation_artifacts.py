#!/usr/bin/env python3
"""Upgrade saved GSM8K generation.json files to the model_output-first layout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write upgraded files; otherwise only report what would change",
    )
    return parser.parse_args()


def _upgrade_artifact(artifact: dict[str, Any]) -> dict[str, Any]:
    generated = dict(artifact["generation"])
    existing_output = artifact.get("model_output", {})
    thinking = existing_output.get("thinking", generated.pop("thinking", None))
    final_response = existing_output.get(
        "final_response",
        generated.pop("final_response", None),
    )
    decoded_text = existing_output.get("text", generated.pop("text", ""))
    raw_text = existing_output.get(
        "raw_text",
        "".join(generated.get("tokens", [])),
    )
    model_output = {
        "raw_text": raw_text,
        "text": decoded_text,
        "thinking": thinking,
        "final_response": final_response,
    }

    terminal_tokens = {"<|endoftext|>", "<|im_end|>"}
    ended_with_eos = bool(
        generated.get("ended_with_eos", False)
        or (
            generated.get("tokens")
            and generated["tokens"][-1] in terminal_tokens
        )
    )
    generated["ended_with_eos"] = ended_with_eos
    generated["hit_max_new_tokens"] = bool(
        generated.get(
            "hit_max_new_tokens",
            generated["n_tokens"] >= generated["max_new_tokens"]
            and not ended_with_eos,
        )
    )
    quality = dict(artifact.get("quality", {}))
    thinking_closed = bool(
        quality.get(
            "thinking_closed",
            thinking and thinking.get("closed", False),
        )
    )
    valid_hash = bool(quality.get("valid_final_hash_answer", False))
    quality["ended_with_eos"] = ended_with_eos
    quality["thinking_closed"] = thinking_closed
    quality["final_response_present"] = bool(
        final_response and final_response.get("text", "").strip()
    )
    quality["hit_max_new_tokens"] = generated["hit_max_new_tokens"]
    quality["complete"] = bool(ended_with_eos and thinking_closed and valid_hash)

    upgraded = {
        "status": artifact["status"],
        "timestamp_utc": artifact["timestamp_utc"],
        "model_output": model_output,
    }
    upgraded.update(
        {
            key: value
            for key, value in artifact.items()
            if key
            not in {
                "status",
                "timestamp_utc",
                "model_output",
                "generation",
                "quality",
            }
        }
    )
    upgraded["generation"] = generated
    upgraded["quality"] = quality
    return upgraded


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _refresh_summary(input_dir: Path) -> None:
    from analyze_gsm8k_hard_with_lens import _summary_record

    rows = [
        _summary_record(sample_dir)
        for sample_dir in sorted(input_dir.glob("wrong-*"))
        if (sample_dir / "generation.json").exists()
    ]
    summary_path = input_dir / "summary.jsonl"
    temporary = summary_path.with_name(f".{summary_path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, summary_path)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    paths = sorted(input_dir.glob("wrong-*/generation.json"))
    if not paths:
        raise FileNotFoundError(f"no generation.json files under {input_dir}")
    for path in paths:
        artifact = json.loads(path.read_text(encoding="utf-8"))
        upgraded = _upgrade_artifact(artifact)
        if args.write:
            _atomic_json(path, upgraded)
        print(f"{'upgraded' if args.write else 'would upgrade'} {path}")
    if args.write:
        _refresh_summary(input_dir)
        print(f"refreshed {input_dir / 'summary.jsonl'}")
    print(f"Done: {len(paths)} artifacts")


if __name__ == "__main__":
    main()
