#!/usr/bin/env python3
"""Download and verify the exact Qwen checkpoint used by the experiments."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

MODEL_REPO = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DEFAULT_DESTINATION = (
    Path(__file__).resolve().parents[1] / "models" / "Qwen3.5-4B"
)
EXPECTED_SHA256 = {
    "LICENSE": "bbedc3fda3305820b977265f01b8619d87570a6739de3a5582c3464840f1e57a",
    "README.md": "1406be1b6b8fd8a6545870da516912804756593628a1d0fb0a7965211e82a7bb",
    "chat_template.jinja": (
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715"
    ),
    "config.json": (
        "ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670"
    ),
    "merges.txt": (
        "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d"
    ),
    "model.safetensors-00001-of-00002.safetensors": (
        "26a93f066e1916adb13453dae5a0c707c0fbc71299ed98779571a907b8e74c61"
    ),
    "model.safetensors-00002-of-00002.safetensors": (
        "cb544bd9bfae93dc59b0f22b292f5933573854a7f9b97835c67060d7d910e188"
    ),
    "model.safetensors.index.json": (
        "cf3f798ee02ba45f9622aa8892a47369ab667d0afbf154ee7c2212de42e6302d"
    ),
    "preprocessor_config.json": (
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
    ),
    "tokenizer.json": (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    ),
    "tokenizer_config.json": (
        "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8"
    ),
    "video_preprocessor_config.json": (
        "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13"
    ),
    "vocab.json": (
        "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path, default=DEFAULT_DESTINATION
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(destination: Path) -> None:
    failures = []
    for relative, expected in EXPECTED_SHA256.items():
        path = destination / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(
                f"checksum: {relative}: expected {expected}, got {actual}"
            )
    if failures:
        raise ValueError("model verification failed:\n" + "\n".join(failures))
    print(
        f"Verified {len(EXPECTED_SHA256)} model files at "
        f"{destination.resolve()}."
    )


def main() -> None:
    args = parse_args()
    destination = args.destination.resolve()
    if not args.verify_only:
        from huggingface_hub import snapshot_download

        destination.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Downloading {MODEL_REPO}@{MODEL_REVISION} to {destination}...",
            flush=True,
        )
        snapshot_download(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            local_dir=destination,
            force_download=args.force_download,
        )
    verify(destination)


if __name__ == "__main__":
    main()
