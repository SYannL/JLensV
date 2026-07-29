#!/usr/bin/env python3
"""Download and verify the official GSM8K train/test source files."""

from __future__ import annotations

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path

SOURCE_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
SOURCE_ROOT = (
    "https://raw.githubusercontent.com/openai/grade-school-math/"
    f"{SOURCE_REVISION}/grade_school_math/data"
)
DEFAULT_DESTINATION = (
    Path(__file__).resolve().parents[1] / "data" / "dataset" / "gms8k"
)
EXPECTED_SHA256 = {
    "train.jsonl": (
        "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"
    ),
    "test.jsonl": (
        "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination", type=Path, default=DEFAULT_DESTINATION
    )
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    temporary = destination.with_name(
        f".{destination.name}.tmp.{os.getpid()}"
    )
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            temporary.write_bytes(response.read())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def verify(destination: Path) -> None:
    failures = []
    for filename, expected in EXPECTED_SHA256.items():
        path = destination / filename
        if not path.is_file():
            failures.append(f"missing: {filename}")
            continue
        actual = sha256(path)
        if actual != expected:
            failures.append(
                f"checksum: {filename}: expected {expected}, got {actual}"
            )
    if failures:
        raise ValueError("GSM8K verification failed:\n" + "\n".join(failures))
    print(
        f"Verified {len(EXPECTED_SHA256)} GSM8K files at "
        f"{destination.resolve()}."
    )


def main() -> None:
    args = parse_args()
    destination = args.destination.resolve()
    if not args.verify_only:
        destination.mkdir(parents=True, exist_ok=True)
        for filename in EXPECTED_SHA256:
            path = destination / filename
            if args.overwrite or not path.exists():
                print(f"Downloading {filename}...", flush=True)
                download(f"{SOURCE_ROOT}/{filename}", path)
    verify(destination)


if __name__ == "__main__":
    main()
