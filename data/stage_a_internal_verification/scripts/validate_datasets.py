#!/usr/bin/env python3
"""Validate Stage A manifests, checksums, counts, IDs, and record schemas."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from _common import DEFAULT_MANIFEST_DIR, validate_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument(
        "--scope",
        choices=("raw", "processed", "all"),
        default="all",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scope in {"raw", "all"}:
        count, records = validate_manifest(
            args.manifest_dir / "raw_manifest.json",
            processed=False,
            expected_schema="stage-a-raw-manifest-v1",
        )
        print(f"Raw: {count} artifacts verified ({records} JSONL rows).")
    if args.scope in {"processed", "all"}:
        count, records = validate_manifest(
            args.manifest_dir / "processed_manifest.json",
            processed=True,
            expected_schema="stage-a-processed-manifest-v1",
        )
        print(f"Processed: {count} artifacts, {records} records verified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
