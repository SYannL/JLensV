#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
REPO_DIR="$(cd -- "${SUITE_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-${REPO_DIR}/.venv/bin/python}"
GPU_ID="${GPU_ID:-0}"
MODEL_ID="${MODEL_ID:-${REPO_DIR}/models/Qwen3.5-4B}"
RUN_NAME="${RUN_NAME:-pilot_qwen35_4b_nonthinking}"
RUN_DIR="${SUITE_DIR}/runs/${RUN_NAME}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/create_selection.py" --overwrite

"${PYTHON_BIN}" "${SCRIPT_DIR}/run_generation.py" \
  --selection "${SUITE_DIR}/selections/pilot.jsonl" \
  --run-dir "${RUN_DIR}" \
  --model "${MODEL_ID}" \
  --gpu "${GPU_ID}" \
  --no-thinking \
  --decoding sample \
  --temperature 0.7 \
  --top-p 0.8 \
  --sampling-top-k 20 \
  --max-new-tokens 2048 \
  --metric-top-k 20

"${PYTHON_BIN}" "${SCRIPT_DIR}/audit_generation.py" \
  --selection "${SUITE_DIR}/selections/pilot.jsonl" \
  --run-dir "${RUN_DIR}" \
  --strict
