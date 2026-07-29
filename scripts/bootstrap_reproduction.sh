#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_DIR}"

uv sync --extra dev

.venv/bin/python scripts/download_model.py

.venv/bin/pip install \
  -r data/stage_a_internal_verification/requirements.txt

.venv/bin/python scripts/download_gsm8k.py --overwrite

.venv/bin/python \
  data/stage_a_internal_verification/scripts/download_datasets.py

.venv/bin/python \
  data/stage_a_internal_verification/scripts/process_datasets.py

.venv/bin/python \
  data/stage_a_internal_verification/scripts/validate_datasets.py

.venv/bin/python -m pytest -q

echo "Reproduction environment, model, datasets, and tests are ready."
