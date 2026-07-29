# Reproducing JLensV

## What the repository contains

The repository tracks the project source, fixed configurations, selection
files, manifests, prompts, logs, Stage A generation artifacts, Stage B saved
smoke hidden states, derived readouts, and analysis reports. Binary research
artifacts under the tracked experiment directories are stored with Git LFS.

The following local or upstream artifacts are intentionally not duplicated:

1. third-party raw and normalized benchmark records under
   `data/stage_a_internal_verification/{raw,processed}` and the legacy
   `data/internal_verification` copy;
2. the public Qwen checkpoint, whose 5.33 GB source shard is larger than
   GitHub's maximum LFS object size;
3. the repository-local `outputs/` directory, including legacy GSM8K outputs
   and locally fitted lens checkpoints.

Both are pinned to immutable revisions and restored by repository scripts.

## One-command setup

Install Git LFS and `uv`, then clone normally:

```bash
git lfs install
git clone git@github.com:SYannL/JLensV.git
cd JLensV
bash scripts/bootstrap_reproduction.sh
```

The bootstrap command:

- creates `.venv` from the committed `uv.lock`;
- downloads and SHA-256 verifies
  `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`;
- downloads every dataset at the revisions in
  `data/stage_a_internal_verification/config/sources.json`;
- restores the official GSM8K train/test files at pinned upstream commit
  `3101c7d5072418e28b9008a6636bde82a006892c`;
- deterministically rebuilds the normalized benchmark files;
- validates their counts and checksums against the committed manifests;
- runs the complete test suite.

## Manual setup

```bash
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
```

To audit an already downloaded model without network access:

```bash
.venv/bin/python scripts/download_model.py --verify-only
```

## Experiment provenance

- Stage A dataset preparation:
  `data/stage_a_internal_verification/README.md`
- Stage A generation and audit:
  `data/stage_a_generation_audit/README.md`
- Stage B frozen solver-state discovery:
  `data/stage_b_solver_state_discovery/README.md`

Each immutable run stores its experiment configuration and source checksums.
Stage B replay reuses the exact saved source token IDs and refuses inputs whose
contracts or hashes differ.

Exact floating-point equality is subject to the recorded CUDA/PyTorch hardware
stack. Dataset membership, prompts, source token IDs, annotations, selection
policy, analysis definitions, and committed outputs are independently
checksum-bound.
