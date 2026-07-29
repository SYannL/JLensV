# Model checkpoint

The experiments use `Qwen/Qwen3.5-4B` at immutable Hugging Face revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.

The upstream checkpoint is not duplicated in GitHub: its largest source shard
is 5.33 GB, above GitHub's maximum Git LFS object size. Download and verify the
exact checkpoint with:

```bash
.venv/bin/python scripts/download_model.py
```

The resulting local path is `models/Qwen3.5-4B`, which is the default consumed
by all Stage A and Stage B commands.
