# Stage A generation-only 与数据审计

本目录承接 `data/stage_a_internal_verification/processed/`，执行研究路线 Stage A
的第二部分：固定样本、让目标模型生成、保存可重放轨迹、运行确定性评分，并判断
哪些样本可以进入后续 JLens、hidden-state probe、CoE 和因果分析。

## 为什么没有直接复用 GSM8K 脚本

现有 `analyze_gsm8k_hard_with_lens.py` 对 GSM8K 的 artifact 保存较完整，但存在
三个不适合直接扩展的问题：

1. prompt、解析器和评分器绑定 GSM8K；
2. generation score 经过 temperature/top-k/top-p 等采样处理，不能作为严格的
   原始模型 MaxProb、entropy、PPL 或 energy；
3. 没有为 ProcessBench 的 verifier-state track 与其他数据的 solver-state
   track 建立统一输出契约。

本目录的新生成器保存 Transformers `output_logits=True` 返回的逐 token
**未采样扭曲原始模型 logits 汇总**，并记录完整 token replay contract。

所有任务共用的 system prompt 只要求简洁 reasoning 和一个 `FINAL:` 行。solver
prompt 仅由规范化 `input` 构造；ProcessBench prompt 仅额外包含源提供的 candidate
steps。每次生成前都会用 redacted gold/metadata 重建 prompt 并比较，若 prompt
依赖标签或分析 metadata 会立即失败。

## 默认 pilot

默认 selection 共 550 条：

| 数据集 | 数量 | 分层 |
|---|---:|---|
| ProcessBench | 150 | split × process correctness × final correctness |
| MATH-500 | 100 | subject × level |
| PrOntoQA | 100 | 当前 ProofsOnly OOD 配置全量 |
| StepGame | 100 | 1–10 hop 各 10 |
| BBEH | 100 | 23 个任务平衡抽样 |

ProcessBench 使用 verifier prompt：找候选过程的首个错误步骤；其他四个数据集让
目标模型自己解题。

## 每条样本保存什么

每个样本目录包含：

- `generation.json`
  - 完整的规范化源记录与 selection group；
  - system/user message、精确 rendered prompt、prompt token IDs；
  - 模型和 tokenizer revision、config/chat-template hash、软件版本；
  - 每样本确定性 seed 与完整 decoding 参数；
  - 原始输出、thinking/final 文本、生成 token IDs 和完整序列 IDs；
  - EOS、截断、格式、解析、评分和 replay-ready 状态；
  - scorer 版本和评分可靠性说明。
- `token_metrics.npz`
  - 原始 chosen-token logprob 与全词表精确 rank；
  - entropy、MaxProb、logsumexp、energy；
  - 原始 top-k token IDs、logits 和 logprobs；
  - token 与 prediction-position 的绝对对齐。
- `error.json`
  - 失败类型、消息与 traceback；单样本失败不会破坏整个 run。

完整 token IDs 足以在固定模型上以 teacher forcing 重放同一轨迹，从而提取全层
hidden states、JLens、SAE 或 CoE。Stage A 不直接保存全层 activation，因为
`sequence × layer × d_model` 会让 550 条 pilot 产生数百 GB 的重复数据。

重放保证的是相同 token trajectory；不同 attention kernel、软件版本或硬件可能
产生浮点级差异。因此模型 config、revision 和软件版本都必须保留。

## 运行

先再次确认处理后数据：

```bash
.venv/bin/python \
  data/stage_a_internal_verification/scripts/validate_datasets.py
```

创建或刷新 550 条 pilot：

```bash
.venv/bin/python \
  data/stage_a_generation_audit/scripts/create_selection.py \
  --overwrite
```

运行 Qwen3.5-4B 非 thinking pilot：

```bash
.venv/bin/python \
  data/stage_a_generation_audit/scripts/run_generation.py \
  --selection data/stage_a_generation_audit/selections/pilot.jsonl \
  --run-dir data/stage_a_generation_audit/runs/pilot_qwen35_4b_nonthinking \
  --model models/Qwen3.5-4B \
  --gpu 0 \
  --no-thinking \
  --decoding sample \
  --temperature 0.7 \
  --top-p 0.8 \
  --sampling-top-k 20 \
  --max-new-tokens 2048 \
  --metric-top-k 20
```

完成后严格审计：

```bash
.venv/bin/python \
  data/stage_a_generation_audit/scripts/audit_generation.py \
  --selection data/stage_a_generation_audit/selections/pilot.jsonl \
  --run-dir data/stage_a_generation_audit/runs/pilot_qwen35_4b_nonthinking \
  --strict
```

也可以用包装脚本一次完成。可通过环境变量覆盖 GPU、模型和 run 名：

```bash
GPU_ID=0 \
MODEL_ID=models/Qwen3.5-4B \
RUN_NAME=pilot_qwen35_4b_nonthinking \
bash data/stage_a_generation_audit/scripts/run_pilot.sh
```

脚本按样本自动续跑。相同 run 目录中的模型、prompt、采样配置、selection 或代码
指纹不一致时会拒绝混合；需要变更实验配置时应使用新的 `RUN_NAME`。

## 小规模 smoke test

首次运行建议使用专用 smoke selection；它从五个数据集各取 1 条，能够覆盖所有
prompt、解析器和评分路径：

```bash
.venv/bin/python \
  data/stage_a_generation_audit/scripts/create_selection.py \
  --mode smoke \
  --output data/stage_a_generation_audit/selections/smoke.jsonl \
  --manifest data/stage_a_generation_audit/selections/smoke_manifest.json \
  --overwrite

.venv/bin/python \
  data/stage_a_generation_audit/scripts/run_generation.py \
  --selection data/stage_a_generation_audit/selections/smoke.jsonl \
  --run-dir data/stage_a_generation_audit/runs/smoke_v2_qwen35_4b \
  --model models/Qwen3.5-4B \
  --gpu 0 \
  --no-thinking \
  --max-new-tokens 1024
```

五条全部生成后可以直接进行严格审计：

```bash
.venv/bin/python \
  data/stage_a_generation_audit/scripts/audit_generation.py \
  --selection data/stage_a_generation_audit/selections/smoke.jsonl \
  --run-dir data/stage_a_generation_audit/runs/smoke_v2_qwen35_4b \
  --strict
```

`--strict` 只有在每条选择样本都满足
`complete + parsed + raw_metrics_complete + replay_ready` 时才返回成功；存在截断、
格式错误、缺失文件或 checksum 问题都会返回非零状态。

## 全量与多 GPU

创建 5,060 条全量 selection：

```bash
.venv/bin/python \
  data/stage_a_generation_audit/scripts/create_selection.py \
  --mode full \
  --output data/stage_a_generation_audit/selections/full.jsonl \
  --manifest data/stage_a_generation_audit/selections/full_manifest.json \
  --overwrite
```

多 GPU 时，各进程必须使用完全相同的 run 目录和实验参数，仅改变
`--num-shards`、`--shard-index` 和 `--gpu`。例如两个 GPU：

```bash
.venv/bin/python \
  data/stage_a_generation_audit/scripts/run_generation.py \
  --selection data/stage_a_generation_audit/selections/pilot.jsonl \
  --run-dir data/stage_a_generation_audit/runs/pilot_2gpu \
  --model models/Qwen3.5-4B \
  --gpu 0 --num-shards 2 --shard-index 0 &

.venv/bin/python \
  data/stage_a_generation_audit/scripts/run_generation.py \
  --selection data/stage_a_generation_audit/selections/pilot.jsonl \
  --run-dir data/stage_a_generation_audit/runs/pilot_2gpu \
  --model models/Qwen3.5-4B \
  --gpu 1 --num-shards 2 --shard-index 1 &

wait
```

`--gpu` 会在进程内部设置 `CUDA_VISIBLE_DEVICES`，普通本地运行不需要再添加环境
变量前缀。

所有 shard 完成后只运行一次 audit。

## 评分边界

- ProcessBench、PrOntoQA、StepGame 和 BBEH 使用确定性规范化 exact match。
- MATH-500 会处理常见 boxed、空白和纯数值形式，但复杂等价表达式的 mismatch
  标记为 provisional；正式 accuracy 需要后续接入 symbolic/official math
  verifier。
- 不论自动评分是否可靠，源 gold、模型原文、解析结果和 scorer 版本均完整保存，
  后续可重新评分，无需重新生成。
- 只有 `complete + parsed + raw_metrics_complete + replay_ready` 的样本会被标为
  `eligible_for_internal_analysis`。
- 报告中的 accuracy 只以 `complete + parsed` 样本为分母；截断输出即使最后一行
  碰巧可解析，也不会污染准确率。
- 完整 JLens 首轮应从 eligible 样本中按数据集、难度和正确/错误平衡选择，而不是
  对 550 条无差别运行。
