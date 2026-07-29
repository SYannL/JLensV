# Stage A：内部推理验证数据流水线

本目录是 JLensV 第一阶段（Stage A）的独立、可复现数据工作区。它依据
`research/internal_verification/README.md` 中的数据路线准备五个公开数据源，
服务于格式审计、标签可靠性检查和小规模 generation-only 可行性实验。

## 范围

| 数据集 | Stage A 范围 | 主要用途 | 预期处理后数量 |
|---|---|---|---:|
| ProcessBench | MATH、OlympiadBench、Omni-MATH；明确排除 GSM8K | 候选过程与首错定位 | 3,000 |
| MATH-500 | test | 自然数学迁移与参考解 | 500 |
| PrOntoQA | 1-hop train / 5-hop test，ProofsOnly OOD 配置 | 精确逻辑链与深度 OOD | 100 |
| StepGame | test，1–10 hop 各固定抽样 100 | 实体绑定与空间状态追踪 | 1,000 |
| BBEH | 官方 mini，任务名由固定 revision 的 full tasks 精确恢复 | 多类型困难推理 | 460 |

完整默认产物应为 7 个 JSONL、5,060 条记录。REVEAL 因 gated
访问和禁止再分发条款不进入自动下载流程；ProcessBench/GSM8K 也按研究设计排除。

## 目录结构

```text
stage_a_internal_verification/
├── config/sources.json          # 数据源、commit/revision、split、许可与抽样参数
├── schema/record.schema.json    # 统一记录 schema（Draft 2020-12）
├── scripts/
│   ├── download_datasets.py     # 只下载固定版本的原始快照
│   ├── process_datasets.py      # 只做确定性规范化与分层抽样
│   ├── validate_datasets.py     # 完全离线的数据完整性验证
│   └── _dataset_common.py       # 原子写入、哈希与 schema 检查
├── raw/                         # 第三方原始数据（默认 Git 忽略）
├── processed/                   # 统一 JSONL（默认 Git 忽略）
├── manifests/                   # SHA-256、字节数、记录数与 provenance
└── tests/test_pipeline.py       # 无网络单元测试
```

所有默认路径均相对本目录解析，命令可从仓库任意工作目录运行。

## 快速开始

建议在仓库虚拟环境内安装数据依赖：

```bash
.venv/bin/pip install \
  -r data/stage_a_internal_verification/requirements.txt
```

从固定 revision 下载原始数据：

```bash
.venv/bin/python \
  data/stage_a_internal_verification/scripts/download_datasets.py
```

规范化、分层抽样并立即执行完整验证：

```bash
.venv/bin/python \
  data/stage_a_internal_verification/scripts/process_datasets.py
```

之后可在不联网的环境中复核所有内容：

```bash
.venv/bin/python \
  data/stage_a_internal_verification/scripts/validate_datasets.py
```

只处理部分数据时，两个阶段使用同一选择：

```bash
.venv/bin/python \
  data/stage_a_internal_verification/scripts/download_datasets.py \
  --datasets math500 prontoqa

.venv/bin/python \
  data/stage_a_internal_verification/scripts/process_datasets.py \
  --datasets math500 prontoqa
```

已存在的文件默认不会被覆盖。源 manifest 发生变化后，处理脚本会要求显式传入
`--overwrite`，防止把新原始数据与旧规范化结果静默混合。Hugging Face 缓存默认
位于 `/tmp/jlensv_stage_a_hf`，可用 `JLENSV_HF_CACHE` 或 `--cache-dir` 修改。

## 数据契约

每条记录包含：

- 稳定 ID、数据集、原始 split 和任务家族；
- `input`：上下文、问题和选项；
- `gold`：答案、源提供的 reasoning、首错、过程/结果正确性和状态轨迹；
- `candidate`：数据源提供的候选答案、步骤和生成器；
- `capabilities`：这条数据实际支持的验证主张；
- `metadata`：原始索引、难度、hop、配置和必要的语义说明。

数据源未提供的字段严格保留为 `null` 或空列表。处理流程不会调用 LLM，不会合成
gold steps，也不会把自然语言参考解伪装成模型内部状态。MATH-500 的
`reasoning_steps` 仅按源文本空行分段，并在 metadata 中明确记录这一限制。

## 可复现性与验收

下载阶段生成 `manifests/raw_manifest.json`，处理阶段生成
`manifests/processed_manifest.json`。验证器检查：

1. 文件存在性、字节数和 SHA-256；
2. JSON/JSONL 可解析性；
3. 处理后记录总数；
4. schema、字段类型和 capability 布尔值；
5. 每个文件内 ID 唯一性；
6. 处理 manifest 与原始 manifest 的内容哈希绑定。

运行无网络测试：

```bash
.venv/bin/python -m unittest discover \
  -s data/stage_a_internal_verification/tests -v
```

## 研究使用边界

ProcessBench 的候选步骤来自其他生成模型，因此适用于
**verifier-state track**：分析本项目 verifier 阅读候选过程时的状态。它们不能被
称为原生成器犯错时的原生 hidden states。

MATH-500、PrOntoQA、StepGame 和 BBEH 适用于后续
**solver-state track**：让目标 backbone 自己生成，并同步采集其 activation。
Stage A 数据准备本身不生成 solver trajectories；generation-only、官方评分、
标签噪声审计和 JLens 选择策略应作为下一层实验产物另行记录。

第三方数据继续受各自许可证和使用条款约束。本目录的 `.gitignore` 默认阻止原始
和处理后数据被意外提交；如需发布快照，应先逐项复核许可与再分发条件。
