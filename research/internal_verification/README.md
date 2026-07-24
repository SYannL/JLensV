# JLens 内部推理验证：研究记录与数据路线

最后更新：2026-07-24

本文档是本项目关于 **interpretable LLM / internal reasoning
verification** 的单一研究记录。它固定当前术语、已有文献、初期方案、
第一项探索及数据决策，避免后续实验把“发现”“方法”和“刷 benchmark”
混为一谈。

## 1. 当前结论

论文应当采用：

> **发现主导，方法承载，性能提升作为结果。**

我们不把工作定义成“再训练一个 hidden-state correctness classifier”，也不只做
JLens 可视化。目标是发现并因果验证：

> LLM 的推理错误何时、以什么形式进入内部可观察表示；该内部偏离是否早于
> 可见文本错误；以及什么计算机制使一次推理可以或不可以被内部验证。

方法的作用是把这个现象变成可测量、可定位、可干预的对象。多数据集、多
backbone 和最新 baseline 是证明普适性的必要条件，但不是论文的中心创新。

### 1.1 暂定中心命题

推理过程中可能形成一条跨层、跨 token 的内部语义状态轨迹。错误答案至少可能
来自四类不同过程：

1. 内部语义状态已经出错，文本随后才显露错误；
2. 内部曾形成正确状态，但被后续错误状态覆盖；
3. 内部状态正确，最终 readout、目标选择或格式化失败；
4. 计算没有以当前观测工具可读的形式进入 verbalizable workspace。

第四类不是普通的“probe 预测错了”，而是需要显式建模的
**observability boundary（可观测性边界）**。

## 2. 术语边界

为避免与既有 verification 工作混淆，固定如下术语。

### 2.1 外部结果验证

\[
V_{\mathrm{outcome}}(q,a,a^*)
\]

由规则、执行器或更强外部 LLM 检查最终答案。它可以很准确，但不能说明答案由
什么内部过程产生。

### 2.2 外部过程验证

\[
V_{\mathrm{process}}(q,r,r^*)
\]

外部 verifier 阅读候选 reasoning，并与规则、证据或 golden reasoning 比较。
用户此前的 verification 工作主要属于这一类和上一类。

### 2.3 内部验证

\[
V_{\mathrm{internal}}
:
(q,h_{1:L,1:T},y)
\rightarrow
(\hat c,\hat t,\hat e,\hat o)
\]

它根据 **被验证模型自身生成本次答案时形成的内部状态** 输出：

- \(\hat c\)：推理或答案是否可靠；
- \(\hat t\)：首次有证据支持的内部偏离位置；
- \(\hat e\)：错误类型；
- \(\hat o\)：本次计算对当前传感器是否具有足够可观测性。

内部验证不是让同一个模型重新做一次题，也不是仅从另一个模型的 hidden state
预测标签。

### 2.4 Golden reasoning 的合法角色

Golden reasoning 可以参与，但不应成为主 verifier。它主要承担：

1. **语义坐标系**：把参考解转成约束或状态序列，而非逐句匹配；
2. **首错 ground truth**：定义候选过程首次违反语义约束的位置；
3. **训练/评估监督**：帮助验证内部信号，但 reference-free setting 在测试时不
   依赖完整参考解。

golden rationale 不是唯一合法路径，也不是模型内部过程的 ground truth。自然
语言表面差异不能被直接当作错误。

## 3. 为什么从 JLens 出发

JLens（Jacobian Lens）读取一个内部激活在模型自身计算下“倾向于使模型说出
什么”。对层 \(\ell\) 的平均传输为：

\[
J_\ell =
\mathbb{E}\left[
\frac{\partial h_{\mathrm{final},t'}}
     {\partial h_{\ell,t}}
\right]
\]

其词表读出近似为：

\[
\operatorname{Lens}_{\ell}(h_{\ell,t})
=
\operatorname{softmax}
\left(
W_U\,\operatorname{norm}(J_\ell h_{\ell,t})
\right).
\]

JLens 的优势不是“它一定比所有 probe 更准”，而是：

- 使用模型自身的下游 Jacobian 与 unembedding，读出与 verbal report 直接相关；
- 可以在 layer × position 上追踪概念的形成、保持、消失与覆盖；
- 与“verbalizable global workspace”假设存在明确理论联系；
- 适合提出时间性问题：内部概念偏离是否早于模型实际说错。

但论文问题必须是工具无关的。JLens 只能作为主要传感器之一，至少需要与
logit/linear probes、SAE/T-SAE 或 activation/path patching 交叉验证。

### 3.1 已知边界

- J-space 目前近似为 token-indexed、稀疏非负概念组合；
- readout 更接近“概念包”，缺少可靠的实体绑定、关系和语法；
- 一个词在内部可读不等于它在当前语义角色中被正确使用；
- 相关 readout 不等于因果计算；
- 熟练或自动化计算可能绕过 verbalizable workspace；
- 干预可能把激活推到分布外状态，不能把任意 patch 结果直接解释为机制。

这正好解释了第一项 GSM8K 探索为什么不能靠末层 gold digit rank 判断错误。

## 4. 可证伪研究假设

当前先固定五个假设，后续允许实验推翻，而不是事后修改定义。

### H1：内部偏离领先

在具有明确中间状态的推理中，首次内部语义偏离
\(t_{\mathrm{internal}}\) 经常早于首次可见文本错误
\(t_{\mathrm{text}}\)。

### H2：计算错误与 readout 错误可分离

存在可重复的 matched cases：

- 内部状态正确但最终答案错误；
- 最终答案相同但内部状态轨迹不同；
- 表面 CoT 相近但内部机制不同。

### H3：结构比词汇可用性更重要

实体绑定、状态更新、约束满足和概念转移，比“正确答案 token 是否在某层可读”
更能定位真实错误。

### H4：可观测性依赖计算机制

新颖、组合性、需要工作记忆的计算更可能进入 J-space；熟练、压缩或自动化计算
可能绕过它。内部 verifier 因此必须预测或校准 observability，并允许 abstain。

### H5：内部证据具有非冗余价值

在控制题目、可见 CoT、token probability、entropy、长度以及 golden semantic
state 后，经因果验证的内部轨迹仍能提供至少一种非冗余能力：

- 更早定位首错；
- 区分 computation 与 readout failure；
- 检测表面正确但机制不忠实的过程；
- 指导更有针对性的重算或修正。

若 H5 不成立，JLens 不应被包装成 verifier；它最多是机制分析工具。

## 5. 初期方法轮廓

暂称 **Mechanistic Verification Trace (MVT)**。这只是研究载体，不是已经确定
的最终模型。

从内部状态读取：

\[
z_{\ell,t} = \phi_{\mathrm{JLens}}(h_{\ell,t}).
\]

但不把每格 top-k token 直接交给外部 LLM 编故事，而是构造：

- 概念出现、持续、消失和覆盖事件；
- 跨层概念形成时间；
- 与当前实体、变量、单位、关系或约束的绑定；
- 内部状态与下一次文本 commitment 的一致性；
- 轨迹突变、冲突、自我修正和回退；
- 当前样本的 observability/coverage。

候选评分分解为：

\[
\mathrm{Validity}(z),\qquad
\mathrm{Commitment}(z,y),\qquad
\mathrm{Observability}(z).
\]

最终输出不只包含 correct/incorrect，还应包括 first-error、error type 和
abstention。

### 5.1 必须避免的弱版本

- 把 JLens top-k 复制到 GPT-4o-mini prompt，然后只报告分类准确率；
- 只训练一个 hidden-state probe，与 ReProbe 类方法做小幅 AUROC 比较；
- 只比较正确与错误答案的最后几层；
- 把 golden rationale 当作唯一合法路径；
- 把相关 token readout 叙述成因果机制；
- 用 GSM8K exact match 直接当作可靠 ground truth。

### 5.2 Oral 级证据链

1. **时间证据**：内部偏离稳定领先文本错误；
2. **增量证据**：控制表面信号后内部轨迹仍增加信息；
3. **因果证据**：删除、交换或修复被识别状态会系统性改变后续错误；
4. **解耦证据**：matched cases 分离答案、CoT 与机制；
5. **边界证据**：明确什么时候可观察、什么时候应 abstain；
6. **闭环应用**：根据错误机制选择重算、局部修正或仅重新 readout。

## 6. Exploration 01：Qwen3.5-4B × GSM8K difficult cases

这是本研究的 **第一个 hypothesis-generation exploration**，不是正式 benchmark
结果。

### 6.1 设置

- Backbone：本地 `Qwen3.5-4B`
- Lens：已拟合的 Qwen3.5-4B Jacobian lens
- 数据：20 个预先选出的 GSM8K difficult/wrong cases
- thinking：关闭
- decoding：sampling，temperature 0.7，top-p 0.8，top-k 20
- 初始长度：512 new tokens
- 自适应重跑：case 2 使用 1024，case 24 使用 2048
- 最终完成：20/20 均 EOS 结束并含合法独立 `####` final answer
- 每个样本保留原始 prompt、原始输出、token score、layer × position trace

主要本地证据：

- [运行目录](../../outputs/gsm8k_nonthinking_20_v2_adaptive/)
- [自动分析报告](../../outputs/gsm8k_nonthinking_20_v2_adaptive/analysis/report.md)
- [13 个官方错误样本的人工复核](../../outputs/gsm8k_nonthinking_20_v2_adaptive/analysis/incorrect_cases_review.md)
- [统一 cases](../../outputs/gsm8k_nonthinking_20_v2_adaptive/analysis/cases.jsonl)

注意：运行目录的顶层 `run_config.json` 来自 512-token base run；两个自适应样本
的实际生成参数以各自 `generation.json` 及 `adaptive_run_manifest.json` 为准。

### 6.2 结果

- GSM8K exact-match：7/20；
- 官方判错 13 题中：
  - 5 题为明确模型推理/理解错误；
  - 3 题存在题意不充分、冲突或模型选择与 benchmark 假设不同；
  - 5 题是单位等价、标签/解析、题目或官方 rationale 错误导致的 false
    negative；
- 至少修正这 5 个 false negative 后，人工裁定分数不低于 12/20。

五个明确错误主要是：

- allocation/quantity tracking；
- semantic scope；
- reference 与 state update；
- modifier scope；
- unsupported inference。

它们共同指向 **binding 和 state tracking**，而不是基本算术能力。

### 6.3 JLens 初步观察

- 错误回答与正确回答在最后几层都高度词汇收敛；
- incorrect 的 chosen-token top-1 rate 约 95.94%，correct 约 95.31%；
- layer-30 与 final readout agreement 在错误组并没有降低；
- gold digit 的 late-layer 可读性也没有把正确和错误可靠分开；
- 模型经常拥有相关数字，却把数字绑定到错误实体、单位、时间范围或问题解释。

因此当前最重要的负结果是：

> **末层词汇收敛和 gold-token availability 不是内部验证器。**

后续应在人工确认的首次语义分歧附近，分析关系、实体和状态转移，而不是继续对
整段输出平均 gold digit rank。

### 6.4 Exploration 01 的限制

- \(n=20\)，不能做总体统计主张；
- 样本由既有 wrong/difficult 列表选取，存在选择偏差；
- GSM8K 官方标签和 rationale 有明显噪声；
- 当前 JLens 输出是 lexical readout，不编码可靠 binding；
- 当前自动 step alignment 较粗；
- 没有 causal intervention；
- 只使用一个 backbone 和一个任务家族。

## 7. 首批数据套件

数据已统一准备到
[data/internal_verification](../../data/internal_verification/)。下载与规范化脚本
为 [prepare_datasets.py](prepare_datasets.py)，固定版本和校验和见
[manifest.json](../../data/internal_verification/manifest.json)。

| 数据集 | 首批范围 | 核心用途 | 关键监督 | 当前状态 |
|---|---:|---|---|---|
| ProcessBench | 非 GSM 的 MATH、OlympiadBench、Omni-MATH，共 3000 | 直接评估首个错误步骤 | 人工 first-error、candidate steps、final correctness | 已准备 |
| PrOntoQA-OOD | 1-hop train / 5-hop test ProofsOnly，共 100 | 形式化逻辑链、长度 OOD、精确 proof state | gold proof chain | 已准备 |
| StepGame | test 按 1–10 hop 分层抽样，每层 100，共 1000 | 空间关系、实体绑定、多跳状态 | story constraints、hop、answer | 已准备 |
| MATH-500 | 500 | 自然数学迁移和官方参考解 | reference solution、answer、level、subject | 已准备 |
| BBEH-mini | 460 | 当前更困难的广义推理、状态/约束/首错任务 | exact target；部分任务在输入中含候选 thoughts | 已准备 |
| REVEAL | eval/open | 开放域 step relevance、attribution、logic correctness | 人工逐步标签和 justification | 需用户接受 gated 条款后单独准备 |

### 7.1 两条不可混淆的 activation 轨道

**Solver-state track** 研究目标模型在自己解题时的内部状态。适用数据包括
MATH-500、PrOntoQA、StepGame、BBEH，以及后续经过可靠裁定的自然题目。我们让
目标 backbone 自己生成答案和过程，并同步保存其原生 activations。

**Verifier-state track** 研究 verifier 阅读候选过程时如何形成错误判断。
ProcessBench 和 REVEAL 直接提供逐步错误监督，适合这一轨道。

ProcessBench 中的 candidate steps 来自数据集记录的其他生成模型。我们没有这些
生成器当时的 hidden states，所以：

- 可以把 candidate steps 输入我们的 verifier，分析 verifier 的内部判断；
- 可以 teacher-force 给目标模型，研究它如何表征一个给定的正确/错误步骤；
- 不能把 teacher-forced activations 说成原生成器产生错误时的内部机制；
- 不能把 ProcessBench 的 first-error 标签直接套到目标 backbone 重新生成的另一
  条 solution 上。

如果论文主线坚持“求解模型验证自身计算”，Solver-state track 必须是主实验，
ProcessBench/REVEAL 只作为互补验证和强 process-verifier baseline。

### 7.2 为什么不是只加更多数学题

Exploration 01 暗示问题是 binding/state，而非算术 token。因此数据需要同时覆盖：

- **过程首错**：ProcessBench；
- **精确 proof chain**：PrOntoQA；
- **实体与关系绑定**：StepGame；
- **自然高难数学迁移**：MATH-500；
- **多类未饱和困难推理**：BBEH；
- **开放域证据与逻辑验证**：REVEAL（受控访问）。

### 7.3 统一数据 schema

每行 JSONL 均包含：

- `id`, `dataset`, `source_split`, `task_family`；
- `input.context`, `input.question`, `input.choices`；
- `gold.answer`, `gold.reasoning_steps`, `gold.first_error_step`；
- `gold.process_correct`, `gold.final_answer_correct`, `gold.state_trace`；
- `candidate.answer`, `candidate.reasoning_steps`, `candidate.generator`；
- `capabilities`：该数据实际支持哪些 verification claim；
- `metadata`：原始索引、配置、难度、hop 等。

不支持的字段使用 `null` 或空列表，不能用外部 LLM 猜测补齐。

### 7.4 数据准备命令

```bash
HF_HOME=/tmp/jlensv_hf \
  .venv/bin/python research/internal_verification/prepare_datasets.py
```

复核已准备的数据而不联网：

```bash
.venv/bin/python \
  research/internal_verification/prepare_datasets.py \
  --validate-only
```

默认不下载 ProcessBench 的 GSM8K split，从数据层保证首批扩展不只是 GSM8K。

### 7.5 已准备快照

- 总计：5060 条，7 个 JSONL；
- ProcessBench：
  - MATH：1000，其中 594 条有 first-error；
  - OlympiadBench：1000，其中 661 条有 first-error；
  - Omni-MATH：1000，其中 759 条有 first-error；
- MATH-500：500；
- PrOntoQA 1-hop→5-hop OOD：100，每条 gold proof 为 11 个源步骤；
- StepGame：1000，1–10 hop 各 100；
- BBEH-mini：460，任务标签通过与固定 revision 的 23 个官方 full task 精确匹配
  恢复；
- 所有文件均已通过 schema、唯一 ID、记录数和 SHA-256 校验。

## 8. 文献与相邻工作

以下按它们在本项目中的作用整理，不代表简单罗列。

### 8.1 JLens、可言语化表示与可解释 readout

- [Verbalizable Representations Form a Global Workspace in Language
  Models](https://transformer-circuits.pub/2026/workspace/index.html)：JLens 及
  verbalizable global workspace 主张；明确留下 binding、J-space ignition 和
  automatic computation 等开放问题。
- [LatentQA, ICLR
  2026](https://iclr.cc/virtual/2026/poster/10007461)：用自然语言读取并控制
  activations，说明自由文本 readout 可以作为另一种传感器。
- [Temporal Sparse Autoencoders, ICLR 2026
  Oral](https://iclr.cc/virtual/2026/poster/10008573)：token-local SAE 会错过跨
  token 的时间结构，支持我们把轨迹而非单点 feature 作为对象。
- [Sparse Feature Circuits, ICLR
  2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3ba4d47a83e498c2b1a0868cba20f6de-Abstract-Conference.html)：
  稀疏特征与 circuit attribution 的组合 baseline。
- [Transcoders Find Interpretable LLM Feature Circuits, NeurIPS
  2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/2b8f4db0464cc5b6e9d5e6bea4b9f308-Abstract-Conference.html)：
  用可解释特征近似 MLP 计算。
- [Propositional Probes, ICLR
  2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3132d0405fabe24b2a7b6cd7ba9de6b5-Abstract-Conference.html)：
  命题与变量绑定表示具有因果证据，是我们补足 JLens “bag of concepts” 的重要
  baseline。
- [Causality is not Invariance, ICLR
  2026](https://iclr.cc/virtual/2026/poster/10010007)：因果 function vector 与
  跨格式 invariant concept vector 可能分离，提醒我们不要把一种 readout 当作
  唯一真实表示。
- [LLMs Know More Than They Show, ICLR
  2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a712d461e57201efe35d429a6f1731c1-Abstract-Conference.html)：
  模型内部可能编码正确信息却输出错误，但内部 truth/error 信号存在跨数据集
  泛化问题。

### 8.2 白盒/activation-based verification

- [ReProbe, ACL
  2026](https://aclanthology.org/2026.acl-long.536/)：小型 verifier 读取 hidden
  states、attention 和 logits 做 step verification，是必须击败或区分的直接
  baseline；其重点是预测而非机制解释。
- [Tracing the Traces, ICLR
  2026](https://mlanthology.org/iclr/2026/vilas2026iclr-tracing/)：latent trajectory
  可提前预测推理成功并节省生成 token，说明时间性内部信号可用于 early exit。
- [Causal Reasoning Verifiers, ICLR 2026
  Oral](https://iclr.cc/virtual/2026/poster/10010813)：用 attribution graph 的
  结构错误特征并做纠错干预，是最接近的机制 verification 工作之一。
- [Internal Causal Mechanisms for OOD Correctness, ICML
  2025](https://proceedings.mlr.press/v267/huang25af.html)：因果变量比普通相关
  特征更有 OOD 预测力。
- [Latent Veracity, ICLR
  2026](https://iclr.cc/virtual/2026/poster/10008278)：在 reasoning steps 上学习
  latent veracity；我们的差异必须落在机制、时间、binding 和 observability。
- [Beyond the Surface, NeurIPS
  2025](https://papers.neurips.cc/paper_files/paper/2025/hash/8629b0fff229b8a27efb1422e990605f-Abstract-Conference.html)：
  使用 judge 自己的跨层表示提高判断，而不是读取被验证模型的机制。

### 8.3 外部 verifier、gold reference 与逐步标注

- [REVEAL, ACL 2024](https://aclanthology.org/2024.acl-long.254/)：逐步标注
  relevance、evidence attribution 与 logical correctness；现有 verifier 尤其
  难以检查逻辑和矛盾。
- [ProcessBench, ACL
  2025](https://aclanthology.org/2025.acl-long.50/)：人工标注数学解的首个错误
  step，直接提供 first-error evaluation。
- [References Improve LLM Alignment in Non-Verifiable Domains, ICLR
  2026](https://iclr.cc/virtual/2026/poster/10009831)：高质量 reference 会明显
  改善 judge，因此实验必须测量 internal evidence 在 strong reference 之上的
  边际贡献。
- [J1, ICLR
  2026](https://iclr.cc/virtual/2026/poster/10008383)：训练 thinking judges 进行
  reference generation 与 self-correction，代表强外部 judge baseline。

### 8.4 CoT faithfulness 与干预风险

- [Thought Branches, ICLR
  2026](https://iclr.cc/virtual/2026/poster/10008605)：单条 CoT 不足以代表策略，
  on-policy resampling 比离线文本编辑更可靠。
- [Measuring the Faithfulness of Thinking Drafts, NeurIPS
  2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/cbd25a91123291348cc8407a38b75080-Abstract-Conference.html)：
  thinking draft 到答案存在选择性忠实。
- [Activation Patching Best Practices, ICLR
  2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html)：
  patching 指标和 corruption 选择会改变结论。
- [Interpretability Illusion for Subspace Activation Patching, ICLR
  2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/70b8505ac79e3e131756f793cd80eb8d-Abstract-Conference.html)：
  子空间干预可能制造看似可解释但不唯一的结果。
- [Intervention Divergence, ICLR 2026
  Oral](https://iclr.cc/virtual/2026/poster/10008487)：干预可能激活 dormant pathways
  并进入分布外状态。
- [Obfuscated Activations, ICLR
  2026](https://iclr.cc/virtual/2026/poster/10007739)：probe/monitor 可被针对性隐藏，
  因而内部 verifier 不能宣称无条件安全保障。
- [Output Supervision Can Obfuscate CoT, ICLR
  2026](https://iclr.cc/virtual/2026/poster/10010196)：即使只监督输出也可能间接使
  CoT 更不透明。

### 8.5 机制形成、自动化与 cognitive compilation

- [Evolution of Concepts in Pretraining, ICLR
  2026](https://iclr.cc/virtual/2026/poster/10007251)：跨 checkpoint 跟踪 feature
  形成，可用于研究 workspace recruitment 如何随熟练度变化。
- [Mechanistic Analysis of Fine-tuning, ICML
  2025](https://proceedings.mlr.press/v267/wang25ak.html)：fine-tuning 后节点可能
  相似但 circuit edges 显著变化。
- [How Transformers Learn Implicit Reasoning, NeurIPS
  2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5f1cb1d23261b19cbd45f90f7b4f251f-Abstract-Conference.html)：
  受控训练中从记忆到 ID/OOD 泛化的阶段变化，为“自动化是否绕过 workspace”
  提供实验范式。
- [From \(f(x),g(x)\) to \(f(g(x))\), ICLR
  2026](https://iclr.cc/virtual/2026/poster/10007845)：RL 如何组合旧技能形成新能力。
- [How Do Language Models Track State?, ICML
  2025](https://proceedings.mlr.press/v267/li25r.html)：状态追踪机制是当前 GSM8K
  观察的直接相邻问题。
- [Language Models Use Lookbacks to Track Beliefs, ICLR
  2026](https://iclr.cc/virtual/2026/poster/10011359)：belief/state tracking 的
  token 间机制可作为 binding 分析 baseline。

## 9. 数据与实验阶段

### Stage A：可行性与标注可靠性

1. 准备并验证五个公开数据集；
2. 在每类数据上先跑小规模 generation-only；
3. 确定格式、完成率、官方评分和标签噪声；
4. 只对正确/错误平衡且可定位首错的样本计算完整 JLens。

### Stage B：发现验证

1. 对齐内部 trace、gold semantic state 和可见 CoT；
2. 测量 \(t_{\mathrm{internal}}-t_{\mathrm{text}}\)；
3. 建立 computation/readout/unobservable 分类；
4. 与 logits、entropy、linear probe、ReProbe-style features 比较；
5. 做 matched corruption 和小规模 causal patching。

### Stage C：方法与泛化

1. 加入 observability-aware verifier；
2. 多 backbone，至少覆盖不同模型家族和规模；
3. 跨任务与 OOD depth；
4. 加入 T-SAE/SAE 或 propositional probe 的代表性比较；
5. 评估 selective prediction、first-error localization、failure type 和 targeted
   correction，而不只看 accuracy/AUROC。

## 10. 当前开放决策

- JLens lexical concepts 如何升级为带 binding 的 structured state；
- observability 是独立估计，还是由多传感器 disagreement 定义；
- reference-free 与 reference-conditioned 两种 setting 的主次；
- causal corruption 应作用于 token、feature、relation subspace 还是 circuit edge；
- 如何构造相同输出但不同内部机制的 matched examples；
- cognitive compilation 是主发现的一部分，还是内部验证适用边界的后续章节。

在完成 Stage A 小规模数据审计前，不锁定复杂模型结构。
