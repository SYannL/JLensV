# JLens 内部推理验证：研究记录与数据路线

最后更新：2026-07-25

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

## 5. 此前首选方案：Workspace Recruitment 与 Cognitive Compilation

这一方向是我们在“低至中等算力、但要求 ICLR Oral 级发现”的第一轮筛选中给出的
**原始首选方案**。它必须被完整保留，不能因为后来把主题收束到 internal
verification 就只剩下一句“可观测性边界”。

### 5.1 原始科学问题

核心问题不是“JLens 能否预测正确率”，而是：

> 当一个模型从不会完成某种推理，到学会、熟练并最终自动化这种推理时，计算
> 是否会从显式、可言语化、可全局访问的 workspace，逐步编译到更局部、更专用
> 的 automatic circuits？遇到新组合或分布外要求时，模型是否会重新招募该
> workspace？

暂称这一过程为 **cognitive compilation**。这里的“编译”是待验证的机制假设，
不是预设结论：

\[
\text{explicit / workspace-mediated computation}
\;\longrightarrow\;
\text{compressed / automatic computation}.
\]

与之相反的方向为 **workspace re-recruitment**：

\[
\text{practiced skill}
+ \text{novel composition}
\;\longrightarrow\;
\text{workspace ignition}.
\]

这直接对应 JLens 论文留下的开放问题：什么任务真正进入 J-space、表示如何进入
J-space，以及熟练计算是否会绕过 verbalizable workspace。

### 5.2 原始预测

对同一种可控技能，跨训练阶段可能出现如下可证伪轨迹：

1. **未掌握阶段**：没有稳定的正确语义轨迹，J-space recruitment 弱或混乱；
2. **能力获得阶段**：显式概念、子目标和中间状态被广泛招募并跨层持续；
3. **熟练阶段**：在保持正确率的同时，显式 workspace 痕迹缩短、变晚或减弱，
   计算更多由专用 circuit 完成；
4. **新组合/OOD 阶段**：旧技能需要以新方式组合，workspace 被重新点燃；
5. **重新熟练阶段**：新组合经过练习后再次压缩或自动化。

最重要的不是某个指标单调下降，而是区分：

- 不会做，因此没有正确 workspace state；
- 会做且显式推理；
- 会做但已自动化；
- 因新颖组合而重新显式推理。

### 5.3 可测量对象

原方案计划把“workspace recruitment”拆成多项而非一个任意分数：

- **ignition**：任务相关表示从何层、何 token 开始稳定出现；
- **persistence**：概念或状态在多少层、多少后续位置保持可读；
- **broadcast**：同一语义信息是否能够影响多个后续位置和输出；
- **explicit state coverage**：关键中间状态有多少进入可言语化表示；
- **compression/bypass**：行为保持正确时，显式轨迹是否缩短或被其他 circuit
  替代；
- **re-recruitment**：新组合、规则变化或 OOD 深度是否恢复上述信号。

JLens 是主要读出工具，但这些概念不能被循环定义成“JLens 看见了，所以存在
workspace”。至少需要三角验证：

- linear/logit/propositional probes；
- SAE/T-SAE 或 transcoder features；
- activation/path patching 或 feature/circuit ablation；
- 行为层面的 novel-composition 与 skill-practice 对照。

### 5.4 原始实验设计

低算力版本不需要大规模 RL 或工业级预训练：

1. 选择可以无限生成、具有精确中间状态的组合任务；
2. 使用约 0.5B–1.5B 模型进行小规模从头训练、continued training 或 LoRA；
3. 保存密集 checkpoints，覆盖“不会—学会—熟练”；
4. 将训练过的原子技能重新组合成未训练组合；
5. 在每个 checkpoint 上读取 JLens、probes 和代表性 circuits；
6. 对 workspace-like state 做删除、交换和修复，测量其在各阶段的因果必要性；
7. 最后用冻结的约 4B backbone 和自然任务做外部有效性验证。

合适的受控任务包括：

- PrOntoQA 式可生成逻辑链和 OOD proof depth；
- shuffled objects、StepGame 或显式 state update；
- 函数组合 \(f(x),g(x)\rightarrow f(g(x))\)；
- 可程序化检查的多步算术、栈或有限状态任务。

关键控制包括：

- 匹配正确率，避免把“更熟练”误写成“准确率更高”；
- 匹配输出长度和 prompt 格式；
- 区分训练步数、任务难度、表示可读性和因果必要性；
- 用 held-out 表述测试词汇不变性；
- 避免只在单一 model family 上定义生命周期。

### 5.5 原始创新主张

如果实验成立，发现层贡献不是“模型熟练后更快”，而是：

1. 给出 LLM 中从 deliberative workspace 到 automatic circuit 的机制生命周期；
2. 证明 novelty/composition 会重新招募全局可访问表示；
3. 解释为什么同一种内部观察工具会在不同熟练阶段得到相反结果；
4. 将 global-workspace 主张从静态表示扩展为随学习变化的动态理论；
5. 给出哪些内部证据具有因果作用、哪些只是 verbalizable readout 的边界。

原始候选标题包括：

- **From Deliberation to Automation: Cognitive Compilation in Language
  Models**
- **When Do Language Models Recruit a Global Workspace?**
- **The Lifecycle of Verbalizable Computation in Language Models**

### 5.6 与 internal verification 的结合

用户随后要求该方向与 interpretable LLM / verification 更紧密关联。两者并非
勉强拼接，cognitive compilation 可以解释内部验证的适用条件：

\[
\text{internal-verification quality}
=
f(\text{workspace recruitment},\text{sensor coverage}).
\]

由此形成当前更完整的命题：

> 一个推理是否能被内部验证，不只取决于 verifier 强弱，还取决于本次计算是否
> 进入了可读取、可归因的内部通道。

具体预测是：

- 新颖、deliberative 计算：workspace witness 较丰富，较容易定位内部偏离；
- 熟练、automatic 计算：JLens witness 可能变弱，但这不等于推理不可靠；
- 新组合：workspace re-recruitment，内部验证能力重新增强；
- 因此 verifier 必须估计 observability，并在证据不足时 abstain。

这一合并方向可暂称：

- **Observability-Conditioned Internal Verification**
- **When Can a Language Model Verify Its Own Reasoning?**
- **Mechanism-Aware Verification across Deliberative and Automatic
  Computation**

cognitive compilation 不应只作为附录中的一个 subgroup。若初步 checkpoint
实验支持它，它可以成为解释“内部验证何时有效”的共同主发现；若不支持，则当前
verification 方案仍可独立推进。

### 5.7 曾讨论的过渡方案：Mechanistic Witnesses for External Verification

在用户进一步澄清“仍然需要内部验证”之前，我们曾把方案改写为：

> **Mechanistic Witnesses for External Reasoning Verification — When Do
> Internal Traces Add Evidence Beyond Gold Rationales?**

其形式为：

\[
V_{\mathrm{hybrid}}
(q,r,a,r^*,a^*,\phi_{\mathrm{JLens}}(h)).
\]

JLens 不直接做 verifier，而是为外部强 LLM 提供 mechanistic witnesses，例如
首次内部偏离、reasoning/readout failure 区分以及 CoT 是否可能是 post-hoc
rationalization。比较条件为：

1. question + answer；
2. 加 candidate CoT；
3. 再加 golden reasoning；
4. candidate CoT + internal trace；
5. golden reasoning + internal trace。

核心边际量为：

\[
\Delta_{\mathrm{internal}}
=
\operatorname{Perf}(\text{gold + internal})
-
\operatorname{Perf}(\text{gold only}).
\]

这一方案不再作为当前论文主线，原因是：

- 外部强 LLM 与 golden reasoning 可能已经足够完成普通 correctness 判断；
- 主贡献容易退化为“给 judge 多一种输入后涨点”；
- 外部 LLM 可能把噪声 JLens tokens 编造成连贯但虚假的机制故事；
- 它不能替代对求解模型自身内部状态的因果验证。

但它仍应保留为重要 baseline 和下游应用：用于测量内部证据在外部 reference
之上的非冗余价值，而不是定义“内部验证”本身。

### 5.8 第一轮方向排序记录

当时在低算力约束下的方向排序为：

1. **Workspace recruitment / cognitive compilation**：发现性、可行性和理论价值
   最平衡，因此成为原始首选；
2. **寻找 vocabulary-independent 的真实 workspace \(W\)-space**：理论上限最高，
   但识别和因果证明风险更大；
3. **Structured propositional/dynamic binding**：与 GSM8K 错误高度相关，但
   propositional probes、state tracking 等相邻工作较拥挤；
4. **JLens causal validity/certification**：必要且有价值，更适合作为所有主方向的
   支撑贡献，而不是单独主线。

当前研究路线不是否定第一项，而是把它进一步收束为：

> **内部验证的可观测性为什么随计算机制、熟练度和新颖性变化。**

## 6. 初期方法轮廓

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

### 6.1 必须避免的弱版本

- 把 JLens top-k 复制到 GPT-4o-mini prompt，然后只报告分类准确率；
- 只训练一个 hidden-state probe，与 ReProbe 类方法做小幅 AUROC 比较；
- 只比较正确与错误答案的最后几层；
- 把 golden rationale 当作唯一合法路径；
- 把相关 token readout 叙述成因果机制；
- 用 GSM8K exact match 直接当作可靠 ground truth。

### 6.2 Oral 级证据链

1. **时间证据**：内部偏离稳定领先文本错误；
2. **增量证据**：控制表面信号后内部轨迹仍增加信息；
3. **因果证据**：删除、交换或修复被识别状态会系统性改变后续错误；
4. **解耦证据**：matched cases 分离答案、CoT 与机制；
5. **边界证据**：明确什么时候可观察、什么时候应 abstain；
6. **闭环应用**：根据错误机制选择重算、局部修正或仅重新 readout。

## 7. Exploration 01：Qwen3.5-4B × GSM8K difficult cases

这是本研究的 **第一个 hypothesis-generation exploration**，不是正式 benchmark
结果。

### 7.1 设置

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

### 7.2 结果

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

### 7.3 JLens 初步观察

- 错误回答与正确回答在最后几层都高度词汇收敛；
- incorrect 的 chosen-token top-1 rate 约 95.94%，correct 约 95.31%；
- layer-30 与 final readout agreement 在错误组并没有降低；
- gold digit 的 late-layer 可读性也没有把正确和错误可靠分开；
- 模型经常拥有相关数字，却把数字绑定到错误实体、单位、时间范围或问题解释。

因此当前最重要的负结果是：

> **末层词汇收敛和 gold-token availability 不是内部验证器。**

后续应在人工确认的首次语义分歧附近，分析关系、实体和状态转移，而不是继续对
整段输出平均 gold digit rank。

### 7.4 Exploration 01 的限制

- \(n=20\)，不能做总体统计主张；
- 样本由既有 wrong/difficult 列表选取，存在选择偏差；
- GSM8K 官方标签和 rationale 有明显噪声；
- 当前 JLens 输出是 lexical readout，不编码可靠 binding；
- 当前自动 step alignment 较粗；
- 没有 causal intervention；
- 只使用一个 backbone 和一个任务家族。

## 8. 首批数据套件

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

### 8.1 两条不可混淆的 activation 轨道

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

### 8.2 为什么不是只加更多数学题

Exploration 01 暗示问题是 binding/state，而非算术 token。因此数据需要同时覆盖：

- **过程首错**：ProcessBench；
- **精确 proof chain**：PrOntoQA；
- **实体与关系绑定**：StepGame；
- **自然高难数学迁移**：MATH-500；
- **多类未饱和困难推理**：BBEH；
- **开放域证据与逻辑验证**：REVEAL（受控访问）。

### 8.3 统一数据 schema

每行 JSONL 均包含：

- `id`, `dataset`, `source_split`, `task_family`；
- `input.context`, `input.question`, `input.choices`；
- `gold.answer`, `gold.reasoning_steps`, `gold.first_error_step`；
- `gold.process_correct`, `gold.final_answer_correct`, `gold.state_trace`；
- `candidate.answer`, `candidate.reasoning_steps`, `candidate.generator`；
- `capabilities`：该数据实际支持哪些 verification claim；
- `metadata`：原始索引、配置、难度、hop 等。

不支持的字段使用 `null` 或空列表，不能用外部 LLM 猜测补齐。

### 8.4 数据准备命令

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

### 8.5 已准备快照

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

## 9. 文献与相邻工作

以下按它们在本项目中的作用整理，不代表简单罗列。

### 9.1 JLens、可言语化表示与可解释 readout

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

### 9.2 白盒/activation-based verification

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

### 9.3 外部 verifier、gold reference 与逐步标注

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

### 9.4 CoT faithfulness 与干预风险

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

### 9.5 机制形成、自动化与 cognitive compilation

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

## 10. Baseline 矩阵

这里把 baseline 按它实际能回答的问题分层。**被验证模型（target/reasoner）**
和**监督/标注模型（judge/annotator）**必须分开记录；一个方法用了
DeepSeek-R1 或 GPT-OSS-120B 标注，不等于它在这些模型的内部状态上做验证。

### 10.1 原论文中的模型与数据

| 类别 | Baseline / 论文 | 原论文被验证或读取的 LLM | 训练数据与监督来源 | 原论文评测数据 | 对我们的角色 |
|---|---|---|---|---|---|
| 输出不确定性 | MaxProb、mean/max entropy、perplexity、temperature scaling、energy | 非特定；[CoE](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b0b1cfc8ede53f452cabf8b9cf4eef76-Abstract-Conference.html) 在 Llama2-7B-Instruct、Llama3-8B/70B-Instruct、Qwen1.5-7B、Qwen2-7B/72B、Mistral-7B-Instruct 上统一比较 | 除 temperature calibration 外无需训练 | GSM8K、MATH、CommonsenseQA、TheoremQA、MMLU、Belebele | **必须复现**；排除“JLens 只是在读 confidence” |
| 简单 hidden-state probe | layer sweep 的 logistic regression；2-layer MLP | [CRV](https://iclr.cc/virtual/2026/poster/10010813) 使用 Llama-3.1-8B-Instruct；[Masked by Consensus](https://aclanthology.org/2026.acl-long.483/) 使用 Llama-3.1-8B、Qwen2.5-7B、Gemma-2-9B，附加 Qwen3-32B | 当前任务 correctness / step-error label；后者用 target 自身正确性训练 self-probe 和 peer-model probe | CRV：Synthetic Boolean、Synthetic Arithmetic、annotated GSM8K；Masked：Mintaka、TriviaQA、HotpotQA（question-only）、MATH、GSM1K | **必须复现**；同时做 self/peer probe 与 disagreement subset |
| 无训练内部轨迹 | [Chain-of-Embedding，CoE-R / CoE-C，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b0b1cfc8ede53f452cabf8b9cf4eef76-Abstract-Conference.html) | Llama2-7B-Instruct、Llama3-8B/70B-Instruct、Qwen1.5-7B、Qwen2-7B/72B、Mistral-7B-Instruct | label-free；直接计算 progressive hidden-state trajectory 的幅度和方向变化 | GSM8K、MATH、CommonsenseQA、TheoremQA、MMLU、Belebele | **必须复现**；最便宜且直接的 trajectory baseline |
| 轻量内部 step verifier | [ReProbe，ACL 2026](https://aclanthology.org/2026.acl-long.536/) | Qwen3-8B、Phi-4；native thinking：Qwen3-1.7B、Qwen3-32B | PRM800K 中 10.8K questions、每题 3 条轨迹，约 32K；structured CoT 用 self-annotation 或 DeepSeek-R1；native thinking 用 GPT-OSS-120B 标注 | ID：MATH、GSM8K、ProofNet；OOD planning：Trip/Meeting/Calendar Planning；OOD QA：StrategyQA、ScienceQA | **主 baseline**；复现 hidden-state 版本，小于 10M 参数 |
| 机制图 verifier | [Circuit-based Reasoning Verification，CRV，ICLR 2026 Oral](https://iclr.cc/virtual/2026/poster/10010813) | 配有逐层 transcoders 的 Llama-3.1-8B-Instruct | 自行生成 solution 与 computation trace；GSM8K step label 由 Llama-3.3-70B-Instruct 辅助并人工复核；在 attribution-graph features 上训练 gradient boosting | Synthetic Boolean、Synthetic Arithmetic、annotated GSM8K | **机制级最强对照**；成本高，先在代表性子集复现 |
| 内部/外部表示对照 | [Masked by Consensus，ACL 2026](https://aclanthology.org/2026.acl-long.483/) | target/source 均取 Llama-3.1-8B、Qwen2.5-7B、Gemma-2-9B；Qwen3-Embedding-8B 仅作 external source；另测 Qwen3-32B | full training split 上训练 L2 logistic regression；每 5 层取 question 最后 token；补充 MLP；标签始终是 target 的 correctness | Mintaka、TriviaQA、HotpotQA（无 supporting docs）、MATH、GSM1K；特别报告 target/source 分歧子集 | **必须复现其控制实验**；检验内部信息是否真的 non-redundant |
| 潜变量过程验证 | [Latent Veracity Inference，ICLR 2026](https://iclr.cc/virtual/2026/poster/10008278) | Qwen3-4B、Qwen3-8B、Llama-3.2-3B、Llama-3-8B | Veracity Search 用 gold final answer likelihood 搜索 step-veracity；AVI 用搜索产生的 pseudo-label SFT，测试时不需要 gold | PrOntoQA、GSM8K、CommonsenseQA，各 1,000 examples | **强文本过程 baseline**；与内部 JLens 对齐但不读取 hidden states |
| PRM | Qwen2.5-Math-7B-PRM800K；Qwen2.5-Math-PRM-7B；Skywork-PRM-1.5B | verifier 自身分别为 Qwen2.5-Math-7B、Qwen2.5-Math-7B、1.5B PRM；不读取被验证 solver 的内部状态 | 约 263K PRM800K、约 860K MC+LLM-judge consensus、Skywork 合成/偏好数据（版本依模型卡） | [ReProbe](https://aclanthology.org/2026.acl-long.536/) 统一测 MATH、GSM8K、ProofNet、三个 NaturalPlan、StrategyQA、ScienceQA；[ProcessBench](https://aclanthology.org/2025.acl-long.50/) 测下列四个数学子集 | **至少选 1.5B 与一个 7B**；代表专用外部 verifier |
| 通用 LLM critic | [ProcessBench，ACL 2025](https://aclanthology.org/2025.acl-long.50/) 中的 prompted critics | 开源：Llama-3/3.1/3.3 8B–70B，Qwen2/2.5/Math/Coder 7B–72B，QwQ-32B-Preview；闭源：GPT-4o-0806、o1-mini | 无专用训练；prompt 逐步批判并输出最早错误段落 | ProcessBench：GSM8K 400、MATH 1,000、OlympiadBench 1,000、Omni-MATH 1,000，共 3,400 条人工首错标注 | **必须有一个本地 critic + 一个强 API judge**；分别测无 reference / 有 gold semantic state |
| 随机与表面控制 | random、majority class、output length、step count、EOS、格式合法性、题目难度/输入表示 probe | 与 target 无关；在我们的 Qwen3.5-4B 生成上计算 | 无训练或仅在训练 split 拟合 | 我们的全部数据 | **必须复现**；防止数据与生成格式泄漏 |

表中 PRM 的精确训练语料并不完全同质，正式实验中必须锁定具体 Hugging Face
revision，并从相应 model card 记录数据许可、样本数和 chat template，不能只写
“7B PRM”。ProcessBench 的阈值还使用 GSM8K 子集选择；跨数据集结果需要另外
报告不调阈值的 AUROC/AUPR，避免把该校准优势带入 OOD 比较。

### 10.2 我们实际要跑的优先级

| 优先级 | 在同一 Qwen3.5-4B 轨迹上运行 | 数据 | 是否训练 | 目的 |
|---|---|---|---|---|
| P0 | random / majority、长度与格式、MaxProb、entropy、PPL、energy | 全部：GSM8K exploration、MATH-500、ProcessBench、PrOntoQA、StepGame、BBEH-mini | 否；temperature scaling 仅用 validation | 建立不可省略的行为与不确定性下界 |
| P0 | layer-wise LR、2-layer MLP：question-only、visible-text、target hidden-state 三种输入 | 同上；按 question/group 切分 | 是，小规模 | 分离题目难度、文本信息和 target-private state |
| P0 | CoE-R、CoE-C | 所有能取得逐层 hidden state 的同模型生成 | 否 | 与 JLens structured trajectory 做最直接比较 |
| P0 | external-source LR/MLP：另一个 3B–9B 模型读取同一问题/可见 trace | MATH、GSM/算术和至少一个事实 QA；额外报告模型分歧子集 | 是，小规模 | 实现 Masked by Consensus 控制，证明或否定 non-redundant internal evidence |
| P1 | ReProbe-hidden-state（固定参数预算，禁止 Attention+Logit 先占算力） | PRM800K train；先测 MATH/GSM8K，再零样本测 PrOntoQA、StepGame、BBEH 与 ProcessBench 非 GSM 子集 | 是，约 10M 参数 | 当前最强、最公平的轻量内部 verifier |
| P1 | Qwen2.5-Math-PRM-7B 或 Qwen2.5-Math-7B-PRM800K；另加 Skywork-PRM-1.5B | ProcessBench、MATH、GSM8K；逻辑/OOD 仅在格式兼容时运行 | 否，直接推理 | 与专用外部过程 verifier 比较 |
| P1 | 本地 7B–14B critic；强 API judge，分别无 reference / 有 gold state | ProcessBench、PrOntoQA、StepGame，以及我们的 matched corruptions | 否 | 与用户此前的外部 verification 路线接轨 |
| P1 | Latent Veracity / 简化 VS | PrOntoQA、GSM8K、CommonsenseQA；若算力有限先各 200–500 条 | VS 否；AVI 暂不训练 | 比较 gold-answer-conditioned 的文本潜变量验证 |
| P2 | CRV | Llama-3.1-8B 上的 synthetic Boolean/Arithmetic 与小规模 GSM8K | 是；需 transcoders/图提取 | 只做机制发现的代表性强对照，不作为全量刷榜项 |
| P2 | SAE/T-SAE、propositional probes、activation/path patching | 受控状态任务和高置信 matched cases | 依方法 | 作为 sensor 与因果交叉验证，不冒充同任务 end-to-end verifier |

### 10.3 公平比较协议

1. **固定 solver trajectories**：同一个 target、prompt、decoding、token budget 和
   原始输出；各 verifier 不得重新生成一套更容易的答案。
2. **三种信息预算分开**：
   `question-only`、`question + visible trace`、`target internal trace`；gold answer 或
   gold semantic state 另成 reference-conditioned track。
3. **分组切分**：同一题的多条 trajectory、corruption 和 paraphrase 必须在同一
   split，禁止 trajectory-level 泄漏。
4. **统一任务**：同时报告 outcome correctness、step correctness、first-error
   localization、failure type、selective risk/coverage；不能拿 AUROC 与另一个方法
   的 final-answer accuracy 横比。
5. **统一预算**：报告 verifier 参数量、训练样本、额外 forward passes、标注器和
   API cost；ReProbe 的外部标注版与 self-annotation 版分开。
6. **主张门槛**：只有当 JLens 在 external-source probe、visible-text probe、
   confidence 和题目难度控制之后仍有增益，并在 matched intervention 上得到因果
   支持，才能称为“内部非冗余验证信号”。

### 10.4 当前推荐的最小主表

第一版论文主结果表不必堆满所有 PRM。最小但足够强的组合是：

1. MaxProb / entropy / PPL；
2. question-only、visible-trace、target-self、external-peer 四种 LR/MLP；
3. CoE-R / CoE-C；
4. ReProbe-hidden-state；
5. 一个 1.5B PRM、一个 7B PRM；
6. 一个本地 LLM critic、一个强 API judge（各自无 reference / 有 reference）；
7. JLens lexical-only、structured JLens、structured JLens + observability；
8. CRV 只在机制子表与代表性数据上比较。

这个设计既不会把“内部传感器”和“70B 外部 judge”当作同成本方法，也能直接回答
审稿人最可能提出的问题：JLens 的增益究竟来自 target-private computation，
还是来自 confidence、题目难度、可见文本或额外监督。

## 11. 数据与实验阶段

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

## 12. 当前开放决策

- JLens lexical concepts 如何升级为带 binding 的 structured state；
- observability 是独立估计，还是由多传感器 disagreement 定义；
- reference-free 与 reference-conditioned 两种 setting 的主次；
- causal corruption 应作用于 token、feature、relation subspace 还是 circuit edge；
- 如何构造相同输出但不同内部机制的 matched examples；
- cognitive compilation 是主发现的一部分，还是内部验证适用边界的后续章节。

在完成 Stage A 小规模数据审计前，不锁定复杂模型结构。
