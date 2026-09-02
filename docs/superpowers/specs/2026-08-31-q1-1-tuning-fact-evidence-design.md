# Q1.1 自动调优事实包与证据引用 MVP 设计规格

> 状态：已完成并通过 Codex 独立验收（2026-09-02）
>
> 日期：2026-08-31
>
> 适用范围：YOLOv8 Detect 的 LLM 自动调优决策链

## 1. 结论

Q1.1 采用“最小证据闭环”，不重做现有 Schema、Guardrails、审计和失败中止能力。系统在调用 LLM 前生成并冻结一个版本化事实包；LLM 的每个参数修改必须引用该事实包中的真实 `fact_id`，并回传完全一致的 `fact_package_id`。代码只校验证据存在、来源属于当前运行、参数合法且响应绑定正确，不尝试判断复杂自然语言推理是否正确。

Q1.1 只约束可能进入训练命令的自动调优决策。智能分析页面中的长篇文本诊断和视觉分析保持现状，后续根据 Q1.1 的效果另行评估。

预计工作量：**3–5 人日**。

## 2. 已验证的现状

2026-08-31 使用 `D:\Program Files\anaconda3\envs\auto_tune\python.exe` 完成现状审计：

- LLM 决策、统一 Schema、Guardrails、调优循环和审计定向测试：`95 passed`。
- 完整自动化套件：`1226 passed, 2 warnings`；两条 warning 均为既有 sklearn PCA 数值警告。
- Studio 可正常启动；首页、智能分析页、运行状态、调优状态和最近训练 API 返回正常，页面无控制台错误。

现有代码已经具备：

- 数据集报告、训练报告的可用性状态及 `reference_run` 关联检查。
- 基础严格 JSON 结构解析和一次 JSON 格式纠错。
- 未知参数拒绝、空修改约束和单轮最多三个参数修改。
- 参数注册表、类型/范围归一化、组合冲突检测和训练前失败中止。
- LLM 原文、建议、护栏、基线、实际参数和命令审计。

当前缺口：

- LLM 输入仍是自由文本摘要，没有稳定、可引用的事实标识。
- 决策无需声明引用了哪些事实，代码无法识别伪造或跨运行证据。
- 审计中的 `baseline` 在 LLM 决策之后写入，不是决策前冻结的正式输入合同。
- Guardrails 能判断参数是否合法，但不能证明建议引用了本轮真实事实。

## 3. 目标与非目标

### 3.1 目标

1. 决策前由代码生成确定性的事实包 v1。
2. 事实包绑定 `detect` 任务、当前 `reference_run` 和报告来源。
3. 每个可执行参数修改至少引用一个事实包内的 `fact_id`。
4. LLM 响应必须回传本轮 `fact_package_id`，禁止使用旧轮次或其他运行的响应。
5. 证据或结构校验失败时最多纠错一次；再次失败不得启动训练。
6. 事实包、响应、引用和校验结果进入现有审计链。
7. 不改变合法建议进入现有 Guardrails、预检、命令构造和训练执行的后续顺序。

### 3.2 非目标

- 不验证 `diagnosis` 自然语言中的每一句话是否正确。
- 不实现指标方向、best/final、早停或曲线趋势的完整语义规则。
- 不建立“某类事实只能修改某些参数”的专家知识图谱。
- 不修改智能分析页面的文本 LLM 或视觉模型输出。
- 不增加第二模型复核、多 Agent 辩论或让 LLM 裁决最终优胜。
- 不接入 HPO、YOLOv8 Classify、YOLOv5 Detect、Docker 或统一任务 API。
- 不新增第三方依赖。

## 4. 总体流程

```text
读取并验证感知报告
        ↓
读取参考运行 args.yaml 与 results.csv
        ↓
生成并冻结 FactPackage v1
        ↓
先写入审计（事实包必须早于 LLM 调用）
        ↓
LLM 输出 TuningDecision v1
        ↓
结构校验 + fact_package_id 校验 + evidence_ids 校验
        ↓ 失败
同一事实包下纠错一次 ──再次失败──→ 记录终态，不启动训练
        ↓ 通过
现有 Guardrails → 预检 → 唯一命令构造 → 训练
```

事实包只由代码生成；LLM 不能新增、删除或修改事实。最终是否执行仍由代码决定。

## 5. 文件与职责边界

### 5.1 新增 `auto_tune/modules/agent_engine/decision_facts.py`

唯一职责：从已验证感知结果、参考运行参数和参考指标构建事实包。

建议接口：

```python
def build_tuning_fact_package(
    perception: dict,
    reference_run: str,
    base_args: dict,
    before_metrics: dict,
    metrics_source: dict,
) -> dict:
    """返回经过规范化、可稳定散列的 FactPackage v1。"""
```

该模块负责：

- 只采集允许进入自动调优决策的字段。
- 不把缺失值伪装成 `0` 或默认值。
- 为事实生成稳定 `fact_id`。
- 生成规范 JSON 的 SHA-256 `fact_package_id`。
- 校验 `reference_run` 非空且与训练报告、指标来源一致。

### 5.2 新增 `auto_tune/modules/agent_engine/decision_contract.py`

唯一职责：解析并校验自动调优专用的 `TuningDecision v1`。

建议接口：

```python
def parse_tuning_decision_response(text: str) -> dict:
    """执行 JSON 与字段结构校验，返回规范化决定或稳定错误。"""


def validate_decision_evidence(decision: dict, fact_package: dict) -> dict:
    """校验事实包绑定、参数引用完整性和 fact_id 存在性。"""
```

该模块不负责参数范围和组合依赖；这些继续由 `parameter_registry.py` 和 `guardrails.py` 处理。

### 5.3 修改 `decision_agent.py`

- 自动调优的 `decide_hyperparameters()` 改为接收冻结的事实包，不再接收自由结构的 `perception` 作为决策合同。
- 自动调优提示词只展示事实包和允许参数，不混入无法引用的自由文本事实。
- `_run_decision_with_retry()` 对结构错误、事实包绑定错误和证据引用错误共用一次受控纠错。
- 网络、鉴权、端点策略和供应商响应错误不重试，保持现有行为。
- `generate_suggestion()` 及智能分析页面使用的现有响应格式保持不变。

### 5.4 修改 `loop.py`

- 将参考 `args.yaml`、参考指标和指标来源的读取提前到 LLM 调用之前。
- 在决策前构造事实包并先写入审计。
- 事实包构建失败形成稳定失败终态，不调用 LLM、不创建训练目录。
- 证据校验通过后再进入现有 Guardrails。
- 不复制或绕过现有 Guardrails、预检与命令构造逻辑。

### 5.5 修改 `audit.py` 和现有审计视图投影

- 审计顶层 Schema 版本升级时保持旧记录可读。
- 每轮增加 `fact_package` 和 `decision_validation`。
- 详情页面只需兼容读取新字段；Q1.1 不新增复杂交互界面。

## 6. FactPackage v1

示例：

```json
{
  "schema_version": "1.0",
  "fact_package_id": "sha256:<hex>",
  "task": "detect",
  "reference_run": "train54",
  "sources": {
    "dataset_report": "dataset_report_20260828.json",
    "training_report": "train54_report.json",
    "metrics": "detect/train54/results.csv",
    "params": "detect/train54/args.yaml"
  },
  "facts": [
    {
      "fact_id": "dataset.total_images",
      "value": 290,
      "source": "dataset_report"
    },
    {
      "fact_id": "training.metrics.mAP50",
      "value": 0.8121,
      "source": "metrics"
    },
    {
      "fact_id": "training.params.lr0",
      "value": 0.001,
      "source": "params"
    },
    {
      "fact_id": "training.issue.overfitting",
      "value": true,
      "source": "training_report"
    }
  ]
}
```

### 6.1 允许事实

首版只纳入现有代码已经确定性产生的事实：

- 数据集：图片数、标注数、标注率、质量评分、类别平衡、小目标比例、平均相对面积、曝光/模糊比例、已检测问题。
- 参考训练：mAP50、mAP50-95、Precision、Recall、已完成 epoch、best epoch、确定性 issue detector 结果和曲线趋势枚举。
- 当前参数：`parameter_registry.py` 中登记且实际存在于参考 `args.yaml` 的参数。
- 来源：数据集报告 basename、训练报告 basename、参考运行、指标来源和参数来源。

### 6.2 缺失值规则

- 原始值缺失、非有限数字或来源不可用时，不创建对应事实。
- 不使用 `0`、`?`、空字符串或默认 YOLO 参数代替未知事实。
- 必需身份字段缺失时整个事实包构建失败。
- 指标来源沿用现有 `_read_reference_before_metrics()` 合同：`error is None` 表示来源有效，`path` 只以 basename 进入事实包；任何非空 `error` 均使事实包构建失败。
- 过滤后事实列表为空时拒绝构建，不允许向 LLM 发送“只有身份、没有事实”的空包。
- 事实排序固定，规范 JSON 使用稳定键序和紧凑分隔符后计算 SHA-256。
- `fact_package_id` 不参与自身散列。

## 7. TuningDecision v1

自动调优响应示例：

```json
{
  "schema_version": "1.0",
  "fact_package_id": "sha256:<hex>",
  "diagnosis": "参考训练存在确定性记录的过拟合问题。",
  "action": "adjust",
  "hyperparameter_changes": {
    "weight_decay": 0.002
  },
  "training_overrides": {},
  "evidence_ids": {
    "weight_decay": [
      "training.issue.overfitting",
      "training.params.weight_decay"
    ]
  }
}
```

规则：

- 根对象只允许以上七个字段，不接受额外字段。
- `schema_version` 固定为 `1.0`。
- `action` 只允许 `adjust` 或 `keep_params`。
- `adjust` 时，两个参数对象合并后必须有 1–3 个参数。
- `keep_params` 时，两个参数对象和 `evidence_ids` 必须全部为空。
- `hyperparameter_changes` 与 `training_overrides` 不得出现重复参数。
- `evidence_ids` 的键必须与本轮修改参数集合完全一致。
- 每个修改参数至少引用一个 `fact_id`，引用列表不得重复。
- 每个 `fact_id` 必须存在于本轮事实包。
- 响应中的 `fact_package_id` 必须与本轮完全一致。
- 参数名仍必须存在于 `parameter_registry.py`；参数值的类型、范围和组合约束仍由现有 Guardrails 最终确认。

Q1.1 不检查某个 `fact_id` 在语义上是否足以支持某种参数调整；该能力属于后续 Q1.2。

## 8. 失败语义与纠错

新增稳定错误码：

| 错误码 | 含义 | 是否允许一次纠错 |
|---|---|---:|
| `FACT_PACKAGE_INVALID` | 输入身份、来源或必需事实无法形成合法事实包 | 否 |
| `DECISION_SCHEMA_INVALID` | JSON 或字段结构不符合 TuningDecision v1 | 是 |
| `DECISION_FACT_PACKAGE_MISMATCH` | 响应绑定了其他事实包 | 是 |
| `DECISION_EVIDENCE_MISSING` | 修改参数没有完整证据映射 | 是 |
| `DECISION_EVIDENCE_UNKNOWN` | 引用了事实包中不存在的 fact_id | 是 |
| `DECISION_GUARDRAIL_REJECTED` | 参数值或组合未通过现有护栏 | 否，保持现有失败语义 |

纠错要求：

- 使用原事实包，不重新采集事实、不改变 `fact_package_id`。
- 纠错提示只包含稳定错误码和允许的字段结构，不要求模型生成新事实。
- 第二次响应必须重新经过完整结构和证据校验。
- 第二次仍失败时写入审计终态，并在创建输出目录、构造命令和启动进程之前返回。

## 9. 审计要求

每轮审计至少增加：

```json
{
  "fact_package": {
    "schema_version": "1.0",
    "fact_package_id": "sha256:<hex>",
    "task": "detect",
    "reference_run": "train54",
    "sources": {},
    "facts": []
  },
  "decision_validation": {
    "valid": true,
    "error_code": null,
    "error_detail": null,
    "retried": false,
    "referenced_fact_ids": []
  }
}
```

约束：

- 事实包必须在首次 LLM 调用前持久化成功；写入失败则禁止调用 LLM 和训练。
- 保留现有 `decision.raw_response`、Guardrails、baseline 和 execution 字段，避免破坏旧审计查看。
- `baseline` 可继续作为执行前后对照，但其数值必须与事实包中同名来源一致。
- 不在审计中新增凭据、请求头或供应商私有错误正文。

## 10. 测试与验收

### 10.1 单元测试

- 相同输入生成相同事实顺序和 `fact_package_id`。
- 任一事实值变化会改变 `fact_package_id`。
- 缺失值、NaN 和 Infinity 不进入事实包。
- 参考运行、训练报告和指标来源不一致时拒绝构建。
- 合法 TuningDecision v1 通过。
- 额外字段、未知参数、重复参数、错误动作和超过三个参数被拒绝。
- 缺少证据映射、伪造 fact_id、额外证据键和事实包 ID 不匹配被拒绝。
- `keep_params` 只允许空参数和空证据。
- 首次证据失败、第二次合法时只纠错一次并通过。
- 第二次仍失败时返回稳定错误，不继续重试。

### 10.2 调优循环测试

- 事实包构建失败时 LLM、Guardrails 和训练启动均未调用。
- 事实包审计失败时 LLM 和训练启动均未调用。
- 伪造、跨运行或缺失证据时训练启动调用次数为 0。
- 合法证据通过后仍必须进入现有 Guardrails。
- Guardrails 拒绝时训练启动调用次数为 0。
- 合法链路只构造一次实际命令，审计参数与执行参数一致。

### 10.3 兼容与回归

- `generate_suggestion()` 的智能分析页面行为和旧响应格式保持不变。
- 旧审计记录仍可在详情页面读取。
- 运行 Q1.1 定向测试、所有受影响既有测试及完整 `auto_tune/tests`。
- 使用最小合法 Detect 数据完成一次 dry-run；最终验收阶段再完成一次短 epoch 合法链路。

### 10.4 验收门槛

1. 非本轮事实包、伪造事实、缺失证据和非法参数启动训练的数量均为 0。
2. 合法建议能完成事实包、响应、校验、Guardrails、命令和终态的完整审计追踪。
3. 完整测试套件不得新增失败或 skipped。
4. 不新增依赖，不改变智能分析长文本和视觉分析业务行为。
5. 本批只解决证据真实性与运行绑定，不宣称已经解决 LLM 语义正确性。

## 11. 分批建议

- **Q1.1-A（1–2 人日）**：事实包构建、规范散列、来源绑定、审计前置和单元测试。
- **Q1.1-B（1–2 人日）**：TuningDecision v1、证据校验、单次纠错和失败终态测试。
- **Q1.1-C（1 人日）**：调优循环接线、审计/详情兼容、完整回归、dry-run 与短训练验收。

每个子批次由 Claude Code 实现并报告改动与测试；Codex 独立审查和复测。Q1.1-C 通过后再根据真实拒绝案例决定是否启动 Q1.2，不自动扩展范围。

## 12. 风险与控制

- **证据存在不等于推理正确**：产品文案和验收结论必须明确 Q1.1 的边界。
- **事实包字段过多导致噪声**：首版只纳入现有确定性字段，不复制整份报告。
- **双 Schema 相互影响**：自动调优使用 TuningDecision v1；智能分析继续使用现有 suggestion 合同。
- **审计文件膨胀**：事实包只保存允许字段，不嵌入完整训练报告或图像内容。
- **旧记录兼容**：新增字段可选读取，不迁移或重写历史审计文件。
- **循环代码继续膨胀**：事实构建和合同校验放入独立模块，`loop.py` 只负责编排。
