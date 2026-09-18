# Q1.2 LLM 调优语义校验设计规格

> 状态：已完成并通过 Codex 独立验收（2026-09-02）
> 日期：2026-09-01
> 适用范围：YOLOv8 Detect 的 LLM 自动调优决策链
> 前置版本：Q1.1 自动调优事实包与证据引用 MVP

## 1. 结论

Q1.2 在 Q1.1 的证据存在性校验之后、现有 Guardrails 之前增加一层确定性语义校验。它验证每个事实引用是否允许支持对应参数、修改方向是否正确、单轮变化幅度是否越界，以及多条证据是否相互冲突。

语义校验不调用第二个模型，不解释开放式自然语言，不自动修改或删除 LLM 建议。首次失败使用 Q1.1 已有的唯一一次纠错机会；再次失败形成明确终态，说明“证据与参数修改语义不一致”，并在创建训练目录、构造命令和启动训练之前停止。

预计代码与测试工作量为 **3–5 人日**；Codex 独立验收和文档回写预计 **1–2 人日**。Q1.1 与 Q1.2 完成联合验收后再统一提交和推送。

## 2. 问题与边界

Q1.1 已经保证：

- LLM 只能接收代码生成、绑定当前参考运行的事实包；
- 每个参数修改必须引用事实包中真实存在的 `fact_id`；
- 响应必须绑定当前 `fact_package_id`；
- 结构或证据存在性失败最多纠错一次，再失败禁止训练；
- 事实包、响应、证据和校验结果进入审计链。

Q1.1 尚不能判断“某条真实事实是否足以支持某种参数修改”。例如，`training.metrics.mAP50` 真实存在，并不代表它可以单独支持调整 `batch`。Q1.2 只补充这一确定性业务边界，不宣称能够证明 LLM 的完整推理正确。

Q1.2 不处理：

- 第二模型复核、知识图谱、多 Agent 辩论或开放式推理；
- HPO 结果交给 LLM 裁决；
- 自动删除不合格参数后部分执行；
- 用户绕过语义失败强制自动训练；
- 自动生成或修改规则；
- Classify 业务规则；
- 模型效果优劣和最终排名，二者继续由确定性代码计算。

## 3. 总体流程

```text
代码生成并冻结 FactPackage v1
        ↓
LLM 输出 TuningDecision v1
        ↓
结构校验 + fact_package_id 校验 + evidence_ids 存在性校验（Q1.1）
        ↓
证据—参数—方向—幅度语义校验（Q1.2）
        ↓ 失败
同一事实包纠错一次 ──再次失败──→ 语义失败终态，不启动训练
        ↓ 通过
现有 Guardrails → 训练预检 → 唯一命令构造 → 训练
```

职责分离：

- `decision_contract.py`：JSON 结构、字段、事实包绑定和证据是否存在；
- `semantic_rules.py`：Detect 语义规则唯一运行时来源；
- `decision_semantics.py`：确定性执行语义规则，不修改参数；
- `guardrails.py`：参数类型、绝对范围和组合冲突；
- `loop.py`：编排校验、失败终态和训练启动边界；
- `audit.py`：持久化语义校验输入摘要与结果。

## 4. 文件与职责

### 4.1 新增 `semantic_rules.py`

定义不可变规则对象及 Detect 规则注册表。每条规则至少包含：

- 稳定 `rule_id`；
- 精确 `fact_id`；
- 可选的精确事实值；
- 允许参数；
- 允许方向或布尔转换；
- 单轮变化幅度限制；
- 用于提示词的受控中文说明。

该注册表同时服务语义校验器和自动调优提示词，禁止在提示词中另写一套独立映射。

### 4.2 新增 `decision_semantics.py`

建议公开接口：

```python
def validate_decision_semantics(
    decision: dict,
    fact_package: dict,
) -> dict:
    """返回结构化语义校验结果；失败不修改 decision。"""
```

返回结果至少包含：

```json
{
  "valid": false,
  "error_code": "DECISION_SEMANTIC_UNSUPPORTED",
  "reason_code": "NO_SUPPORTING_RULE",
  "parameter": "batch",
  "rule_ids": [],
  "supporting_fact_ids": [],
  "conflicting_fact_ids": [],
  "neutral_fact_ids": ["training.metrics.mAP50"],
  "current_value": 16,
  "suggested_value": 32,
  "change_direction": "increase"
}
```

合法结果使用相同字段结构，`error_code`、`reason_code` 和 `parameter` 为 `null`。结果必须可 JSON 序列化，并且不得包含原始 LLM 响应、凭据、绝对路径或未受控错误正文。

### 4.3 修改 `decision_agent.py`

- 自动调优提示词从 Detect 规则注册表生成允许关系摘要；
- 首次响应通过 Q1.1 校验后进入 Q1.2 校验；
- 语义失败复用现有唯一一次纠错，不增加第三次模型调用；
- 纠错提示包含稳定错误码、失败参数和受控的允许方向/幅度；
- 纠错继续使用完全相同的事实包和 `fact_package_id`；
- `generate_suggestion()` 及智能分析页面保持原合同。

### 4.4 修改 `loop.py`

- 只有结构、证据存在性和语义校验全部通过，才允许进入 Guardrails；
- 语义失败必须写入审计后形成终态；
- 语义失败不得创建输出目录、构造命令或启动训练；
- 不从失败决策中挑选部分参数执行；
- `keep_params` 无参数变化，语义校验直接通过。

### 4.5 修改审计与投影

- 审计 Schema 从 `1.1` 升级为 `1.2`；
- 每轮增加 `semantic_validation`；
- 旧 1.0/1.1 审计继续可读；
- 旧记录缺少该字段时显示“未执行语义校验”，不得显示为“通过”；
- Studio 首版只显示语义状态、稳定原因和失败参数，不展示完整事实包或原始响应。

## 5. Detect 首版规则

### 5.1 允许关系

| 事实 | 参数 | 允许动作 |
|---|---|---|
| `training.issue.overfitting` | `weight_decay` | 增加 |
| `training.issue.overfitting` | `epochs` | 减少 |
| `training.curve.val_box_loss=rising` | `weight_decay` | 增加 |
| `training.curve.val_box_loss=rising` | `epochs` | 减少 |
| `training.curve.val_cls_loss=rising` | `weight_decay` | 增加 |
| `training.curve.val_cls_loss=rising` | `epochs` | 减少 |
| `training.issue.underfitting` | `weight_decay` | 减少 |
| `training.issue.underfitting` | `epochs` | 增加 |
| `training.issue.plateau` | `lr0` | 降低 |
| `training.issue.plateau` | `cos_lr` | `false → true` |
| `training.curve.mAP50=saturated` | `lr0` | 降低 |
| `training.curve.mAP50=saturated` | `cos_lr` | `false → true` |
| `training.issue.unstable_training` | `lr0` | 降低 |
| `training.issue.unstable_training` | `warmup_epochs` | 增加 |
| `training.issue.nan_loss` | `lr0` | 降低 |
| `training.issue.nan_loss` | `warmup_epochs` | 增加 |
| `training.issue.early_stop_too_soon` | `patience` | 增加 |
| `dataset.issue.tiny_bbox_high_ratio` | `imgsz` | 增加 |
| `dataset.issue.tiny_bbox_high_ratio` | `box` | 增加 |
| `dataset.issue.long_tail_class` | `cls` | 增加 |
| `dataset.issue.center_spatial_bias` | `translate` | 增加 |

事实值条件必须精确匹配。其他值即使具有相同 `fact_id` 也不能命中该规则。

### 5.2 单轮幅度限制

| 参数 | 限制 |
|---|---|
| `lr0` | 新值为当前值的 `25%–80%` |
| `weight_decay` 增加 | 当前值大于 0 时，新值不超过当前值 4 倍；当前值为 0 时，新值不超过 `0.001` |
| `weight_decay` 减少 | 当前值必须大于 0，新值不得低于当前值的 25% |
| `epochs` 减少 | 新值不得低于当前值的 50% |
| `epochs` 增加 | 新值不得超过当前值的 2 倍 |
| `patience` 增加 | 当前值大于 0 时不超过 2 倍；当前值为 0 时新值不超过 20 |
| `warmup_epochs` 增加 | 新值不超过当前值加 3 |
| `imgsz` 增加 | 新值不超过当前值 2 倍 |
| `box`、`cls` 增加 | 新值不超过当前值 2 倍 |
| `translate` 增加 | 新值不超过当前值加 `0.2` |
| `cos_lr` | 只允许 `false → true` |

边界值允许，超过边界拒绝。绝对上下限仍由 `PARAMETER_REGISTRY` 和 Guardrails 校验，Q1.2 不复制绝对范围。

### 5.3 不支持自动修改的事实和参数

以下事实不能单独支持修改参数：

- `dataset.issue.low_label_coverage`；
- `dataset.issue.high_blur_ratio`；
- `training.issue.parse_error`；
- `training.issue.low_final_map`；
- 原始 `mAP50`、`mAP50_95`、`precision`、`recall`；
- `descending`、`improving`、`degrading`等首版未列入表格的趋势。

以下注册参数首版没有任何语义允许关系：

- `model`、`optimizer`、`batch`；
- `mosaic`、`mixup`、`copy_paste`；
- 未在 5.1 中列出的其他参数。

它们仍可存在于参数注册表和事实包中，但作为 LLM 修改项时必须返回语义不支持。此限制不影响直接训练和未来 HPO。

## 6. 校验算法

对每个修改参数独立收集其 `evidence_ids`，并按下列顺序校验：

1. 从事实包读取 `training.params.<parameter>` 当前值；缺失则失败；
2. 按注册参数类型规范化建议值；无法比较则失败；
3. 计算 `increase`、`decrease`、`unchanged` 或布尔转换；
4. 将证据划分为支持、冲突和中性；
5. 至少一条证据必须明确支持当前参数和动作；
6. 任一证据若明确要求相反方向，则整体失败；
7. 支持关系存在后再校验单轮幅度；
8. 所有参数均通过时，整个决策才通过。

分类规则：

- **支持**：同一事实规则明确允许当前参数和实际动作；
- **冲突**：同一事实存在该参数规则，但规则要求相反动作；
- **中性**：该事实没有当前参数规则。

组合原则：

- 支持 + 中性：通过语义关系判断，继续校验幅度；
- 支持 + 冲突：拒绝；
- 全部中性：拒绝；
- 多参数仅一个失败：整个决策拒绝；
- `keep_params`：直接通过，不要求参数当前值。

若建议值等于当前值，返回 `DECISION_SEMANTIC_UNSUPPORTED`，理由为 `UNCHANGED_VALUE`；LLM 不得把未变化值声明为调参动作。

## 7. 错误码与用户提示

稳定错误码：

| 错误码 | 含义 |
|---|---|
| `DECISION_SEMANTIC_UNSUPPORTED` | 没有规则支持该事实与参数动作，或值未变化 |
| `DECISION_SEMANTIC_DIRECTION_CONFLICT` | 建议方向与支持规则相反 |
| `DECISION_SEMANTIC_CHANGE_TOO_LARGE` | 单轮变化超过规则幅度 |
| `DECISION_SEMANTIC_CURRENT_VALUE_MISSING` | 事实包缺少可比较的当前参数值 |
| `DECISION_SEMANTIC_EVIDENCE_CONFLICT` | 多条证据对同一参数提出相反方向 |

受控 `reason_code` 至少包括：

- `NO_SUPPORTING_RULE`；
- `UNCHANGED_VALUE`；
- `DIRECTION_NOT_SUPPORTED`；
- `CHANGE_LIMIT_EXCEEDED`；
- `CURRENT_VALUE_MISSING`；
- `CONFLICTING_EVIDENCE`。

首次失败进入现有一次纠错流程。第二次仍失败时，用户统一看到：

> LLM 建议引用了真实证据，但证据与参数修改之间缺少受支持的语义关系。本轮自动调优已停止，未启动训练。

审计和研发日志可以记录稳定错误码、原因码、参数名、规则 ID 和受控事实 ID，但不得拼接原始响应、绝对路径、凭据或任意异常正文。

## 8. 审计合同

`semantic_validation` 至少包含：

```json
{
  "valid": true,
  "error_code": null,
  "reason_code": null,
  "retried": false,
  "parameters": [
    {
      "parameter": "epochs",
      "current_value": 50,
      "suggested_value": 25,
      "change_direction": "decrease",
      "rule_ids": ["detect.overfitting.epochs.decrease.v1"],
      "supporting_fact_ids": ["training.issue.overfitting"],
      "conflicting_fact_ids": [],
      "neutral_fact_ids": []
    }
  ]
}
```

事实包必须仍在第一次 LLM 调用前持久化。每次模型响应的 Q1.1 和 Q1.2 校验结果都必须在决定是否继续前写入审计；审计写入失败继续使用现有 fail-closed 语义。

## 9. 提示词与纠错

初始提示词从规则注册表生成简洁允许表，只列出首版支持关系和幅度，不重新引入自由文本专家规则。未列出的参数明确说明不能自动修改。

纠错提示必须：

- 使用同一事实包和同一 `fact_package_id`；
- 只提供一个稳定语义错误码；
- 指明失败参数；
- 提供该参数受控的允许证据类型、方向和幅度；
- 要求重新输出完整 TuningDecision v1；
- 不提供或猜测替代参数值；
- 不增加调用次数。

供应商、网络、鉴权和端点策略错误仍不重试。

## 10. 测试与验收

### 10.1 单元测试

- 每条注册规则的合法动作；
- 每条注册规则的相反动作；
- 每项幅度限制的边界值和越界值；
- 数值当前值为 0 的明确分支；
- 整数、浮点和布尔比较；
- 当前参数事实缺失；
- 支持 + 中性、支持 + 冲突、全部中性；
- 多参数中单项失败导致整体拒绝；
- `keep_params` 直接通过；
- `model/optimizer/batch` 和其他未开放参数拒绝；
- 提示词摘要完全来自规则注册表；
- 规则 ID、事实 ID 和错误码稳定且不含任意输入。

### 10.2 决策与编排测试

- 首次语义失败、纠错成功；
- 第二次语义失败形成明确终态；
- Q1.1 失败时不进入 Q1.2；
- Q1.2 失败时不进入 Guardrails；
- 语义失败时输出目录、命令构造和训练启动均为零；
- 合法语义决策继续进入原 Guardrails、预检和训练链；
- 不发生部分参数执行；
- 审计写入失败继续禁止训练。

### 10.3 兼容和安全测试

- Audit Schema 1.0、1.1、1.2 均可读取；
- 旧记录显示“未执行语义校验”；
- 新记录显示通过或明确失败原因；
- API/UI 不暴露原始响应、完整事实包、绝对路径、命令或凭据；
- Q1.1 全部测试和完整自动化套件无回归。

### 10.4 发布级验收

Codex 独立执行：

1. Q1.2 定向测试；
2. Q1.1 + Q1.2 聚合回归；
3. 完整自动化套件；
4. 一次合法语义决策的 Detect 最小真实短训练；
5. 至少三种非法语义决策的零启动验证；
6. Studio 状态、历史详情和审计详情检查；
7. 机器规则与人工规则说明逐条比对；
8. Git 文件范围、`.gitignore` 和敏感信息检查。

## 11. 人工规则说明文档

Q1.2 代码通过独立验收后，由 Codex 创建：

```text
docs/llm_tuning_semantic_rules.md
```

文档必须说明目标、流程位置、完整规则表、幅度限制、不支持项、多证据处理、错误码、扩展方式和能力边界。该文档不得由 Claude Code提前编写，最终内容必须以验收通过的 `semantic_rules.py` 为准。

Classify 后续使用独立规则集合；人工说明中必须明确 Detect 规则不能直接套用于 Classify。

## 12. 完成定义

Q1.2 只有同时满足以下条件才完成：

- 规则注册表是提示词和校验器的唯一运行时来源；
- 每个修改参数具有至少一条支持证据且无冲突证据；
- 方向、幅度和当前值均通过确定性校验；
- 语义失败最多纠错一次，再失败明确说明原因并零启动；
- 合法决策仍通过现有 Guardrails、预检和唯一命令构造；
- Audit 1.0/1.1/1.2兼容，UI/API不泄露敏感信息；
- 自动化回归、真实短训练、非法零启动和页面验收全部通过；
- `docs/llm_tuning_semantic_rules.md` 与最终机器规则一致；
- Q1.1 与 Q1.2 联合完成提交前检查后，才允许统一提交和推送。
