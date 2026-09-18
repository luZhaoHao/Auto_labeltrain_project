# Hyperparameter Tuning Reliability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不引入模型结构调整的前提下，使一期“大模型建议超参数 → 安全校验 → YOLO 实际执行 → 指标复盘”闭环可验证、可追溯且与后续在线平台兼容。

**Architecture:** 将参数定义、类型转换、边界约束和执行映射收敛到单一注册表；LLM 只产生候选建议，执行器只能接收清洗后的参数。Module B 继续负责事实提取，Module C 负责决策与实验闭环，任务持久化/GPU 调度留给在线平台阶段，但一期生成稳定的 `job_id` 与执行快照接口。

**Tech Stack:** Python 3.10、pytest、FastAPI、Ultralytics CLI、YAML/JSON；不新增第三方依赖。

## Global Constraints

- 一期只允许修改训练超参数，不允许修改 Backbone、Neck、Head、模型 YAML 或自定义模块。
- 不新增依赖；优先使用 `dataclasses`、标准库和现有 PyYAML。
- 所有执行参数必须来自参考参数或经过验证的大模型建议。
- `hyperparameter_changes` 与 `training_overrides` 必须经过同一套校验。
- 保持当前 JSON 报告兼容；新增字段只能向后兼容。
- 当前本地模式仍可使用文件存储，但审计记录必须携带稳定的 `session_id`、`iteration`、`train_name`。
- 后续在线平台可以把同一审计记录迁移到数据库，不在本计划中建设认证、队列或 GPU 租约。
- 模型结构解析和结构自调整不在本计划范围内，保留到后续模型架构实验阶段。

---

## Audit Summary and Roadmap Alignment

### 一期必须立即修复

1. Guardrails 计算了截断值，但 Loop 合并并执行原始值。
2. `training_overrides` 绕过第一轮验证，第二轮错误只记录、不阻止。
3. `optimizer=auto` 时 Ultralytics 会忽略 `lr0`、`momentum`，但界面仍可能声称修改已生效。
4. LLM JSON 缺少严格类型、字段和参数白名单校验。
5. Prompt、Guardrails、CLI 参数映射各自维护参数集合，已经出现 `fl_gamma` 等不一致。
6. Module B 读取完整配置却按平铺配置取阈值，`train_analyzer` 自定义阈值可能失效。
7. Early Stopping 接收的数据层级错误，Stale 检测未进入有效路径。
8. 历史反馈使用不存在的 `result` 字段，下一轮 LLM 不能获得可靠 before/after/delta。
9. 正式 pytest 未覆盖 Module C；当前 `test_modelc.py` 主要是人工集成脚本。

### 可为后续功能预留、但不应现在重构

- 用 `TuningAuditRecord` 明确审计结构，为阶段 B 的数据库 `training_jobs` 表预留字段。
- 执行器保留 `train_name`/`output_dir` 接口，为未来任务队列与 GPU Worker 复用。
- 参数注册表预留 `tasks`、`versions` 字段，为 YOLO11、分割、OBB 扩展服务。
- 不在一期拆出数据库、认证、GPU 调度；这些仍归 `implementation_plan_20260814.md` 阶段 B。
- 不在一期加入模型结构动作空间；结构解析与结构自调整统一放入后续模型架构实验阶段。

### 对现有路线图的校准建议

- `roadmap_20260814.md` 中“核心层复用，无需改动”应改为“核心理念复用，Module B/C 先完成可靠性加固”。
- 阶段 A 开始前插入 A0“当前闭环可靠性加固”，然后再做 task/version 框架、Linux 迁移和上传体验。
- “优化 LLM 分析能力”不能只列为 Prompt 工程；应包含事实层、Schema、Guardrails 和执行审计。
- 模型架构实验平台继续后置，但一期参数注册表应避免把结构动作混入超参数命名空间。

---

### Task 1: Establish Formal Module C Regression Tests

**Files:**
- Create: `auto_tune/tests/test_guardrails.py`
- Create: `auto_tune/tests/test_decision_agent.py`
- Create: `auto_tune/tests/test_tuning_loop.py`
- Preserve: `test_modelc.py` as a manual/API integration script

**Interfaces:**
- Consumes: existing `validate_and_clamp`, `merge_params`, `decide_hyperparameters`, `run_tuning_loop`.
- Produces: deterministic pytest coverage that never calls external LLM APIs or starts real training.

- [ ] **Step 1: Write failing Guardrails execution-safety tests**

```python
def test_out_of_range_value_is_the_value_returned_for_execution():
    result = validate_and_clamp({"lr0": 2.0})
    assert result.valid is True
    assert result.params["lr0"] == 0.1


def test_training_override_uses_same_validation_path():
    result = sanitize_tuning_parameters(
        {"lr0": 0.005},
        {"epochs": 5000, "batch": 0},
    )
    assert result.params["epochs"] == 1000
    assert result.params["batch"] == 1
```

- [ ] **Step 2: Write failing parameter-dependency tests**

```python
def test_auto_optimizer_rejects_explicit_learning_rate():
    result = sanitize_tuning_parameters(
        {"lr0": 0.0025},
        {"optimizer": "auto"},
    )
    assert result.valid is False
    assert any("optimizer=auto" in error for error in result.errors)
```

- [ ] **Step 3: Write failing Loop tests with all external boundaries mocked**

```python
def test_loop_executes_only_sanitized_parameters(monkeypatch, tmp_path):
    monkeypatch.setattr(loop, "decide_hyperparameters", lambda *a, **k: {
        "diagnosis": "test",
        "action": "reduce lr",
        "hyperparameter_changes": {"lr0": 2.0},
        "training_overrides": {"optimizer": "AdamW"},
        "error": None,
    })
    captured = {}
    monkeypatch.setattr(loop, "launch_training", lambda name, path, params: captured.update(params) or FakeProcess())
    result = loop.run_tuning_loop(make_config(tmp_path), skip_execute=False)
    assert captured["lr0"] == 0.1
```

- [ ] **Step 4: Run the new tests and verify they fail for the audited reasons**

Run: `python -m pytest auto_tune/tests/test_guardrails.py auto_tune/tests/test_decision_agent.py auto_tune/tests/test_tuning_loop.py -v -p no:cacheprovider`

Expected: failures show missing sanitized `params`, missing unified override validation, and raw values reaching the Loop.

- [ ] **Step 5: Commit the regression tests**

```bash
git add auto_tune/tests/test_guardrails.py auto_tune/tests/test_decision_agent.py auto_tune/tests/test_tuning_loop.py
git commit -m "test: define hyperparameter tuning safety contract"
```

---

### Task 2: Create a Single Hyperparameter Registry

**Files:**
- Create: `auto_tune/modules/agent_engine/parameter_registry.py`
- Modify: `auto_tune/modules/agent_engine/guardrails.py`
- Modify: `auto_tune/modules/agent_engine/executor.py`
- Test: `auto_tune/tests/test_guardrails.py`

**Interfaces:**
- Produces: `ParameterSpec`, `PARAMETER_REGISTRY`, `get_tunable_parameter_names()`, `build_cli_parameters(params)`.
- Consumed by: Decision Prompt, Guardrails, executor and future task/version adapters.

- [ ] **Step 1: Define the registry contract in a failing test**

```python
def test_every_tunable_parameter_has_cli_mapping():
    for name, spec in PARAMETER_REGISTRY.items():
        if spec.llm_tunable:
            assert spec.cli_name
            assert spec.kind in {"int", "float", "bool", "choice", "string"}
```

- [ ] **Step 2: Implement `ParameterSpec` and register一期 parameters**

```python
@dataclass(frozen=True)
class ParameterSpec:
    kind: str
    cli_name: str
    minimum: float | None = None
    maximum: float | None = None
    choices: frozenset[str] = frozenset()
    llm_tunable: bool = True
    group: str = "hyperparameter"
    tasks: frozenset[str] = frozenset({"detect"})


PARAMETER_REGISTRY = {
    "lr0": ParameterSpec("float", "lr0", 1e-5, 0.1),
    "lrf": ParameterSpec("float", "lrf", 1e-5, 1.0),
    "batch": ParameterSpec("int", "batch", 1, 256, group="training"),
    "epochs": ParameterSpec("int", "epochs", 1, 1000, group="training"),
    "optimizer": ParameterSpec(
        "choice", "optimizer",
        choices=frozenset({"SGD", "Adam", "AdamW", "Adamax", "NAdam", "RAdam", "auto"}),
        group="training",
    ),
}
```

- [ ] **Step 3: Replace duplicated bounds and CLI maps with registry reads**

`guardrails.py` must no longer own an independent `BOUNDS`; `executor.py` must obtain allowed CLI fields from the same registry. Non-tunable runtime parameters such as `project`, `name`, `device` remain executor-owned and cannot be emitted by the LLM.

- [ ] **Step 4: Remove unsupported Prompt actions**

Delete `fl_gamma` from the mapping rule unless it is explicitly registered and supported by the installed Ultralytics version. Add a test asserting every parameter name found in the expert mapping table exists in the registry.

- [ ] **Step 5: Run registry and existing executor tests**

Run: `python -m pytest auto_tune/tests/test_guardrails.py auto_tune/tests -q -p no:cacheprovider`

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add auto_tune/modules/agent_engine/parameter_registry.py auto_tune/modules/agent_engine/guardrails.py auto_tune/modules/agent_engine/executor.py auto_tune/tests/test_guardrails.py
git commit -m "refactor: centralize tunable parameter definitions"
```

---

### Task 3: Make Guardrails Produce the Only Executable Parameter Set

**Files:**
- Modify: `auto_tune/modules/agent_engine/guardrails.py`
- Modify: `auto_tune/modules/agent_engine/loop.py`
- Test: `auto_tune/tests/test_guardrails.py`
- Test: `auto_tune/tests/test_tuning_loop.py`

**Interfaces:**
- Produces: `SanitizationResult.params`, the only dictionary accepted by `merge_params` and `launch_training`.

- [ ] **Step 1: Extend the result object**

```python
@dataclass
class SanitizationResult:
    valid: bool
    params: dict[str, int | float | str | bool]
    original: dict[str, object]
    adjusted: dict[str, dict[str, object]]
    warnings: list[str]
    errors: list[str]
```

- [ ] **Step 2: Implement strict type conversion and rejection**

Accept numeric strings only when lossless (`"0.005"` → `0.005`). Reject nested objects, lists, NaN, Infinity, unknown parameters and booleans supplied for numeric fields.

- [ ] **Step 3: Validate changes and overrides together**

```python
candidate = {**hyperparameter_changes, **training_overrides}
sanitized = sanitize_tuning_parameters(candidate, dataset_info=dataset_info)
if not sanitized.valid:
    stop_iteration(sanitized.errors)
merged = merge_params(base_args, sanitized.params)
```

- [ ] **Step 4: Enforce semantic dependencies**

- Reject `optimizer=auto` combined with an explicit `lr0` or `momentum` change.
- Require `close_mosaic < epochs` when both are known.
- Reject simultaneous `resume=true` and a change to model/data/imgsz.
- Warn when batch changes without an explicit learning-rate policy.

- [ ] **Step 5: Remove the second warning-only validation path**

The Loop must perform one authoritative sanitization after combining changes and overrides. It must never merge or execute the original LLM dictionaries.

- [ ] **Step 6: Run tests**

Run: `python -m pytest auto_tune/tests/test_guardrails.py auto_tune/tests/test_tuning_loop.py -v -p no:cacheprovider`

Expected: all safety-contract tests pass.

- [ ] **Step 7: Commit**

```bash
git add auto_tune/modules/agent_engine/guardrails.py auto_tune/modules/agent_engine/loop.py auto_tune/tests/test_guardrails.py auto_tune/tests/test_tuning_loop.py
git commit -m "fix: execute only sanitized tuning parameters"
```

---

### Task 4: Validate LLM Decisions Before Guardrails

**Files:**
- Modify: `auto_tune/modules/agent_engine/decision_agent.py`
- Test: `auto_tune/tests/test_decision_agent.py`

**Interfaces:**
- Produces: `parse_decision_response(text) -> DecisionResult` with typed dictionaries and structured errors.

- [ ] **Step 1: Write malformed-response tests**

```python
@pytest.mark.parametrize("payload", [
    '{"hyperparameter_changes": [1, 2]}',
    '{"hyperparameter_changes": {"lr0": "降低"}}',
    '{"hyperparameter_changes": {"unknown": 1}}',
])
def test_invalid_decision_is_rejected(payload):
    result = parse_decision_response(payload)
    assert result.valid is False
```

- [ ] **Step 2: Implement structural validation without new dependencies**

Require `diagnosis` and `action` to be strings; require both parameter sections to be dictionaries; allow no more than three changed parameters per iteration in normal mode; reject unknown keys using `PARAMETER_REGISTRY`.

- [ ] **Step 3: Generate the Prompt parameter list from the registry**

The Prompt must show only currently supported parameters, ranges and choices. Do not hand-maintain a second list.

- [ ] **Step 4: Add an explicit no-change result**

```json
{
  "diagnosis": "当前指标已稳定",
  "action": "keep_params",
  "hyperparameter_changes": {},
  "training_overrides": {}
}
```

An empty change set is valid only when `action == "keep_params"`; otherwise return an actionable validation error.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest auto_tune/tests/test_decision_agent.py -v -p no:cacheprovider`

```bash
git add auto_tune/modules/agent_engine/decision_agent.py auto_tune/tests/test_decision_agent.py
git commit -m "fix: validate structured LLM tuning decisions"
```

---

### Task 5: Repair Module B Fact Generation

**Files:**
- Modify: `auto_tune/modules/train_analyzer/analyzer.py`
- Modify: `auto_tune/modules/train_analyzer/curve_analysis.py`
- Modify: `auto_tune/modules/train_analyzer/issue_detector.py`
- Modify: `auto_tune/modules/agent_engine/perception.py`
- Test: `auto_tune/tests/test_train_analyzer.py`
- Test: `auto_tune/tests/test_curve_analysis.py`
- Test: `auto_tune/tests/test_issue_detector.py`

**Interfaces:**
- Produces: stable Stage-1 facts consumed by Decision Agent; no change to existing top-level report keys.

- [ ] **Step 1: Add a failing nested-config test**

```python
def test_analyzer_uses_train_analyzer_section(tmp_path):
    config = {"train_analyzer": {"min_acceptable_map": 0.1, "stale_threshold": 3}}
    report = analyze_training_results(str(make_run(tmp_path, final_map=0.2)), config)
    assert "low_final_map" not in issue_types(report)
```

- [ ] **Step 2: Normalize configuration once in the orchestrator**

```python
analysis_config = config.get("train_analyzer", config)
curve_analysis = analyze_loss_curves(run_data["results"], analysis_config)
metric_analysis = analyze_metric_curves(run_data["results"], analysis_config)
early_stop = detect_early_stopping(run_data["results"], run_data["args"], analysis_config)
```

- [ ] **Step 3: Correct Early Stopping interface**

Change signature to:

```python
def detect_early_stopping(results: dict, args: dict, config: dict) -> dict:
```

Use actual row count from `results["total_epochs"]` and planned epochs/patience from `args`.

- [ ] **Step 4: Implement recent-window plateau and stale facts**

Plateau is true only when the best improvement inside the last `plateau_epochs` is below a configurable epsilon. Stale is true when `total_epochs - best_epoch >= stale_threshold`. Pass these facts explicitly to `detect_issues`; do not use hidden `_results` fields.

- [ ] **Step 5: Add fact consistency fields for vision analysis**

Expose numerical Precision/Recall and available error counts beside vision text. Mark vision conclusions as `hypotheses`, not facts, until prediction-vs-GT matching is implemented in the later analysis enhancement.

- [ ] **Step 6: Run Module B tests**

Run: `python -m pytest auto_tune/tests/test_train_analyzer.py auto_tune/tests/test_curve_analysis.py auto_tune/tests/test_issue_detector.py -v -p no:cacheprovider`

Expected: all pass, including nested config, Early Stopping and stale-window cases.

- [ ] **Step 7: Commit**

```bash
git add auto_tune/modules/train_analyzer auto_tune/modules/agent_engine/perception.py auto_tune/tests/test_train_analyzer.py auto_tune/tests/test_curve_analysis.py auto_tune/tests/test_issue_detector.py
git commit -m "fix: make training diagnosis facts configuration-aware"
```

---

### Task 6: Add End-to-End Parameter Audit Records

**Files:**
- Create: `auto_tune/modules/agent_engine/audit.py`
- Modify: `auto_tune/modules/agent_engine/loop.py`
- Modify: `auto_tune/modules/agent_engine/executor.py`
- Test: `auto_tune/tests/test_tuning_loop.py`

**Interfaces:**
- Produces: `TuningAuditRecord.to_dict()` and `log/tuning_audit_<session_id>.json`.
- Future consumer: stage-B database repository can persist the same dictionary unchanged.

- [ ] **Step 1: Define the audit record**

```python
@dataclass
class TuningAuditRecord:
    session_id: str
    iteration: int
    reference_run: str
    original_params: dict
    llm_raw_response: str | None
    proposed_changes: dict
    sanitized_changes: dict
    actual_params: dict
    command: list[str]
    before_metrics: dict
    after_metrics: dict
    metric_delta: dict
    status: str
```

- [ ] **Step 2: Build the command before launch and persist the exact command**

Refactor executor to expose:

```python
def launch_training(command: list[str], log_path: str) -> subprocess.Popen:
```

The command stored in the audit record must be byte-for-byte equivalent to the executed argument list.

- [ ] **Step 3: Record before/after/delta metrics**

Use mAP50, mAP50-95, Precision and Recall. Missing metrics remain `None`; never convert missing values to zero because that changes experiment meaning.

- [ ] **Step 4: Feed actual history back to the next LLM iteration**

Replace the nonexistent generic `result` field with `before_metrics`, `after_metrics`, `metric_delta`, `probe_verdict` and `status`.

- [ ] **Step 5: Make writes atomic**

Write JSON to `<path>.tmp`, flush and close, then use `os.replace(tmp, path)`. This local implementation can later be replaced by a database repository.

- [ ] **Step 6: Run Loop tests and commit**

Run: `python -m pytest auto_tune/tests/test_tuning_loop.py -v -p no:cacheprovider`

```bash
git add auto_tune/modules/agent_engine/audit.py auto_tune/modules/agent_engine/loop.py auto_tune/modules/agent_engine/executor.py auto_tune/tests/test_tuning_loop.py
git commit -m "feat: add traceable tuning audit records"
```

---

### Task 7: UI and Documentation Truthfulness

**Files:**
- Modify: `auto_tune/ui/app.py`
- Modify: `auto_tune/ui/templates/single_page.html`
- Modify: `docs/roadmap_20260814.md`
- Modify: `docs/implementation_plan_20260814.md`
- Modify: `docs/module_c_testing.md`
- Test: `auto_tune/tests/test_tuning_loop.py`

**Interfaces:**
- Consumes: audit records from Task 6.
- Produces: UI comparison of proposed, sanitized and actual execution parameters.

- [ ] **Step 1: Show three parameter states**

The UI must distinguish:

```text
AI 建议值 → 护栏调整值 → 实际训练值
```

If Ultralytics overrides a parameter, show it as “运行时自动决定”，not as an applied Agent change.

- [ ] **Step 2: Show result delta after Module B completes**

Display before/after/delta for mAP50, mAP50-95, Precision and Recall. Do not describe an iteration as improved unless the selected evaluation score increased.

- [ ] **Step 3: Add A0 to roadmap and implementation plan**

Insert “一期闭环可靠性加固” before current A1. State that Module B/C concepts are reusable but require reliability work before online-platform development. Keep model-structure parsing and self-adjustment in the later architecture-experiment phase.

- [ ] **Step 4: Update Module C testing guide**

Document dry-run validation, 1-epoch smoke training, command/audit comparison and `optimizer=auto` conflict test.

- [ ] **Step 5: Run complete automated verification**

Run: `python -m pytest auto_tune/tests -q -p no:cacheprovider`

Expected: all existing and newly added tests pass.

- [ ] **Step 6: Run one controlled smoke training**

Use the existing small validated dataset, `yolov8n.pt`, `epochs=1`, `imgsz=320`, `batch=8`. Verify:

- audit `actual_params` equals generated `args.yaml` for controlled fields;
- audit command equals the launched command;
- UI shows actual values;
- completion triggers Module B and fills metric delta;
- no auto-loop starts unless explicitly enabled.

- [ ] **Step 7: Commit**

```bash
git add auto_tune/ui docs/roadmap_20260814.md docs/implementation_plan_20260814.md docs/module_c_testing.md
git commit -m "docs: align tuning reliability with product roadmap"
```

---

## Deferred Work Boundaries

The following are intentionally excluded and remain in later roadmap phases:

- authentication, users and roles;
- SQLite/PostgreSQL job persistence;
- multi-process training worker and GPU lease scheduler;
- SSE reconnect and process recovery after service restart;
- YOLO model YAML parsing;
- Backbone/Neck/Head/Attention/Loss structure changes;
- automatic architecture search or ablation orchestration;
- segmentation, OBB and classification-specific parameter registries.

The一期 registry and audit record must make these additions possible without changing the meaning of existing fields.

## Self-Review

- **Spec coverage:** covers Module B fact correctness, LLM decision Schema, unified parameter registry, Guardrails enforcement, actual CLI execution, history feedback, UI truthfulness and roadmap alignment.
- **Scope:** model structure parsing/self-adjustment, online users/database/GPU queue remain explicitly deferred.
- **Placeholder scan:** every task contains concrete files, interfaces, commands and expected outcomes.
- **Type consistency:** `SanitizationResult.params` is the sole executable parameter dictionary; `TuningAuditRecord` names are consistent across Tasks 3, 6 and 7.
- **Dependency order:** tests → registry → sanitization → LLM validation → facts → audit → UI/docs.
