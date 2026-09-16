# Q1.1 自动调优事实包与证据引用 MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 YOLOv8 Detect 的 LLM 自动调优建立最小证据闭环，使每个可执行参数修改都绑定本轮事实包并引用真实事实，伪造、缺失、过期或跨运行证据不能启动训练。

**Architecture:** 新增独立的事实包模块和自动调优决策合同模块；`loop.py` 在 LLM 调用前读取参考参数/指标、冻结事实包并持久化审计，`decision_agent.py` 只在自动调优路径使用 TuningDecision v1。智能分析页面的 `generate_suggestion()` 保持旧合同，合法调优决定继续进入现有 Guardrails、预检、唯一命令构造和训练执行链。

**Tech Stack:** Python 3.10、标准库 `dataclasses/json/hashlib/math`、pytest、现有 FastAPI/Ultralytics 项目代码；不新增第三方依赖。

**Spec:** `docs/superpowers/specs/2026-08-31-q1-1-tuning-fact-evidence-design.md`

**Estimated Effort:** 3–5 人日；Task 1–2 约 1–2 人日，Task 3–4 约 1–2 人日，Task 5 与交付验收约 1 人日。

## Global Constraints

- 只修改 YOLOv8 Detect 的 LLM 自动调优决策链，不接入 HPO、Classify、YOLOv5、Docker 或统一任务 API。
- `generate_suggestion()`、智能分析长文本和视觉分析业务行为保持不变。
- 不验证复杂自然语言推理；Q1.1 只验证事实包身份、证据存在性和参数合同。
- 缺失事实不得转换为 `0`、`?`、空字符串或默认 YOLO 参数。
- 事实包必须在首次 LLM 调用前成功写入审计；失败时不得调用 LLM 或启动训练。
- 结构或证据错误最多纠错一次；网络、鉴权、端点和 Guardrails 错误不重试。
- 不新增依赖，不修改真实 `auto_tune/config.yaml`，不提交数据集、权重、训练产物、日志、审计运行文件或凭据。
- 所有 Python 和 pytest 命令使用 `D:\Program Files\anaconda3\envs\auto_tune\python.exe`。
- Claude Code 不修改 README、路线图、交接、DOCX 或其他项目规划文档，不提交、不推送 GitHub。
- 当前完整回归基线：`1226 passed, 2 warnings, 0 skipped`；不得新增失败或 skipped。

---

## File Map

| 文件 | 操作 | 单一职责 |
|---|---|---|
| `auto_tune/modules/agent_engine/decision_facts.py` | 新增 | 生成、校验并稳定散列 FactPackage v1 |
| `auto_tune/modules/agent_engine/decision_contract.py` | 新增 | 解析 TuningDecision v1 并校验事实包/证据引用 |
| `auto_tune/modules/agent_engine/decision_agent.py` | 修改 | 构造事实包提示词、执行一次受控纠错；保留智能分析旧路径 |
| `auto_tune/modules/agent_engine/loop.py` | 修改 | 在决策前冻结事实包、持久化审计并编排失败终态 |
| `auto_tune/modules/agent_engine/audit.py` | 修改 | 为每轮审计增加事实包和决策校验槽位，Schema 升级为 1.1 |
| `auto_tune/modules/presentation/experiment_views.py` | 原则上不改 | 现有投影忽略未知审计字段；只有兼容测试失败时才做最小读取兼容修正 |
| `auto_tune/tests/test_decision_facts.py` | 新增 | 事实过滤、来源绑定和稳定散列测试 |
| `auto_tune/tests/test_decision_contract.py` | 新增 | TuningDecision v1 与证据校验测试 |
| `auto_tune/tests/test_decision_agent.py` | 修改 | 自动调优提示词、纠错和智能分析兼容测试 |
| `auto_tune/tests/test_tuning_loop.py` | 修改 | 事实包前置、失败禁止训练和合法链路测试 |
| `auto_tune/tests/test_audit.py` | 修改 | Schema 1.1 新字段、原子持久化和脱敏测试 |
| `auto_tune/tests/test_experiment_audit_view.py` | 修改 | 旧 1.0/新 1.1 审计均可安全读取 |

---

### Task 1: FactPackage v1 构建与稳定身份

**Files:**
- Create: `auto_tune/modules/agent_engine/decision_facts.py`
- Create: `auto_tune/tests/test_decision_facts.py`

**Interfaces:**
- Consumes: `perception: dict`、`reference_run: str`、`base_args: dict`、`before_metrics: dict`、`metrics_source: dict`。
- Produces: `build_tuning_fact_package(...) -> dict`。
- Produces: `FactPackageError(error_code="FACT_PACKAGE_INVALID", detail: str)`。
- Later tasks rely on exact keys: `schema_version`、`fact_package_id`、`task`、`reference_run`、`sources`、`facts`。

- [ ] **Step 1: 新建失败测试，固定事实包最小合同**

Create `auto_tune/tests/test_decision_facts.py` with fixtures and the first contract tests:

```python
import copy

import pytest

from auto_tune.modules.agent_engine.decision_facts import (
    FactPackageError,
    build_tuning_fact_package,
)


def _perception():
    return {
        "dataset": {
            "total_images": 290,
            "total_annotations": 128,
            "label_rate": 0.441,
            "quality_score": 0.98,
            "bbox_analysis": {"tiny_bbox_ratio": 0.35, "avg_relative_area": 0.008},
            "image_quality": {"blur_ratio": None, "overexposure_ratio": 0.1},
            "key_issues": ["small_objects"],
        },
        "training": {
            "reference_run": "train54",
            "per_run": {
                "train54": {
                    "issues": [{"type": "overfitting", "severity": "medium"}],
                    "curve_trends": {"mAP50": "plateau", "val_cls_loss": "rising"},
                }
            },
        },
        "sources": {
            "dataset_report": {"status": "available", "basename": "dataset_report_1.json"},
            "training_report": {"status": "available", "basename": "train54_report.json"},
        },
    }


def _build(perception=None, reference_run="train54"):
    return build_tuning_fact_package(
        perception or _perception(),
        reference_run,
        {"lr0": 0.001, "weight_decay": 0.0005, "data": "D:/secret/data.yaml"},
        {"mAP50": 0.8121, "mAP50_95": 0.368, "precision": 0.777, "recall": 0.657},
        {
            "type": "results_csv",
            "path": "D:/secret/detect/train54/results.csv",
            "epoch_scope": "final",
            "error": None,
        },
    )


def test_fact_package_is_bound_and_deterministic():
    first = _build()
    second = _build()
    assert first == second
    assert first["schema_version"] == "1.0"
    assert first["task"] == "detect"
    assert first["reference_run"] == "train54"
    assert first["fact_package_id"].startswith("sha256:")
    assert [f["fact_id"] for f in first["facts"]] == sorted(
        f["fact_id"] for f in first["facts"]
    )


def test_missing_values_and_unregistered_params_are_not_facts():
    package = _build()
    facts = {f["fact_id"]: f["value"] for f in package["facts"]}
    assert "dataset.image_quality.blur_ratio" not in facts
    assert "training.params.data" not in facts
    assert facts["training.params.lr0"] == 0.001


def test_fact_change_changes_package_id():
    changed = _perception()
    changed["dataset"]["total_images"] = 291
    assert _build()["fact_package_id"] != _build(changed)["fact_package_id"]


@pytest.mark.parametrize("bad_run", [None, "", "train53"])
def test_reference_identity_mismatch_is_rejected(bad_run):
    with pytest.raises(FactPackageError) as excinfo:
        _build(reference_run=bad_run)
    assert excinfo.value.error_code == "FACT_PACKAGE_INVALID"
```

- [ ] **Step 2: 运行测试并确认因模块不存在而失败**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_decision_facts.py -v -p no:cacheprovider
```

Expected: collection fails with `ModuleNotFoundError: ...decision_facts`.

- [ ] **Step 3: 实现规范化、过滤和 SHA-256 身份**

Create `decision_facts.py` with these exact public definitions:

```python
from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Any

from .parameter_registry import PARAMETER_REGISTRY

FACT_PACKAGE_SCHEMA_VERSION = "1.0"


class FactPackageError(ValueError):
    def __init__(self, detail: str):
        super().__init__(detail)
        self.error_code = "FACT_PACKAGE_INVALID"
        self.detail = detail


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _append(facts: list[dict], fact_id: str, value: Any, source: str) -> None:
    if value is None or value == "":
        return
    if isinstance(value, float) and not math.isfinite(value):
        return
    facts.append({"fact_id": fact_id, "value": value, "source": source})


def _canonical_payload(package: dict) -> bytes:
    payload = {k: v for k, v in package.items() if k != "fact_package_id"}
    return json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
```

Implement `build_tuning_fact_package()` so that it:

1. Requires a non-empty `reference_run` matching `perception["training"]["reference_run"]`.
2. Requires both source statuses to be `available` and both basenames to be non-empty.
3. Requires `metrics_source["error"] is None` and a non-empty `metrics_source["path"]`, matching the existing `_read_reference_before_metrics()` contract.
4. Stores only `os.path.basename(...)` for metric/parameter sources; never stores absolute paths.
5. Adds dataset facts only from the allowlist in spec §6.1.
6. Adds metrics only for `mAP50`, `mAP50_95`, `precision`, `recall` when finite.
7. Adds parameters only when the key exists in `PARAMETER_REGISTRY` and the value is not missing/non-finite.
8. Adds issue facts as `training.issue.<type> = true` and curve facts as `training.curve.<name> = <trend>` from the reference run only.
9. Sorts facts by `fact_id`, rejects duplicate IDs and an empty fact list, then computes `sha256:<hex>` over `_canonical_payload()`.

Return exactly:

```python
{
    "schema_version": FACT_PACKAGE_SCHEMA_VERSION,
    "fact_package_id": package_id,
    "task": "detect",
    "reference_run": reference_run,
    "sources": {
        "dataset_report": dataset_source["basename"],
        "training_report": training_source["basename"],
        "metrics": os.path.basename(str(metrics_source.get("path") or "results.csv")),
        "params": "args.yaml",
    },
    "facts": facts,
}
```

- [ ] **Step 4: 增加边界测试**

Add parameterized tests for:

```python
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_metric_is_omitted(bad):
    package = build_tuning_fact_package(
        _perception(), "train54", {"lr0": 0.001},
        {"mAP50": bad, "recall": 0.6},
        {"type": "results_csv", "path": "results.csv", "epoch_scope": "final", "error": None},
    )
    ids = {f["fact_id"] for f in package["facts"]}
    assert "training.metrics.mAP50" not in ids
    assert "training.metrics.recall" in ids


def test_absolute_paths_never_enter_fact_package():
    blob = json.dumps(_build(), ensure_ascii=False)
    assert "D:/secret" not in blob
    assert "D:\\\\secret" not in blob
```

Also cover unavailable reports, a non-empty `metrics_source["error"]`, duplicate issue IDs, bool-as-number rejection and an empty fact list. Every case must assert `FactPackageError.error_code == "FACT_PACKAGE_INVALID"` where the whole package is invalid.

- [ ] **Step 5: 运行 Task 1 测试**

Run the same Task 1 command. Expected: all `test_decision_facts.py` tests pass.

- [ ] **Step 6: 向艾卡/Codex报告 Task 1 结果，不提交代码**

Report changed files, exact pytest output, any deviation from the specified fact allowlist, and remaining risks. Do not run `git commit` or `git push`.

---

### Task 2: TuningDecision v1 与证据引用校验

**Files:**
- Create: `auto_tune/modules/agent_engine/decision_contract.py`
- Create: `auto_tune/tests/test_decision_contract.py`

**Interfaces:**
- Consumes: LLM raw `text: str` and Task 1 `fact_package: dict`.
- Produces: `parse_tuning_decision_response(text: str) -> dict`.
- Produces: `validate_decision_evidence(decision: dict, fact_package: dict) -> dict`.
- Produces: `DecisionContractError(error_code: str, detail: str)`.

- [ ] **Step 1: 写合同失败测试**

Create `auto_tune/tests/test_decision_contract.py` with a valid fixture:

```python
import json
import pytest

from auto_tune.modules.agent_engine.decision_contract import (
    DecisionContractError,
    parse_tuning_decision_response,
    validate_decision_evidence,
)


PACKAGE = {
    "schema_version": "1.0",
    "fact_package_id": "sha256:abc",
    "task": "detect",
    "reference_run": "train54",
    "sources": {},
    "facts": [
        {"fact_id": "training.issue.overfitting", "value": True, "source": "training_report"},
        {"fact_id": "training.params.weight_decay", "value": 0.0005, "source": "params"},
    ],
}


def _decision(**overrides):
    value = {
        "schema_version": "1.0",
        "fact_package_id": "sha256:abc",
        "diagnosis": "存在过拟合",
        "action": "adjust",
        "hyperparameter_changes": {"weight_decay": 0.002},
        "training_overrides": {},
        "evidence_ids": {"weight_decay": ["training.issue.overfitting"]},
    }
    value.update(overrides)
    return value


def test_valid_decision_is_normalized():
    parsed = parse_tuning_decision_response(json.dumps(_decision()))
    validated = validate_decision_evidence(parsed, PACKAGE)
    assert validated["fact_package_id"] == "sha256:abc"
    assert validated["evidence_ids"] == {"weight_decay": ["training.issue.overfitting"]}
```

Add parameterized failures asserting stable error codes for extra root fields, wrong schema, invalid action, duplicate parameter buckets, unknown parameter, more than three changes, missing evidence key, extra evidence key, empty evidence list, duplicated fact ID, unknown fact ID and mismatched package ID.

- [ ] **Step 2: 运行测试并确认模块不存在**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_decision_contract.py -v -p no:cacheprovider
```

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: 实现结构合同**

Create these public definitions:

```python
DECISION_SCHEMA_VERSION = "1.0"
ROOT_FIELDS = frozenset({
    "schema_version", "fact_package_id", "diagnosis", "action",
    "hyperparameter_changes", "training_overrides", "evidence_ids",
})


class DecisionContractError(ValueError):
    def __init__(self, error_code: str, detail: str):
        super().__init__(detail)
        self.error_code = error_code
        self.detail = detail
```

`parse_tuning_decision_response()` must reuse or mirror the existing fenced/bare JSON extraction behavior, then enforce:

- exact root fields;
- non-empty string `diagnosis`;
- `action in {"adjust", "keep_params"}`;
- both parameter buckets and `evidence_ids` are dicts;
- no parameter appears in both buckets;
- all parameter names exist in `get_tunable_parameter_names()`;
- `adjust` has 1–3 changes;
- `keep_params` has empty changes and evidence.

All structural failures raise `DecisionContractError("DECISION_SCHEMA_INVALID", safe_detail)`; `safe_detail` names fields but never includes raw model text.

- [ ] **Step 4: 实现事实包和证据引用校验**

`validate_decision_evidence()` must:

```python
if decision["fact_package_id"] != fact_package["fact_package_id"]:
    raise DecisionContractError(
        "DECISION_FACT_PACKAGE_MISMATCH", "fact_package_id does not match current facts"
    )

changed = set(decision["hyperparameter_changes"]) | set(decision["training_overrides"])
evidence = decision["evidence_ids"]
if set(evidence) != changed:
    raise DecisionContractError(
        "DECISION_EVIDENCE_MISSING", "evidence keys must exactly match changed parameters"
    )

known = {item["fact_id"] for item in fact_package["facts"]}
for parameter, ids in evidence.items():
    if not isinstance(ids, list) or not ids or any(not isinstance(x, str) for x in ids):
        raise DecisionContractError("DECISION_EVIDENCE_MISSING", f"invalid evidence for {parameter}")
    if len(ids) != len(set(ids)):
        raise DecisionContractError("DECISION_EVIDENCE_MISSING", f"duplicate evidence for {parameter}")
    unknown = sorted(set(ids) - known)
    if unknown:
        raise DecisionContractError(
            "DECISION_EVIDENCE_UNKNOWN", f"unknown evidence for {parameter}: {', '.join(unknown)}"
        )
return decision
```

- [ ] **Step 5: 运行 Task 2 测试和既有参数测试**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_decision_contract.py auto_tune\tests\test_guardrails.py -v -p no:cacheprovider
```

Expected: all tests pass; existing Guardrails semantics remain unchanged.

- [ ] **Step 6: 报告 Task 2 结果，不提交代码**

Include the error-code matrix and exact test output. Do not commit or push.

---

### Task 3: Decision Agent 自动调优专用合同与一次纠错

**Files:**
- Modify: `auto_tune/modules/agent_engine/decision_agent.py`
- Modify: `auto_tune/tests/test_decision_agent.py`
- Modify: `auto_tune/tests/test_decision_unified.py`

**Interfaces:**
- Consumes: Task 1 FactPackage v1 and Task 2 contract functions.
- Produces: `build_tuning_decision_prompt(fact_package: dict, previous_attempts: list[dict] | None) -> str`.
- Produces: `decide_hyperparameters(fact_package: dict, config: dict, previous_attempts: list[dict] | None = None) -> dict`.
- Keeps unchanged: `generate_suggestion(summary_text, project_info, config) -> dict`.

- [ ] **Step 1: 写自动调优专用失败/纠错测试**

Add tests that patch `call_decision_llm` with two responses:

```python
def test_tuning_decision_retries_once_on_unknown_evidence(monkeypatch):
    package = _fact_package()
    bad = _tuning_json(evidence_ids={"lr0": ["invented.fact"]})
    good = _tuning_json(evidence_ids={"lr0": ["training.params.lr0"]})
    replies = iter([bad, good])
    monkeypatch.setattr(decision_agent, "call_decision_llm", lambda *a, **k: next(replies))

    result = decision_agent.decide_hyperparameters(package, _config())

    assert result["error"] is None
    assert result["retried"] is True
    assert result["validation"]["valid"] is True
    assert result["evidence_ids"] == {"lr0": ["training.params.lr0"]}


def test_tuning_decision_second_evidence_failure_is_stable(monkeypatch):
    package = _fact_package()
    bad = _tuning_json(evidence_ids={"lr0": ["invented.fact"]})
    monkeypatch.setattr(decision_agent, "call_decision_llm", lambda *a, **k: bad)

    result = decision_agent.decide_hyperparameters(package, _config())

    assert result["error"] == "DECISION_EVIDENCE_UNKNOWN"
    assert result["retried"] is True
    assert result["validation"]["valid"] is False
```

Also add a spy test proving `generate_suggestion()` still accepts its current four-field schema and does not require `fact_package_id` or `evidence_ids`.

- [ ] **Step 2: 运行新增测试并确认旧实现失败**

Run the named new tests. Expected: fail because `decide_hyperparameters()` still consumes perception and returns no evidence validation fields.

- [ ] **Step 3: 增加事实包提示词**

Implement `build_tuning_decision_prompt()` with:

- the complete FactPackage JSON serialized with `ensure_ascii=False, sort_keys=True`;
- the exact TuningDecision v1 output example;
- the allowed parameter list from `get_tunable_parameter_names()`;
- explicit instruction that no evidence outside `facts` may be invented;
- no free-form perception summary and no hard-coded fact values.

Keep existing `build_decision_prompt()` unchanged for `generate_suggestion()`.

- [ ] **Step 4: 增加自动调优专用执行和纠错函数**

Add:

```python
def _run_tuning_decision_with_retry(
    prompt: str, config: dict, fact_package: dict
) -> tuple[str | None, dict | None, dict]:
    """Return raw, normalized decision, validation metadata."""
```

Validation metadata shape is fixed:

```python
{
    "valid": bool,
    "error_code": str | None,
    "error_detail": str | None,
    "retried": bool,
    "referenced_fact_ids": list[str],
}
```

Algorithm:

1. Call provider once; transport/provider exceptions return the existing safe provider error and `retried=False`.
2. Parse with `parse_tuning_decision_response()` and validate with `validate_decision_evidence()`.
3. On `DecisionContractError`, build a correction prompt containing only the stable error code/detail, same `fact_package_id`, exact schema and original task.
4. Call provider exactly once more and repeat full validation.
5. Do not retry any other exception or a third response.

- [ ] **Step 5: 改造 `decide_hyperparameters()`，保留兼容路径**

Change only the auto-tuning function signature to consume `fact_package`. Return the existing keys plus:

```python
{
    "schema_version": decision.get("schema_version"),
    "fact_package_id": decision.get("fact_package_id"),
    "evidence_ids": decision.get("evidence_ids", {}),
    "validation": validation,
}
```

On contract failure set `error` to the stable error code, not raw LLM content. Preserve `raw_response` only for the existing redacted audit path.

- [ ] **Step 6: 运行 Decision Agent 回归**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_decision_agent.py auto_tune\tests\test_decision_unified.py auto_tune\tests\test_decision_contract.py -v -p no:cacheprovider
```

Expected: all pass; intelligent-analysis contract tests remain unchanged.

- [ ] **Step 7: 报告 Task 3 结果，不提交代码**

Explicitly report provider call counts for first-pass success, corrected success, repeated contract failure and transport failure.

---

### Task 4: 调优循环前置冻结、审计与失败终态

**Files:**
- Modify: `auto_tune/modules/agent_engine/audit.py`
- Modify: `auto_tune/modules/agent_engine/loop.py`
- Modify: `auto_tune/tests/test_audit.py`
- Modify: `auto_tune/tests/test_tuning_loop.py`

**Interfaces:**
- Consumes: Task 1 `build_tuning_fact_package()` and Task 3 `decide_hyperparameters(fact_package, ...)`。
- Produces audit iteration fields: `fact_package: dict | None` and `decision_validation: dict`。
- Preserves existing baseline, Guardrails, execution and result fields.

- [ ] **Step 1: 先写审计 Schema 失败测试**

Add to `test_audit.py`:

```python
def test_audit_iteration_persists_fact_package_before_decision(tmp_path):
    audit = TuningAuditSession("s1", str(tmp_path), "train54", 1)
    audit.start_iteration(1)
    package = {
        "schema_version": "1.0", "fact_package_id": "sha256:abc",
        "task": "detect", "reference_run": "train54", "sources": {}, "facts": [],
    }
    audit.update_iteration(1, fact_package=package)
    audit.update_iteration(1, decision_validation={
        "valid": False, "error_code": "DECISION_EVIDENCE_UNKNOWN",
        "error_detail": "unknown evidence", "retried": True,
        "referenced_fact_ids": [],
    })
    saved = json.loads(Path(audit.path).read_text(encoding="utf-8"))
    assert saved["schema_version"] == "1.1"
    assert saved["iterations"][0]["fact_package"] == package
```

Expected before implementation: `Unknown audit iteration field: fact_package`.

- [ ] **Step 2: 升级审计默认结构**

In `audit.py`:

```python
AUDIT_SCHEMA_VERSION = "1.1"
```

Add to `_new_iteration()`:

```python
"fact_package": None,
"decision_validation": {
    "valid": None,
    "error_code": None,
    "error_detail": None,
    "retried": False,
    "referenced_fact_ids": [],
},
```

Do not migrate or rewrite existing audit files.

- [ ] **Step 3: 写循环失败测试，证明事实错误不会调用 LLM/训练**

Add focused tests using existing loop fixtures and spies:

```python
def test_fact_package_failure_stops_before_llm_and_training(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.build_tuning_fact_package",
        lambda *a, **k: (_ for _ in ()).throw(FactPackageError("reference mismatch")),
    )
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.decide_hyperparameters",
        lambda *a, **k: calls.append("llm"),
    )
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.launch_training",
        lambda *a, **k: calls.append("train"),
    )

    result = _run_loop_fixture(tmp_path, reference_run="train54")

    assert calls == []
    assert result["failure"]["error_type"] == "fact_package_invalid"
```

Add corresponding tests for audit write failure, package mismatch, missing evidence, unknown evidence and second retry failure. Each must assert `launch_training` and `build_yolo_command` call count is zero.

- [ ] **Step 4: 将参考事实读取移到决策前**

In each loop iteration, after `perception_blocking_error()` passes and before any LLM call:

1. Resolve `ref_dir` and `base_args` once.
2. Call `_read_reference_before_metrics(reference_run, detect_dir)` once.
3. Build FactPackage v1 once.
4. Persist `audit.update_iteration(iteration, fact_package=fact_package)`.
5. Only then call `decide_hyperparameters(fact_package, config, prev_changes)`.

Remove the later duplicate reads. Reuse the same `base_args`, `before_metrics` and `metrics_source` when writing `baseline` and merging parameters.

If package construction fails, persist stage `facts`, error type `fact_package_invalid`, abort the session and return before LLM invocation.

If fact-package audit persistence fails, use existing `audit_persistence_error`; do not call LLM.

- [ ] **Step 5: 持久化完整决策合同和校验结果**

Extend the existing decision audit write with:

```python
"schema_version": decision.get("schema_version"),
"fact_package_id": decision.get("fact_package_id"),
"evidence_ids": decision.get("evidence_ids", {}),
```

Then persist `decision_validation=decision["validation"]` before checking `decision["error"]`.

Map stable contract errors without substring guessing:

```python
CONTRACT_FAILURE_TYPES = {
    "DECISION_SCHEMA_INVALID": "decision_schema_invalid",
    "DECISION_FACT_PACKAGE_MISMATCH": "decision_fact_package_mismatch",
    "DECISION_EVIDENCE_MISSING": "decision_evidence_missing",
    "DECISION_EVIDENCE_UNKNOWN": "decision_evidence_unknown",
}
```

Provider errors keep the existing `decision_api_error` classification.

- [ ] **Step 6: 保持 `keep_params` 显式人工路径可用**

The existing user-selected `keep_params=True` path does not call LLM. It must still build and audit the fact package, then create a local TuningDecision-shaped record bound to that package:

```python
{
    "schema_version": "1.0",
    "fact_package_id": fact_package["fact_package_id"],
    "diagnosis": "按原有参数训练，不做超参数调整",
    "action": "keep_params",
    "hyperparameter_changes": {},
    "training_overrides": {},
    "evidence_ids": {},
    "validation": {
        "valid": True, "error_code": None, "error_detail": None,
        "retried": False, "referenced_fact_ids": [],
    },
}
```

This record is code-generated and does not consume an LLM retry.

- [ ] **Step 7: 运行循环与审计测试**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_audit.py auto_tune\tests\test_tuning_loop.py auto_tune\tests\test_decision_unified.py -v -p no:cacheprovider
```

Expected: all pass; every negative decision test proves training launch count is zero.

- [ ] **Step 8: 报告 Task 4 结果，不提交代码**

Include one saved audit fixture with paths and secrets redacted, plus test call-count evidence. Do not include real audit files in Git.

---

### Task 5: 旧审计兼容、集成回归与交付证据

**Files:**
- Modify: `auto_tune/tests/test_experiment_audit_view.py`
- Modify only if tests prove necessary: `auto_tune/modules/presentation/experiment_views.py`
- Test: all files from Tasks 1–4

**Interfaces:**
- Consumes: Audit Schema 1.0 and 1.1 files.
- Produces: unchanged bounded audit view response; no raw LLM response, absolute path, full command or credential reaches the client.

- [ ] **Step 1: 增加 1.0/1.1 双版本兼容测试**

Parameterize existing audit-view fixture:

```python
@pytest.mark.parametrize("schema_version", ["1.0", "1.1"])
def test_audit_view_accepts_old_and_new_schema(schema_version, tmp_path):
    iteration = _iteration()
    if schema_version == "1.1":
        iteration["fact_package"] = {
            "schema_version": "1.0", "fact_package_id": "sha256:abc",
            "task": "detect", "reference_run": "train52", "sources": {}, "facts": [],
        }
        iteration["decision_validation"] = {
            "valid": True, "error_code": None, "error_detail": None,
            "retried": False, "referenced_fact_ids": [],
        }
    payload = _audit_payload("sess1", iterations=[iteration])
    payload["schema_version"] = schema_version
    audit_path = _write(tmp_path / "log" / "tuning_audit_sess1.json", payload)
    view = build_audit_view(
        _experiment(audit_path=audit_path), artifact_roots=_roots(tmp_path)
    )
    assert view["session"]["session_id"] == "sess1"
```

Also extend the leak test so `fact_package.sources` containing a malicious absolute path is not projected to the client.

- [ ] **Step 2: 运行兼容测试，只有失败才最小修改投影**

Run `test_experiment_audit_view.py`. If it passes without product change, leave `experiment_views.py` untouched. If it fails because the parser rejects unknown 1.1 fields, change only the version/optional-field validation needed to accept both versions; do not expose the full fact package or raw evidence details in Q1.1 UI.

- [ ] **Step 3: 运行 Q1.1 聚合定向套件**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_decision_facts.py auto_tune\tests\test_decision_contract.py auto_tune\tests\test_decision_agent.py auto_tune\tests\test_decision_unified.py auto_tune\tests\test_guardrails.py auto_tune\tests\test_tuning_loop.py auto_tune\tests\test_audit.py auto_tune\tests\test_experiment_audit_view.py -q -p no:cacheprovider
```

Expected: zero failures, zero skipped.

- [ ] **Step 4: 运行完整自动化回归**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests -q -p no:cacheprovider
```

Expected: at least the 1226-test baseline plus new Q1.1 tests, zero failures, zero skipped; only the two known sklearn PCA warnings are acceptable.

- [ ] **Step 5: 执行无外部训练副作用的 dry-run 验证**

Using a controlled temporary directory and mocked provider response:

1. Bind a valid dataset/training report and `reference_run`.
2. Return one valid TuningDecision v1 referencing real fact IDs.
3. Run auto-tuning in dry-run mode.
4. Assert audit order contains fact package before decision, `fact_package_id` matches, evidence IDs exist, Guardrails is valid and no YOLO process is launched.
5. Repeat with an invented fact ID and assert terminal failure plus zero training launch.

Do not call a real external LLM and do not write fixtures into project `log/`.

- [ ] **Step 6: Claude Code 交付报告**

Claude Code must provide:

- exact changed-file list;
- test commands and complete pass/fail counts;
- whether `experiment_views.py` required modification;
- any deviation from the approved interfaces;
- known risks, especially evidence existence not equaling semantic correctness;
- confirmation that no dependency, planning document, real config, logs, data, weights or training output was added;
- confirmation that no commit or push was performed.

- [ ] **Step 7: Codex 独立验收（Claude Code 不执行）**

Codex will independently:

1. Review every diff against the design spec and this plan.
2. Re-run the Q1.1 aggregate suite and complete suite in the `auto_tune` Conda environment.
3. Inspect one valid and four invalid audit chains: package mismatch, missing evidence, unknown evidence and second retry failure.
4. Run a controlled dry-run.
5. After confirming a valid dataset snapshot, run one minimal real Detect short-epoch chain with a controlled valid decision.
6. Verify the Studio page and audit detail remain usable and no sensitive data is exposed.
7. Only after explicit acceptance, update authoritative docs and perform a separately approved Git commit/push check.

---

## Completion Definition

Q1.1 is complete only when all conditions hold:

- FactPackage v1 is deterministic, source-bound, free of unknown defaults and written before LLM invocation.
- Every changed parameter has at least one existing fact reference and the response matches the current `fact_package_id`.
- Contract failure is retried at most once; repeated failure and fact-package failure launch zero training processes.
- Valid decisions still pass through existing Guardrails, preflight and one-time command construction.
- Old Schema 1.0 audits remain readable; new 1.1 audits do not leak raw responses, paths, commands or credentials through the view.
- Q1.1 aggregate and complete test suites pass with zero failures and zero skipped.
- No code or product copy claims that evidence existence proves LLM reasoning is correct.
