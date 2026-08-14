# A0 Audit Closure and Failure Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为自动调优闭环增加独立、原子持久化的审计记录，并保证决策、护栏、预检或审计失败时绝不启动 YOLO 训练。

**Architecture:** 新增无业务副作用的 `audit.py` 负责 Schema、脱敏和原子 JSON 写入；`executor.py` 提供可测试的训练预检，并允许 Loop 将已经审计过的准确命令交给启动函数；`loop.py` 负责阶段状态、fatal 失败控制和兼容返回结构。现有 `tuning_history.json` 保留，但改用同一原子写入函数。

**Tech Stack:** Python 3.10、pytest、FastAPI 项目现有模块、标准库 `json/os/tempfile/pathlib/datetime`、PyYAML（项目现有依赖）。

## Global Constraints

- 本批只处理调优审计闭环、执行预检和失败策略。
- 不新增第三方依赖，不引入 SQLite，不重构 UI，不实现数据快照或密钥迁移。
- 保留 `run_tuning_loop()` 现有顶层字段与 `iterations` 结构。
- `tuning_history.json` 必须保持可被现有 `TuningHistory.from_json()` 和 UI 读取。
- 命令必须以 `list[str]` 保存和执行，不生成可直接执行的 shell 字符串。
- 决策、Guardrails、预检、命令构造或审计持久化失败均为 fatal，不进入下一轮，不调用训练启动函数。
- 探针返回 `RETRY` 时维持现有下一轮重试行为。
- 任何代码改动先写失败测试，再写最小实现。
- Claude Code 不得自行初始化仓库、提交或推送；每个任务结束只输出改动文件和测试结果。Codex 独立审查并验收通过后，再按项目根目录 `AGENTS.md` 的规则提交和推送到艾卡指定的 GitHub 私有仓库。

---

## File Map

- Create: `auto_tune/modules/agent_engine/audit.py` — 审计 Schema、递归脱敏、原子写入和会话状态。
- Create: `auto_tune/tests/test_audit.py` — 审计模块单元测试。
- Modify: `auto_tune/modules/agent_engine/executor.py` — 训练预检、命令复用和启动契约。
- Modify: `auto_tune/tests/test_executor.py` — 预检与命令一致性测试。
- Modify: `auto_tune/modules/agent_engine/loop.py` — 审计接入、fatal 控制、结构化失败和兼容历史写入。
- Modify: `auto_tune/tests/test_tuning_loop.py` — Loop 失败阻断和审计集成测试。
- Modify: `docs/implementation_plan_20260814.md` — 完成后只更新本批实际完成状态和测试证据。
- Modify: `docs/roadmap_20260814.md` — 完成后只更新 A0 审计闭环状态。

---

### Task 1: 原子 JSON 写入与敏感字段脱敏

**Files:**
- Create: `auto_tune/modules/agent_engine/audit.py`
- Create: `auto_tune/tests/test_audit.py`

**Interfaces:**
- Produces: `utc_now_iso() -> str`
- Produces: `redact_sensitive(value: object) -> object`
- Produces: `atomic_write_json(path: str | os.PathLike, payload: object) -> None`
- Consumes: only Python standard library

- [ ] **Step 1: Write failing redaction tests**

Create `auto_tune/tests/test_audit.py` with:

```python
import json
from pathlib import Path

import pytest

from auto_tune.modules.agent_engine.audit import atomic_write_json, redact_sensitive


def test_redact_sensitive_recurses_without_mutating_input():
    source = {
        "llm": {
            "api_key": "secret-key",
            "headers": {"Authorization": "Bearer abc"},
        },
        "items": [
            {"access_token": "token-value", "model": "deepseek"},
            ("plain", {"PASSWORD": "p@ss"}),
        ],
    }

    redacted = redact_sensitive(source)

    assert redacted["llm"]["api_key"] == "***REDACTED***"
    assert redacted["llm"]["headers"]["Authorization"] == "***REDACTED***"
    assert redacted["items"][0]["access_token"] == "***REDACTED***"
    assert redacted["items"][0]["model"] == "deepseek"
    assert redacted["items"][1][1]["PASSWORD"] == "***REDACTED***"
    assert source["llm"]["api_key"] == "secret-key"


def test_redact_sensitive_masks_partial_case_insensitive_key_names():
    assert redact_sensitive({"DeepSeekApiKey": "x"}) == {
        "DeepSeekApiKey": "***REDACTED***"
    }
    assert redact_sensitive({"client_secret_value": "x"}) == {
        "client_secret_value": "***REDACTED***"
    }
```

- [ ] **Step 2: Run redaction tests and verify import failure**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_audit.py -v -p no:cacheprovider
```

Expected: collection fails because `auto_tune.modules.agent_engine.audit` does not exist.

- [ ] **Step 3: Implement minimal redaction and timestamp helpers**

Create `audit.py` with these exact constants and semantics:

```python
"""Durable audit records for Module C tuning sessions."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

AUDIT_SCHEMA_VERSION = "1.0"
REDACTED = "***REDACTED***"
SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_sensitive_key(key: object) -> bool:
    normalized = str(key).lower()
    return any(part in normalized for part in SENSITIVE_KEY_PARTS)


def redact_sensitive(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: REDACTED if _is_sensitive_key(key) else redact_sensitive(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive(item) for item in value)
    return value
```

- [ ] **Step 4: Run redaction tests and verify pass**

Run the Task 1 test command. Expected: the two redaction tests pass; atomic writer import exists but is not yet exercised.

- [ ] **Step 5: Add failing atomic-write tests**

Append:

```python
def test_atomic_write_json_replaces_existing_file(tmp_path):
    target = tmp_path / "audit.json"
    target.write_text('{"old": true}', encoding="utf-8")

    atomic_write_json(target, {"new": True, "text": "中文"})

    assert json.loads(target.read_text(encoding="utf-8")) == {
        "new": True,
        "text": "中文",
    }
    assert list(tmp_path.glob(".audit.json.*.tmp")) == []


def test_atomic_write_json_preserves_old_file_when_replace_fails(tmp_path, monkeypatch):
    target = tmp_path / "audit.json"
    target.write_text('{"stable": true}', encoding="utf-8")

    def fail_replace(src, dst):
        raise OSError("replace failed")

    monkeypatch.setattr("auto_tune.modules.agent_engine.audit.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        atomic_write_json(target, {"stable": False})

    assert json.loads(target.read_text(encoding="utf-8")) == {"stable": True}
    assert list(tmp_path.glob(".audit.json.*.tmp")) == []
```

- [ ] **Step 6: Run tests and verify `atomic_write_json` failure**

Expected: tests fail because the function is missing or incomplete.

- [ ] **Step 7: Implement atomic JSON replacement**

Add:

```python
def atomic_write_json(path: str | os.PathLike, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(
                redact_sensitive(payload),
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target)
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
```

- [ ] **Step 8: Run Task 1 tests**

Expected: all four tests pass.

- [ ] **Step 9: Record task checkpoint**

Report modified files and exact pytest summary. Do not initialize Git or push GitHub.

---

### Task 2: 审计会话状态机

**Files:**
- Modify: `auto_tune/modules/agent_engine/audit.py`
- Modify: `auto_tune/tests/test_audit.py`

**Interfaces:**
- Consumes: `atomic_write_json`, `utc_now_iso`, `AUDIT_SCHEMA_VERSION`
- Produces: `TuningAuditSession` methods defined in the approved design

- [ ] **Step 1: Write failing session lifecycle test**

Append:

```python
from auto_tune.modules.agent_engine.audit import TuningAuditSession


def test_audit_session_persists_iteration_lifecycle(tmp_path):
    session = TuningAuditSession("session-1", str(tmp_path), "train38", 2)
    session.flush()
    session.start_iteration(1)
    session.update_iteration(
        1,
        baseline={"reference_run": "train38", "params": {"lr0": 0.01}, "metrics": {}},
        decision={"raw_response": "{}", "action": "keep_params"},
    )
    session.complete_iteration(1)
    session.finalize("completed")

    saved = json.loads(Path(session.path).read_text(encoding="utf-8"))
    assert saved["schema_version"] == "1.0"
    assert saved["session_id"] == "session-1"
    assert saved["status"] == "completed"
    assert saved["finished_at"].endswith("Z")
    assert saved["iterations"][0]["iteration"] == 1
    assert saved["iterations"][0]["status"] == "completed"
    assert saved["iterations"][0]["baseline"]["params"] == {"lr0": 0.01}


def test_audit_session_records_structured_failure(tmp_path):
    session = TuningAuditSession("session-2", str(tmp_path), None, 1)
    session.start_iteration(1)
    session.fail_iteration(
        1,
        stage="decision",
        error_type="decision_schema_error",
        message="invalid JSON",
        fatal=True,
    )
    session.finalize("failed", session.to_dict()["iterations"][0]["error"])

    saved = json.loads(Path(session.path).read_text(encoding="utf-8"))
    error = saved["iterations"][0]["error"]
    assert error["stage"] == "decision"
    assert error["error_type"] == "decision_schema_error"
    assert error["fatal"] is True
    assert saved["error"] == error
```

- [ ] **Step 2: Run tests and verify missing class failure**

Use the Task 1 pytest command. Expected: new tests fail because `TuningAuditSession` is missing.

- [ ] **Step 3: Implement session defaults and iteration lookup**

Add a class whose `data` begins with:

```python
{
    "schema_version": AUDIT_SCHEMA_VERSION,
    "session_id": session_id,
    "status": "running",
    "started_at": utc_now_iso(),
    "finished_at": None,
    "reference_run": reference_run,
    "max_retries": max_retries,
    "iterations": [],
    "error": None,
}
```

`path` must be `str(Path(log_dir) / f"tuning_audit_{session_id}.json")`. `start_iteration()` must reject duplicate iteration numbers with `ValueError`. Each new iteration must contain every field from section 5 of the approved design with empty defaults.

- [ ] **Step 4: Implement lifecycle methods**

Rules:

```python
def update_iteration(self, iteration: int, **fields: object) -> None:
    record = self._get_iteration(iteration)
    for key, value in fields.items():
        if key not in record:
            raise KeyError(f"Unknown audit iteration field: {key}")
        record[key] = value
    self.flush()
```

`fail_iteration()` sets iteration status to `failed`, `finished_at`, and the exact structured error. `complete_iteration()` sets `completed` and `finished_at`. `finalize()` only accepts `completed`, `failed`, or `cancelled`, sets top-level status/error/finished_at, and flushes. `to_dict()` returns `copy.deepcopy(self.data)` so callers cannot mutate internal state.

- [ ] **Step 5: Run audit tests**

Expected: all Task 1 and Task 2 tests pass.

- [ ] **Step 6: Add invariant tests**

Test duplicate iterations, unknown update fields and invalid final status. Expected exception types: `ValueError`, `KeyError`, `ValueError` respectively.

- [ ] **Step 7: Run audit tests again and record checkpoint**

Report exact pytest summary.

---

### Task 3: 执行预检与准确命令复用

**Files:**
- Modify: `auto_tune/modules/agent_engine/executor.py`
- Modify: `auto_tune/tests/test_executor.py`

**Interfaces:**
- Produces: `validate_training_preflight(reference_run, reference_dir, merged_params) -> list[str]`
- Modifies: `launch_training(..., command: list[str] | None = None) -> subprocess.Popen`
- Preserves: `build_yolo_command(...) -> list[str]`

- [ ] **Step 1: Write failing preflight tests**

Append to `test_executor.py`:

```python
from pathlib import Path

from auto_tune.modules.agent_engine.executor import validate_training_preflight


def test_preflight_rejects_missing_reference_directory(tmp_path):
    errors = validate_training_preflight(
        reference_run="train38",
        reference_dir=str(tmp_path / "missing"),
        merged_params={"model": "yolov8n.pt", "data": str(tmp_path / "data.yaml")},
    )
    assert any("reference" in error.lower() for error in errors)


def test_preflight_rejects_missing_local_data_yaml(tmp_path):
    errors = validate_training_preflight(
        reference_run=None,
        reference_dir=None,
        merged_params={"model": "yolov8n.pt", "data": str(tmp_path / "missing.yaml")},
    )
    assert any("data" in error.lower() for error in errors)


def test_preflight_accepts_existing_yaml_and_builtin_weight_name(tmp_path, monkeypatch):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("path: .\ntrain: images/train\nval: images/val\n", encoding="utf-8")
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.resolve_yolo_executable",
        lambda: "yolo",
    )

    errors = validate_training_preflight(
        reference_run=None,
        reference_dir=None,
        merged_params={"model": "yolov8n.pt", "data": str(data_yaml)},
    )
    assert errors == []
```

- [ ] **Step 2: Run targeted tests and verify missing function failure**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_executor.py -v -p no:cacheprovider
```

- [ ] **Step 3: Implement deterministic preflight**

Implementation rules:

- If `reference_run` is truthy, `reference_dir` must exist as a directory and contain readable `args.yaml`.
- `model` and `data` must be non-empty strings.
- Treat a model value as a local path only if it is absolute or contains `/` or `\`; local model paths must exist. A bare value such as `yolov8n.pt` is allowed.
- `data` must resolve to an existing readable `.yaml` or `.yml` file.
- Call `resolve_yolo_executable()` inside `try/except`; append `YOLO executable unavailable: <message>` on failure.
- Return all errors in deterministic check order, never raise for validation failures.

- [ ] **Step 4: Run preflight tests and verify pass**

- [ ] **Step 5: Write failing exact-command reuse test**

Add a test monkeypatching `subprocess.Popen` and `build_yolo_command`:

```python
def test_launch_training_uses_supplied_command_without_rebuilding(tmp_path, monkeypatch):
    args_path = tmp_path / "args.yaml"
    args_path.write_text("epochs: 1\n", encoding="utf-8")
    supplied = ["yolo", "train", "epochs=1"]
    captured = {}

    class FakeProcess:
        pass

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return FakeProcess()

    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.executor.build_yolo_command",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not rebuild")),
    )
    monkeypatch.setattr("auto_tune.modules.agent_engine.executor.subprocess.Popen", fake_popen)

    proc = launch_training("train1", str(args_path), {"epochs": 1}, command=supplied)

    assert isinstance(proc, FakeProcess)
    assert captured["command"] == supplied
```

- [ ] **Step 6: Change `launch_training` signature and behavior**

Use:

```python
cmd = list(command) if command is not None else build_yolo_command(
    train_name, args_path, merged_params
)
```

The copied list must be used both in the log line and `subprocess.Popen`. Do not mutate it.

- [ ] **Step 7: Run executor tests and record checkpoint**

Expected: all executor tests pass.

---

### Task 4: Loop 接入审计与 fatal 失败策略

**Files:**
- Modify: `auto_tune/modules/agent_engine/loop.py`
- Modify: `auto_tune/tests/test_tuning_loop.py`

**Interfaces:**
- Consumes: `TuningAuditSession`, `atomic_write_json`, `validate_training_preflight`, `build_yolo_command`
- Produces: `run_tuning_loop()` result adds `session_id`, `audit_path`, `failure`
- Preserves: existing result keys and iteration summary

- [ ] **Step 1: Extract a reusable structured failure helper with tests**

Add a private helper to `loop.py`:

```python
def _failure(stage: str, error_type: str, message: str, fatal: bool = True) -> dict:
    return {
        "stage": stage,
        "error_type": error_type,
        "message": message,
        "fatal": fatal,
    }
```

Test exact dict equality. Timestamp belongs to the audit record, not this compatibility return helper.

- [ ] **Step 2: Add failing decision-error integration test**

Use a minimal config, `tmp_path`, and monkeypatch:

```python
def test_decision_error_is_fatal_and_never_launches_training(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.build_perception",
        lambda **kwargs: {"dataset": {"total_images": 10}},
    )
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.decide_hyperparameters",
        lambda *args, **kwargs: {
            "diagnosis": None,
            "action": None,
            "hyperparameter_changes": {},
            "training_overrides": {},
            "raw_response": "not json",
            "error": "Failed to parse JSON from LLM response",
        },
    )
    launched = []
    monkeypatch.setattr(
        "auto_tune.modules.agent_engine.loop.launch_training",
        lambda *args, **kwargs: launched.append(True),
    )

    result = run_tuning_loop(
        {"probe": {"max_retries": 3}},
        reference_run=None,
        log_dir=str(tmp_path),
    )

    assert launched == []
    assert len(result["iterations"]) == 1
    assert result["failure"]["stage"] == "decision"
    assert result["failure"]["error_type"] == "decision_schema_error"
    assert result["failure"]["fatal"] is True
    audit = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    assert audit["status"] == "failed"
    assert audit["iterations"][0]["decision"]["raw_response"] == "not json"
```

Import `json`, `Path`, and `run_tuning_loop` in the test file.

- [ ] **Step 3: Initialize audit session and compatible result fields**

Immediately after creating `tuning_session_id`:

```python
audit = TuningAuditSession(
    tuning_session_id,
    log_dir,
    reference_run,
    max_retries,
)
audit.flush()
```

Add to `tuning_result`:

```python
"session_id": tuning_session_id,
"audit_path": audit.path,
"failure": None,
```

If initial `audit.flush()` fails, return a result with `failure.error_type == "audit_persistence_error"`; do not enter the iteration loop.

- [ ] **Step 4: Record perception and decision facts**

At the start of each loop call `audit.start_iteration(iteration)`. After perception, update baseline metrics and reference. After decision, store these exact fields:

```python
{
    "raw_response": decision.get("raw_response"),
    "diagnosis": decision.get("diagnosis"),
    "action": decision.get("action"),
    "hyperparameter_changes": decision.get("hyperparameter_changes", {}),
    "training_overrides": decision.get("training_overrides", {}),
}
```

Do not include full application config.

- [ ] **Step 5: Replace decision `continue` with fatal return**

When `decision.get("error")` is truthy:

- classify messages starting with `DeepSeek API error`, request exceptions, timeouts, or connection errors as `decision_api_error`;
- classify all remaining decision errors as `decision_schema_error`;
- append one compatible iteration summary;
- call `audit.fail_iteration(...)` and `audit.finalize("failed", error)`;
- set top-level `failure` and human-readable `error`;
- atomically write `tuning_history.json`;
- return immediately.

- [ ] **Step 6: Add failing guardrail fatal test**

Monkeypatch `sanitize_and_merge_tuning_params` to return `(None, fake_guard)` where `valid=False`. Assert one iteration, no launch, `guardrail_rejected`, and stored guardrail errors.

- [ ] **Step 7: Record Guardrails comparison and return on rejection**

Audit `guardrails` must contain:

```python
{
    "valid": guard_result.valid,
    "warnings": list(guard_result.warnings),
    "errors": list(guard_result.errors),
    "clamped": dict(guard_result.clamped),
    "sanitized_changes": dict(guard_result.params),
}
```

Guardrail rejection is fatal and returns immediately.

- [ ] **Step 8: Add failing preflight and audit-persistence tests**

Preflight test: monkeypatch `validate_training_preflight` to return `["data yaml missing"]`; assert `preflight_error` and no output directory/launch.

Audit failure test: allow decision and Guardrails to pass, then monkeypatch `audit.atomic_write_json` or the session instance `flush` to raise `OSError("disk full")` before command launch. Assert `audit_persistence_error` and no launch.

- [ ] **Step 9: Integrate preflight before output directory creation**

After merged params are available and before `os.makedirs(output_dir, exist_ok=False)`:

```python
reference_dir = os.path.join(detect_dir, reference_run) if reference_run else None
preflight_errors = validate_training_preflight(
    reference_run,
    reference_dir,
    merged,
)
```

On errors, record `preflight_error` and return.

- [ ] **Step 10: Build, audit, flush, then launch the exact command**

The order must be:

1. create fresh output directory;
2. write `args.yaml`;
3. build command exactly once with `build_yolo_command`;
4. update audit `execution.actual_params`, `args_yaml_path`, `command`, `train_name`;
5. successfully flush audit;
6. call `launch_training(..., command=command)`.

If command construction fails, record `command_build_error`. If audit flush fails, remove no files in this batch, but return `audit_persistence_error` and do not launch. If `launch_training` raises, record `training_launch_error` and return.

- [ ] **Step 11: Make history writes atomic**

Replace `TuningHistory.to_json()` implementation with:

```python
atomic_write_json(path, self.attempts)
```

Do not change `from_json()` or the serialized list structure.

- [ ] **Step 12: Run Loop tests**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_tuning_loop.py auto_tune\tests\test_audit.py auto_tune\tests\test_executor.py -v -p no:cacheprovider
```

Expected: all tests pass; no real training starts.

---

### Task 5: 完成结果、指标差值与取消路径审计

**Files:**
- Modify: `auto_tune/modules/agent_engine/loop.py`
- Modify: `auto_tune/tests/test_tuning_loop.py`

**Interfaces:**
- Consumes: existing `TuningResult` metric fields and `ProbeDecision`
- Produces: completed/cancelled audit statuses and `result.metric_delta`

- [ ] **Step 1: Add a pure metric-delta helper and failing tests**

Add:

```python
def _metric_delta(before: dict, after: dict) -> dict:
    delta = {}
    for key, after_value in after.items():
        before_value = before.get(key)
        if before_value is None or after_value is None:
            continue
        delta[key] = round(float(after_value) - float(before_value), 10)
    return delta
```

Test normal values, missing values and zero values. Zero must not be treated as missing.

- [ ] **Step 2: Record result facts at each terminal iteration path**

For completed, abort and retry paths, update audit result with:

```python
{
    "before_metrics": before_metrics,
    "after_metrics": after_metrics,
    "metric_delta": _metric_delta(before_metrics, after_metrics),
    "probe": {
        "verdict": probe_decision.verdict,
        "reason": probe_decision.reason,
        "suggestion": probe_decision.suggestion,
    },
    "analysis": full_diagnosis_or_none,
}
```

Use only metrics actually present; do not synthesize missing metrics as zero.

- [ ] **Step 3: Preserve probe retry semantics**

For `ProbeDecision.RETRY`, record iteration failure with:

```text
stage=probe
error_type=probe_retry
fatal=false
```

Then continue to the next iteration. For `ABORT`, use `probe_abort`; preserve current product behavior about whether the next iteration runs, but make `fatal` match that control flow.

- [ ] **Step 4: Audit cancellation**

When cancellation is detected before an iteration, finalize session as `cancelled`. When detected during training, terminate the process, fail the current iteration with `stage=execute`, `error_type=user_cancelled`, `fatal=true`, finalize `cancelled`, atomically persist history, and return.

- [ ] **Step 5: Finalize successful and exhausted sessions**

- If the loop returns a valid final result, call `audit.finalize("completed")`.
- If retries are exhausted without a valid final result, finalize `failed` with `error_type=retries_exhausted`, without changing prior per-iteration errors.
- Every return path after session creation must leave audit status different from `running`.

- [ ] **Step 6: Add terminal-state invariant test**

Parametrize mocked decision failure, guardrail failure, preflight failure, cancellation and dry-run success. Read each audit file and assert `status in {"completed", "failed", "cancelled"}`.

- [ ] **Step 7: Run targeted tests and record checkpoint**

Use Task 4 test command. Expected: all pass.

---

### Task 6: 回归验证、文档回写与交付给 Codex 测试

**Files:**
- Modify: `docs/implementation_plan_20260814.md`
- Modify: `docs/roadmap_20260814.md`

**Interfaces:**
- Produces: implementation evidence for subsequent independent Codex verification

- [ ] **Step 1: Run the complete automated suite**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests -q -p no:cacheprovider
```

Expected: existing 81 tests plus all new tests pass. Warnings may remain only if they already existed; report their exact text and count.

- [ ] **Step 2: Run an explicit negative-contract test group**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_audit.py auto_tune\tests\test_tuning_loop.py auto_tune\tests\test_executor.py -v -p no:cacheprovider
```

Confirm test names visibly cover decision failure, Guardrails rejection, preflight failure, audit write failure and exact command reuse.

- [ ] **Step 3: Inspect produced audit fixtures**

Use a temporary test directory or a dry-run invocation. Verify:

- valid JSON;
- top-level terminal status;
- no unredacted keys matching `api_key|apikey|authorization|token|secret|password|passwd|credential`;
- `execution.command` is a JSON array;
- `failure` and iteration `error` carry the agreed stage/type/fatal fields.

Do not use a real API key in this check.

- [ ] **Step 4: Update planning documents with facts only**

In both roadmap documents, mark only these completed items if tests prove them:

- independent tuning audit record;
- atomic audit/history writes;
- fatal decision/guardrail/preflight/audit failure policy;
- exact command audit before launch;
- new test count and full-suite result.

Do not mark dataset snapshots, secret migration, unified normal-training history or UI audit comparison complete.

- [ ] **Step 5: Prepare handoff report**

Return:

1. changed file list;
2. summary of each behavior change;
3. exact test commands and results;
4. any deviations from this plan with reasons;
5. remaining risks;
6. one example audit JSON path generated during testing.

Do not perform a real LLM call or real YOLO training. Codex will independently review the code and decide whether a one-epoch smoke training is necessary.

---

## Plan Self-Review

- **Spec coverage:** covers independent audit Schema, atomic writes, sensitive redaction, exact command reuse, structured failure, fatal pre-launch failures, preflight, metric delta, cancellation and compatibility.
- **Scope control:** explicitly excludes snapshotting, secret migration, database, UI redesign and normal-training history.
- **Type consistency:** `TuningAuditSession`, `validate_training_preflight`, `launch_training(command=...)`, `failure` and error field names match the approved design.
- **No dependency expansion:** implementation uses only standard library and existing project dependencies.
