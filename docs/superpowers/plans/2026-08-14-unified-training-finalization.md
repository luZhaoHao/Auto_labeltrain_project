# Unified Training Finalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让普通训练与自动调优训练共用 Module B 收尾、KPI 提取和统一实验历史，并在同一 UI 中自然展示。

**Architecture:** 保留普通训练与自动调优各自的启动和控制流程，在训练结束后汇入 `finalize_training_run()`。统一历史由独立存储类原子、幂等维护；`tuning_history.json` 继续只服务 LLM 反馈，UI 改读统一实验历史。

**Tech Stack:** Python 3.10、FastAPI、Jinja2、原生 JavaScript、PyYAML、pytest；不新增第三方依赖。

## Global Constraints

- 不建设数据库、用户体系、任务队列或 GPU 调度。
- 不重构自动调优的 LLM、Guardrails、探针、重试和审计状态机。
- 不实现 SSE 断线恢复或服务器重启后的任务接管。
- 不改变导航结构、视觉主题或整体页面布局。
- 缺失指标保持缺失，真实 `0` 必须保留。
- 训练状态与分析状态必须分离；分析失败不得覆盖训练成功事实。
- JSON 报告与历史必须原子写入；相同 `run_id` 必须幂等更新。
- 不删除或覆盖 `tuning_history.json`。
- 不新增依赖、不删除文件、不执行真实 LLM 请求。
- 项目暂不使用本地 Git 提交；每个 Task 用测试结果作为检查点，全部验收后由 Codex 首次上传 GitHub。

---

## File Map

- Create `auto_tune/modules/train_analyzer/experiment_history.py`：统一历史 Schema、原子读写、幂等 upsert、旧调优历史兼容。
- Create `auto_tune/modules/train_analyzer/training_finalizer.py`：统一 Module B 分析、KPI、状态和报告持久化。
- Create `auto_tune/tests/test_experiment_history.py`：历史存储与兼容测试。
- Create `auto_tune/tests/test_training_finalizer.py`：统一收尾服务测试。
- Modify `auto_tune/ui/app.py`：普通训练接入、页面数据加载、SSE 统一结果。
- Modify `auto_tune/modules/agent_engine/loop.py`：自动调优复用统一收尾结果。
- Modify `auto_tune/ui/components/tuning_panel.py`：保留内部调优历史读取；不得混入普通训练。
- Create `auto_tune/ui/components/experiment_panel.py`：UI 统一实验历史读取与兼容视图。
- Modify `auto_tune/ui/templates/single_page.html`：完成结果、统一历史、筛选、条件详情。
- Modify `auto_tune/ui/i18n.py`：新增来源、分析状态和错误文案。
- Modify `auto_tune/tests/test_ui_training_results.py`：SSE 与 UI 数据契约测试。
- Modify `auto_tune/tests/test_tuning_loop.py`：调优收尾复用及审计一致性测试。
- Modify `docs/implementation_plan_20260814.md`、`docs/roadmap_20260814.md`：只在验收后回写实际证据。

---

### Task 1: Unified Experiment History Store

**Files:**
- Create: `auto_tune/modules/train_analyzer/experiment_history.py`
- Create: `auto_tune/tests/test_experiment_history.py`
- Reuse: `auto_tune/modules/agent_engine/audit.py` 中的 `atomic_write_json`

**Interfaces:**
- Produces: `ExperimentHistoryStore(path: str, legacy_tuning_path: str | None = None)`
- Produces: `load() -> dict`
- Produces: `list_experiments(include_legacy: bool = True) -> list[dict]`
- Produces: `upsert(record: dict) -> dict`
- Produces: `make_run_id(source: str, run_name: str, session_id: str | None = None) -> str`

- [ ] **Step 1: Write failing Schema and stable-ID tests**

```python
def test_make_run_id_is_stable():
    assert make_run_id("manual", "train39") == "manual:train39"
    assert make_run_id("tuning", "autotune_1", "s1") == "tuning:s1:autotune_1"

def test_upsert_creates_versioned_history(tmp_path):
    store = ExperimentHistoryStore(str(tmp_path / "experiment_history.json"))
    record = {"run_id": "manual:train39", "run_name": "train39", "source": "manual"}
    store.upsert(record)
    saved = json.loads((tmp_path / "experiment_history.json").read_text("utf-8"))
    assert saved["schema_version"] == "1.0"
    assert saved["experiments"] == [record]
```

- [ ] **Step 2: Run the new tests and verify they fail because the module does not exist**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_experiment_history.py -v -p no:cacheprovider
```

Expected: collection fails with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the minimal versioned store and stable IDs**

Use this top-level structure exactly:

```python
EMPTY_HISTORY = {"schema_version": "1.0", "experiments": []}

def make_run_id(source, run_name, session_id=None):
    if source == "manual":
        return f"manual:{run_name}"
    if source == "tuning" and session_id:
        return f"tuning:{session_id}:{run_name}"
    raise ValueError("tuning source requires session_id")
```

`upsert()` must validate `run_id` and `source`, replace an existing matching `run_id`, sort by `finished_at` with missing values last, call `atomic_write_json`, and return a deep copy.

- [ ] **Step 4: Add idempotency and atomic-failure tests**

```python
def test_upsert_replaces_same_run_id(tmp_path):
    store = ExperimentHistoryStore(str(tmp_path / "history.json"))
    store.upsert({"run_id": "manual:train1", "source": "manual", "status": "running"})
    store.upsert({"run_id": "manual:train1", "source": "manual", "status": "completed"})
    assert len(store.list_experiments(False)) == 1
    assert store.list_experiments(False)[0]["status"] == "completed"

def test_atomic_failure_keeps_old_history(tmp_path, monkeypatch):
    path = tmp_path / "history.json"
    store = ExperimentHistoryStore(str(path))
    store.upsert({"run_id": "manual:train1", "source": "manual"})
    old = path.read_text("utf-8")
    monkeypatch.setattr("auto_tune.modules.train_analyzer.experiment_history.atomic_write_json", lambda *a: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        store.upsert({"run_id": "manual:train2", "source": "manual"})
    assert path.read_text("utf-8") == old
```

- [ ] **Step 5: Add legacy tuning compatibility without mutating the legacy file**

Map only records that have enough identity. A legacy record without `train_name` must receive deterministic ID `legacy-tuning:<index>:<timestamp>` and keep missing fields missing. New records win when their `run_name` matches a legacy record.

Test that `list_experiments(True)` includes legacy records once, `list_experiments(False)` excludes them, and the legacy file bytes are unchanged.

- [ ] **Step 6: Run Task 1 tests**

Expected: all `test_experiment_history.py` tests pass.

---

### Task 2: Shared Training Finalizer

**Files:**
- Create: `auto_tune/modules/train_analyzer/training_finalizer.py`
- Create: `auto_tune/tests/test_training_finalizer.py`
- Reuse: `auto_tune/modules/train_analyzer/analyzer.py`
- Reuse: `auto_tune/modules/agent_engine/audit.py`

**Interfaces:**
- Consumes: `ExperimentHistoryStore.upsert(record)` and `make_run_id(...)`
- Produces:

```python
def finalize_training_run(
    run_dir: str,
    run_name: str,
    source: str,
    config: dict,
    log_dir: str = "log",
    training_status: str = "completed",
    session_id: str | None = None,
    audit_path: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
) -> dict:
    ...
```

- [ ] **Step 1: Write a failing successful-finalization test**

Build a temporary `detect/train1/args.yaml` and `results.csv`, disable LLM and vision, then assert:

```python
result = finalize_training_run(...)
assert result["status"] == "completed"
assert result["analysis_status"] == "completed"
assert result["metrics"] == {
    "mAP50": 0.2,
    "mAP50_95": 0.1,
    "precision": 0.3,
    "recall": 0.4,
}
assert Path(result["artifacts"]["report_path"]).exists()
assert len(history["experiments"]) == 1
```

- [ ] **Step 2: Run and verify failure due to missing implementation**

Run the single test with the project Python. Expected: import or attribute failure.

- [ ] **Step 3: Implement completed-run analysis**

Call `analyze_training_results(os.path.dirname(run_dir), config, run_name=run_name, enable_llm=False, enable_vision=False)`. Persist the returned report as `log/<run_name>_report.json` with `atomic_write_json`. Extract final metrics from `report["runs"][run_name]["results"]["final_metrics"]` using canonical names.

The return record must contain exactly the fields defined in the design: identity, source, two statuses, timestamps, params, metrics, epochs, artifacts, `error`, and `analysis_error`.

- [ ] **Step 4: Write and implement status-separation tests**

Cover:

```python
assert finalize_completed_with_bad_csv()["status"] == "completed"
assert finalize_completed_with_bad_csv()["analysis_status"] == "failed"
assert finalize_failed_training()["analysis_status"] == "skipped"
assert finalize_cancelled_training()["status"] == "cancelled"
```

Errors must be objects with `stage`, `error_type`, `message`, `timestamp`.

- [ ] **Step 5: Write and implement zero/missing metric tests**

Assert a CSV value `0.0` survives and a missing metric column is absent from `metrics`.

- [ ] **Step 6: Write and implement history-persistence failure behavior**

If report analysis succeeds but `ExperimentHistoryStore.upsert()` raises, return the truthful training and analysis statuses plus `history_error` with `error_type=history_persistence_error`; do not report full finalization success and do not damage the old history.

- [ ] **Step 7: Run finalizer and analyzer regression tests**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_training_finalizer.py auto_tune\tests\test_train_analyzer.py auto_tune\tests\test_results_parser.py -q -p no:cacheprovider
```

Expected: all pass.

---

### Task 3: Integrate Manual Training SSE

**Files:**
- Modify: `auto_tune/ui/app.py` around `_build_training_result_payload()` and `/api/training/start`
- Modify: `auto_tune/tests/test_ui_training_results.py`

**Interfaces:**
- Consumes: `finalize_training_run(...) -> dict`
- Produces: SSE `done`, `error`, or `cancelled` event with `result` using the unified structure.

- [ ] **Step 1: Write failing SSE completion-contract tests**

Mock the subprocess and finalizer. Assert a zero return code calls finalizer once with `source="manual"`, `training_status="completed"`, and the emitted `done` event contains the finalizer result unchanged.

- [ ] **Step 2: Write failing failure-contract tests**

For non-zero return code, assert the finalizer is called with `training_status="failed"`; the emitted event is `error` and includes the persisted unified record.

- [ ] **Step 3: Replace the small payload helper with the finalizer call**

Keep command construction and SSE streaming behavior unchanged. Capture one `started_at` before process launch and one `finished_at` after exit. Do not call Module B twice.

- [ ] **Step 4: Implement partial-success presentation contract**

When result has `status=completed` and `analysis_status=failed`, emit `status="done"`, `level="warning"`, and message `训练完成，结果分析失败`; never emit training `error` for this case.

- [ ] **Step 5: Run UI training tests**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_ui_training_results.py -v -p no:cacheprovider
```

Expected: all pass.

---

### Task 4: Integrate Auto-Tuning Without Weakening A0 Audit

**Files:**
- Modify: `auto_tune/modules/agent_engine/loop.py` around the successful full-training Module B block
- Modify: `auto_tune/tests/test_tuning_loop.py`

**Interfaces:**
- Consumes: `finalize_training_run(... source="tuning", session_id=..., audit_path=...)`
- Preserves: audit `before_metrics`, `after_metrics`, `metric_delta`, terminal status and fatal policies.

- [ ] **Step 1: Write a failing shared-finalizer integration test**

Mock `finalize_training_run` to return canonical metrics. Assert it is called once after full training with the current `train_name`, `session_id`, and audit path.

- [ ] **Step 2: Write a failing audit-consistency test**

Assert finalizer metrics are copied to `iter_result`, audit `after_metrics`, and `_metric_delta(before, after)` without renaming or zero loss.

- [ ] **Step 3: Replace duplicated Module B assembly in the successful full-training branch**

Remove only the duplicated report construction that the finalizer now owns. Do not change decision, guardrail, command, preflight, probe, cancellation, retry, or audit-finalization order.

- [ ] **Step 4: Define finalizer failure behavior**

- Training completed but analysis failed: audit iteration completes with real training status, empty/partial metrics, and structured analysis error in `result.analysis`; do not retry training solely because Module B failed.
- History persistence failure: preserve A0 audit truth and return a structured finalization warning; do not recursively call audit failure handlers.

- [ ] **Step 5: Run A0 contract tests**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_audit.py auto_tune\tests\test_tuning_loop.py auto_tune\tests\test_executor.py -q -p no:cacheprovider
```

Expected: all pass and no training starts in existing fatal-path tests.

---

### Task 5: Expose Unified Experiment History to the UI

**Files:**
- Create: `auto_tune/ui/components/experiment_panel.py`
- Modify: `auto_tune/ui/app.py` in `_load_data()` and `_common_context()`
- Modify: `auto_tune/tests/test_ui_training_results.py`

**Interfaces:**
- Produces: `get_experiment_history(log_dir: str = "log") -> list[dict]`
- Produces template context key: `experiment_history`
- Keeps template context key: `tuning_history` for tuning-only panels until Task 6 replaces the history page.

- [ ] **Step 1: Write failing component tests**

Assert the component reads `experiment_history.json`, merges compatible legacy tuning entries, sorts newest first for UI, and returns `[]` when files are absent.

- [ ] **Step 2: Implement `experiment_panel.py` as a thin adapter**

It may normalize display-only fields but must not persist data. Persistence remains owned by `ExperimentHistoryStore` and `training_finalizer`.

- [ ] **Step 3: Add `experiment_history` to cached page context**

Update cache invalidation tests so a completed manual training makes the new record visible after `_invalidate_cache("load_data")`.

- [ ] **Step 4: Run component/UI context tests**

Expected: unified history is available without changing tuning-only suggestion behavior.

---

### Task 6: Update the Unified History and Completion UI

**Files:**
- Modify: `auto_tune/ui/templates/single_page.html`
- Modify: `auto_tune/ui/i18n.py`
- Modify: `auto_tune/tests/test_ui_training_results.py`

**Interfaces:**
- Consumes: `experiment_history` records from Task 5.
- Consumes: unified SSE `result` from Task 3.

- [ ] **Step 1: Write failing template assertions**

Render records for both sources and assert:

```python
assert "普通训练" in html
assert "自动调优" in html
assert "mAP50" in html
assert "分析失败" in html
```

Also assert manual details omit LLM/Guardrails sections and tuning details include them when data exists.

- [ ] **Step 2: Replace the history table data source**

The “Training History” page must iterate over `experiment_history`, showing source, run name, model, dataset, training status, analysis status, four KPI values, finish time and details. Missing metrics render `—`, while numeric zero renders `0` or `0.0000`.

- [ ] **Step 3: Add source and status filters in native JavaScript**

Use `data-source` and `data-status` attributes on rows. Filtering must be client-side and must not add a new endpoint or dependency.

- [ ] **Step 4: Implement conditional detail sections**

Render public fields for every record. Render LLM decision, changes, Guardrails and audit link only for `source == "tuning"` and only when the field exists.

- [ ] **Step 5: Update the training completion card**

Consume unified SSE result; show training status, analysis status, four KPI values, completed/best epoch, and report/run-directory links when present. Use warning styling for partial success.

- [ ] **Step 6: Add all Chinese and English translation keys**

At minimum: Manual Training, Auto-Tuning, Analysis Status, Analysis Failed, Analysis Skipped, All Sources, All Statuses, Training completed but result analysis failed.

- [ ] **Step 7: Run UI tests**

Expected: source/status filters, zero display and conditional sections pass.

---

### Task 7: Contract, Regression, and Real Short-Run Verification

**Files:**
- Modify only if evidence requires: tests or implementation files from Tasks 1–6
- Modify after verification: `docs/implementation_plan_20260814.md`
- Modify after verification: `docs/roadmap_20260814.md`

- [ ] **Step 1: Run focused unified-finalization tests**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_experiment_history.py auto_tune\tests\test_training_finalizer.py auto_tune\tests\test_ui_training_results.py auto_tune\tests\test_tuning_loop.py -q -p no:cacheprovider
```

Expected: all pass.

- [ ] **Step 2: Run the complete suite**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests -q -p no:cacheprovider
```

Expected: all pass; existing sklearn PCA warnings may remain but no new warnings are accepted without explanation.

- [ ] **Step 3: Run one real 1-epoch manual training**

Use the existing UI/API and currently configured local dataset. Do not call an LLM. Verify:

- YOLO exits successfully;
- Module B report exists;
- `experiment_history.json` contains one `source=manual` record;
- report, history, SSE result and UI show identical four KPI values;
- no dataset, weight, log or secret is prepared for GitHub.

- [ ] **Step 4: Verify tuning integration**

First run a tuning dry-run to verify identity/history structure. If dry-run cannot establish `after_metrics`, run one 1-epoch `keep_params=True` tuning training. Verify audit after metrics equal the finalizer/history metrics.

- [ ] **Step 5: Verify idempotency against the real run**

Call the finalizer a second time for the same run and assert `experiment_history.json` still contains exactly one matching `run_id`.

- [ ] **Step 6: Perform UI browser smoke testing**

Check the training monitor completion card, source/status filters, manual details, tuning details, zero/missing metric rendering and partial-success warning state. Record screenshots or a concise observed-results table in the delivery report; do not commit generated screenshots unless explicitly requested.

- [ ] **Step 7: Update roadmap documents with actual evidence only**

Record changed files, exact test counts, real run names, KPI comparison, deviations and remaining risks. Do not mark the task complete before Codex independently re-runs the evidence.

- [ ] **Step 8: Produce the Claude Code delivery report**

The report must contain:

1. changed files;
2. behavior and Schema changes;
3. focused/full test commands and counts;
4. real-run comparison of CSV/report/history/UI metrics;
5. failure-policy evidence;
6. deviations and remaining risks;
7. confirmation that no Git initialization, commit or push was performed.

---

## Codex Acceptance Gate and GitHub Boundary

Claude Code stops after Task 7 and hands the workspace to Codex. Codex independently reviews the implementation, re-runs focused and full tests, performs UI/short-run verification as needed, and lists all remaining issues as copyable text.

Only after Codex accepts this batch may Codex prepare the first GitHub upload. Before upload, Codex must verify `.gitignore` and exclude at least datasets, `detect/`, `runs/`, `log/`, model weights, generated reports, `config.yaml`, `qwen.txt`, API keys and other secrets. GitHub publication is not part of Claude Code's implementation tasks.
