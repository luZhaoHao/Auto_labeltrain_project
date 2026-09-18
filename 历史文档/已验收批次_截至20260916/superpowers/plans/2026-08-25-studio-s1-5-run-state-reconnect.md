# Studio S1.5 Run State and Reconnect Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让普通训练与自动调优使用同一版本化运行状态契约，并让刷新/重连可靠说明运行、完成、失败、取消、中断和未知状态。

**Architecture:** 新建独立 `run_state` 领域模块负责 Schema、原子持久化、旧格式兼容与 PID 创建身份校验；普通训练和自动调优由独立后台控制器拥有执行生命周期，SSE 仅作为订阅者。API、SSE 和 UI 使用同一状态投影，并通过有界 EventBroker 与短期终态保留支持 `after_seq` 重连；不实现进程接管、断点续训或持久化完整事件日志。

**Tech Stack:** Python 3.10、FastAPI、标准库 `dataclasses/json/uuid/ctypes/pathlib`、Jinja2/原生 JavaScript、pytest、Chromium。

**Spec:** `docs/superpowers/specs/2026-08-25-studio-s1-5-run-state-reconnect-design.md`

## Global Constraints

- 只使用 `D:\Program Files\anaconda3\envs\auto_tune\python.exe` 运行 Python 与 pytest。
- 不新增依赖、不删除遗留状态读取、不建设 SQLite、任务队列、进程接管或断点续训。
- 旧 running 状态没有可验证身份时必须降级为 unknown，不能继续报告 running。
- 所有新状态写入必须原子化；损坏旧文件不得被静默覆盖。
- Claude Code 只修改业务代码和对应测试，不修改 README、路线图、交接记录、规格、计划或 DOCX。
- 完成后不得自行提交或推送；先输出交付报告，由 Codex 独立审查和验收。

---

### Task 1: 统一运行状态领域模块

**Files:**
- Create: `auto_tune/modules/run_state/__init__.py`
- Create: `auto_tune/modules/run_state/models.py`
- Create: `auto_tune/modules/run_state/service.py`
- Test: `auto_tune/tests/test_run_state.py`

**Interfaces:**
- Produces: `RunState`, `LastEvent`, `ProcessIdentity`, `new_run_state()`, `read_run_state()`, `write_run_state()`, `update_run_state()`, `project_public_state()`。
- Consumes: 仅标准库和现有 `log` 目录。

- [ ] **Step 1: 先写 Schema 与唯一身份失败测试**

```python
def test_new_manual_run_has_unique_versioned_identity():
    first = new_run_state("manual", run_name="train1")
    second = new_run_state("manual", run_name="train1")
    assert first.schema_version == "1.0"
    assert first.run_id.startswith("manual:")
    assert first.run_id != second.run_id
    assert first.status == "starting"
    assert first.phase == "preparing"
```

- [ ] **Step 2: 运行单测确认失败**

Run:
`D:\Program Files\anaconda3\envs\auto_tune\python.exe -m pytest auto_tune\tests\test_run_state.py -k "unique or schema" -v -p no:cacheprovider`

Expected: FAIL，模块尚不存在。

- [ ] **Step 3: 实现冻结模型与严格枚举校验**

`RunState` 至少包含设计文档第 4 节字段；`LastEvent.seq` 为非负整数；状态和阶段不在白名单时抛 `RunStateValidationError`。使用 `uuid.uuid4()` 生成身份，UTC 时间统一为 `Z` 后缀 ISO-8601。

- [ ] **Step 4: 写原子写入与半写失败测试**

```python
def test_atomic_failure_preserves_last_valid_state(tmp_path, monkeypatch):
    path = tmp_path / "training_running.json"
    original = new_run_state("manual", run_name="train1")
    write_run_state(path, original)
    monkeypatch.setattr(os, "replace", lambda *_: (_ for _ in ()).throw(OSError("boom")))
    with pytest.raises(RunStatePersistenceError):
        write_run_state(path, replace(original, phase="training"))
    assert read_run_state(path).phase == "preparing"
```

- [ ] **Step 5: 实现同目录临时文件、flush、fsync、replace 和清理**

临时文件名包含目标文件名、PID 和随机后缀；任何异常都删除临时文件但保留原文件。禁止以 `open(path, "w")` 直接覆盖目标。

- [ ] **Step 6: 写旧格式和损坏文件兼容测试**

覆盖普通训练旧 running、调优旧 running、旧终态、非法 JSON、未知字段。断言旧 running 投影为 `unknown/legacy_identity_unverifiable`，损坏文件投影为 `unknown/state_corrupt`，原字节不变。

- [ ] **Step 7: 实现兼容读取与公共响应投影**

`project_public_state(state)` 必须始终返回 `running/run_id/run_kind/status/phase/last_event/terminal_reason/run_name/updated_at`；无记录也返回稳定形状。

- [ ] **Step 8: 运行 Task 1 全部测试**

Run:
`D:\Program Files\anaconda3\envs\auto_tune\python.exe -m pytest auto_tune\tests\test_run_state.py -v -p no:cacheprovider`

Expected: PASS。

### Task 2: PID 创建身份与保守复核

**Files:**
- Create: `auto_tune/modules/run_state/process_identity.py`
- Modify: `auto_tune/modules/run_state/service.py`
- Test: `auto_tune/tests/test_run_state_process.py`

**Interfaces:**
- Produces: `capture_process_identity(pid: int) -> ProcessIdentity | None`、`compare_process_identity(expected: ProcessIdentity) -> IdentityMatch`、`reconcile_persisted_state(state, controller_owned: bool) -> RunState`。
- Consumes: Task 1 的 `RunState/ProcessIdentity/update_run_state`。

- [ ] **Step 1: 写当前进程身份稳定测试**

连续两次捕获 `os.getpid()`，断言 PID 和 `process_create_token` 相同且 token 非空。

- [ ] **Step 2: 写进程消失、PID 复用和无法校验测试**

使用 monkeypatch 分别返回 `missing/mismatch/unverifiable`，断言结果为：

```text
missing       -> interrupted/process_missing
mismatch      -> interrupted/pid_reused
unverifiable  -> unknown/process_identity_unverifiable
```

- [ ] **Step 3: 实现 Windows 与 Linux 标准库身份读取**

Windows token 使用 `GetProcessTimes` 的创建 FILETIME；Linux token 使用 `/proc/<pid>/stat` 第 22 字段。关闭所有 Windows handle。其他平台或权限错误返回 `unverifiable`，不得只以 `os.kill(pid, 0)` 判定匹配。

- [ ] **Step 4: 实现控制器丢失规则**

持久化身份仍匹配但 `controller_owned=False` 时返回 `interrupted/controller_lost`；只有当前服务内存仍持有对应子进程且身份匹配，才保持 `running=true`。

- [ ] **Step 5: 运行 PID 定向测试**

Run:
`D:\Program Files\anaconda3\envs\auto_tune\python.exe -m pytest auto_tune\tests\test_run_state_process.py -v -p no:cacheprovider`

Expected: Windows 当前进程测试实际执行，0 skipped；其余平台分支通过 monkeypatch 覆盖。

### Task 3: 普通训练接入统一状态

**Files:**
- Modify: `auto_tune/ui/app.py:2356-2575`
- Modify: `auto_tune/modules/agent_engine/training_log.py`
- Test: `auto_tune/tests/test_run_state_training_api.py`
- Modify: `auto_tune/tests/test_ui_training_results.py`

**Interfaces:**
- Consumes: `new_run_state/write_run_state/update_run_state/capture_process_identity/project_public_state`。
- Produces: `/api/training/start` 每条 SSE 的 `run_id/phase/event_seq`，以及 `/api/training/running` 的统一公共状态。

- [ ] **Step 1: 写启动前持久化失败不得创建子进程测试**

monkeypatch `write_run_state` 抛 `RunStatePersistenceError`，监视 `asyncio.create_subprocess_exec` 未调用；接口返回 `500` 和 `RUN_STATE_PERSIST_FAILED`。

- [ ] **Step 2: 写启动、事件递增与终态一致性测试**

使用假异步进程输出两条日志并退出 0，断言 SSE 中 run_id 全部相同、event_seq 单调递增、最终状态文件为 `completed/terminal`，且 API 返回同一 run_id 和状态。

- [ ] **Step 3: 替换普通训练旧直写/删除逻辑**

请求校验完成后创建 `manual:<uuid4>`；子进程创建后写入 PID/token；`_process_training_output_line` 产生事件时同步推进 seq 和 last_event。移除以 `_remove_status_file()` 表达终态的逻辑，但保留旧文件读取兼容。

- [ ] **Step 4: 写停止、失败、SSE 客户端断开测试**

断言用户停止为 `cancelled/terminal`，非零退出为 `failed/terminal`；客户端停止读取 SSE 不得自动把状态改为 completed 或删除记录。

- [ ] **Step 5: 重构 `/api/training/running`**

优先复核内存子进程与持久化身份；服务内存丢失时按 Task 2 规则返回 interrupted/unknown。保留顶层 `running` 和 `status` 字段，不再返回旧文件 running=true。

- [ ] **Step 6: 运行普通训练定向套件**

Run:
`D:\Program Files\anaconda3\envs\auto_tune\python.exe -m pytest auto_tune\tests\test_run_state_training_api.py auto_tune\tests\test_executor.py auto_tune\tests\test_training_log.py auto_tune\tests\test_ui_training_results.py -v -p no:cacheprovider`

Expected: PASS。

### Task 4: 自动调优接入统一状态

**Files:**
- Modify: `auto_tune/ui/app.py:1056-1059,1245-1485`
- Modify: `auto_tune/modules/agent_engine/loop.py:423-930`
- Modify: `auto_tune/modules/agent_engine/executor.py:335-375`
- Modify: `auto_tune/ui/components/tuning_panel.py`
- Test: `auto_tune/tests/test_run_state_tuning_api.py`
- Modify: `auto_tune/tests/test_tuning_loop.py`

**Interfaces:**
- Consumes: Task 1/2 的统一状态服务。
- Produces: 调优 `run_id`、阶段回调、内部训练 PID 身份绑定、统一 `/api/tuning/status`。

- [ ] **Step 1: 写调优身份、阶段和终态失败测试**

dry-run 覆盖 `preparing -> analyzing/finalizing -> completed`；异常覆盖 failed；取消覆盖 cancelled。所有事件与状态文件使用同一 `tuning:<uuid4>`。

- [ ] **Step 2: 为调优循环增加显式状态回调**

在 `run_tuning_loop` 增加可选 `on_state(phase, event_type, message, process_identity=None)`，默认 `None` 保持旧调用兼容。循环只报告事实，不直接写 UI 状态文件。

- [ ] **Step 3: 暴露 TrainingProcess 的 PID 身份**

`TrainingProcess` 提供只读 `pid`；实际子进程启动后由上层捕获创建 token并通过 on_state 绑定。不要让 UI 模块访问 executor 私有 `_proc`。

- [ ] **Step 4: 替换 tuning_running.json 旧直写/删除**

`start_tuning` 创建统一状态并把 `on_progress/on_state` 转换为递增事件；终态同步写入。`get_tuning_status()` 改为统一公共投影，不再只返回字符串。

- [ ] **Step 5: 写服务重启和内部 PID 复用测试**

清空内存控制器后读取新状态：即使 PID/token 仍匹配也必须为 `interrupted/controller_lost`；token 不同必须为 `interrupted/pid_reused`。

- [ ] **Step 6: 运行自动调优定向套件**

Run:
`D:\Program Files\anaconda3\envs\auto_tune\python.exe -m pytest auto_tune\tests\test_run_state_tuning_api.py auto_tune\tests\test_tuning_loop.py auto_tune\tests\test_executor.py -v -p no:cacheprovider`

Expected: PASS。

### Task 5: API、SSE 与 UI 刷新/重连一致性

**Files:**
- Modify: `auto_tune/ui/templates/single_page.html:1568-1610` 及训练监控脚本
- Modify: `auto_tune/ui/i18n.py`
- Modify: `auto_tune/ui/app.py`
- Test: `auto_tune/tests/test_run_state_ui.py`
- Modify: `auto_tune/tests/test_s11_performance.py`
- Modify: `auto_tune/tests/test_template_xss.py`

**Interfaces:**
- Consumes: 两个状态 API 的统一响应和 SSE 的 `run_id/phase/event_seq`。
- Produces: `renderRunState(state)`、`refreshRunState()`、当前页面 `_activeRunId/_lastEventSeq`。

- [ ] **Step 1: 写六态 UI 模板契约测试**

检查中英文键和值，覆盖 running/completed/failed/cancelled/interrupted/unknown；断言停止按钮只由 `state.running === true` 显示，状态消息使用 `textContent`。

- [ ] **Step 2: 实现统一页面状态渲染**

页面加载和切换监控页均调用两个状态 API，按 `updated_at` 选择最近运行。interrupted 文案必须说明无法继续原进程；unknown 文案必须说明无法确认。不得出现“恢复成功”“继续运行”等承诺。

- [ ] **Step 3: 写旧 SSE 串线和重复事件测试**

当前 run_id 为 B 时丢弃 A 的事件；`event_seq <= _lastEventSeq` 时丢弃重复事件；新 run_id 启动时清零序号。终态事件必须隐藏停止按钮并立即刷新状态 API。

- [ ] **Step 4: 实现 SSE 身份过滤且保持 S1.1 性能边界**

不得改回 `innerHTML +=`；默认/完整日志仍分别限制 500/2000 行；队列仍有界；终态不能因背压丢失。

- [ ] **Step 5: 运行 UI 与性能定向套件**

Run:
`D:\Program Files\anaconda3\envs\auto_tune\python.exe -m pytest auto_tune\tests\test_run_state_ui.py auto_tune\tests\test_ui_training_results.py auto_tune\tests\test_s11_performance.py auto_tune\tests\test_template_xss.py -v -p no:cacheprovider`

Expected: PASS。

### Task 6: Claude Code 交付前回归检查

**Files:**
- Modify only if a failing S1.5 test identifies an in-scope defect.
- Test: all files modified above and `auto_tune/tests/` full suite.

**Interfaces:**
- Produces: 未提交、未推送的业务代码与测试交付报告。

- [ ] **Step 1: 检查解释器**

Run:
`D:\Program Files\anaconda3\envs\auto_tune\python.exe -c "import sys; print(sys.executable); print(sys.version)"`

Expected: 解释器绝对路径指向 `auto_tune` 环境，Python 3.10.x。

- [ ] **Step 2: 运行 S1.5 聚合套件**

Run:
`D:\Program Files\anaconda3\envs\auto_tune\python.exe -m pytest auto_tune\tests\test_run_state.py auto_tune\tests\test_run_state_process.py auto_tune\tests\test_run_state_training_api.py auto_tune\tests\test_run_state_tuning_api.py auto_tune\tests\test_run_state_ui.py auto_tune\tests\test_executor.py auto_tune\tests\test_training_log.py auto_tune\tests\test_tuning_loop.py auto_tune\tests\test_ui_training_results.py auto_tune\tests\test_s11_performance.py auto_tune\tests\test_template_xss.py -q -p no:cacheprovider`

Expected: 全部通过，0 skipped；既有两条 sklearn PCA warning 可保留。

- [ ] **Step 3: 运行完整测试**

Run:
`D:\Program Files\anaconda3\envs\auto_tune\python.exe -m pytest auto_tune\tests -q -p no:cacheprovider`

Expected: 不少于当前基线 460 passed，0 skipped，仅允许既有两条 PCA warning。

- [ ] **Step 4: 形成交付报告并停止**

报告必须包含：实际新增/修改文件；每条测试命令和完整结果；首次失败及修复；与规格偏离；平台分支；遗留风险。明确确认未修改项目文档、未新增依赖、未删除文件、未提交、未推送、未泄露凭据或本机业务路径。

---

## Codex 独立验收门

Claude Code 交付后由 Codex 执行，Claude Code 不代替：

1. 审查全部差异，重点检查原子写入、旧状态保守降级、PID token 比对、终态不被进度覆盖、SSE run_id 串线和 XSS。
2. 使用同一 `auto_tune` 解释器独立运行 S1.5 聚合套件和完整套件。
3. 启动真实服务，用 Chromium 验证普通训练和调优的运行、完成、失败、取消、中断、未知六态；覆盖刷新、切页、SSE 断开后重载和按钮状态。
4. 用最小合法数据集完成一次短 epoch 普通训练；若实际调优 PID 绑定路径有改动，再完成一次最短真实调优冒烟。
5. 检查 `.gitignore`、待提交文件、状态 JSON、日志、训练产物、API Key、本机绝对路径和大文件。
6. 只有审查与验收均通过，才由 Codex 提交、推送并创建 PR；随后更新最新研发文档和交接记录。

---

## 最终执行结果（2026-08-26）

- 状态：已完成业务实现、返修和 Codex 独立验收，待提交推送。
- 实现补充：训练与调优执行已从 SSE 请求生命周期解耦；RunManager 对已完成控制器执行最多 20 个、TTL 30 分钟的有界保留；EventBroker 每运行保留最近 2000 条事件并支持 `after_seq` 补发。
- 停止语义：覆盖启动前、子进程创建中和运行中竞态；只有实际终止成功才写 `cancelled`，否则按真实退出码写 `completed` 或 `failed`。
- 重放语义：活动运行发生缓冲滚动时发送无序号、非终态 `replay_truncated` 传输控制消息；服务重启或保留期结束后只返回持久化终态并标记重放不完整。
- 自动化证据：`test_run_manager_reconnect.py` 28 passed；S1.5 六文件聚合 84 passed；完整套件 546 passed、2 条既有 sklearn PCA warning、0 skipped。
- 真实验证：客户端断开后后台调优继续并落盘真实终态，浏览器刷新显示一致状态。
- 验收限制：当前登记的数据集快照损坏，未绕过 S1.2 门禁执行新的短 epoch；进入后续真实训练前先重建并验证合法快照。
