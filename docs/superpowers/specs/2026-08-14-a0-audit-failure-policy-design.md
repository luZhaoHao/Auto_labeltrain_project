# A0 第一批：调优审计闭环与失败策略设计

## 1. 目标

为 Module C 自动调优闭环增加一份独立、可追溯、可原子落盘的会话审计记录，并统一训练启动前的失败策略。任何未通过决策解析、参数护栏、执行预检或审计持久化的迭代都不得启动 YOLO 训练。

本批不改造数据库、不建设普通训练历史、不实现数据快照、不迁移密钥，也不大规模修改 UI。

## 2. 现状与问题

当前 `loop.py` 使用 `TuningHistory` 记录面向下一轮 LLM 和 UI 的摘要，但它不是完整审计：

- `decision_agent.py` 已返回 `raw_response`，`TuningResult.to_dict()` 没有保留该字段。
- Guardrails 的 warning、error 和 clamped 值存在，但缺少候选值、清洗值和实际执行值的统一对照。
- `executor.py` 内部构造并启动命令，Loop 无法在启动前持久化“即将执行的准确参数列表”。
- 部分失败使用自由文本写入 `error`，不能稳定识别失败阶段和错误类型。
- `TuningHistory.to_json()` 直接覆盖目标文件，进程中断可能留下截断 JSON。
- 当前 Loop 遇到单轮失败通常继续下一轮；对于基础设施和契约错误，这会重复失败并可能浪费 LLM 调用。

## 3. 方案选择

采用独立审计模块，生成 `log/tuning_audit_<session_id>.json`。审计与 `tuning_history.json` 职责分离：

- 审计记录保存执行事实、失败分类和前后差异，是后续迁移数据库的可信来源。
- 历史记录继续服务现有 UI 和下一轮 LLM 上下文，本批只将其写入改为原子写入。
- 不提前引入 SQLite，避免侵入阶段 B。

## 4. 文件与职责

### 4.1 新增 `auto_tune/modules/agent_engine/audit.py`

该模块只负责审计数据结构、敏感字段净化和原子 JSON 写入，不调用 LLM、不构造训练参数、不启动进程。

公开接口：

```python
AUDIT_SCHEMA_VERSION = "1.0"

def utc_now_iso() -> str: ...

def redact_sensitive(value: object) -> object: ...

def atomic_write_json(path: str | os.PathLike, payload: object) -> None: ...

class TuningAuditSession:
    def __init__(
        self,
        session_id: str,
        log_dir: str,
        reference_run: str | None,
        max_retries: int,
    ): ...

    @property
    def path(self) -> str: ...

    def start_iteration(self, iteration: int) -> dict: ...
    def update_iteration(self, iteration: int, **fields: object) -> None: ...
    def fail_iteration(
        self,
        iteration: int,
        stage: str,
        error_type: str,
        message: str,
        fatal: bool,
    ) -> None: ...
    def complete_iteration(self, iteration: int) -> None: ...
    def finalize(self, status: str, error: dict | None = None) -> None: ...
    def flush(self) -> None: ...
    def to_dict(self) -> dict: ...
```

`flush()` 必须先在目标目录创建同文件系统临时文件，完成 `json.dump`、`flush` 和 `os.fsync` 后调用 `os.replace`。任一步失败都抛出异常；不得吞掉异常或退回普通覆盖写入。

### 4.2 修改 `executor.py`

保持 `build_yolo_command(train_name, args_path, merged_params) -> list[str]` 为命令的唯一构造入口。

调整启动接口，使 Loop 能先获得并审计准确命令，再启动进程：

```python
def launch_training(
    train_name: str,
    args_path: str,
    merged_params: dict,
    command: list[str] | None = None,
) -> subprocess.Popen:
    cmd = command if command is not None else build_yolo_command(...)
```

Loop 必须调用一次 `build_yolo_command()`，成功写入审计后，将同一个列表对象或内容完全相同的副本传入 `launch_training(..., command=command)`。不得在 `launch_training` 内追加、删除或重排参数。

### 4.3 修改 `loop.py`

`run_tuning_loop()` 创建会话 ID 后立即创建 `TuningAuditSession`。每个阶段完成后更新内存记录，并在关键边界调用 `flush()`：

1. 会话创建；
2. 决策完成或失败；
3. Guardrails 完成或失败；
4. 执行预检与命令构造完成，且训练启动之前；
5. 探针或训练结束；
6. 会话返回之前。

审计文件无法创建或无法更新属于 `audit_persistence_error`，必须停止整个会话且不得启动训练。

`TuningHistory.to_json()` 改用 `atomic_write_json()`，但历史数据结构保持兼容。

### 4.4 修改测试

主要扩展：

- `auto_tune/tests/test_tuning_loop.py`
- 新增 `auto_tune/tests/test_audit.py`
- 必要时扩展 `auto_tune/tests/test_executor.py`

不依赖真实网络或真实 YOLO 训练，使用 `tmp_path`、monkeypatch 和假进程验证。

## 5. 审计 Schema

顶层结构：

```json
{
  "schema_version": "1.0",
  "session_id": "1723600000000",
  "status": "running",
  "started_at": "2026-08-14T12:00:00Z",
  "finished_at": null,
  "reference_run": "train38",
  "max_retries": 3,
  "iterations": [],
  "error": null
}
```

每轮结构：

```json
{
  "iteration": 1,
  "status": "running",
  "started_at": "2026-08-14T12:00:01Z",
  "finished_at": null,
  "baseline": {
    "reference_run": "train38",
    "params": {},
    "metrics": {}
  },
  "decision": {
    "raw_response": null,
    "diagnosis": null,
    "action": null,
    "hyperparameter_changes": {},
    "training_overrides": {}
  },
  "guardrails": {
    "valid": null,
    "warnings": [],
    "errors": [],
    "clamped": {},
    "sanitized_changes": {}
  },
  "execution": {
    "actual_params": {},
    "args_yaml_path": null,
    "command": [],
    "train_name": null
  },
  "result": {
    "before_metrics": {},
    "after_metrics": {},
    "metric_delta": {},
    "probe": null,
    "analysis": null
  },
  "error": null
}
```

失败对象统一为：

```json
{
  "stage": "decision",
  "error_type": "decision_schema_error",
  "message": "diagnosis must be a non-empty string",
  "fatal": true,
  "timestamp": "2026-08-14T12:00:02Z"
}
```

## 6. 敏感信息策略

`redact_sensitive()` 必须递归处理 dict、list 和 tuple。键名不区分大小写；包含以下任一片段时值替换为 `"***REDACTED***"`：

```text
api_key, apikey, authorization, token, secret, password, passwd, credential
```

审计中保存命令参数列表，不保存拼接后可直接执行的 shell 字符串。日志和异常不得包含请求头或完整私密配置。

## 7. 失败分类与控制流

失败阶段仅使用以下值：

```text
session, perception, decision, guardrails, preflight, audit, execute, probe, analysis
```

训练启动前的失败类型及行为：

| 场景 | error_type | fatal | 行为 |
|---|---|---:|---|
| LLM API 失败 | `decision_api_error` | 是 | 停止会话，不启动训练 |
| JSON/Schema/未知参数错误 | `decision_schema_error` | 是 | 停止会话，不启动训练 |
| Guardrails 拒绝 | `guardrail_rejected` | 是 | 停止会话，不启动训练 |
| 参考运行、模型、data YAML、YOLO 入口无效 | `preflight_error` | 是 | 停止会话，不启动训练 |
| 审计无法落盘 | `audit_persistence_error` | 是 | 停止会话，不启动训练 |
| 命令构造失败 | `command_build_error` | 是 | 停止会话，不启动训练 |
| `Popen` 启动失败 | `training_launch_error` | 是 | 记录失败并停止会话 |

本批将这些契约性和基础设施错误全部视为 fatal，不再消耗下一轮重试。探针判定 `RETRY` 仍可进入下一轮，因为它代表训练已经启动后的可恢复模型行为。

`keep_params=True` 时可以使用空修改；LLM 返回的空修改只有在 `action == "keep_params"` 时合法。其他动作的空修改在决策解析阶段失败。

## 8. 执行预检

在创建输出目录和启动进程之前验证：

- `reference_run` 对应目录存在且包含可解析的 `args.yaml`；若明确是无参考基线的合法流程，则必须由调用方提供完整 `model` 和 `data`。
- `model` 非空；本地路径形式的模型必须存在。内置权重名称允许交给 Ultralytics 解析。
- `data` 非空；本地 YAML 路径必须存在且可读取。
- `resolve_yolo_executable()` 成功返回可执行入口。
- `build_yolo_command()` 返回非空 `list[str]`，每个元素均为非空字符串。

预检函数应保持独立且易测试，建议放在 `executor.py`：

```python
def validate_training_preflight(
    reference_run: str | None,
    reference_dir: str | None,
    merged_params: dict,
) -> list[str]:
    """Return validation errors; empty list means the training may proceed."""
```

## 9. 返回结果兼容性

保留现有 `run_tuning_loop()` 顶层字段和 `iterations` 结构，避免破坏 UI。新增：

```json
{
  "session_id": "1723600000000",
  "audit_path": "log/tuning_audit_1723600000000.json",
  "failure": {
    "stage": "decision",
    "error_type": "decision_schema_error",
    "message": "...",
    "fatal": true
  }
}
```

成功时 `failure` 为 `null`。旧的顶层 `error` 字段继续保留人类可读摘要。

## 10. 验收标准

- 审计文件通过临时文件、`fsync` 和 `os.replace` 原子写入。
- 决策原始响应、候选修改、护栏调整、sanitized 参数、实际参数和执行命令可在同一轮对照。
- 审计中的 `execution.command` 与传给 `subprocess.Popen` 的参数列表逐项一致。
- 审计中的 `actual_params` 与生成的 `args.yaml` 中受控字段一致。
- LLM API、Schema、未知参数、Guardrails、预检和审计落盘失败均不会调用 `launch_training`。
- fatal 失败不进入下一轮；探针 `RETRY` 保留现有重试行为。
- 敏感字段在任意嵌套层级均被遮盖。
- 原有返回结构和 `tuning_history.json` 读取保持兼容。
- 新增测试与原有完整 pytest 全部通过，不新增第三方依赖。

## 11. 后续批次边界

下一批再处理不可变数据快照和不移动原文件的数据划分；密钥轮换与配置迁移单独处理；普通训练与调优历史统一、UI 审计对照展示不纳入本批。

