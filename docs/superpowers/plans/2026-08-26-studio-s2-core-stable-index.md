# Studio S2 Core Stable Index Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 Studio 建立稳定的 SQLite 本地数据集与实验索引，使训练/调优历史在应用重启后可查询和重新打开，同时保持 JSON 审计与 S1.5 运行状态为事实来源。

**Architecture:** 新增 `auto_tune.modules.local_index`，将连接/迁移、Repository 和 Service 分层；现有训练收尾与数据集快照只通过 Service 写入索引。SQLite 是可重建查询投影，写入失败只产生 `index_error`，不得改变训练真实终态、覆盖旧 JSON 或阻止报告生成。

**Tech Stack:** Python 3.10、标准库 `sqlite3`/`dataclasses`/`hashlib`/`json`、FastAPI、Jinja2、pytest；不新增依赖。

**Spec:** `docs/superpowers/specs/2026-08-26-studio-s2-core-stable-index-design.md`

## Global Constraints

- 只能使用 `D:\Program Files\anaconda3\envs\auto_tune\python.exe` 运行 Python 和 pytest，并在每份测试报告中记录该解释器路径。
- 不新增依赖，不修改真实 `auto_tune/config.yaml`；只修改脱敏的 `auto_tune/config.template.yaml`。
- SQLite 只保存索引、必要摘要和路径引用，不保存数据集二进制、模型权重、日志全文、API Key 或完整训练命令。
- `experiment_history.json`、`tuning_history.json`、审计文件、S1.5 状态文件和报告继续作为事实文件；迁移只读，不改写、不移动、不删除。
- SQLite 故障不得改变 `completed`、`failed`、`cancelled`、`interrupted` 或 `unknown` 的真实状态。
- 正式数据入口继续使用目录；遗留 ZIP/JSON 上传保持 410，不恢复 ZIP 安全提取开发。
- 所有 SQL 必须参数化；业务调用方不得直接依赖表结构或散落 SQL。
- Claude Code 不修改 README、路线图、交接记录、实施计划、规格、DOCX，不执行 `git add`、`git commit`、`git push` 或 PR 操作。
- 不创建 Git worktree，不复制数据集、权重或训练结果。

## File Structure

**新增业务文件**

- `auto_tune/modules/local_index/__init__.py`：仅导出公共领域类型和 `LocalIndexService`。
- `auto_tune/modules/local_index/models.py`：冻结配置、记录、查询、导入结果和稳定错误类。
- `auto_tune/modules/local_index/database.py`：连接、PRAGMA、Schema v1、事务、迁移、完整性检查和有界备份。
- `auto_tune/modules/local_index/repository.py`：数据集、实验、产物与迁移记录的参数化 CRUD/查询。
- `auto_tune/modules/local_index/service.py`：配置解析、领域校验、JSON 投影、幂等导入和 API 使用场景。

**新增测试文件**

- `auto_tune/tests/test_local_index_database.py`
- `auto_tune/tests/test_local_index_repository.py`
- `auto_tune/tests/test_local_index_import.py`
- `auto_tune/tests/test_local_index_integration.py`
- `auto_tune/tests/test_local_index_api.py`
- `auto_tune/tests/test_local_index_ui.py`

**修改文件**

- `auto_tune/config.template.yaml`：新增脱敏 `local_index` 配置段。
- `auto_tune/modules/train_analyzer/training_finalizer.py`：JSON 成功后尽力写 SQLite，并保留独立 `index_error`。
- `auto_tune/modules/agent_engine/loop.py`：把 S1.5 调优运行身份传入训练收尾。
- `auto_tune/ui/app.py`：创建 Service、数据集快照登记、API 和最小 UI 上下文。
- `auto_tune/ui/components/experiment_panel.py`：优先读取 SQLite，索引不可用时诚实回退 JSON。
- `auto_tune/ui/templates/single_page.html`：数据集索引摘要、实验数据集筛选、索引状态和旧记录导入结果。
- `auto_tune/ui/i18n.py`：S2 Core 中英文文案。
- `.gitignore`：显式忽略 SQLite 主文件、WAL/SHM 和数据库备份；现有 `/log/` 忽略继续保留。
- 相关既有测试：`test_training_finalizer.py`、`test_tuning_loop.py`、`test_ui_training_results.py`。

---

### Task 1: SQLite 基础、Schema v1 与故障保护

**Files:**
- Create: `auto_tune/modules/local_index/__init__.py`
- Create: `auto_tune/modules/local_index/models.py`
- Create: `auto_tune/modules/local_index/database.py`
- Test: `auto_tune/tests/test_local_index_database.py`
- Modify: `auto_tune/config.template.yaml`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `config.template.yaml` 中的 `local_index` 字典；标准库 `sqlite3`。
- Produces:
  - `LocalIndexConfig(database_path: Path, backup_dir: Path, backup_max_files: int = 3, busy_timeout_ms: int = 5000)`
  - `load_local_index_config(config: dict, base_dir: Path | None = None) -> LocalIndexConfig`
  - `connect_database(config: LocalIndexConfig) -> sqlite3.Connection`
  - `initialize_database(config: LocalIndexConfig) -> int`
  - `check_database_integrity(config: LocalIndexConfig) -> None`
  - `transaction(connection: sqlite3.Connection) -> ContextManager[sqlite3.Connection]`
  - 稳定错误：`LocalIndexError`、`LocalIndexConfigError`、`LocalIndexCorruptError`、`LocalIndexMigrationError`、`LocalIndexPersistenceError`。

- [ ] **Step 1: 写配置与领域错误失败测试**

在 `test_local_index_database.py` 写明：相对路径按传入 `base_dir` 解析；`backup_max_files` 必须为 1–10；`busy_timeout_ms` 必须为 100–60000；数据库与备份目录不能相同；冻结配置不可修改。

```python
def test_load_local_index_config_resolves_paths(tmp_path):
    cfg = load_local_index_config({"local_index": {
        "database_path": "log/auto_tune.db",
        "backup_dir": "log/db_backups",
        "backup_max_files": 3,
        "busy_timeout_ms": 5000,
    }}, base_dir=tmp_path)
    assert cfg.database_path == tmp_path / "log" / "auto_tune.db"
    assert cfg.backup_dir == tmp_path / "log" / "db_backups"
```

- [ ] **Step 2: 运行 Task 1 首组测试并确认红灯**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_local_index_database.py -v -p no:cacheprovider
```

预期：`ModuleNotFoundError: auto_tune.modules.local_index`。

- [ ] **Step 3: 实现冻结配置和稳定错误**

`models.py` 至少包含以下契约，不在错误消息中放完整业务路径：

```python
@dataclass(frozen=True)
class LocalIndexConfig:
    database_path: Path
    backup_dir: Path
    backup_max_files: int = 3
    busy_timeout_ms: int = 5000

class LocalIndexError(Exception):
    error_code = "LOCAL_INDEX_ERROR"
    status_code = 500

class LocalIndexCorruptError(LocalIndexError):
    error_code = "LOCAL_INDEX_CORRUPT"
```

- [ ] **Step 4: 写 Schema、PRAGMA 和重复初始化失败测试**

测试必须断言：`PRAGMA foreign_keys=ON`、`journal_mode=WAL`、`busy_timeout=5000`；表为 `schema_migrations`、`datasets`、`experiments`、`artifacts`、`legacy_imports`；重复初始化仍为版本 1；外键生效。

Schema v1 固定为：

```sql
CREATE TABLE schema_migrations (
  version INTEGER PRIMARY KEY,
  checksum TEXT NOT NULL,
  applied_at TEXT NOT NULL
);
CREATE TABLE datasets (
  dataset_id TEXT PRIMARY KEY,
  display_name TEXT NOT NULL,
  canonical_path TEXT NOT NULL,
  data_yaml_path TEXT,
  snapshot_id TEXT UNIQUE,
  snapshot_digest TEXT,
  validation_status TEXT NOT NULL,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  last_used_at TEXT
);
CREATE UNIQUE INDEX datasets_canonical_path_uq ON datasets(canonical_path);
CREATE TABLE experiments (
  run_id TEXT PRIMARY KEY,
  source TEXT NOT NULL,
  run_name TEXT,
  dataset_id TEXT REFERENCES datasets(dataset_id) ON DELETE SET NULL,
  status TEXT NOT NULL,
  phase TEXT,
  model_name TEXT,
  task_type TEXT,
  started_at TEXT,
  finished_at TEXT,
  params_json TEXT NOT NULL,
  metrics_json TEXT NOT NULL,
  analysis_status TEXT,
  error_json TEXT,
  updated_at TEXT NOT NULL
);
CREATE TABLE artifacts (
  artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id TEXT NOT NULL REFERENCES experiments(run_id) ON DELETE CASCADE,
  kind TEXT NOT NULL,
  path TEXT NOT NULL,
  exists_state TEXT NOT NULL,
  created_at TEXT NOT NULL,
  UNIQUE(run_id, kind, path)
);
CREATE TABLE legacy_imports (
  import_id INTEGER PRIMARY KEY AUTOINCREMENT,
  source_path TEXT NOT NULL,
  content_sha256 TEXT NOT NULL,
  result TEXT NOT NULL,
  imported_count INTEGER NOT NULL,
  error_code TEXT,
  imported_at TEXT NOT NULL,
  UNIQUE(source_path, content_sha256)
);
```

- [ ] **Step 5: 实现连接、事务与迁移**

`connect_database()` 只能创建父目录，不能在已存在损坏文件上创建新空库。`initialize_database()` 先对已存在数据库执行 `PRAGMA quick_check`，再按 `MIGRATIONS` 顺序在 `BEGIN IMMEDIATE` 事务内迁移；迁移校验和不一致抛 `LOCAL_INDEX_MIGRATION_FAILED`。

- [ ] **Step 6: 写损坏库、迁移回滚和有界备份测试**

覆盖：随机字节数据库拒绝且原字节不变；人为注入第二版失败 SQL 后 Schema 仍为 v1；迁移前备份；备份按 mtime 只保留最新 3 份；备份失败时不执行迁移。

- [ ] **Step 7: 实现迁移前备份与清理**

备份命名使用 `auto_tune.db.v{old_version}.{UTC timestamp}.bak`，通过 SQLite backup API 写入临时文件后 `os.replace` 发布；只清理 `backup_dir` 内严格匹配当前数据库前缀的旧 `.bak`，不得递归删除。

- [ ] **Step 8: 更新脱敏配置与忽略规则**

```yaml
local_index:
  database_path: log/auto_tune.db
  backup_dir: log/db_backups
  backup_max_files: 3
  busy_timeout_ms: 5000
```

`.gitignore` 增加 `*.db`、`*.db-wal`、`*.db-shm`、`*.db-journal`、`*.bak`；不得放宽现有忽略规则。

- [ ] **Step 9: 运行 Task 1 全部测试**

预期：`test_local_index_database.py` 全部通过，0 skipped。

---

### Task 2: 数据集、实验与产物 Repository

**Files:**
- Create: `auto_tune/modules/local_index/repository.py`
- Modify: `auto_tune/modules/local_index/models.py`
- Modify: `auto_tune/modules/local_index/__init__.py`
- Test: `auto_tune/tests/test_local_index_repository.py`

**Interfaces:**
- Consumes: Task 1 的 `LocalIndexConfig`、`connect_database()`、`transaction()`。
- Produces:
  - `DatasetRecord`、`ExperimentRecord`、`ArtifactRecord` 冻结 dataclass。
  - `ExperimentQuery(dataset_id: str | None, source: str | None, status: str | None, limit: int = 100)`。
  - `LocalIndexRepository.upsert_dataset(record) -> DatasetRecord`
  - `LocalIndexRepository.get_dataset(dataset_id) -> DatasetRecord | None`
  - `LocalIndexRepository.find_dataset_by_data_yaml(path) -> DatasetRecord | None`
  - `LocalIndexRepository.list_datasets() -> list[DatasetRecord]`
  - `LocalIndexRepository.upsert_experiment(record, artifacts=()) -> ExperimentRecord`
  - `LocalIndexRepository.get_experiment(run_id) -> dict | None`
  - `LocalIndexRepository.list_experiments(query) -> list[dict]`
  - `LocalIndexRepository.record_legacy_import(...) -> None`
  - `LocalIndexRepository.has_legacy_import(source_path, content_sha256) -> bool`

- [ ] **Step 1: 写数据集幂等和冲突失败测试**

覆盖：同 `dataset_id` 更新；同 `canonical_path` 不重复；相同 `snapshot_id` 不能绑定两个数据集；`last_used_at` 允许为空；按 `last_used_at/updated_at` 倒序。

- [ ] **Step 2: 运行数据集测试确认红灯**

预期：导入 `LocalIndexRepository` 失败。

- [ ] **Step 3: 实现数据集 Repository**

使用 `INSERT ... ON CONFLICT(dataset_id) DO UPDATE`；冲突映射为 `LocalIndexPersistenceError`，不把原始 SQL 或完整路径写入异常消息。

- [ ] **Step 4: 写实验/产物幂等和筛选失败测试**

```python
def test_upsert_experiment_replaces_same_runtime_run_id(repo):
    repo.upsert_experiment(experiment(status="running"))
    repo.upsert_experiment(experiment(status="completed"))
    rows = repo.list_experiments(ExperimentQuery(limit=100))
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
```

覆盖 `source in {manual,tuning}`、六态状态白名单、按数据集/来源/状态筛选、时间倒序、limit 只允许 1–500、artifact 重放不重复、缺失文件返回 `exists_state=missing` 但不删除记录。

- [ ] **Step 5: 实现实验和产物 Repository**

Repository 输出把 `params_json`、`metrics_json`、`error_json` 解码为 `params`、`metrics`、`error`；JSON 非法时抛 `LOCAL_INDEX_CORRUPT`，不能吞掉并返回空列表。

- [ ] **Step 6: 写数据库锁与并发事务测试**

一个连接持有写锁时，第二连接必须在 `busy_timeout_ms` 后抛 `LocalIndexPersistenceError`；释放锁后可继续写。测试使用短超时配置，不能 sleep 30 秒。

- [ ] **Step 7: 运行 Task 1+2 测试**

预期：两个测试文件全部通过，0 skipped。

---

### Task 3: Service、旧 JSON 幂等导入与稳定错误投影

**Files:**
- Create: `auto_tune/modules/local_index/service.py`
- Modify: `auto_tune/modules/local_index/__init__.py`
- Test: `auto_tune/tests/test_local_index_import.py`

**Interfaces:**
- Consumes: Task 2 的 Repository；现有 `experiment_history.json`、`tuning_history.json`、`latest_dataset.json` 结构。
- Produces:
  - `ImportSummary(source_files: int, imported: int, skipped: int, failed: int, failures: tuple[ImportFailure, ...])`
  - `LocalIndexService.initialize() -> int`
  - `LocalIndexService.index_dataset(payload: dict) -> DatasetRecord`
  - `LocalIndexService.index_experiment(record: dict, runtime_run_id: str | None = None) -> dict`
  - `LocalIndexService.import_legacy_files(paths: Sequence[Path]) -> ImportSummary`
  - `LocalIndexService.list_datasets() -> list[dict]`
  - `LocalIndexService.get_dataset(dataset_id: str) -> dict | None`
  - `LocalIndexService.list_experiments(query: ExperimentQuery) -> list[dict]`
  - `LocalIndexService.get_experiment(run_id: str) -> dict | None`
  - `LocalIndexService.status() -> dict`

- [ ] **Step 1: 写数据集投影失败测试**

`index_dataset()` 从 `latest_dataset.json` 形状提取 `source_dataset_path`、`data_yaml_path`、`snapshot_id`、`snapshot_manifest_digest` 和校验状态。`dataset_id` 为 `sha256("snapshot:" + snapshot_id)`；没有快照时为 `sha256("path:" + normcase(canonical_path))`。

- [ ] **Step 2: 实现路径规范化和数据集投影**

只接受绝对路径；使用 `os.path.abspath`、`normpath`、Windows `normcase`；不得解析或遍历数据集成员。调用方必须先经过 S1.4/S1.2 校验，Service 只验证索引字段。

- [ ] **Step 3: 写实验投影与 runtime_run_id 测试**

当传入 `runtime_run_id="tuning:<uuid>"` 时，SQLite `experiments.run_id` 使用该值；原 JSON `record["run_id"]` 保持不变并存入 `params_json` 的兼容元数据 `legacy_record_run_id`。同一调优会话多轮写入只更新同一实验，产物按实际路径累积去重。

- [ ] **Step 4: 实现实验投影**

允许状态固定为 `starting/running/completed/failed/cancelled/interrupted/unknown`；旧 `done/error/aborted` 映射为 `completed/failed/cancelled`。模型、任务和数据路径从 `params` 提取；按 `data_yaml_path` 查找数据集关联；未匹配时 `dataset_id=None`，不伪造数据集。

- [ ] **Step 5: 写旧 JSON 导入红灯测试**

覆盖：

- `experiment_history.json` v1 正常导入；
- `tuning_history.json` 列表按现有 `_legacy_status/_legacy_metrics` 语义导入；
- 相同路径和 SHA-256 重复执行跳过；
- 文件内容更新后只 upsert，不重复；
- 单个损坏文件记 `LEGACY_IMPORT_INVALID_JSON`，其他文件继续；
- 非法 Schema 记 `LEGACY_IMPORT_INVALID_SCHEMA`；
- 单文件超过 16 MiB 记 `LEGACY_IMPORT_TOO_LARGE`；
- 源文件字节、mtime 和 SHA-256 在导入前后不变。

- [ ] **Step 6: 实现有界、只读、幂等导入**

以 1 MiB 块读取并计算 SHA-256，总读取上限 16 MiB；先计算摘要再解析 UTF-8 JSON。`legacy_imports` 对成功、跳过和失败均记录；失败信息只含 basename 和稳定错误码，不泄漏完整本机路径。

- [ ] **Step 7: 实现 status()**

返回固定字段：`enabled`、`available`、`schema_version`、`dataset_count`、`experiment_count`、`last_import`、`error_code`；损坏库返回 `available=false`，不自动重建。

- [ ] **Step 8: 运行 Task 1–3 聚合测试**

预期：三个新测试文件全部通过，0 skipped。

---

### Task 4: 接入数据集快照、普通训练与自动调优收尾

**Files:**
- Modify: `auto_tune/modules/train_analyzer/training_finalizer.py`
- Modify: `auto_tune/modules/agent_engine/loop.py`
- Modify: `auto_tune/ui/app.py`
- Modify: `auto_tune/tests/test_training_finalizer.py`
- Modify: `auto_tune/tests/test_tuning_loop.py`
- Modify: `auto_tune/tests/test_ui_training_results.py`
- Create: `auto_tune/tests/test_local_index_integration.py`

**Interfaces:**
- Consumes: Task 3 `LocalIndexService.index_dataset()` / `index_experiment()`。
- Produces:
  - `finalize_training_run(..., runtime_run_id: str | None = None, local_index_service: LocalIndexService | None = None) -> dict`
  - 返回记录新增固定字段 `index_error: dict | None`。
  - `run_tuning_loop(..., runtime_run_id: str | None = None)` 将 S1.5 调优身份传给 finalizer。

- [ ] **Step 1: 写“SQLite 失败不改变训练事实”测试**

分别让 `index_experiment()` 抛损坏、锁和持久化错误，断言 JSON 历史仍写入，报告仍存在，返回 `status=completed`、`analysis_status=completed`，仅 `index_error.error_type == "local_index_persistence_error"`。

- [ ] **Step 2: 修改训练 finalizer**

顺序固定：训练事实 → Module B/报告 → JSON history → SQLite index。JSON 失败仍写 `history_error`；SQLite 失败写独立 `index_error`，两者不能互相覆盖。

```python
record = {..., "history_error": None, "index_error": None}
try:
    store.upsert(record)
except Exception as exc:
    record["history_error"] = _error("history", "history_persistence_error", str(exc))
try:
    if local_index_service is not None:
        local_index_service.index_experiment(record, runtime_run_id=runtime_run_id)
except LocalIndexError as exc:
    record["index_error"] = _error("index", "local_index_persistence_error", exc.error_code)
```

- [ ] **Step 3: 写普通训练 S1.5 run_id 接入测试**

`_manual_finalize_cb()` 必须把 `controller.run_id` 作为 `runtime_run_id` 传入；旧测试 fake finalizer 接收 `**kwargs` 并断言该值。不得用 `manual:train_name` 代替 S1.5 UUID 身份。

- [ ] **Step 4: 写调优多轮同身份测试**

`app.py` 的 `loop_runner` 把 `run_state.run_id` 传给 `run_tuning_loop`；两轮训练 finalizer 都收到同一 `runtime_run_id`，SQLite 最终只有一个调优实验，JSON 仍保留现有每轮记录语义。

- [ ] **Step 5: 实现运行身份透传**

新增参数必须有默认 `None`，保持既有直接调用和旧测试可读。不要把 S1.5 controller 或 EventBroker 依赖注入 `local_index` 模块。

- [ ] **Step 6: 写数据集快照登记测试**

在 `/api/dataset/split` 成功原子写入 `latest_dataset.json` 后调用 `index_dataset()`；索引失败时快照仍成功，响应增加：

```json
{"index_warning": {"error_code": "LOCAL_INDEX_PERSIST_FAILED", "message": "数据集已创建，但本地索引更新失败"}}
```

快照失败时不得调用 SQLite；重复创建/复用同一快照只更新一条数据集记录。

- [ ] **Step 7: 实现数据集登记和缓存失效**

仅在 S1.2 快照完整验证和 `latest_dataset.json` 原子发布后登记；登记成功后调用 `_invalidate_cache("load_data")`。不得扫描或复制数据集。

- [ ] **Step 8: 运行 S1.2、S1.5、finalizer 和 tuning 定向回归**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_local_index_integration.py auto_tune\tests\test_training_finalizer.py auto_tune\tests\test_tuning_loop.py auto_tune\tests\test_ui_training_results.py auto_tune\tests\test_dataset_snapshot_api.py auto_tune\tests\test_run_state_training_api.py auto_tune\tests\test_run_state_tuning_api.py auto_tune\tests\test_run_manager_reconnect.py -q -p no:cacheprovider
```

预期：全部通过，0 skipped；既有 S1.5 状态、停止竞态和重放契约不变。

---

### Task 5: 最小查询 API、JSON 回退与 UI

**Files:**
- Modify: `auto_tune/ui/app.py`
- Modify: `auto_tune/ui/components/experiment_panel.py`
- Modify: `auto_tune/ui/templates/single_page.html`
- Modify: `auto_tune/ui/i18n.py`
- Create: `auto_tune/tests/test_local_index_api.py`
- Create: `auto_tune/tests/test_local_index_ui.py`

**Interfaces:**
- Consumes: Task 3 的 `LocalIndexService` 查询和导入接口。
- Produces API：
  - `GET /api/local-index/status`
  - `GET /api/datasets?limit=100`
  - `GET /api/datasets/{dataset_id}`
  - `GET /api/experiments?dataset_id=&source=&status=&limit=100`
  - `GET /api/experiments/{run_id}`
  - `POST /api/local-index/import-legacy`

- [ ] **Step 1: 写 API 成功和稳定错误测试**

覆盖：参数筛选、URL 编码的 `run_id`、不存在返回 404、非法 limit/source/status 返回 400、损坏库返回 503 + `LOCAL_INDEX_CORRUPT`、锁/查询失败返回 503 + `LOCAL_INDEX_UNAVAILABLE`。响应不得包含 SQL、traceback 或完整数据库路径。

- [ ] **Step 2: 实现 API 与统一错误映射**

`POST import-legacy` 只读取固定 `log/experiment_history.json` 和 `log/tuning_history.json`，不接受客户端自定义路径；沿用现有 CSRF 检查。不要恢复 `/api/dataset/upload` 或 `/api/training/analyze`。

- [ ] **Step 3: 写 JSON 回退测试**

`get_experiment_history()` 在 SQLite 可用时返回 SQLite；数据库不存在且尚未初始化时执行初始化后返回 SQLite；损坏或不可用时回退现有 JSON/legacy adapter，同时返回可供页面显示的 `index_warning`。为兼容旧调用，保留 `get_experiment_history(log_dir="log") -> list`，新增：

```python
def get_experiment_history_view(log_dir: str = "log", service=None) -> dict:
    return {"experiments": [...], "source": "sqlite|json_fallback", "index_warning": None | {...}}
```

- [ ] **Step 4: 实现最小 UI 上下文**

`_load_data()` 使用 `get_experiment_history_view()`；上下文新增 `experiment_history_source`、`experiment_index_warning`、`dataset_index`。缓存键仍为 `load_data`，索引写入/导入后必须失效。

- [ ] **Step 5: 写 UI 契约和 XSS 测试**

断言：

- 历史页增加数据集筛选，不删除现有来源/状态筛选；
- 显示“SQLite 索引”或“JSON 回退”状态；
- 数据集页只显示索引数量、名称、校验状态和最近使用时间，不显示复杂批量按钮；
- 导入结果使用 `textContent`；
- 路径、名称、错误码和报告链接均经 Jinja 转义或现有 `esc()`；
- 页面不存在“重命名”“批量删除”“自动归档”“导出 Excel”等未实现操作。

- [ ] **Step 6: 实现最小 UI**

复用现有历史表，不创建新前端框架。数据集筛选使用 `data-dataset-id`；筛选函数同时应用 dataset/source/status。索引警告不隐藏 JSON 回退记录，不把不可用显示成空历史。

- [ ] **Step 7: 添加中英文文案**

至少加入：`Local index`、`SQLite index`、`JSON fallback`、`Index unavailable`、`Import legacy history`、`Imported`、`Skipped`、`Failed`、`Dataset filter`、`All datasets`、`Missing artifact`。

- [ ] **Step 8: 运行 API/UI/XSS 定向测试**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_local_index_api.py auto_tune\tests\test_local_index_ui.py auto_tune\tests\test_template_xss.py auto_tune\tests\test_s11_performance.py auto_tune\tests\test_ui_training_results.py -q -p no:cacheprovider
```

预期：全部通过，0 skipped。

---

### Task 6: 完整回归、真实短训练与交付边界

**Files:**
- No business-code changes unless a failure identifies an S2 regression.
- Verify: all files listed above.

**Interfaces:**
- Consumes: Tasks 1–5 完整实现。
- Produces: Claude Code 交付报告和供 Codex 独立验收的可复验证据。

- [ ] **Step 1: 运行 S2 Core 聚合测试**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_local_index_database.py auto_tune\tests\test_local_index_repository.py auto_tune\tests\test_local_index_import.py auto_tune\tests\test_local_index_integration.py auto_tune\tests\test_local_index_api.py auto_tune\tests\test_local_index_ui.py -q -p no:cacheprovider
```

预期：全部通过，0 skipped。

- [ ] **Step 2: 运行完整测试套件**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests -q -p no:cacheprovider
```

预期：0 failed；只允许既有两条 sklearn PCA warning；不得新增 skipped。

- [ ] **Step 3: 执行最小真实训练验证**

先重建并验证合法 S1.2 快照，再用最小合法 YOLOv8 Detect 数据执行 1 epoch 普通训练。验证：

- S1.5 终态与 SQLite 实验状态一致；
- SQLite `run_id` 等于 S1.5 UUID 身份；
- 数据集外键指向实际快照；
- 指标摘要与 `results.csv`、JSON history、Module B 报告一致；
- 报告/产物存在性正确；
- 应用重启后历史仍可查询和打开；
- 源数据集成员、大小、mtime 和 SHA-256 不变。

- [ ] **Step 4: 执行一次调优短链路验证**

使用同一合法快照执行最小非 dry-run 调优；确认调优 controller 的 S1.5 `run_id` 写入 SQLite，SSE 断开不停止训练，重连/刷新显示真实终态，SQLite 失败注入不改变 S1.5 状态。

- [ ] **Step 5: 执行 Chromium 最小 UI 验收**

验证数据集筛选、来源/状态筛选、实验详情、报告打开、索引状态、JSON 回退、旧记录导入结果、空库、缺失产物和损坏库提示；控制台无新增错误。

- [ ] **Step 6: 检查发布边界**

```powershell
git status --short
git diff --check
git diff --name-only
```

逐项确认没有 `.db`、`.db-wal`、`.db-shm`、`.bak`、数据集、图片、标签、权重、训练产物、日志、凭据、真实 `config.yaml` 或本机业务路径进入待交付文件。

- [ ] **Step 7: 输出交付报告并停止**

报告必须列出：实际新增/修改文件；每条测试命令和结果；首次失败与修复；与规格偏离；Windows/Linux 分支和 skipped；真实训练/UI 证据；SQLite 备份上限；遗留风险；敏感信息与大文件确认。明确写明“未修改项目文档、未提交、未推送”。等待 Codex 审查和独立验收。

---

## Codex 独立验收门

Claude Code 停止后，Codex 按以下顺序验收：

1. 审查所有业务和测试 diff，重点检查事务、损坏库、迁移回滚、SQL 参数化、运行身份和 JSON 回退。
2. 独立运行 S2 Core 聚合、S1.2/S1.5 回归和完整测试。
3. 独立执行短训练、短调优和 Chromium 验收。
4. 检查 `.gitignore`、敏感信息、大文件、数据库实例和无关工作区文件。
5. 仅在艾卡批准发布且 Codex 明确验收通过后，选择本批相关源码、测试和必要当前文档提交；不得使用 `git add .` 或 `git add -A`。
