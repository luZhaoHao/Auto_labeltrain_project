# Studio S1.2 不可变数据集快照实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用确定性、可校验、原子发布的完整物化快照替换现有破坏性数据划分，新训练只能使用校验通过的快照，原始数据始终不变。

**Architecture:** 新增 `dataset_snapshot` 领域模块统一负责样本发现、标签校验、摘要、确定性划分、空间预检、复制复核、manifest、复用、锁和原子发布。FastAPI 只适配 HTTP 与原子登记 latest 状态，SPA 只展示服务端确认的快照状态并将快照 `data.yaml` 交给训练。

**Tech Stack:** Python 3.10+ 标准库、项目既有 PyYAML/FastAPI/Jinja2/pytest，不新增依赖。

**Spec:** `docs/superpowers/specs/2026-08-21-studio-s1-2-immutable-dataset-snapshot-design.md`

## Global Constraints

- 源目录路径、成员、内容和修改时间不得变化。
- 必须完整复制；禁用硬链接、符号链接、junction 和仅引用 manifest。
- 本批只支持 YOLOv8 Detect；不扩展其他任务/版本、SQLite、多数据集或提示词。
- 不自动删除历史快照、临时目录或锁文件。
- 保持 A0 审计、统一历史、调优历史、KPI、fatal 策略和 S1.1 日志契约。
- 目录选择为正式入口，ZIP 为遗留入口；二者共用快照服务。
- Claude Code 只修改业务代码和测试，不修改 README、路线、交接、规格、计划、PDF/DOCX 或发布说明。
- 不提交数据集、快照、权重、日志、报告、真实配置或凭据，不推送 GitHub。
- 先运行 `git status --short`，保留当前工作区已有改动；计划文件若存在不可隔离的重叠改动，停止并报告。

---

## File Map

**Create**

- `auto_tune/modules/dataset_snapshot/__init__.py`：公共导出。
- `auto_tune/modules/dataset_snapshot/models.py`：不可变类型和错误。
- `auto_tune/modules/dataset_snapshot/service.py`：全部快照领域逻辑。
- `auto_tune/tests/test_dataset_snapshot.py`：服务层测试。
- `auto_tune/tests/test_dataset_snapshot_api.py`：API/latest/训练边界测试。

**Modify**

- `auto_tune/ui/app.py`：替换破坏性划分、原子 latest、训练快照校验。
- `auto_tune/ui/templates/single_page.html`：快照创建和状态 UI。
- `auto_tune/ui/i18n.py`：新增中英文文案。
- `auto_tune/tests/test_ui_training_results.py`：UI 与训练路径回归。
- `.gitignore`：仅当现有 `log/` 规则未覆盖快照时补充。

所有测试均使用本计划各步骤给出的完整 PowerShell 命令和既有 auto_tune 环境，不自行改用系统 Python。

---

### Task 1: 定义领域契约与错误边界

**Files:**

- Create: `auto_tune/modules/dataset_snapshot/__init__.py`
- Create: `auto_tune/modules/dataset_snapshot/models.py`
- Create: `auto_tune/tests/test_dataset_snapshot.py`

**Interfaces:**

- Produces: `SnapshotError`、`SnapshotValidationError`、`SnapshotConflictError`、`SnapshotInsufficientSpaceError`、`SnapshotIOError`。
- Produces: frozen `SnapshotSample`、`DatasetSnapshot`。
- Consumers: 后续所有任务从包根导入这些名称。

- [ ] **Step 1: 记录初始状态**

Run: `git status --short`

Expected: 允许既有脏文件；记录到交付报告，不修改/回退无关项。

- [ ] **Step 2: 写失败测试**

在 `test_dataset_snapshot.py` 写：

```python
from dataclasses import FrozenInstanceError
from pathlib import Path
import pytest

from auto_tune.modules.dataset_snapshot import (
    DatasetSnapshot, SnapshotSample, SnapshotError,
    SnapshotValidationError, SnapshotConflictError,
    SnapshotInsufficientSpaceError, SnapshotIOError,
)

def test_snapshot_error_codes_are_stable():
    cases = [
        (SnapshotValidationError("bad"), "SNAPSHOT_VALIDATION_FAILED", 400),
        (SnapshotConflictError("changed"), "SNAPSHOT_CONFLICT", 409),
        (SnapshotInsufficientSpaceError("full"), "SNAPSHOT_INSUFFICIENT_SPACE", 507),
        (SnapshotIOError("write"), "SNAPSHOT_IO_FAILED", 500),
    ]
    for error, code, status in cases:
        assert isinstance(error, SnapshotError)
        assert (error.error_code, error.status_code) == (code, status)

def test_dataset_snapshot_is_immutable(tmp_path):
    sample = SnapshotSample(
        source_image="a.jpg", source_label=None,
        snapshot_image="images/train/x_a.jpg", snapshot_label=None,
        split="train", is_background=True, image_size=3, label_size=None,
        image_sha256="abc", label_sha256=None,
    )
    snapshot = DatasetSnapshot(
        schema_version="1.0", snapshot_id="sid",
        snapshot_path=tmp_path/"sid", manifest_path=tmp_path/"sid"/"manifest.json",
        data_yaml_path=tmp_path/"sid"/"data.yaml", source_root=tmp_path/"source",
        source_layout="flat", seed=42, val_ratio=0.2,
        train_count=1, val_count=1, background_count=1, total_bytes=3,
        manifest_digest="digest", reused=False, samples=(sample,),
    )
    with pytest.raises(FrozenInstanceError):
        snapshot.snapshot_id = "changed"
```

- [ ] **Step 3: 验证测试先失败**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_dataset_snapshot.py -v -p no:cacheprovider
```

Expected: `ModuleNotFoundError`。

- [ ] **Step 4: 实现最小契约**

错误基类默认 `SNAPSHOT_ERROR/500`；子类使用测试中的固定 code/status。两个 dataclass 必须 `frozen=True`，字段与测试完全一致；`__init__.py` 用 `__all__` 导出公共名称。

- [ ] **Step 5: 运行测试**

Expected: `2 passed`。

- [ ] **Step 6: 审查差异**

Run: `git diff -- auto_tune/modules/dataset_snapshot auto_tune/tests/test_dataset_snapshot.py`

Expected: 仅 Task 1 文件。

- [ ] **Step 7: 记录提交边界**

未获 Codex 本地提交授权则不提交。建议提交信息：`feat: define immutable dataset snapshot contracts`。

---

### Task 2: 安全发现并校验 Detect 样本

**Files:**

- Create: `auto_tune/modules/dataset_snapshot/service.py`
- Modify: `auto_tune/modules/dataset_snapshot/__init__.py`
- Modify: `auto_tune/tests/test_dataset_snapshot.py`

**Interfaces:**

- Produces: `discover_samples(source_dir: Path, class_names: dict[int, str]) -> tuple[str, tuple[SourceSample, ...]]`。
- Produces: `sha256_file(path: Path, chunk_size: int = 1048576) -> str`。
- Internal: frozen `SourceSample` 含实际路径、规范相对路径、预分配 split、背景状态、大小和摘要。

- [ ] **Step 1: 添加 fixture 与失败测试**

实现测试辅助函数：

```python
def write_sample(root: Path, relative_image: str,
                 label_text: str | None = "0 0.5 0.5 0.2 0.2\n"):
    image = root / relative_image
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes((relative_image + "-image").encode())
    label = image.with_suffix(".txt")
    if label_text is not None:
        label.write_text(label_text, encoding="utf-8")
    return image, label if label_text is not None else None
```

覆盖：

- flat 按 POSIX 相对路径稳定排序；
- 空标签与无标签均为背景；
- 标准 `images/{train,val}` + `labels/{train,val}` 保持预分配；
- 少于 2 张图拒绝；
- 非法标签逐项拒绝：非 5 列、负/未知 class、NaN、Infinity、坐标越界、宽高为 0；
- symlink/reparse point 拒绝；无创建权限时该条测试可明确 skip。

- [ ] **Step 2: 验证失败**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_dataset_snapshot.py -k "discover" -v -p no:cacheprovider
```

Expected: 导入或断言失败。

- [ ] **Step 3: 最小实现**

要求：

- `source_dir.resolve(strict=True)` 且必须为目录；
- Windows 通过 `st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT` 检测 reparse point；
- 图片扩展名按规格且大小写不敏感；
- presplit 只读取 train/val，并按对应 labels 相对路径配对；
- flat 递归发现，规范路径使用 `PurePosixPath`；
- 每张图片最多一个明确标签；
- 标签每个非空行恰好 5 列，`math.isfinite`，class 存在，坐标范围合法；
- SHA-256 以 1 MiB 分块；hash 前后比较 size 与 `mtime_ns`，变化抛 `SnapshotConflictError`；
- 返回按 `relative_image` 排序。

- [ ] **Step 4: 运行发现测试**

Expected: 全通过；symlink 测试只允许因 OS 权限明确 skip。

- [ ] **Step 5: 回归 Module A**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_analyzer.py auto_tune\tests\test_bbox_geometry.py auto_tune\tests\test_class_stats.py -q -p no:cacheprovider
```

Expected: 全通过。

- [ ] **Step 6: 审查/记录提交**

建议提交信息：`feat: validate dataset snapshot sources`。

---

### Task 3: 生成确定性划分、身份和 manifest 计划

**Files:**

- Modify: `auto_tune/modules/dataset_snapshot/service.py`
- Modify: `auto_tune/modules/dataset_snapshot/__init__.py`
- Modify: `auto_tune/tests/test_dataset_snapshot.py`

**Interfaces:**

- Produces: `build_snapshot_plan(source_dir, val_ratio, seed, class_names) -> SnapshotPlan`。
- Internal: frozen `SnapshotPlan`、`_canonical_json_bytes`、`_assign_splits`、`_snapshot_filename`、`_compute_manifest_digest`。
- Consumers: Task 4。

- [ ] **Step 1: 写失败测试**

必须覆盖：

```python
def make_four_samples(root):
    for name in ("a.jpg", "b.jpg", "c.jpg", "d.jpg"):
        write_sample(root, name)

def test_plan_is_deterministic_without_global_random_side_effect(tmp_path):
    import random
    make_four_samples(tmp_path)
    random.seed(999)
    before = random.getstate()
    first = build_snapshot_plan(tmp_path, 0.25, 42, {0: "defect"})
    after = random.getstate()
    second = build_snapshot_plan(tmp_path, 0.25, 42, {0: "defect"})
    assert before == after
    assert first.snapshot_id == second.snapshot_id
    assert first.manifest_digest == second.manifest_digest
    assert first.train_count == 3 and first.val_count == 1
    assert [(x.source_image, x.split) for x in first.samples] == [
        (x.source_image, x.split) for x in second.samples
    ]
```

另写明确测试：flat 的 seed/ratio 变化导致不同 ID；presplit 的实际归属相同则请求 seed 变化不改变 ID；不同目录同名图片的目标文件名唯一且稳定；布尔 seed、非整数 seed、边界 ratio 拒绝。

- [ ] **Step 2: 验证失败**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_dataset_snapshot.py -k "plan or identity or basename" -v -p no:cacheprovider
```

- [ ] **Step 3: 实现**

- 严格要求 `0 < val_ratio < 1`、`type(seed) is int`；
- 使用局部 `random.Random(seed)`；
- `val_count=min(total-1,max(1,round(total*val_ratio)))`；
- presplit 保持归属且两组都非空；
- 文件名前缀为规范源相对路径 SHA-256 前 12 位；
- ratio 身份字符串用 `format(Decimal(str(val_ratio)), "f")`；
- ID 包含 Schema、布局、类别、实际划分、内容摘要、缺失/空标签状态；flat 还包含 seed/ratio；
- 绝对路径、mtime、创建时间不进入 ID/digest；
- canonical JSON 固定 `ensure_ascii=False, sort_keys=True, separators=(",", ":")`。

- [ ] **Step 4: 运行全部快照测试**

Expected: Task 1-3 全通过。

- [ ] **Step 5: 审查/记录提交**

建议提交信息：`feat: plan deterministic dataset snapshots`。

---

### Task 4: 复制、复核、复用、锁和原子发布

**Files:**

- Modify: `auto_tune/modules/dataset_snapshot/service.py`
- Modify: `auto_tune/modules/dataset_snapshot/__init__.py`
- Modify: `auto_tune/tests/test_dataset_snapshot.py`
- Inspect/conditional modify: `.gitignore`

**Interfaces:**

- Produces: `create_dataset_snapshot(source_dir: Path, snapshot_root: Path, val_ratio: float, seed: int, class_names: dict[int, str]) -> DatasetSnapshot`。
- Produces: `validate_dataset_snapshot(snapshot_dir: Path) -> DatasetSnapshot`。

- [ ] **Step 1: 写 happy-path 与源不变测试**

```python
def fingerprint(root):
    return {
        p.relative_to(root).as_posix(): (p.read_bytes(), p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*")) if p.is_file()
    }

def test_create_snapshot_copies_without_mutating_source(tmp_path):
    source, snapshots = tmp_path/"source", tmp_path/"snapshots"
    make_four_samples(source)
    before = fingerprint(source)
    result = create_dataset_snapshot(source, snapshots, 0.25, 42, {0: "defect"})
    assert fingerprint(source) == before
    assert result.snapshot_path.is_dir()
    assert result.data_yaml_path.is_file()
    assert result.manifest_path.is_file()
    assert (result.train_count, result.val_count, result.reused) == (3, 1, False)
    assert validate_dataset_snapshot(result.snapshot_path).snapshot_id == result.snapshot_id
```

再测试相同请求复用且快照文件时间/内容不变。

- [ ] **Step 2: 写失败路径测试**

通过命名 helper 的 monkeypatch 覆盖：

- 可用空间不足：无 temp/final；
- `shutil.copy2` 中途失败：无 final；
- 来源在计划后/复制中变化：409 类型错误；
- 复制后被篡改：摘要冲突；
- 同 ID 正式快照损坏：校验和复用均失败且不覆盖；
- 源与快照根任一方向嵌套：拒绝；
- 并发创建：后到请求复用已发布快照或锁超时冲突。

- [ ] **Step 3: 验证失败**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_dataset_snapshot.py -k "create_snapshot or validate_dataset_snapshot or space or copy or nested or lock" -v -p no:cacheprovider
```

- [ ] **Step 4: 实现空间与路径预检**

```python
SAFETY_MIN_BYTES = 64 * 1024 * 1024
SAFETY_RATIO = 0.05

def required_snapshot_bytes(source_bytes: int) -> int:
    return source_bytes + max(SAFETY_MIN_BYTES, math.ceil(source_bytes * SAFETY_RATIO))
```

用 `shutil.disk_usage(snapshot_root_parent).free`；空间不足在创建 temp 前拒绝。

- [ ] **Step 5: 实现复制与 data.yaml**

- `shutil.copy2`；
- 复制后重新 hash，并复查来源未变化；
- `data.yaml` 只写 `path: .`、相对 train/val、排序 names 和 nc；
- 用 `yaml.safe_load` 重读精确验证。

- [ ] **Step 6: 实现 manifest 与原子发布**

- UTC 带时区 `created_at`；
- digest 排除 `created_at/source_root/manifest_digest`；
- UTF-8 稳定 JSON；
- data/manifest 均 flush + `os.fsync`；
- temp 内完整自校验后以 `os.replace` 发布；
- 失败不自动删除 temp，只写受控服务端日志。

- [ ] **Step 7: 实现锁**

锁路径由 `snapshot_root / ".locks" / f"{snapshot_id}.lock"` 构造；以 `mode="x"` 独占创建，内容含 ID、PID、UTC 时间和随机 token；最多等待 30 秒，每次复查 final；只删除当前 token 对应的自有锁；不清理既有陈旧锁。

- [ ] **Step 8: 实现严格校验/复用**

必须验证 reparse point、Schema/ID、canonical digest、所有文件 size/hash、计数/split、data.yaml 只能指向本快照。校验成功才返回 `reused=True`。

- [ ] **Step 9: 运行服务测试**

Expected: 全通过，无新增未解释 warning。

- [ ] **Step 10: 检查忽略规则**

Run: `git check-ignore -v log/dataset_snapshots/example/manifest.json`

Expected: 被现有 `log/` 覆盖；否则仅补 `/log/dataset_snapshots/`。

- [ ] **Step 11: 审查/记录提交**

建议提交信息：`feat: materialize immutable dataset snapshots`。

---

### Task 5: 替换破坏性 API 并原子登记 latest

**Files:**

- Modify: `auto_tune/ui/app.py`
- Create: `auto_tune/tests/test_dataset_snapshot_api.py`
- Modify: `auto_tune/tests/test_ui_training_results.py`

**Interfaces:**

- Produces: `_read_latest_dataset() -> dict | None`。
- Produces: `_write_json_atomic(path: Path, payload: dict) -> None`。
- Produces: `_snapshot_to_latest_info(existing, snapshot) -> dict`。
- Preserves: `POST /api/dataset/split`、`GET /api/dataset/latest`。

- [ ] **Step 1: 写 API 失败测试**

使用项目现有 app/TestClient fixture 方式，禁止创建第二套应用实例。覆盖：

- 旧请求 `{"val_ratio":0.2,"seed":42}` 成功；
- 响应精确包含 status/reused/ID/路径/count/bytes；
- ratio 0、1、字符串；seed True、浮点均 400 + 固定 error code；
- 四类 SnapshotError 映射 400/409/507/500；
- 服务失败时 latest 字节不变。

- [ ] **Step 2: 写 latest 兼容测试**

- 旧 JSON 可读、字节不变、`snapshot_valid=false`；
- 新 JSON 校验通过为 true；
- 缺失/损坏快照为 false，`_common_context` 不崩；
- 模拟 `os.replace` 失败，旧 latest 仍可读且 API 返回失败。

- [ ] **Step 3: 验证失败**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_dataset_snapshot_api.py -v -p no:cacheprovider
```

- [ ] **Step 4: 添加常量/原子 helper**

```python
LATEST_DATASET_PATH = Path("log") / "latest_dataset.json"
DATASET_SNAPSHOT_ROOT = Path("log") / "dataset_snapshots"
```

`_write_json_atomic` 使用同目录唯一 temp、UTF-8、flush、fsync、replace，失败不删 temp。读取旧文件不自动重写。

- [ ] **Step 5: 替换 split 路由**

删除路由内 random/glob/mkdir/`shutil.move` 逻辑。新路由：

1. 严格解析参数，不用会接受模糊值的强制转换；
2. 从 `source_dataset_path` 或旧 `dataset_path` 取来源；
3. 从现有真实来源取得 class names，不在已有映射时发明默认类；
4. 使用 `await asyncio.to_thread(create_dataset_snapshot, source_path, DATASET_SNAPSHOT_ROOT, val_ratio, seed, class_names)`；
5. 服务成功后才原子写 latest；
6. typed error 直接映射状态码/error_code；
7. 返回规格定义的成功结构。

- [ ] **Step 6: 每次读取都校验新快照**

新记录调用校验后才置 `snapshot_valid=true`；旧记录非致命地为 false；来源信息不能因快照损坏而消失。

- [ ] **Step 7: 运行 API/UI 路由测试**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_dataset_snapshot_api.py auto_tune\tests\test_ui_training_results.py -q -p no:cacheprovider
```

- [ ] **Step 8: 证明破坏性移动已移除**

Run: `rg -n "shutil\.move|os\.rename" auto_tune/ui/app.py auto_tune/modules/dataset_snapshot`

Expected: 数据划分流无 `shutil.move`；仅允许已审查的原子发布/状态替换。

- [ ] **Step 9: 审查/记录提交**

建议提交信息：`feat: serve dataset snapshot creation`。

---

### Task 6: 新训练与调优必须验证快照

**Files:**

- Modify: `auto_tune/ui/app.py`
- Modify: `auto_tune/tests/test_dataset_snapshot_api.py`
- Modify: `auto_tune/tests/test_ui_training_results.py`

**Interfaces:**

- Produces: `_resolve_validated_snapshot_data_yaml(latest_info: dict) -> Path`。
- Applies: 普通训练与自动调优从 latest 数据集推导路径的入口。

- [ ] **Step 1: 写失败测试**

覆盖：

1. valid latest 只把快照绝对 `data.yaml` 传给训练；
2. 快照缺失/manifest 或样本被改，命令构造和训练入口调用次数为 0；
3. 旧 latest 不被标为 ready，不能隐式选择；
4. 若当前明确支持手工 `data_yaml`，保持该兼容但不得标作快照；
5. 自动调优调用同一 resolver，不得旁路。

- [ ] **Step 2: 验证失败**

运行带 `snapshot and (training or tuning or launch or ready)` 的定向测试，确认旧逻辑会失败。

- [ ] **Step 3: 实现单一 resolver**

它必须校验目录、ID、manifest digest 和 data.yaml 位于快照内，返回绝对路径；普通训练和调优统一调用。

- [ ] **Step 4: 明确保留手工路径边界**

不静默改写用户显式传入的手工路径。只有现有测试证明是正式兼容时保留；可在内存上下文区分 manual/snapshot，但不得为此破坏稳定历史 Schema。

- [ ] **Step 5: 回归关键链路**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_dataset_snapshot_api.py auto_tune\tests\test_ui_training_results.py auto_tune\tests\test_executor.py auto_tune\tests\test_tuning_loop.py auto_tune\tests\test_audit.py auto_tune\tests\test_training_finalizer.py -q -p no:cacheprovider
```

Expected: 全通过，非法快照启动进程数为 0。

- [ ] **Step 6: 审查/记录提交**

建议提交信息：`feat: require validated snapshots for training`。

---

### Task 7: 更新 UI 与国际化

**Files:**

- Modify: `auto_tune/ui/templates/single_page.html`
- Modify: `auto_tune/ui/i18n.py`
- Modify: `auto_tune/tests/test_ui_training_results.py`
- Conditional modify: `auto_tune/tests/test_s11_performance.py`

- [ ] **Step 1: 写静态 UI 失败测试**

断言存在：

- “创建训练快照”；
- ratio 输入严格避开 0/1；
- integer seed 默认 42；
- 原目录不变提示；
- 额外磁盘占用提示；
- 快照 ID、复用和校验状态；
- 旧/损坏记录不显示 ready；
- JS 同时发送 ratio 和 seed；
- 请求中按钮禁用，任何失败恢复。

- [ ] **Step 2: 验证失败**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_ui_training_results.py -k "snapshot or dataset or split" -v -p no:cacheprovider
```

- [ ] **Step 3: 实现 UI**

- 可保留函数名 `splitDataset()` 以减少无关改动，但用户文案全部改为快照；
- 客户端严格验证 finite ratio 和 integer seed；
- fetch 前禁用按钮，所有错误路径恢复；
- 使用安全文本/既有 `esc` 显示服务端字段；
- ready 条件改为 `latest_dataset.snapshot_valid`；
- 不在热路径增加 `innerHTML +=`。

- [ ] **Step 4: 补齐 i18n**

每个新增可见字符串均有中英文键，不散落未翻译中文。

- [ ] **Step 5: UI/S1.1 回归**

Run:

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_ui_training_results.py auto_tune\tests\test_s11_performance.py auto_tune\tests\test_training_log.py -q -p no:cacheprovider
```

- [ ] **Step 6: 审查/记录提交**

建议提交信息：`feat: expose immutable snapshots in studio`。

---

### Task 8: Claude Code 自动化总验证与交付

**Files:** 仅在发现真实缺陷时修改业务/测试；不得修改项目文档。

- [ ] **Step 1: S1.2 定向套件**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_dataset_snapshot.py auto_tune\tests\test_dataset_snapshot_api.py auto_tune\tests\test_ui_training_results.py -q -p no:cacheprovider
```

Expected: 全通过，无新增未解释 warning。

- [ ] **Step 2: 关键兼容套件**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_executor.py auto_tune\tests\test_tuning_loop.py auto_tune\tests\test_audit.py auto_tune\tests\test_training_finalizer.py auto_tune\tests\test_training_log.py auto_tune\tests\test_s11_performance.py -q -p no:cacheprovider
```

- [ ] **Step 3: 完整套件**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests -q -p no:cacheprovider
```

Expected: 大于现有 205 passed、0 failures；除因 OS 权限明确记录的 symlink fixture skip 外不得新增 skip；warning 只允许两条既有 sklearn PCA 数值 warning。

- [ ] **Step 4: 检查差异与敏感信息**

Run:

```powershell
git status --short
git diff --stat
git diff -- auto_tune/modules/dataset_snapshot auto_tune/ui/app.py auto_tune/ui/templates/single_page.html auto_tune/ui/i18n.py auto_tune/tests .gitignore
rg -n "api[_-]?key|token|password|Bearer\s+[A-Za-z0-9]" auto_tune/modules/dataset_snapshot auto_tune/tests/test_dataset_snapshot.py auto_tune/tests/test_dataset_snapshot_api.py
```

Expected: 仅授权文件；无凭据、数据、快照、权重、日志、报告或新增本机绝对路径。

- [ ] **Step 5: 提交 Claude Code 交付报告**

报告必须依次包含“改动文件、定向测试、完整测试、计划偏离、遗留风险、Git”六节。改动文件逐项写真实路径；测试逐项粘贴实际命令以及 passed/failed/skipped/warnings 数量；偏离为零时明确写“无”；遗留风险必须说明 Windows junction/reparse point 与并发锁覆盖状态，并注明真实短 epoch 和浏览器检查由 Codex 执行；Git 节明确未推送，并列出实际存在的本地提交哈希或明确“未提交”。

不得自称验收通过；只有 Codex 可以给出“审查通过/验收通过”。

---

### Task 9: Codex 独立验收（不得交给 Claude Code）

- [ ] **Step 1:** 逐文件审查实现与规格：源不变、身份、canonical manifest、reparse 拒绝、空间、复制复核、锁、原子发布、latest 原子性、训练门禁、旧记录兼容。
- [ ] **Step 2:** 独立重跑 Task 8 定向与完整测试，不引用 Claude Code 结果代替。
- [ ] **Step 3:** 在受控临时目录用最小合法 Detect 数据创建快照；前后比对成员、SHA-256、大小和 mtime。
- [ ] **Step 4:** 使用快照 `data.yaml` 完成真实 1 epoch YOLOv8 Detect；核对命令、`args.yaml`、results CSV、Module B、统一历史和审计兼容。
- [ ] **Step 5:** 真实 Chromium 验证创建、重复点击、复用、非法参数、错误显示、刷新、损坏快照拒绝和训练路径。
- [ ] **Step 6:** 全部通过才明确“审查通过/验收通过”；否则形成问题清单交 Claude Code 修复并重新验收。
- [x] **Step 7:** 验收后由 Codex 更新交接/路线状态和 DOCX；按艾卡 2026-08-21 的决定不再维护 PDF。提交与推送仍需另行授权。

---

## 执行完成记录（2026-08-21）

> 本节记录真实完成事实。上文任务内的未勾选框保留为原始执行模板，不倒填无法独立证明的历史 TDD 时序；以下验收记录是当前完成状态的权威依据。

- [x] Claude Code 已提交两轮交付报告，并明确未提交、未推送。
- [x] Codex 已逐文件审查源不变、身份、manifest、路径/reparse、空间、复制复核、锁、原子发布、latest 和训练门禁。
- [x] S1.2 定向套件最终结果：`110 passed`。
- [x] 关键兼容套件最终结果：`88 passed`。
- [x] 完整套件最终结果：`285 passed, 2 warnings, 0 failed, 0 skipped`。
- [x] 受控临时数据验证源目录成员、大小、mtime、SHA-256 不变；复用与损坏拒绝通过。
- [x] 修复 `path: .` 后，从项目工作目录使用快照完成真实 1 epoch YOLOv8 Detect。
- [x] Chromium 验证快照区块、无快照状态、创建表单和非法参数反馈；自动化覆盖创建、复用、刷新、损坏拒绝及训练路径。
- [x] Codex 已于 2026-08-21 明确给出“审查通过、验收通过”。
- [x] Git 忽略规则覆盖 `log/dataset_snapshots/`，敏感信息扫描未发现真实凭据。
- [ ] 提交与推送：S1.3 已完成验收；按艾卡要求暂缓，先完成 S1.2+S1.3 联合提交前检查并另行确认。

### 实际计划偏离

1. `data.yaml.path` 由 `.` 改为受严格校验的快照根绝对路径，以兼容 Ultralytics 8.3.253 从任意工作目录启动训练。
2. 新增 `auto_tune/modules/agent_engine/loop.py` 的执行层数据路径绑定，防止已校验快照与实际调优训练输入分离。
3. Chromium 未在真实 363 张业务数据上创建新快照，避免未经单独确认复制较大业务数据；创建/复用/损坏/刷新/门禁由自动化覆盖，真实最小数据完成快照与训练验收。
