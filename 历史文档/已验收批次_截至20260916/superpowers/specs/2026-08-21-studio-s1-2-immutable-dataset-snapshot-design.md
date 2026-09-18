# Studio S1.2 不可变数据集快照设计规格

## 1. 结论

Studio S1.2 将现有“直接移动原始图片和标签完成 train/val 划分”的流程替换为“复制到受控目录并原子发布的完整物化快照”。训练只能使用校验通过的快照 `data.yaml`，原始数据目录不得被移动、改名、删除或改写。

本批只支持现有 YOLOv8 Detect 数据格式，不扩展 Classify、Segment、OBB、YOLO11、YOLO26、SQLite、多数据集管理、快照自动清理、API 密钥迁移或容量限制通用框架。两个大模型分析提示词的优化另列后续独立批次。

## 2. 背景与问题

当前 `POST /api/dataset/split` 在 `auto_tune/ui/app.py` 中使用 `shutil.move` 把原始图片和标签搬入 `images/train`、`images/val`、`labels/train` 和 `labels/val`。这会改变用户原目录，且缺少可验证的快照身份、文件摘要、原子发布、失败隔离和可复现契约。

2026-08-20 研发执行版把 Studio S1.2 定义为当前下一批：不可变数据快照，不移动原始文件；验收重点是原目录不变、划分可校验、运行可复现。S1.1 训练日志分层已经完成，不得重复实现或破坏其兼容边界。

## 3. 目标

1. 原始数据目录的路径、成员、文件内容和修改时间在创建快照前后保持不变。
2. 将训练所需图片和标签完整复制到受控快照目录。
3. 相同源数据、相同划分参数和相同 Schema 版本生成相同快照身份和相同 train/val 归属。
4. 通过 SHA-256 摘要验证来源、复制结果和 manifest 的一致性。
5. 使用临时目录和原子重命名发布快照，半成品不得被训练或 UI 视为可用。
6. 相同且完整的快照可复用，避免重复复制和重复占用空间。
7. 旧 `latest_dataset.json`、训练历史、调优历史和审计记录保持可读。
8. 普通训练和自动调优只能使用已校验快照的 `data.yaml`。

## 4. 非目标

- 不删除或自动清理历史快照、临时目录或锁文件。
- 不以硬链接、符号链接、junction 或引用清单代替完整物化复制。
- 不迁移到 SQLite，不实现多数据集 CRUD。
- 不增加登录、用户、角色、队列、GPU 调度或 Cloud 能力。
- 不改变 A0 审计 Schema、统一历史 Schema、KPI 口径或 fatal 策略。
- 不扩展 YOLO 任务或版本范围。
- 不在本批修改文本 LLM、视觉 LLM 或决策 Agent 提示词。

## 5. 方案选择

### 5.1 采用：完整物化快照

图片和标签复制到 `log/dataset_snapshots/<snapshot_id>/`。该方案额外占用约一份数据集空间，但能保证源数据之后被修改或删除时，已经发布的训练输入仍然完整可用。

### 5.2 不采用：轻量引用快照

仅保存路径和摘要不能在源文件变化或删除后复现旧训练，因此不满足完整不可变快照要求。

### 5.3 不采用：硬链接快照

硬链接可能随原文件的原地写入一起变化，且受文件系统、跨盘和 Windows 权限限制，不作为正式默认方案。

## 6. 模块边界

新增独立模块，避免继续扩大 UI 路由文件：

```text
auto_tune/modules/dataset_snapshot/
├─ __init__.py
├─ models.py
└─ service.py
```

职责如下：

- `models.py`：定义不可变的数据对象、异常类型和序列化契约。
- `service.py`：负责发现样本、校验标签、计算摘要、确定性划分、空间预检、复制核验、manifest 生成、快照校验与原子发布。
- `auto_tune/ui/app.py`：仅处理 HTTP 输入输出、读取 latest 状态、调用快照服务和把成功结果登记到 `latest_dataset.json`。
- `auto_tune/ui/templates/single_page.html`：展示创建状态、快照身份、容量提示和校验状态，不承担快照正确性判断。

核心接口：

```python
def create_dataset_snapshot(
    source_dir: Path,
    snapshot_root: Path,
    val_ratio: float,
    seed: int,
    class_names: dict[int, str],
) -> DatasetSnapshot:
    ...


def validate_dataset_snapshot(snapshot_dir: Path) -> DatasetSnapshot:
    ...
```

`create_dataset_snapshot` 同步执行，便于服务层独立测试；FastAPI 路由负责在线程中调用，避免阻塞事件循环。

## 7. 快照目录和文件契约

```text
log/dataset_snapshots/<snapshot_id>/
├─ images/
│  ├─ train/
│  └─ val/
├─ labels/
│  ├─ train/
│  └─ val/
├─ data.yaml
└─ manifest.json
```

创建期间使用同一父目录下的临时目录：

```text
log/dataset_snapshots/<snapshot_id>.tmp-<random-token>/
```

只有完成复制、摘要复核、`data.yaml` 写入和 manifest 自校验后，才能用原子重命名发布为正式目录。

快照文件名格式：

```text
<source-relative-path-sha256-prefix-12>_<original-filename>
```

图片和标签使用同一摘要前缀，避免不同子目录中的同名文件冲突。manifest 始终保存原相对路径和快照相对路径。

## 8. 数据模型

`DatasetSnapshot` 至少包含：

```python
@dataclass(frozen=True)
class DatasetSnapshot:
    schema_version: str
    snapshot_id: str
    snapshot_path: Path
    manifest_path: Path
    data_yaml_path: Path
    source_root: Path
    seed: int
    val_ratio: float
    train_count: int
    val_count: int
    background_count: int
    total_bytes: int
    manifest_digest: str
    reused: bool
```

manifest 顶层字段：

```json
{
  "schema_version": "1.0",
  "snapshot_id": "sha256-hex",
  "created_at": "2026-08-21T00:00:00+00:00",
  "source_root": "absolute-local-path",
  "source_layout": "flat-or-presplit",
  "seed": 42,
  "val_ratio": 0.2,
  "train_count": 80,
  "val_count": 20,
  "background_count": 3,
  "total_bytes": 123456789,
  "samples": [],
  "manifest_digest": "sha256-hex"
}
```

每个 `samples` 条目包含：

```json
{
  "source_image": "relative/source/image.jpg",
  "source_label": "relative/source/image.txt",
  "snapshot_image": "images/train/prefix_image.jpg",
  "snapshot_label": "labels/train/prefix_image.txt",
  "split": "train",
  "is_background": false,
  "image_size": 12345,
  "label_size": 123,
  "image_sha256": "sha256-hex",
  "label_sha256": "sha256-hex"
}
```

无标签图片的 `source_label`、`snapshot_label` 和 `label_sha256` 为 `null`；空标签文件仍需复制，摘要为该空文件的 SHA-256，`is_background` 为 `true`。

`manifest_digest` 对排除 `created_at`、`source_root`、`manifest_digest` 自身后的规范 JSON 计算。规范 JSON 使用 UTF-8、键排序和固定分隔符，确保不受本机路径和创建时间影响。

## 9. 快照身份与确定性划分

### 9.1 快照身份

`snapshot_id` 对以下规范数据计算 SHA-256：

- Schema 版本；
- 按 POSIX 风格相对路径排序的样本清单；
- 每个图片和标签的 SHA-256；
- 标签缺失或空标签状态；
- 来源布局类型；
- seed；
- `val_ratio` 的固定十进制字符串；
- 类别名称映射。

绝对源路径、文件修改时间和创建时间不进入身份摘要。同样内容复制到不同本机目录仍得到相同身份。

### 9.2 未划分来源

1. 以规范相对路径稳定排序样本。
2. 使用局部 `random.Random(seed)`，禁止修改全局随机状态。
3. 对稳定列表执行确定性 shuffle。
4. train 与 val 均必须至少一个样本。
5. `val_count` 采用四舍五入后约束到 `[1, total-1]`；`train_count = total - val_count`。

### 9.3 已划分来源

当来源同时存在 `images/train` 和 `images/val` 时，保持已有归属，不重新随机划分。manifest 记录 `source_layout="presplit"`。请求中的 seed 和比例保留在请求审计字段中，但快照身份以实际归属为准，避免参数变化制造内容相同的重复快照。

只有 train、没有 val 的来源视为未完成划分；服务从可发现的完整样本集合重新生成快照，不改动来源。

## 10. 输入发现与校验

本批支持：

- 平铺布局：图片与同 stem 的 `.txt` 标签位于同一目录或受支持的子目录。
- 标准 Detect 布局：`images/train`、`images/val` 及对应 `labels/train`、`labels/val`。
- 图片扩展名：`.jpg`、`.jpeg`、`.png`、`.bmp`、`.tif`、`.tiff`、`.webp`，大小写不敏感。
- 无标签图片和空标签文件作为合法背景样本。

每个非空标签行必须满足：

- 恰好五列：`class_id x_center y_center width height`；
- `class_id` 是非负整数，并存在于类别映射；
- 四个坐标是有限数值；
- `x_center`、`y_center`、`width`、`height` 均在 `[0, 1]`；
- `width` 和 `height` 大于零。

发现以下输入时拒绝创建：

- 没有支持的图片；
- 样本数少于 2；
- 非法标签；
- 同一图片存在不明确的多个候选标签；
- 路径解析后逃逸源根目录；
- 图片、标签或中间目录是符号链接、junction 或其他 reparse point；
- 快照根目录位于源目录内部；
- 源目录位于快照根目录内部。

## 11. 空间预检

预估空间为所有待复制图片和现存标签的字节总和，加上以下安全余量：

```text
required_bytes = source_bytes + max(64 MiB, ceil(source_bytes * 0.05))
```

通过 `shutil.disk_usage(snapshot_root_parent).free` 获取目标卷可用空间。可用空间小于 `required_bytes` 时在创建临时目录前拒绝。空间预检只降低风险；复制期间空间耗尽仍按失败策略处理。

## 12. 创建、复用与发布流程

1. 规范化并验证请求参数。
2. 解析源根目录和快照根目录，执行路径与 reparse point 检查。
3. 发现样本并校验标签。
4. 计算来源文件大小和 SHA-256；摘要前后复查文件大小与修改时间，检测读取期间变化。
5. 生成确定性 split、`snapshot_id` 和目标目录。
6. 若正式快照存在，调用 `validate_dataset_snapshot`；校验通过则返回 `reused=true`，校验失败则报错，不覆盖。
7. 执行磁盘余量预检。
8. 获取以 `snapshot_id` 为粒度的独占创建锁。
9. 获取锁后再次检查正式快照，以处理并发创建完成的情况。
10. 创建唯一临时目录并复制文件。
11. 对复制文件重新计算 SHA-256，并与来源摘要逐项比较。
12. 写入仅引用快照内相对目录的 `data.yaml`。
13. 写入 manifest，重新读取并进行自校验。
14. 原子重命名临时目录为正式快照目录。
15. 释放锁并返回结果。
16. HTTP 路由在服务成功后原子更新 `latest_dataset.json`。

## 13. 并发与锁

每个快照使用独立锁文件：

```text
log/dataset_snapshots/.locks/<snapshot_id>.lock
```

锁必须通过独占创建获得，并记录 PID、创建时间和 snapshot ID。等待者在有限超时内定期复查正式快照；正式快照已发布且校验通过时直接复用。超时返回结构化冲突错误。

本批不自动删除陈旧锁或临时目录。错误响应应提供受控的相对标识，不向浏览器暴露无关系统路径。清理属于后续运维能力，执行删除前必须由艾卡另行确认。

## 14. `data.yaml` 契约

生成文件只引用快照内部路径：

```yaml
path: .
train: images/train
val: images/val
nc: 2
names:
  0: class_a
  1: class_b
```

不得写入原始目录的绝对路径。类别映射按整数 ID 稳定排序。生成后重新读取，验证 `path`、`train`、`val`、`nc` 和 `names` 与 manifest 一致。

## 15. API 契约

保留现有接口：

```http
POST /api/dataset/split
```

兼容现有请求：

```json
{
  "val_ratio": 0.2,
  "seed": 42
}
```

成功响应：

```json
{
  "status": "success",
  "reused": false,
  "snapshot_id": "sha256-hex",
  "snapshot_path": "log/dataset_snapshots/sha256-hex",
  "manifest_path": "log/dataset_snapshots/sha256-hex/manifest.json",
  "data_yaml_path": "log/dataset_snapshots/sha256-hex/data.yaml",
  "train_count": 80,
  "val_count": 20,
  "background_count": 3,
  "total_bytes": 123456789
}
```

错误分类：

- 参数或数据格式错误：HTTP 400；
- 空间不足：HTTP 507；
- 同快照创建冲突或锁超时：HTTP 409；
- 文件在创建期间变化：HTTP 409；
- 读取、复制、写入或原子发布失败：HTTP 500。

错误响应统一包含稳定错误码和中文信息：

```json
{
  "error": "可读错误信息",
  "error_code": "SNAPSHOT_INSUFFICIENT_SPACE"
}
```

`GET /api/dataset/latest` 保留旧结构并增加：

```json
{
  "dataset": {
    "dataset_path": "原始数据目录",
    "source_dataset_path": "原始数据目录",
    "split": true,
    "snapshot_id": "sha256-hex",
    "snapshot_path": "...",
    "manifest_path": "...",
    "data_yaml_path": "...",
    "train_count": 80,
    "val_count": 20,
    "snapshot_valid": true
  }
}
```

`dataset_path` 在本批继续表示来源目录。训练入口必须显式使用快照 `data_yaml_path`。

## 16. latest 状态与原子写入

新版 `latest_dataset.json` 增加可选字段：

- `source_dataset_path`
- `snapshot_id`
- `snapshot_path`
- `manifest_path`
- `data_yaml_path`
- `snapshot_schema_version`
- `snapshot_manifest_digest`
- `snapshot_created_at`
- `train_count`
- `val_count`
- `background_count`
- `split=true`

写入使用同目录临时文件、flush、`os.fsync` 和 `os.replace`。快照成功但 latest 更新失败时，正式快照保留为未登记快照，接口返回失败，不能声称数据集已就绪。相同请求可在下次校验并复用该快照。

旧 latest 文件没有快照字段时继续可读，UI 显示“尚未创建不可变训练快照”。禁止在读取时自动重写旧文件。

## 17. UI 行为

原“开始划分”改为“创建训练快照”。创建前显示：

- 验证集比例；
- 随机种子，默认 42；
- 原始数据不会被移动或修改；
- 完整快照会额外占用磁盘空间。

请求期间禁用按钮并显示“正在校验并复制”。成功后显示：

- snapshot ID 短格式；
- train、val 和背景样本数量；
- 新建或复用状态；
- 快照目录；
- manifest 校验通过状态；
- 训练使用的 `data.yaml`。

训练弹窗只在 `snapshot_valid=true` 时显示“数据集已就绪”并自动填入快照 `data.yaml`。快照目录、manifest 或数据文件缺失或摘要不匹配时，不能自动带入路径。

动态错误信息继续通过安全文本 API 或现有转义函数显示，禁止引入未经转义的 `innerHTML` 拼接。

## 18. 失败策略

以下情况在复制前拒绝：

- 参数、路径、样本数量或标签格式非法；
- reparse point 或路径边界不安全；
- 预计磁盘空间不足。

以下情况中止且不发布：

- 来源文件在摘要或复制期间变化；
- 复制后摘要不一致；
- `data.yaml` 或 manifest 写入失败；
- manifest 自校验失败；
- 原子发布失败；
- 同 ID 正式快照存在但校验失败。

所有失败必须满足：

- 不修改原始目录；
- 不创建或覆盖可用的正式快照；
- 不让 latest 状态指向半成品；
- 不启动训练；
- 返回结构化错误。

## 19. 兼容边界

- 旧 `latest_dataset.json` 保持可读。
- 不迁移或重写历史 JSON。
- 不改变 `tuning_history.json` 与 `experiment_history.json` 的职责。
- 不改变 A0 审计格式、精确命令复用或 fatal 失败策略。
- 不改变训练成功、分析失败属于部分成功的语义。
- 缺失指标与真实零值继续区分。
- 普通训练和自动调优共用的训练收尾、KPI、日志事件及 S1.1 日志分层不得回归。
- ZIP 继续作为遗留兼容入口；目录选择仍为正式数据接入流程。两种入口都必须经过同一快照服务。

## 20. 测试设计

新增：

```text
auto_tune/tests/test_dataset_snapshot.py
auto_tune/tests/test_dataset_snapshot_api.py
```

修改：

```text
auto_tune/tests/test_ui_training_results.py
```

### 20.1 服务层测试

1. 创建前后来源路径、成员、内容摘要和修改时间不变。
2. 相同输入、seed 和比例生成相同 ID、划分和 manifest 摘要。
3. seed 或比例变化生成不同的未划分快照身份。
4. 快照包含正确的 train/val 图片、标签、背景样本和 `data.yaml`。
5. 已划分来源保持原 train/val 归属。
6. 不同子目录中的同名文件不冲突。
7. 空标签作为合法背景并被复制。
8. 无标签图片作为合法背景且 manifest 字段为 `null`。
9. 非法列数、NaN、Infinity、负类别 ID、未知类别 ID 和非正宽高被拒绝。
10. 符号链接、junction、reparse point 和路径逃逸被拒绝。
11. 快照根目录与来源目录嵌套时被拒绝。
12. 磁盘余量不足时未创建临时或正式快照。
13. 复制中断后没有正式快照。
14. 来源在创建期间变化时失败。
15. 复制后摘要不一致时失败。
16. 相同完整快照被复用，不重复复制。
17. 同 ID 损坏快照被拒绝且不覆盖。
18. manifest 使用稳定排序和规范 JSON。
19. train 与 val 均至少一个样本。
20. 服务不修改全局随机状态。

### 20.2 API 测试

1. 旧请求格式继续可用。
2. 成功响应包含所有新字段。
3. 参数错误返回 400。
4. 空间不足返回 507。
5. 文件变化和并发冲突返回 409。
6. 服务失败不更新 latest 状态。
7. latest 原子更新失败时接口返回失败，正式快照仍可在后续复用。
8. 旧版 latest 文件可读且不被自动重写。
9. 新版 latest 返回 `snapshot_valid`。
10. 损坏快照返回 `snapshot_valid=false`。

### 20.3 UI 测试

1. 创建按钮、空间提示和固定 seed 输入存在。
2. 请求期间禁止重复提交。
3. 新建和复用结果显示正确。
4. 损坏快照不显示“数据集已就绪”。
5. 训练弹窗使用快照 `data.yaml`。
6. 错误文本安全转义。
7. S1.1 训练日志节点上限、分层和 SSE 行为不回归。

## 21. 验证与验收

Claude Code 完成后先运行定向测试：

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_dataset_snapshot.py -v -p no:cacheprovider
```

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_dataset_snapshot_api.py auto_tune\tests\test_ui_training_results.py -q -p no:cacheprovider
```

Codex 独立验收时重新运行完整套件：

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests -q -p no:cacheprovider
```

验收还必须包含：

1. 使用最小合法 YOLOv8 Detect 数据集创建快照。
2. 对比创建前后来源目录的成员、内容摘要和修改时间。
3. 使用快照 `data.yaml` 完成一次真实 1 epoch 训练。
4. 重新执行相同请求并确认复用相同快照。
5. 人工破坏测试副本中的快照文件，确认校验失败且训练不启动。
6. 使用真实 Chromium 验证创建、复用、错误提示、刷新状态和训练路径。
7. 完整测试结果不得低于现有 205 passed 基线；新增测试应使通过数增加，既有两条 sklearn PCA warning 可保留但不得新增未解释 warning。

## 22. Claude Code 交付边界

Claude Code 只实现业务代码和对应测试：

- 可新增 `auto_tune/modules/dataset_snapshot/`；
- 可新增或修改本规格列出的测试；
- 可修改 `auto_tune/ui/app.py`、`auto_tune/ui/templates/single_page.html` 和必要的 `auto_tune/ui/i18n.py`；
- 不修改 README、路线图、交接文档、本规格、实施计划、研发执行版 PDF/DOCX 或发布说明；
- 不新增第三方依赖；
- 不删除文件；
- 不提交数据集、快照、权重、日志、训练产物或真实配置；
- 不自行推送 GitHub；
- 完成后报告改动文件、测试结果、计划偏离和遗留风险。

## 23. 提示词后续批次记录

提示词优化不进入 S1.2。后续独立批次至少处理：

1. 文本训练诊断提示词增加事实证据字段、缺失数据约束、置信度、建议适用条件和“诊断不等于执行”边界。
2. 视觉诊断不再要求模型从图片自行猜测类别映射和精确百分比；同时提供确定性类别名称和矩阵统计，只让视觉模型解释模式与样本现象。
3. 统一输出 Schema，减少依赖正则清洗自然语言。
4. 修复决策提示词“最多 5-8 个参数”与解析器“最多 3 个参数”的冲突。
5. 建立固定输入样例、事实一致性断言、JSON 合规率和建议稳定性回归测试。

## 24. 文档维护策略

本规格与实施计划批准后，Codex 更新《Auto-Tune 后续研发路线与实施评估》：

- 将 S1.1 标记为已完成并补入 v0.2、205 passed 和浏览器验收证据；
- S1.2 实施前曾标记为当前批次；验收后已回写为“完成验收、待与 S1.3 一并提交”，并记录完整物化快照边界与实际偏离；
- 把提示词优化列为 S1.2 之后的独立持续优化批次，不与快照编码混合；
- 以 `.docx` 作为可编辑真源；该批验收时曾导出同名 PDF，后按艾卡 2026-08-21 的决定停止维护 PDF 并将旧 PDF 归档；
- 保留 Research 与 Cloud 暂缓边界，不扩张近期范围。

## 25. 实际实现与验收记录（2026-08-21）

### 25.1 完成结论

Studio S1.2 已完成 Codex 独立审查与验收。功能代码、测试和文档保留在本地未提交工作区；S1.3 现也已完成验收，按艾卡要求进入 S1.2+S1.3 联合提交前检查，不将 S1.2 单独发布。

### 25.2 规格必要偏离

1. `data.yaml` 的 `path` 最终使用快照根绝对路径，而不是原设计的 `.`。项目当前 Ultralytics 8.3.253 会把 `path: .` 按训练进程工作目录解析，导致真实训练错误读取项目根目录下的 `images/val`。绝对路径由严格校验器约束为当前快照根，`train`/`val` 仍只能为快照内部相对目录。
2. 为保证自动调优实际训练输入与已校验快照一致，除 FastAPI 注入请求级配置副本外，执行循环在构建命令前还会用 `training.data_yaml` 强制覆盖合并参数中的 `data`。共享 `APP_CONFIG` 不被修改。
3. 临时目录自校验允许 `data.yaml` 预先指向即将发布的正式目录；原子发布后立即按正式根执行完整校验。

### 25.3 验收证据

- S1.2 定向套件：`110 passed, 0 failed, 0 skipped, 0 warnings`。
- 关键兼容套件：`88 passed, 0 failed, 0 skipped, 0 warnings`。
- 完整套件：`285 passed, 2 warnings, 0 failed, 0 skipped`；两条 warning 为既有 sklearn PCA 数值警告。
- 受控最小 Detect 数据创建前后，源目录成员、大小、mtime 与 SHA-256 完全一致；相同请求复用，人工损坏快照后严格校验拒绝。
- 从项目工作目录使用发布快照的 `data.yaml` 完成真实 1 epoch YOLOv8 Detect，正确读取 3 张 train、1 张 val，并生成 `best.pt`、`last.pt` 和训练结果。
- Chromium 验证快照区块、无快照状态、创建表单和非法比例反馈；API/UI 自动化覆盖创建、复用、latest 原子登记、刷新状态、损坏拒绝和训练门禁。

### 25.4 遗留风险

- Windows junction 由 reparse-point 检测逻辑覆盖，自动化使用 symlink 验证；真实 junction 构造仍可在后续安全专项补充。
- Ultralytics 会在快照 `labels` 目录生成 `train.cache`/`val.cache`。manifest 管理的图片与标签不变且校验仍通过；后续可评估把缓存迁移到训练运行目录，以实现更严格的目录级只读语义。
- 陈旧锁、失败临时目录和历史快照的运维清理不属于 S1.2，保持后续独立能力。
