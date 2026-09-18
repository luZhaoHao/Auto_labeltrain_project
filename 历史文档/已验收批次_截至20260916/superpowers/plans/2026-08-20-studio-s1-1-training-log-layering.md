# Studio S1.1 训练日志分层实施计划

> **供 Claude Code 执行：** 逐项使用复选框跟踪；先写失败测试，再实现最小改动。Claude Code 只修改本计划列出的业务代码和测试，不修改本计划、路线图、README 或其他项目文档，不推送 GitHub。

**目标：** 将普通训练与自动调优的训练输出统一为“默认关键摘要 + 可展开完整日志”，保证长训练不淹没页面、错误诊断信息不丢失、SSE 行为不回归。

**架构：** 新增一个无 UI 依赖的训练日志模块，负责 ANSI/控制字符清理、行分类、epoch/验证指标摘要生成、完整日志追加和 SSE 事件构造。普通训练流直接消费该模块；自动调优沿用现有阶段进度事件，同时把底层训练输出接入同一完整日志契约。浏览器只根据服务端事件字段渲染，不再自行猜测 YOLO 原始输出格式。

**技术栈：** Python 3.10、FastAPI `StreamingResponse`、异步 subprocess、原子/追加式 UTF-8 文本日志、Jinja2、原生 JavaScript、pytest、FastAPI TestClient。

**规格来源：** `docs/Auto-Tune后续研发路线与实施评估_研发执行版_20260820.docx` 的 3.1、8、9、12.1；兼容基线来自 `docs/development_handoff_20260814.md`。

## 全局约束

- 本批次只实现 Studio S1.1；不进入 S1.2 数据快照、S1.3 密钥迁移、S1.4 容量限制、S2 SQLite、S2.5 任务扩展、Research 或 Cloud。
- 不新增第三方依赖，不修改真实 `auto_tune/config.yaml`，不删除文件，不复制数据集、权重或训练产物。
- 保持 v0.1 已验收边界：审计失败仍为 fatal；训练成功但分析失败仍为部分成功；缺失指标与真实 `0` 必须区分；普通训练与自动调优继续共用统一收尾。
- 正式数据入口仍为目录选择；ZIP 仅作遗留兼容。本批不得改变数据接入行为。
- 完整日志必须使用 UTF-8，逐行保留清理 ANSI 后的原始文本；默认摘要不得成为训练事实来源。
- 浏览器输出必须按文本处理，禁止用训练输出拼接 `innerHTML`；必须避免日志内容触发 HTML/脚本注入。
- SSE 必须正确处理一个 JSON 事件跨多个网络 chunk、以及一个 chunk 含多个事件的情况。
- 现有配置、命令、审计、最终 KPI、Module B 报告、统一历史和停止训练行为不得回归。

---

## 〇、硬性性能约束（本批必须满足）

以下 12 条为 S1.1 的硬性约束（由艾卡指定），优先级高于本节各 Task 内的既有实现细节；冲突时以本节为准。每条必须有自动化测试或真实浏览器验收证据，缺项视为本批未完成。本批性能目标：长时间训练不淹没页面、不无限增长内存、SSE 行为可度量。

<!-- markdownlint-disable MD029 -->

### P1. 前端 DOM 渲染（约束 1–6）

1. **禁止使用 `innerHTML +=` 渲染训练日志。** `monitorLog`、`tuningLog`、`monitorFullLog`、`tuningFullLog` 任何日志行（含 `message`/`detail`）都不得用 `innerHTML +=` 或字符串拼接写入；现有 `monitorLog.innerHTML += ...` 相关代码必须移除，改用文本节点写入。
2. **统一使用 `createElement` + `textContent`。** 全站复用统一函数 `appendTrainingLogLine(container, text, cssClass)`；普通训练与自动调优两套渲染必须共用，不得复制两套实现。
3. **默认摘要区最多保留最近 500 行。** `monitorLog`/`tuningLog` 子节点行数 ≤ 500。
4. **完整日志区最多保留最近 2000 行。** `monitorFullLog`/`tuningFullLog` 子节点行数 ≤ 2000。
5. **超出上限时删除最旧节点（环形缓冲）。** 用固定容量节点引用队列实现；append 新行若超限先 `removeChild` 最旧节点再插入，删除与追加必须 O(1)，禁止重建整个日志区。折叠/展开切换不得清空或丢失已有行。
6. **使用 `DocumentFragment` 批量插入。** SSE 消费端累积待渲染行，按 50–100ms 时间窗或每次最多 20 行，用 `DocumentFragment` 一次性 append，禁止逐行同步追加导致的连续回流。

### P2. 服务端事件与内存边界（约束 7–10）

7. **tqdm/detail 行默认不进入默认摘要 SSE。** 分类为 `detail` 的事件，SSE payload 的 `message` 为 `null`，只进入完整日志面板与 `training.log`；不得把 detail 行作为默认摘要消息发送。
8. **SSE 事件必须批量发送。** 普通训练后端禁止对每条原始 stdout 行立即 `yield`；必须按行数窗口（≤20 行）或时间窗（50–100ms）聚合成一个 SSE chunk（同一 chunk 内含多个 `data:` 事件）。批量缓冲为有界窗口，受约束 9 约束。
9. **后端不得把完整日志保存到 list、字符串或队列中。** 完整日志只通过 `append_training_log` 追加到 `<train_dir>/training.log`；内存中不得存在全量日志的 list/字符串/无限增长队列。SSE 批处理缓冲与 `msg_queue` 必须有界，客户端变慢时不得无限积累。
10. **同一个 subprocess stdout 只能有一个消费者。** 普通训练协程内"读取一行 → 清理 → 落盘 → 分类 → 构造事件"必须在同一处且只执行一次；自动调优侧若为 `TrainingProcess` 增加输出消费（`drain_output`），必须与 `yolo_train.log` 文件写入、probe/`read_results_csv` 轮询互斥，同一 fd 不得存在两个并发 reader；调优与普通训练不得各自建立一套重复消费者。

### P3. 性能测试与最终度量（约束 11–12）

11. **必须新增并跑通的性能测试：**
    - **a. 行数上限**：模拟 10,000 行日志输入后，默认区 DOM 行数 ≤ 500、完整区 ≤ 2000。环形缓冲逻辑必须提取为可独立执行的纯函数（JS），用本机 Node 运行时直接执行该函数验证；不新增项目第三方依赖，若 Node 不可用则该项必须纳入真实浏览器验收。
    - **b. 处理耗时非二次增长**：对 1k / 5k / 10k 行日志测量单行处理耗时，断言处理 10,000 行后单行平均耗时与 1,000 行时同一数量级（不随历史行数线性/二次增长）；环形缓冲插入与删除必须 O(1)。
    - **c. SSE 队列有界**：客户端慢于服务端时，后端待发送队列/批处理缓冲长度存在明确上限；构造慢客户端场景，断言队列不随训练时长无限增长。
12. **完成后必须报告三项度量**（并入交付报告，Codex 验收复核）：每秒 SSE 事件数（批量前后对比）、前端日志区 DOM 节点数、训练前后 Python/浏览器进程内存变化。

<!-- markdownlint-enable MD029 -->

---

## 一、现状结论与本批设计

### 1. 当前代码事实

- `auto_tune/ui/app.py:1748-1893` 的普通训练接口逐行读取合并后的 stdout/stderr，但所有原始行均以 `status=running, level=info` 发送，没有事件种类、摘要字段或完整日志索引。
- `auto_tune/ui/templates/single_page.html:2255-2292` 在浏览器端用正则过滤 YOLO 文本，能减少部分 tqdm 噪声，但协议仍是非结构化字符串；YOLO 输出变化会直接破坏页面摘要。
- 普通训练页面只有一个 `monitorLog`，没有“完整日志”折叠区，也没有运行结束后重新查询完整日志的接口。
- 自动调优页面展示阶段消息和每 epoch 摘要，但底层训练原始输出未形成与普通训练一致的完整日志契约。
- 当前前端直接把 `data.message` 拼进 `innerHTML`，训练输出若包含 HTML 字符存在注入风险。
- 当前 `reader.read()` 后直接 `split('\n')`，未保留不完整 SSE chunk 尾部，存在事件被网络分片后静默丢失的风险。

### 2. 本批选择的实现边界

采用“服务端统一日志协议”方案：日志分类、摘要生成和完整日志落盘均在 Python 侧完成；前端只负责展示结构化事件。此方案比继续扩展前端正则多改一个小模块，但能让普通训练、自动调优、测试和后续 Linux 交付共用同一事实契约。

不在本批实现：日志轮转/压缩、跨重启续传、数据库索引、历史页日志入口、WebSocket、任务队列、多用户日志隔离。这些分别属于后续运行恢复、SQLite 或 Cloud 范围。

### 3. 固定日志契约

新增以下 Python 接口，名称和字段在本批中不得自行改名：

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

LogKind = Literal[
    "epoch", "validation", "lifecycle", "warning", "error", "detail"
]

@dataclass(frozen=True)
class TrainingLogEvent:
    kind: LogKind
    level: Literal["info", "success", "warning", "error", "debug"]
    raw: str
    summary: str | None
    epoch: int | None = None
    total_epochs: int | None = None

def sanitize_training_line(line: str) -> str: ...
def classify_training_line(line: str) -> TrainingLogEvent: ...
def append_training_log(path: Path, line: str) -> None: ...
def build_training_sse_payload(event: TrainingLogEvent, train_name: str) -> dict: ...
```

`build_training_sse_payload` 固定返回：

```python
{
    "status": "running",
    "event": "training_log",
    "train_name": "train87",
    "log_kind": "epoch",
    "level": "info",
    "message": "Epoch 1/100: box_loss=1.234 cls_loss=0.456 dfl_loss=0.789",
    "detail": "1/100 ...清理 ANSI 后的原始行...",
    "epoch": 1,
    "total_epochs": 100,
}
```

其中 `message` 是默认面板可显示的摘要；`detail` 是完整日志面板显示的原始行。`detail` 事件的 `message` 为 `None`，但仍必须写入日志文件和完整日志面板。最终 `done/error` 事件继续使用现有 `_finalize_and_build_event` 契约，不得改写。

每次运行的完整日志固定为 `<train_dir>/training.log`。文件按进程输出顺序追加，每行结尾统一为 `\n`；启动新运行时创建空文件，禁止覆盖同名既有运行目录中的日志。若日志写入失败，训练可继续，但必须立即发出 `event=log_persistence_error, level=warning`，并且同一路径只提示一次，避免刷屏；训练进程失败仍按现有规则进入失败终态。

### 4. 分类规则与显示规则

- `epoch`：识别 Ultralytics Detect 训练 epoch 数据行，摘要至少含 `epoch/total`、`box_loss`、`cls_loss`，存在 `dfl_loss` 时保留；同一 epoch 的 tqdm 重绘只在默认摘要显示一次，但每一原始行仍进入完整日志。
- `validation`：识别 `all Images Instances P R mAP50 mAP50-95` 数据行，摘要必须区分缺失值和真实 `0`，不得使用 `value || "—"`。
- `lifecycle`：启动训练、数据集、模型、结果目录、训练结束等用户可行动状态，进入默认摘要和完整日志。
- `warning`：包含 `WARNING`、`WARN`、弃用、资源不足等警告标识，进入默认摘要和完整日志。
- `error`：包含 traceback、exception、error、CUDA OOM、进程启动失败等错误信息。错误与紧邻的 traceback 行必须在默认摘要可见，完整日志保留全部上下文。
- `detail`：tqdm 批次进度、模型层表、环境横幅和其他诊断细节；只进入完整日志。
- 所有类型先清理 ANSI CSI/OSC、回车重绘符和不可显示控制字符；中文、路径、冒号、方括号等正常字符不得丢失。

---

## 二、Claude Code 分批实施任务

### Task 1：建立日志分类与持久化的纯 Python 契约

**文件：**

- 新建：`auto_tune/modules/agent_engine/training_log.py`
- 新建：`auto_tune/tests/test_training_log.py`

**接口：**

- 输入：subprocess 解码后的单行字符串、目标 `Path`、训练名。
- 输出：上文固定的 `TrainingLogEvent` 与 SSE payload 字典。
- 不依赖 FastAPI、Jinja2、全局配置或浏览器代码。

- [ ] **Step 1：先写清理和分类失败测试**

至少覆盖以下可直接执行的用例；如真实 YOLO 行格式与样例不同，只能补充样例，不得删除断言语义：

```python
def test_sanitize_removes_ansi_and_carriage_return():
    raw = "\x1b[32m  1/100  1.20G  1.234  0.456  0.789\x1b[0m\r"
    assert sanitize_training_line(raw) == "  1/100  1.20G  1.234  0.456  0.789"


def test_classify_epoch_preserves_zero_values():
    event = classify_training_line("  1/100  1.20G  0  0.456  0.789")
    assert event.kind == "epoch"
    assert event.epoch == 1
    assert event.total_epochs == 100
    assert "box_loss=0" in event.summary


def test_classify_validation_preserves_zero_map():
    event = classify_training_line("all 10 50 0.123 0.456 0 0.111")
    assert event.kind == "validation"
    assert "mAP50=0" in event.summary


def test_traceback_and_cuda_oom_are_errors():
    assert classify_training_line("Traceback (most recent call last):").kind == "error"
    assert classify_training_line("torch.cuda.OutOfMemoryError: CUDA out of memory").kind == "error"


def test_tqdm_batch_progress_is_detail_only():
    event = classify_training_line("1/100 50%|█████| 5/10 [00:01<00:01, 4.5it/s]")
    assert event.kind == "detail"
    assert event.summary is None
```

- [ ] **Step 2：运行测试，确认因模块不存在而失败**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_training_log.py -v -p no:cacheprovider
```

预期：测试收集或导入失败，明确指向 `training_log` 尚不存在。

- [ ] **Step 3：实现最小分类器**

实现固定接口。正则必须集中为模块级已编译常量；先清理再分类；解析失败降级为 `detail`，不得抛异常中断训练。摘要格式由测试锁定，不能把原始整行直接作为所有类型的摘要。

- [ ] **Step 4：补写持久化与 payload 测试**

```python
def test_append_training_log_is_utf8_and_ordered(tmp_path):
    path = tmp_path / "training.log"
    append_training_log(path, "第一行")
    append_training_log(path, "second")
    assert path.read_text("utf-8") == "第一行\nsecond\n"


def test_build_payload_separates_summary_and_detail():
    event = classify_training_line("  1/100  1.20G  1.234  0.456  0.789")
    payload = build_training_sse_payload(event, "train87")
    assert payload["event"] == "training_log"
    assert payload["message"].startswith("Epoch 1/100:")
    assert payload["detail"] == event.raw
    assert payload["epoch"] == 1
```

- [ ] **Step 5：运行 Task 1 测试**

预期：`auto_tune/tests/test_training_log.py` 全部通过；临时目录外无新增日志。

### Task 2：普通训练 SSE 接入统一日志协议

**文件：**

- 修改：`auto_tune/ui/app.py:1748-1893`
- 修改：`auto_tune/tests/test_ui_training_results.py`

**接口：**

- 消费：Task 1 的四个固定接口。
- 产出：`/api/training/start` 的结构化 `training_log` SSE 事件，以及 `<train_dir>/training.log`。
- 保持：现有启动参数、`training_running.json`、停止接口、最终收尾事件。

- [ ] **Step 1：写普通训练事件流失败测试**

将事件流中的“读取并处理单行”提取为可单测的小函数，例如：

```python
def _process_training_output_line(line: str, train_name: str, log_path: Path) -> dict:
    ...
```

测试必须模拟 epoch、tqdm detail、warning、traceback 和中文路径，断言每行均落盘、默认摘要只出现允许类型、payload 字段完整。另写日志写入异常测试，断言返回一次 `log_persistence_error` 警告且不改变 subprocess 终态规则。再写 SSE 批量断言（约束 8/9）：模拟 50 行输入，断言输出 SSE chunk 数 ≤ 原始行数的 1/20 且单 chunk 内 `data:` 事件 ≤ 20；`detail` 行 `message` 为 `null`；客户端未及时消费时待发送缓冲有界。

- [ ] **Step 2：运行新增测试，确认当前非结构化事件失败**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_ui_training_results.py auto_tune\tests\test_training_log.py -v -p no:cacheprovider
```

- [ ] **Step 3：接入普通训练事件流**

在训练目录确定后建立 `training.log`；每条 stdout/stderr 行依次执行清理、追加、分类、构造 payload。不得改变命令生成、进程启动、停止和 `_finalize_and_build_event` 调用顺序。对同一 epoch 的默认摘要去重只能影响 `message` 展示，不得跳过日志落盘或 `detail` 字段。

硬性约束（〇节）：单一消费者（约束 10）——读取→清理→落盘→分类→构造必须在同一协程同一处只执行一次；SSE 批量发送（约束 8）——按 ≤20 行或 50–100ms 窗口聚合成一个含多个 `data:` 事件的 chunk 后 `yield`，禁止逐行立即 `yield`；批量缓冲有界（约束 9）。

- [ ] **Step 4：锁定失败路径**

增加以下断言：

- subprocess 非零退出时，错误行仍可在 `training.log` 找到；
- 最终 SSE 仍为 `status=error` 且保留结构化 `training_process_failed`；
- 日志写入失败只降级日志可用性，不误报训练成功或失败；
- 客户端断开不应把既有审计/最终 KPI 伪造成完成。

- [ ] **Step 5：运行 Task 2 测试**

预期：日志模块和 UI 训练结果测试全部通过。

### Task 3：普通训练 UI 增加默认摘要与完整日志折叠

**文件：**

- 修改：`auto_tune/ui/templates/single_page.html:1176-1196,2174-2318`
- 修改：`auto_tune/ui/i18n.py`
- 修改：`auto_tune/tests/test_ui_training_results.py`

**接口：**

- 消费：`training_log` SSE payload 的 `message`、`detail`、`log_kind`、`level`。
- 产出：`monitorLog` 默认摘要区、`monitorFullLog` 折叠区、可见的展开/收起控件。

- [ ] **Step 1：写渲染结构失败测试**

渲染 `single_page.html` 后至少断言：

```python
assert 'id="monitorLog"' in html
assert 'id="monitorFullLog"' in html
assert 'id="toggleMonitorFullLog"' in html
assert 'function consumeSseChunk' in html
assert 'textContent' in html
```

并断言新增中文和英文文案都在 `i18n.py` 中存在：`完整日志`、`展开完整日志`、`收起完整日志`、`日志保存失败`。

再断言行数上限与批量渲染契约存在（〇节 P1）：模板含 `function appendBounded`、`function appendTrainingLogLine`、`DocumentFragment`，默认区上限常量 500、完整区上限常量 2000；并断言模板中不存在用于训练日志的 `innerHTML +=`（约束 1）。

- [ ] **Step 2：先运行测试并确认 DOM/翻译缺失**

- [ ] **Step 3：实现安全 DOM 渲染与有界行缓冲**

新增统一方法，只使用 `document.createElement` 和 `textContent`：

```javascript
function appendTrainingLogLine(container, text, cssClass) {
  if (!container || text == null) return;
  var row = document.createElement('div');
  row.className = cssClass || 'info';
  row.textContent = String(text);
  container.appendChild(row);
}
```

硬性约束（〇节 P1）：禁止将 `data.message` 或 `data.detail` 拼接到 `innerHTML`（约束 1/2）；默认区只追加非空 `message`，完整区追加所有非空 `detail`；错误和警告在两个区域采用对应样式；折叠区默认关闭，切换时不丢失已经接收的行。

行数上限与环形缓冲（约束 3/4/5）：默认区 `monitorLog` 保留最近 500 行、完整区 `monitorFullLog` 保留最近 2000 行；用固定容量节点引用队列实现，追加新行时若超限先删除最旧节点再插入，删除与追加 O(1)，禁止 `innerHTML` 重建。缓冲逻辑需提取为可独立执行的纯函数（如 `appendBounded` 中的队列操作），供约束 11a 性能测试直接执行。

批量刷新（约束 6）：SSE 消费端累积待渲染行，按 50–100ms 或最多 20 行，用 `DocumentFragment` 一次性 append，禁止逐行同步追加导致连续回流。

- [ ] **Step 4：修复 SSE chunk 缓冲**

提取纯函数式缓冲逻辑 `consumeSseChunk(state, chunk, onEvent)`：将上一次残留与新 chunk 合并，以空行分隔完整 SSE event，只解析完整 `data:` 事件，并把尾部残留保存到 `state.buffer`。普通训练和自动调优两处 fetch 流都必须复用同一实现，避免各自维护易错的 `split('\n')`。

测试至少用以下三种输入进行浏览器或 JavaScript 单测/页面脚本契约测试：一个事件拆成两个 chunk、两个事件位于一个 chunk、中文 JSON 在多字节边界附近分片。不得通过吞掉 JSON 异常来假装成功。

- [ ] **Step 5：运行模板和 UI 测试**

预期：结构、翻译、安全文本渲染和 SSE 缓冲契约通过。

### Task 4：自动调优复用日志分层契约

**文件：**

- 修改：`auto_tune/modules/agent_engine/executor.py`
- 修改：`auto_tune/modules/agent_engine/loop.py`
- 修改：`auto_tune/ui/app.py:681-861`
- 修改：`auto_tune/ui/templates/single_page.html:1024-1053,1739-2064`
- 修改：`auto_tune/tests/test_executor.py`
- 修改：`auto_tune/tests/test_tuning_loop.py`
- 修改：`auto_tune/tests/test_ui_training_results.py`

**接口：**

- 消费：Task 1 日志协议和 Task 3 SSE 缓冲/安全 DOM 方法。
- 产出：自动调优训练阶段的完整日志文件与 `training_log` UI 事件。
- 保持：现有 perception → decision → guardrails → execute → probe 阶段消息、审计、取消、keep_params、dry_run 和最终结果事件。

- [ ] **Step 1：写 executor 输出回调失败测试**

为 `TrainingProcess` 增加可选、向后兼容的输出消费能力，建议固定接口：

```python
def drain_output(self, on_line: Callable[[str], None] | None = None) -> list[str]:
    """Non-blockingly consume newly available decoded lines exactly once."""
```

测试使用伪进程输出，断言每行只消费一次、UTF-8 解码替换非法字节、无回调时保持现有调用兼容、进程结束前后的尾行不丢失。

- [ ] **Step 2：运行 executor 与 tuning loop 测试并确认失败**

- [ ] **Step 3：接入调优训练输出**

`run_tuning_loop` 在启动每个候选训练后，把每一底层输出行写入该候选 `<train_dir>/training.log`，并通过现有 `on_progress` 的新增可选字段传递：

```python
on_progress(
    iteration,
    event.summary,
    step="execute",
    event="training_log",
    log_kind=event.kind,
    level=event.level,
    detail=event.raw,
    epoch=event.epoch,
    total_epochs=event.total_epochs,
)
```

不得用新日志回调替代现有 `results.csv` 指标事实读取；探针判断和最终 KPI 仍以确定性结果文件为准。

硬性约束（〇节）：单一消费者（约束 10）——`TrainingProcess.drain_output` 只允许一个消费循环读取 stdout，且与 `yolo_train.log` 文件写入、probe/`read_results_csv` 轮询互斥，同一 fd 不得存在两个并发 reader。事件经 `on_progress` 传出后由有界 `msg_queue` 排队（约束 9），并按约束 8 批量发送；detail 行 `message` 为 `null`（约束 7）。

- [ ] **Step 4：调优 UI 增加完整日志折叠区**

在 `tuningLog` 旁新增 `tuningFullLog` 与展开/收起按钮。阶段事件继续进入默认调优日志；`training_log.message` 进入默认区，`training_log.detail` 进入完整区。两区均使用 Task 3 安全 DOM 方法，并遵守同一硬性约束（〇节 P1）：`tuningLog` ≤ 500 行、`tuningFullLog` ≤ 2000 行（约束 3/4）、环形缓冲删除最旧（约束 5）、`DocumentFragment` 批量刷新（约束 6）、禁止 `innerHTML +=`（约束 1/2）。

- [ ] **Step 5：增加负向与回归测试**

至少覆盖：dry-run 不创建伪训练日志；keep_params 实训创建日志；非法参数在训练前 fatal 时不创建“已启动”日志；取消后已接收日志仍存在；候选训练失败时 traceback/OOM 在完整日志可查询；`on_progress` 老签名调用仍能工作或所有测试调用已显式兼容。另补约束 11a/11b 的行数上限与耗时非二次增长验证（环形缓冲纯函数经本机 Node 执行，或明确纳入真实浏览器验收）。

- [ ] **Step 6：运行 Task 4 测试**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_executor.py auto_tune\tests\test_tuning_loop.py auto_tune\tests\test_training_log.py auto_tune\tests\test_ui_training_results.py -v -p no:cacheprovider
```

### Task 5：完整回归与 Claude Code 交付报告

**文件：** 不新增业务文件；仅修复本批引入的失败。

- [ ] **Step 1：运行日志与 UI 定向测试**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_training_log.py auto_tune\tests\test_ui_training_results.py auto_tune\tests\test_executor.py auto_tune\tests\test_tuning_loop.py -q -p no:cacheprovider
```

- [ ] **Step 2：运行安全和统一收尾回归**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_audit.py auto_tune\tests\test_guardrails.py auto_tune\tests\test_training_finalizer.py auto_tune\tests\test_upload_security.py -q -p no:cacheprovider
```

- [ ] **Step 3：运行完整套件**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests -q -p no:cacheprovider
```

预期：不得少于当前 160 个既有测试加本批新增测试；除既有 2 条 sklearn PCA warning 外，不得新增 warning、error 或 skipped。

- [ ] **Step 4：静态敏感信息与范围检查**

```powershell
rg -n --hidden -g '!auto_tune/config.yaml' -g '!log/**' -g '!detect/**' -g '!runs/**' "(api[_-]?key|token|password)\s*[:=]\s*['\"][^'\"]+" auto_tune
git status --short
```

若当前目录仍没有 `.git`，交付报告必须写明“无法执行 Git 待提交清单检查”，不得声称 Git 检查通过。

- [ ] **Step 5：向艾卡/Codex提交交付报告，不推送**

报告必须包含：

1. 实际修改文件清单；
2. 每个测试命令的退出码、passed/warnings/skipped 数；
3. 与本计划的任何偏离及原因；
4. 日志文件实际路径与示例事件字段；
5. 遗留风险，尤其是不同 Ultralytics 版本输出格式；
6. 未修改路线图、README、执行版 Word 文档和真实配置的确认；
7. 未推送 GitHub 的确认；
8. 约束 12 三项度量：每秒 SSE 事件数（批量前后对比）、前端日志区 DOM 节点数、训练前后 Python/浏览器内存变化。

---

## 三、Claude Code 完成后的 Codex 独立验收计划

Claude Code 完成后，Codex 必须重新读取实际 diff/文件内容，不能只接受交付报告或复用 Claude 的测试结论。

### A. 代码审查门

- 日志分类器无 FastAPI/UI 依赖，正则和字段集中定义；解析失败安全降级。
- 训练原始输出没有通过 `innerHTML` 注入页面；新增日志文件和接口无路径遍历。
- 训练日志渲染不存在 `innerHTML +=` 拼接；环形缓冲插入/删除 O(1)；默认区 ≤500 行、完整区 ≤2000 行。
- SSE 为批量发送且待发送缓冲有界；同一个 subprocess stdout 只有单一消费者。
- 普通训练和调优确实共用分类/持久化协议，不是复制两套正则。
- 完整日志写入失败不会绕过审计、护栏、预检或最终训练状态。
- `results.csv`、`args.yaml`、审计和 finalizer 仍是事实层；摘要文本不参与 KPI 或 keep/discard 判断。
- 无新增依赖、无真实配置/凭据修改、无数据集/权重/产物进入待提交清单。

任一项不满足，结论为“审查不通过”，先出问题清单，不进行发布。

### B. 自动化测试门

Codex 独立运行 Task 5 的三组 pytest。验收要求：

- 新增日志测试全部通过；
- 当前 160 个基线测试无回归；
- 新增测试数与 Claude 报告一致；
- 只允许既有 2 条 sklearn PCA warning；
- 无意外 skipped/xfailed；
- 约束 11 三组性能测试（行数上限、耗时非二次增长、SSE 队列有界）通过。

### C. 真实浏览器验收门

使用本机最小合法 YOLOv8 Detect 数据集启动 2 epoch 普通训练，再启动一次最小自动调优训练。逐项检查：

1. 默认面板每 epoch 只出现一组关键训练/验证指标，长批次 tqdm 不刷屏；
2. 展开完整日志可看到环境、模型层、批次进度、结果路径等原始行；收起再展开内容仍在；
3. 中文路径、`<tag>`、`&` 等文本按字面显示，不生成 HTML；
4. 停止训练后默认摘要显示真实停止状态，完整日志保留停止前输出；
5. 制造一个可控失败（例如测试替身或无效但能通过请求层的训练条件），错误摘要可见、traceback/错误上下文在完整日志可查；
6. 浏览器开发者网络节流下 SSE 事件不丢失、不重复，完成事件仍能到达；
7. 普通训练和调优页面均满足以上日志分层行为；
8. 批量 SSE 后网络面板 SSE 事件/秒明显低于原始 stdout 行/秒；长训练中默认区 DOM 行数 ≤500、完整区 ≤2000，展开/收起切换后行数仍在上限内，浏览器内存不随训练时长持续线性增长。

浏览器验收必须记录页面截图或逐项文本结果；仅模板字符串测试不能替代真实浏览器检查。

### D. 最小真实训练与产物一致性门

真实训练采用最小合法数据集、2 epoch，避免完整训练。核对：

- `<train_dir>/training.log` 存在、UTF-8 可读、行序与页面完整日志一致；
- `args.yaml`、实际命令、`results.csv`、页面最终 KPI、Module B 报告和统一历史一致；
- 真实 `0` 指标显示为 `0`/`0.0000`，缺失指标显示为缺失标记，两者不混淆；
- 普通训练成功但分析失败时仍显示“训练完成、分析失败”的部分成功语义；
- 非零退出仍为失败，日志分层不得吞掉错误终态。

### E. 最终验收结论格式

Codex 只能给出以下两种结论之一：

- **验收通过：** 代码审查、定向测试、完整测试、真实浏览器、2 epoch 训练和安全/待提交检查全部通过，并列出证据。
- **验收不通过：** 列出可复现问题、影响、证据、建议责任人和复验命令；不得上传版本。

只有“验收通过”后，才进入 `.gitignore`、敏感信息、大文件、待提交文件复核以及由 Codex 执行的独立版本提交/推送流程。

### F. 验收通过后的文档回写与 GitHub 发布

此阶段仅由 Codex 执行，Claude Code 不得代做。执行顺序固定如下：

1. 在 `docs/Auto-Tune后续研发路线与实施评估_研发执行版_20260820.docx` 的 Studio S1 表格中，将 S1.1 从“下一批候选”更新为“已完成”，补充实际完成日期、最终 pytest 数量、真实浏览器结果、最小短 epoch 结果和遗留风险；不得提前修改或只依据 Claude Code 报告标记完成。
2. 同步更新 `docs/implementation_plan_20260814.md` 中 A0.2 第 3 项，以及 `docs/development_handoff_20260814.md` 的当前稳定能力、验证基线和下一开发入口。若 `docs/roadmap_20260814.md` 存在与执行版冲突的 S1.1 状态，只做最小状态同步，不改写 Research/Cloud 方向。
3. 如 Word 文档被修改，必须按 documents 技能执行 DOCX 渲染，并逐页检查 PNG，确认表格、分页、字体和中文无错位后才可提交。
4. 重新运行受文档状态影响的检查，并确认所有文档中的测试数字、日期、状态和遗留风险互相一致。
5. 检查 `.gitignore`、`git status --short`、待提交文件清单和敏感信息；明确排除数据集、图片、标注、模型权重、训练产物、日志、审计运行文件、上传缓存、虚拟环境、构建产物和真实 `auto_tune/config.yaml`。
6. 确认远端唯一为 `https://github.com/luZhaoHao/Auto_labeltrain_project`，且不执行强制推送、历史重写、远端分支删除或仓库可见性变更。
7. 使用一个独立且可解释的 S1.1 版本提交，提交说明同时概括日志分层功能和验证结果；由 Codex 推送到艾卡指定的 GitHub 仓库。
8. 发布后向艾卡报告 commit SHA、分支、远端地址、实际提交文件、最终测试证据和仍存在的风险。

若当前工作目录届时仍不是 Git 仓库、远端不匹配、存在无法判定归属的用户改动、敏感信息或大文件，Codex 必须停止发布并向艾卡报告具体阻塞，不得初始化新仓库、覆盖改动或改推其他远端。

---

## 四、计划自检

- **规格覆盖：** 覆盖执行版 S1.1 的默认 epoch 指标、完整日志折叠、错误栈可查询和 SSE 不回归；未夹带 S1.2 及后续范围。
- **兼容性：** 明确保留 A0 审计、fatal 策略、统一收尾、零值/缺失值、部分成功语义和旧历史可读边界。
- **类型一致：** `TrainingLogEvent`、payload 字段、普通训练、调优与 UI 使用同一命名。
- **无占位项：** 每个实现任务均给出文件、接口、失败测试、运行命令和可判断的完成条件。
- **性能可度量：** 硬性性能约束（〇节）均有测试或浏览器验收证据，交付报告含 SSE 事件数、DOM 节点数、内存变化三项度量。
- **风险聚焦：** 已将服务端结构化协议、SSE chunk 缓冲、日志注入、持久化降级和 Ultralytics 输出差异纳入测试。
