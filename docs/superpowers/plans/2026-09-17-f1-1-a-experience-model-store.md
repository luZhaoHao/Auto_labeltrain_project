# F1.1-A Experience Optimization and Controlled Model Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复大模型调优模式提交阻断，建立安全的受控权重库，让直接训练和 HPO 可选择初始权重，并用持久事实丰富 HPO 动态进度。

**Architecture:** 新增独立 `ModelStore` 作为权重列举、上传、身份解析和提交前复核的唯一入口；UI 只传不可伪造路径的 `model_id`，训练边界再解析成文件路径。HPO 状态继续由既有 `/api/hpo/studies/{study_id}` 轮询提供，服务端从已持久化 trial/attempt 事实形成状态轨道和最近事件，前端不新增计时器。LLM 调优不新增权重字段，始终继承参考运行的 `args.yaml`。

**Tech Stack:** Python 3.10、FastAPI、Pydantic、Jinja2、原生 JavaScript、pytest、Node `vm`/minidom。

**Spec:** `docs/f1_1_a_experience_model_store_spec_20260917.md`

## Global Constraints

- Claude Code 只修改业务代码和对应测试；不得修改本计划、规格、README、路线图、交接记录或其他项目文档。
- 不新增依赖、不删除文件、不提交、不推送；每项完成后只报告检查点并保持改动未提交。
- 所有 Python 命令必须使用 `D:\Program Files\anaconda3\envs\auto_tune\python.exe`。
- 每项先运行指定 RED 测试并保留真实失败，再做最小实现转 GREEN；不得先改业务代码再补测试。
- 不回退或覆盖工作区既有改动。若同一文件存在不明改动，先核对差异再编辑。
- 权重上传不反序列化 `.pt`、不访问网络、不自动下载模型、不扫描训练产物目录。
- LLM 调优不得出现权重选择器、`model_id` 请求字段或客户端模型覆盖；只继承参考运行权重。
- HPO 不新增 loss/mAP 曲线，也不新增 GPU、显存、利用率、温度等资源图表。
- 自动化通过不等于浏览器或真实 GPU 验收通过；真实复验由 Codex 独立执行。

---

## Task 1: 修复模式表单关联并把主操作放回对应配置区

**Files:**

- Modify: `auto_tune/ui/templates/single_page.html`
- Modify: `auto_tune/ui/static/hpo.js`
- Modify: `auto_tune/tests/js/minidom.js`
- Modify: `auto_tune/tests/test_hpo_ui_behaviour.py`
- Modify: `auto_tune/tests/test_hpo_ui_rework.py`

- [ ] **Step 1: 写模板契约 RED 测试**

在 `test_hpo_ui_rework.py` 增加断言：

```python
def test_tuning_mode_is_explicitly_associated_with_the_shared_form(page_html):
    select = _element(page_html, "tuningModeSelect")
    assert select["name"] == "mode"
    assert select["form"] == "tuningForm"


def test_primary_action_hosts_are_inside_their_visible_configuration_areas(page_html):
    assert _is_descendant(page_html, "startTuningBtn", "llmPrimaryActionHost")
    assert _is_descendant(page_html, "hpoCreateAndStartBtn", "hpoPrimaryActionHost")
    assert page_html.count('id="startTuningBtn"') == 1
    assert page_html.count('id="hpoCreateAndStartBtn"') == 1
```

扩充 minidom 场景，真实执行 `new FormData(tuningForm)` 后依次切换四种模式并记录请求：

```javascript
result.modeValues = ['dry_run', 'keep_params', 'hpo', 'full'].map(function (mode) {
  document.getElementById('tuningModeSelect').value = mode;
  return new FormData(document.getElementById('tuningForm')).get('mode');
});
```

行为断言固定为：

```python
assert result["modeValues"] == ["dry_run", "keep_params", "hpo", "full"]
assert result["llmRequests"] == ["dry_run", "keep_params", "full"]
assert result["hpoCreateCount"] == 1
assert result["llmNullModes"] == 0
assert result["duplicateClicksWhilePending"] == 0
```

- [ ] **Step 2: 运行 RED**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_hpo_ui_rework.py auto_tune\tests\test_hpo_ui_behaviour.py -k "form or primary_action or mode_submission" -q -p no:cacheprovider
```

预期至少出现：`KeyError: 'form'` 或 `assert None == 'tuningForm'`，并且按钮宿主断言失败。

- [ ] **Step 3: 最小修复模板与按钮宿主**

给模式选择器增加显式表单关联：

```html
<select name="mode" id="tuningModeSelect" form="tuningForm"
        onchange="window.onTuningModeChange && window.onTuningModeChange(this.value)">
```

在各配置区放置唯一宿主，不复制按钮：

```html
<div id="llmPrimaryActionHost" class="non-hpo-only" style="margin-top:12px;"></div>
<div id="hpoPrimaryActionHost" class="hpo-only hpo-mode hidden" style="margin-top:12px;"></div>
```

`startTuningBtn` 初始放入 `llmPrimaryActionHost`，位置在评估模式之后；`hpoCreateAndStartBtn` 初始放入 `hpoPrimaryActionHost`，位置在 HPO 草稿底部、当前任务之前。`dry_run`、`keep_params`、`full` 继续复用同一个 `startTuningBtn`；`hpo` 只使用 `hpoCreateAndStartBtn`。

在模式切换函数中只切换可见性，不克隆节点：

```javascript
var hpoMode = mode === 'hpo';
setHidden(document.getElementById('startTuningBtn'), hpoMode);
setHidden(document.getElementById('hpoCreateAndStartBtn'), !hpoMode);
```

保留后端 `/tuning/start` 的严格白名单，不增加 `None` 默认映射。

- [ ] **Step 4: 运行 GREEN 与静态检查**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_hpo_ui_rework.py auto_tune\tests\test_hpo_ui_behaviour.py -q -p no:cacheprovider
node --check auto_tune/ui/static/hpo.js
node --check auto_tune/tests/js/minidom.js
```

- [ ] **Step 5: 检查点报告（不得提交）**

列出 RED 失败、GREEN 结果、实际修改文件和按钮 DOM 位置；执行 `git status --short`，不得运行 `git commit` 或 `git push`。

---

## Task 2: 实现受控权重库核心服务

**Files:**

- Create: `auto_tune/modules/model_store/__init__.py`
- Create: `auto_tune/modules/model_store/service.py`
- Create: `auto_tune/tests/test_model_store.py`
- Modify: `auto_tune/config.template.yaml`
- Modify: `.gitignore`

- [ ] **Step 1: 写核心服务 RED 测试**

测试必须覆盖合法上传、同名同哈希幂等、同名异内容冲突、大小上限、空文件、非法文件名、错误扩展名、符号链接/reparse、并发同名上传、写盘失败清理、文件被替换后的解析拒绝，以及项目根遗留 `.pt` 只读兼容。

固定公共接口：

```python
from auto_tune.modules.model_store import ModelStore, ModelStoreError

store = ModelStore(
    root=tmp_path / "models" / "weights",
    legacy_roots=[tmp_path],
    max_upload_bytes=1024,
    max_models=100,
)
record = store.import_stream("yolov8n.pt", io.BytesIO(b"weights"))
assert record.model_id == "sha256:" + hashlib.sha256(b"weights").hexdigest()
assert store.resolve(record.model_id) == record.path
```

公开投影必须精确为：

```python
assert record.public_dict() == {
    "model_id": record.model_id,
    "name": "yolov8n.pt",
    "size_bytes": 7,
    "sha256": hashlib.sha256(b"weights").hexdigest(),
    "origin": "managed",
    "available": True,
}
```

- [ ] **Step 2: 运行 RED**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_model_store.py -q -p no:cacheprovider
```

预期失败：`ModuleNotFoundError: No module named 'auto_tune.modules.model_store'`。

- [ ] **Step 3: 实现数据类型与稳定错误**

`service.py` 使用以下接口，不暴露路径到 JSON：

```python
@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    name: str
    size_bytes: int
    sha256: str
    origin: Literal["managed", "legacy"]
    path: Path = field(repr=False, compare=False)
    available: bool = True

    def public_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "name": self.name,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "origin": self.origin,
            "available": self.available,
        }


class ModelStoreError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
```

稳定错误码固定为：

```text
MODEL_UPLOAD_INVALID_NAME
MODEL_UPLOAD_INVALID_TYPE
MODEL_UPLOAD_EMPTY
MODEL_UPLOAD_TOO_LARGE
MODEL_NAME_CONFLICT
MODEL_STORE_UNAVAILABLE
MODEL_NOT_FOUND
MODEL_CHANGED
MODEL_PATH_UNSAFE
MODEL_PATH_FORBIDDEN
```

- [ ] **Step 4: 实现安全上传、列表与解析**

公共方法固定为：

```python
class ModelStore:
    def __init__(self, root: Path, *, legacy_roots: Sequence[Path] = (),
                 max_upload_bytes: int = 2_147_483_648,
                 max_models: int = 100): ...

    def list_models(self) -> list[ModelRecord]: ...
    def import_stream(self, filename: str, stream: BinaryIO) -> ModelRecord: ...
    def resolve(self, model_id: str) -> Path: ...
```

实现约束：

```python
name = Path(filename).name
if name != filename or not name or name in {".", ".."}:
    raise ModelStoreError("MODEL_UPLOAD_INVALID_NAME", "权重文件名不合法。")
if Path(name).suffix.lower() != ".pt":
    raise ModelStoreError("MODEL_UPLOAD_INVALID_TYPE", "只允许上传 .pt 权重文件。")
```

流式写入使用 1 MiB 块；累计字节超过上限立即报错；写入同目录唯一临时文件，完成后 `flush()`、`os.fsync()`。Windows/NTFS 上用 `os.link(temp_path, target_path)` 原子创建最终名称：目标已存在时该调用必须失败而不能覆盖，再重新计算目标哈希决定“同内容幂等”或 `MODEL_NAME_CONFLICT`；链接成功后删除临时名称。任何异常都删除本次临时文件，不使用存在检查后 `os.replace()` 这种会在竞争窗口覆盖文件的实现。

`model_id` 固定为 `sha256:<64位小写十六进制>`。服务实例维护由 `list_models()`/`import_stream()` 填充的受控 `_known` 映射（只存模型 ID、名称、路径和当时哈希，不写业务索引文件）；`resolve()` 根据该映射重新核对普通文件事实与 SHA-256。从未列举或导入的 ID 报 `MODEL_NOT_FOUND`；已知文件被删除报 `MODEL_NOT_FOUND`；已知路径仍存在但内容哈希失配报 `MODEL_CHANGED`。服务重启后页面必须先重新获取列表，客户端不能凭旧页面 ID 直接绕过当前事实建立选择。

列表扫描规则：

- `models/weights/` 非递归扫描 `.pt`；
- 项目根遗留目录非递归扫描 `.pt`；
- 最多返回 `max_models` 条，按 `name.lower(), sha256` 稳定排序；
- 不扫描 `detect/`、`trainN/weights/` 或任意递归子目录；
- 拒绝链接、junction/reparse point 和非普通文件；
- 不调用 `torch.load`、`pickle`、Ultralytics 或任何网络函数。

- [ ] **Step 5: 配置默认值与 Git 忽略**

在 `config.template.yaml` 增加：

```yaml
model_store:
  root: models/weights
  max_upload_bytes: 2147483648
  max_models: 100
```

在 `.gitignore` 增加目录级保护：

```gitignore
models/weights/
```

不创建或提交任何二进制权重。

- [ ] **Step 6: 运行 GREEN**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_model_store.py -q -p no:cacheprovider
```

- [ ] **Step 7: 检查点报告（不得提交）**

报告所有错误码覆盖、临时文件残留检查和并发测试结果；不得提交或推送。

---

## Task 3: 增加模型库 API 并完成安全投影

**Files:**

- Create: `auto_tune/ui/model_store_api.py`
- Create: `auto_tune/tests/test_model_store_api.py`
- Modify: `auto_tune/ui/app.py`

- [ ] **Step 1: 写 API RED 测试**

使用 `TestClient` 覆盖：

```python
response = client.get("/api/models")
assert response.status_code == 200
assert set(response.json()["models"][0]) == {
    "model_id", "name", "size_bytes", "sha256", "origin", "available"
}
assert "path" not in response.text.lower()

response = client.post(
    "/api/models/upload",
    files={"file": ("custom.pt", b"weights", "application/octet-stream")},
    headers={"Origin": "http://testserver", "X-CSRF-Token": csrf_token},
)
assert response.status_code == 201
assert response.json()["model"]["name"] == "custom.pt"
```

还要断言无 CSRF、跨源、非法类型、超限、冲突和存储不可用返回稳定码；响应不含绝对路径、堆栈或异常原文。用 monkeypatch 证明上传不会调用 `torch.load`、网络客户端或 Ultralytics 模型构造器。

- [ ] **Step 2: 运行 RED**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_model_store_api.py -q -p no:cacheprovider
```

预期失败：`404 Not Found` 或 `ModuleNotFoundError`。

- [ ] **Step 3: 实现路由工厂**

`model_store_api.py` 暴露：

```python
def create_model_store_router(*, store: ModelStore,
                              require_security: Callable[[Request], None]) -> APIRouter:
    router = APIRouter(prefix="/api/models", tags=["models"])

    @router.get("")
    def list_models():
        return {"models": [row.public_dict() for row in store.list_models()]}

    @router.post("/upload", status_code=201)
    async def upload_model(request: Request, file: UploadFile = File(...)):
        require_security(request)
        record = await run_in_threadpool(store.import_stream, file.filename or "", file.file)
        return {"status": "created", "model": record.public_dict()}

    return router
```

捕获 `ModelStoreError` 并返回：

```python
JSONResponse(
    {"error_code": exc.code, "error": exc.message},
    status_code=exc.status_code,
)
```

不要返回 `repr(exc)`、完整路径或 traceback。

- [ ] **Step 4: 在 app.py 中建立唯一服务实例并注册**

从 `APP_CONFIG["model_store"]` 读取根、大小和数量，根路径相对项目工作目录解析；遗留根只传项目根，不传 Detect 目录。注册 `create_model_store_router(...)`，复用现有 `_require_security`。

保留 `list_local_models()` 名称作为短期兼容包装，但其实现只能调用同一 `ModelStore.list_models()`，不得继续自行扫描训练产物：

```python
def list_local_models() -> list[dict]:
    return [row.public_dict() for row in _MODEL_STORE.list_models()]
```

- [ ] **Step 5: 运行 GREEN**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_model_store.py auto_tune\tests\test_model_store_api.py auto_tune\tests\test_upload_security.py -q -p no:cacheprovider
```

- [ ] **Step 6: 检查点报告（不得提交）**

报告 API 状态码、CSRF/同源检查、路径泄露断言和“不反序列化/不联网”证据；不得提交或推送。

---

## Task 4: 将受控权重接入直接训练和 HPO，冻结 LLM 继承规则

**Files:**

- Modify: `auto_tune/ui/app.py`
- Modify: `auto_tune/ui/hpo_api.py`
- Modify: `auto_tune/ui/static/hpo.js`
- Modify: `auto_tune/ui/templates/single_page.html`
- Modify: `auto_tune/tests/js/minidom.js`
- Modify: `auto_tune/tests/test_hpo_api.py`
- Modify: `auto_tune/tests/test_hpo_ui_behaviour.py`
- Modify: `auto_tune/tests/test_ui_training_results.py`
- Modify: `auto_tune/tests/test_tuning_loop.py`

- [ ] **Step 1: 写三路线契约 RED 测试**

直接训练测试发送：

```python
payload = {
    "model_id": uploaded["model_id"],
    "epochs": 1,
    "imgsz": 96,
    "batch": 1,
}
```

并断言控制器参数中的 `model` 是服务端解析的受控路径，而响应不回显路径。上传后替换文件，再提交必须返回 `MODEL_CHANGED` 且没有创建 run 目录、状态文件或子进程。

直接向 `/api/training/start` 发送客户端 `model` 或绝对路径必须返回 `MODEL_PATH_FORBIDDEN`（422）；不带 `model_id` 的默认训练只能按配置中的 basename 唯一解析当前受控库/项目根遗留文件，文件不存在时不得联网下载。

HPO 创建请求改为：

```python
{
    "snapshot_id": snapshot_id,
    "model_id": uploaded["model_id"],
    "study_config": {...},
    "execution_config": {...},
}
```

断言 study 仍冻结规范化路径、字节数、mtime 与 SHA-256；客户端发送 `model_path` 必须因 `extra="forbid"` 返回 422。

LLM 测试断言：

```python
assert "model_id" not in submitted_body
assert "model" not in submitted_body
assert executed_params["model"] == reference_args["model"]
```

即使恶意请求向 `/tuning/start` 加入 `model_id` 或 `model`，业务也不得用它覆盖参考运行模型；为避免误导，返回 `LLM_MODEL_OVERRIDE_FORBIDDEN`（422），而不是静默忽略。

- [ ] **Step 2: 运行 RED**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_hpo_api.py auto_tune\tests\test_hpo_ui_behaviour.py auto_tune\tests\test_ui_training_results.py auto_tune\tests\test_tuning_loop.py -k "model_id or model_store or inherited_model or override_forbidden" -q -p no:cacheprovider
```

预期失败包括 HPO 请求缺少旧 `model_path`、直接训练未识别 `model_id`、LLM 覆盖请求未被明确拒绝。

- [ ] **Step 3: 修改服务端提交边界**

HPO 请求模型改为：

```python
class CreateStudyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    snapshot_id: str = Field(min_length=1, max_length=128)
    model_id: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    study_config: CreateStudyConfig = Field(default_factory=CreateStudyConfig)
    execution_config: ExecutionConfig = Field(default_factory=ExecutionConfig)
```

路由工厂新增 `resolve_model: Callable[[str], Path]` 注入；创建时执行：

```python
model_path = resolve_model(payload.model_id)
study = service.create_study(
    payload.study_config,
    snapshot_dir=snapshot_dir,
    model_path=model_path,
)
```

直接训练规则：

```python
model_id = body.get("model_id")
if model_id:
    model = str(_MODEL_STORE.resolve(model_id))
else:
    if "model" in body:
        return JSONResponse(
            {"error_code": "MODEL_PATH_FORBIDDEN",
             "error": "请从受控权重库选择初始权重。"},
            status_code=422,
        )
    configured = project_cfg.get("model") or training_cfg.get("model", "yolov8n.pt")
    model = str(_MODEL_STORE.resolve_configured_name(configured))
```

`resolve_configured_name()` 只允许从当前受控库或项目根遗留 `.pt` 中按完整 basename 唯一匹配；不存在时返回 `MODEL_NOT_FOUND`，不得触发 Ultralytics 隐式下载。客户端任意 `model`/路径字段不再作为训练输入；`source_hpo` 正式验证分支仍先按既有权威 HPO 绑定处理。解析必须发生在创建 run 目录、状态文件和控制器之前。

`/tuning/start` 在读取 reference run 前增加：

```python
if "model_id" in body or "model" in body:
    return JSONResponse(
        {"error_code": "LLM_MODEL_OVERRIDE_FORBIDDEN",
         "error": "大模型调优必须继承参考运行的初始权重。"},
        status_code=422,
    )
```

- [ ] **Step 4: 修改 UI 权重选择与上传**

直接训练把 `trainModel` 自由文本改为 `trainModelSelect`；HPO 保留 `hpoModelSelect`。两者复用 `GET /api/models` 的安全投影，option 的 value 只能是 `model_id`：

```javascript
option.value = row.model_id;
option.textContent = row.name + ' · ' + formatBytes(row.size_bytes) +
  (row.origin === 'legacy' ? ' · 兼容来源' : ' · 权重库');
```

新增一个文件控件和上传按钮，放在权重选择器下方；上传成功后重新刷新两个选择器，并选中新返回的 `model_id`。请求使用 `FormData`，只设置 CSRF header，不手写 multipart content-type。上传 pending 时禁用按钮，重复点击只发一次；失败不清空既有合法选项。

HPO `createStudy()` 请求字段从 `model_path` 改为 `model_id`。直接训练 `startTraining()` 从选择器提交 `model_id`。LLM 配置区不得出现模型选择器或上传后自动改写参考运行。

- [ ] **Step 5: 保留旧研究读取兼容**

旧 HPO study 的 `model_binding.model_path` 和哈希继续由 `HpoService.validate_binding()` 校验；不要迁移、改写或按当前模型库重新绑定旧记录。新创建研究才从 `model_id` 解析后冻结旧有 `ModelBinding` 字段。

- [ ] **Step 6: 运行 GREEN**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_model_store.py auto_tune\tests\test_model_store_api.py auto_tune\tests\test_hpo_api.py auto_tune\tests\test_hpo_ui_behaviour.py auto_tune\tests\test_ui_training_results.py auto_tune\tests\test_tuning_loop.py -q -p no:cacheprovider
node --check auto_tune/ui/static/hpo.js
node --check auto_tune/tests/js/minidom.js
```

- [ ] **Step 7: 检查点报告（不得提交）**

分别报告直接训练、HPO、LLM 的实际请求字段和解析结果，明确 LLM 没有模型覆盖入口；不得提交或推送。

---

## Task 5: 用持久事实丰富 HPO 动态进度

**Files:**

- Modify: `auto_tune/ui/hpo_api.py`
- Modify: `auto_tune/ui/static/hpo.js`
- Modify: `auto_tune/ui/templates/single_page.html`
- Modify: `auto_tune/tests/js/minidom.js`
- Modify: `auto_tune/tests/test_hpo_api.py`
- Modify: `auto_tune/tests/test_hpo_ui_behaviour.py`
- Modify: `auto_tune/tests/test_hpo_ui_rework.py`

- [ ] **Step 1: 写服务端投影 RED 测试**

在状态响应中固定增加：

```python
assert body["trial_states"] == [
    {"trial_number": 1, "state": "SUCCESS"},
    {"trial_number": 2, "state": "RUNNING"},
    {"trial_number": 3, "state": "WAITING"},
]
assert len(body["recent_events"]) <= 3
assert set(body["recent_events"][0]) == {
    "event_id", "kind", "trial_number", "state", "message"
}
```

状态映射固定为：trial `SUCCESS/FAILED/CANCELLED/INTERRUPTED/PENDING` 保持原值；当前未终态 attempt 对应的 trial 显示 `RUNNING`；预算内尚未创建的槽位显示 `WAITING`。

最近事件只从持久事实确定性重建，按以下优先级和顺序生成后取最后三条：

1. trial 终态：`trial:<number>:<state>`；
2. 当前运行 trial：`trial:<number>:RUNNING`；
3. 当前最佳变化：`best:<trial_number>:<value>`；
4. 研究终态：`study:<execution_revision>:<execution_status>`。

不得使用当前时间生成 ID，也不得把浏览器已显示内容回传服务端。

- [ ] **Step 2: 写前端 RED 场景**

minidom 场景必须验证：

```python
assert running["rail"] == ["成功", "运行中", "等待", "等待"]
assert completed["rail"] == ["成功", "失败", "成功", "成功"]
assert len(completed["events"]) == 3
assert repeated["events"] == completed["events"]
assert stale_reply["rail"] == completed["rail"]
assert result["intervalCount"] == 1
```

另测 READY、PAUSED、INTERRUPTED、BLOCKED、COMPLETED、FAILED 的中文状态和轨道 class；刷新后第一次权威响应可完整重建，不依赖先前内存。

- [ ] **Step 3: 运行 RED**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_hpo_api.py auto_tune\tests\test_hpo_ui_behaviour.py auto_tune\tests\test_hpo_ui_rework.py -k "trial_states or recent_events or progress_rail" -q -p no:cacheprovider
```

预期失败：状态响应缺少 `trial_states`/`recent_events`，模板缺少轨道和事件容器。

- [ ] **Step 4: 实现服务端纯投影函数**

在 `hpo_api.py` 增加：

```python
def _trial_state_projection(study, attempts, budget: int) -> list[dict]: ...


def _recent_event_projection(study, execution_status: str,
                             execution_revision: int, attempts,
                             ranked) -> list[dict]: ...
```

函数只读，不写 storage，不启动/停止任务。`_build_status_payload()` 复用已读取的 `study`、`attempts`、`ranked`，增加：

```python
"trial_states": _trial_state_projection(study, attempts, study.config.budget),
"recent_events": _recent_event_projection(
    study, execution_status, execution_revision, attempts, ranked
),
```

事件 `message` 使用固定中文模板，不包含路径、命令或异常原文。没有事实时返回空数组，不伪造“已开始”。

- [ ] **Step 5: 实现模板和前端渲染**

在现有 `hpoProgress` 内、进度条下增加：

```html
<div id="hpoTrialRail" class="hpo-trial-rail" aria-label="试验状态轨道"></div>
<div id="hpoRecentEvents" class="hpo-recent-events" aria-live="polite"></div>
```

`renderProgress(body, counts)` 每次从响应整体重建两处 DOM，不采用 append-only 内存队列：

```javascript
renderTrialRail(body.trial_states || []);
renderRecentEvents(body.recent_events || []);
```

轨道每格显示序号与状态，class 只允许 `waiting/running/success/failed/cancelled/interrupted`。事件使用 `textContent`，最多显示响应中的三条。既有 generation/study ID 保护继续负责丢弃旧研究和乱序响应；不得创建新的 `setInterval`、`setTimeout` 或 SSE。

- [ ] **Step 6: 运行 GREEN**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_hpo_api.py auto_tune\tests\test_hpo_ui_behaviour.py auto_tune\tests\test_hpo_ui_rework.py auto_tune\tests\test_hpo_ui_lifecycle.py -q -p no:cacheprovider
node --check auto_tune/ui/static/hpo.js
node --check auto_tune/tests/js/minidom.js
```

- [ ] **Step 7: 检查点报告（不得提交）**

报告七种研究状态、刷新恢复、重复/乱序幂等和定时器数量；不得提交或推送。

---

## Task 6: 集成回归与 Claude Code 交付报告

**Files:**

- Verify all files changed in Tasks 1-5
- Do not modify project documentation

- [ ] **Step 1: 检查解释器**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -c "import sys; print(sys.executable); print(sys.version)"
```

必须显示 `D:\Program Files\anaconda3\envs\auto_tune\python.exe` 和 Python 3.10.x。

- [ ] **Step 2: 运行定向测试组**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_model_store.py auto_tune\tests\test_model_store_api.py auto_tune\tests\test_upload_security.py -q -p no:cacheprovider
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_hpo_api.py auto_tune\tests\test_hpo_ui.py auto_tune\tests\test_hpo_ui_behaviour.py auto_tune\tests\test_hpo_ui_rework.py auto_tune\tests\test_hpo_ui_lifecycle.py auto_tune\tests\test_hpo_formal_training.py -q -p no:cacheprovider
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_ui_training_results.py auto_tune\tests\test_run_manager_reconnect.py -q -p no:cacheprovider
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_tuning_loop.py auto_tune\tests\test_decision_agent.py auto_tune\tests\test_guardrails.py -q -p no:cacheprovider
```

- [ ] **Step 3: 运行完整套件和静态检查**

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests -q -p no:cacheprovider
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pip check
node --check auto_tune/ui/static/hpo.js
node --check auto_tune/tests/js/minidom.js
```

- [ ] **Step 4: 检查渲染契约和文件卫生**

用现有 TestClient 渲染真实首页，检查：

- `FormData(tuningForm)` 四种 mode 均为完整值；
- `startTuningBtn`、`hpoCreateAndStartBtn`、`hpoTrialRail`、`hpoRecentEvents` 各只有一个 ID；
- LLM 配置区没有权重选择器；
- HPO 和直接训练选择器 value 为 `sha256:<digest>`，页面不出现绝对模型路径；
- HTML 的 `div/form/details/select` 平衡。

检查本批实际修改文件：

```powershell
git diff --check
rg -n "<<<<<<<|=======|>>>>>>>" auto_tune/modules/model_store auto_tune/ui/model_store_api.py auto_tune/ui/app.py auto_tune/ui/hpo_api.py auto_tune/ui/static/hpo.js auto_tune/ui/templates/single_page.html auto_tune/tests/test_model_store.py auto_tune/tests/test_model_store_api.py auto_tune/tests/test_hpo_api.py auto_tune/tests/test_hpo_ui_behaviour.py auto_tune/tests/test_hpo_ui_rework.py auto_tune/tests/test_ui_training_results.py auto_tune/tests/js/minidom.js
```

如全局 `git diff --check` 命中既有无关文件，单独列出并证明不属于本批；不要擅自修复无关改动。

- [ ] **Step 5: 输出最终交付报告并停止**

报告必须包含：

1. 每项根因；
2. 实际修改文件；
3. 每项真实 RED→GREEN 证据；
4. 精确测试命令、通过数、warnings、skips 和耗时；
5. 三路线权重规则的最终事实；
6. 安全边界和稳定错误码；
7. 偏离计划之处；
8. 剩余风险；
9. 完整 `git status --short`，区分本批与既有脏工作区；
10. 明确声明未新增依赖、未删除文件、未修改项目文档、未真实浏览器/GPU验收、未提交、未推送。

不得声称 F1.1-A 已验收；等待 Codex 独立代码审查、浏览器检查和必要的短 GPU 验证。

- [ ] **Step 6: 最终检查点（不得提交）**

不得运行 `git commit` 或 `git push`。将交付报告交给艾卡/Codex 后停止。
