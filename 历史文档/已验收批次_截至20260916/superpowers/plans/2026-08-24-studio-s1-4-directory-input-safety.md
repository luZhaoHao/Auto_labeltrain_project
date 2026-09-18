# Studio S1.4 Directory Input Safety Implementation Plan

> **完成状态（2026-08-24）：** Task 1–8 已完成并通过 Codex 独立验收。定向套件 176 passed；完整套件 460 passed、2 warnings、0 skipped；Chromium 验证不存在/越界路径、成员超限、容量超限、浏览错误、按钮恢复和稳定错误码。验收追加了两个分析入口显示 `error_code` 的 TDD 修复。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在数据集或训练结果进入递归处理前实施统一、有界的目录安全预检，并停用遗留 ZIP/JSON 上传 API。

**Architecture:** 新建无第三方依赖的 `input_safety` 领域模块，集中解析策略、校验路径、枚举安全子目录和执行有界扫描。FastAPI 路由只调用公共接口并映射稳定错误，S1.2 快照及 Module A/B 保持原有职责。

**Tech Stack:** Python 3.10、FastAPI、Jinja2、pytest、标准库 `pathlib/os/stat`。

**Spec:** `docs/superpowers/specs/2026-08-24-studio-s1-4-directory-input-safety-design.md`

## Global Constraints

- Python/pytest 固定使用 `D:\Program Files\anaconda3\envs\auto_tune\python.exe`。
- 不新增依赖、不删除文件、不提交、不推送。
- Claude Code 不修改 README、路线图、交接记录、规格、计划或 DOCX。
- ZIP/JSON 上传只改为 `410 Gone`，不实现安全解压，也不永久删除遗留实现。
- 不修改 S1.2 Schema、A0 审计、训练命令、指标口径或历史兼容语义。
- 所有目录扫描必须有界并提前终止。

---

### Task 1: 领域模型与严格配置

**Files:**
- Create: `auto_tune/modules/input_safety/__init__.py`
- Create: `auto_tune/modules/input_safety/models.py`
- Create: `auto_tune/modules/input_safety/service.py`
- Create: `auto_tune/tests/test_input_safety.py`
- Modify: `auto_tune/config.template.yaml`

**Interfaces:**
- Produces: `InputSafetyPolicy`、`DirectoryScanResult`、`InputSafetyError`、`load_input_safety_policy(config)`。
- Consumes: 顶层配置映射的 `input_safety` 段。

- [ ] **Step 1: 写失败测试**：覆盖默认值、合法字段、布尔值冒充整数、零值、超上限、非绝对允许根和未知字段。
- [ ] **Step 2: 运行并确认失败**：

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_input_safety.py -v -p no:cacheprovider
```

- [ ] **Step 3: 实现冻结模型、规格中的稳定错误码及严格配置解析**。
- [ ] **Step 4: 在 `config.template.yaml` 加入四个字段及字节单位注释，不写本机业务路径**。
- [ ] **Step 5: 重跑 `test_input_safety.py`，确认 Task 1 测试通过**。

### Task 2: 路径校验与允许根

**Files:**
- Modify: `auto_tune/modules/input_safety/service.py`
- Modify: `auto_tune/tests/test_input_safety.py`

**Interfaces:**
- Produces: `validate_directory_path(path, policy) -> Path`。
- Consumes: Task 1 的 `InputSafetyPolicy`。

- [ ] **Step 1: 写失败测试**：空值、相对路径、不存在、普通文件、允许根内外、Windows 大小写、UNC 默认拒绝、符号链接、junction/reparse point、权限错误。
- [ ] **Step 2: 运行定向测试并确认失败**：

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_input_safety.py -k "path or root or link or permission" -v -p no:cacheprovider
```

- [ ] **Step 3: 使用不跟随链接的 stat 和路径组件比较实现校验**；禁止字符串前缀判断允许根。
- [ ] **Step 4: 重跑定向测试并确认通过**。

### Task 3: 有界扫描与安全子目录枚举

**Files:**
- Modify: `auto_tune/modules/input_safety/service.py`
- Modify: `auto_tune/tests/test_input_safety.py`

**Interfaces:**
- Produces: `scan_directory_bounded(path, policy) -> DirectoryScanResult`、`list_safe_subdirectories(path, policy) -> tuple[Path, ...]`。

- [ ] **Step 1: 写失败测试**：精确等于上限、超过一个成员/字节、嵌套累计、链接、成员消失/变型、权限变化、超限后不访问陷阱目录。
- [ ] **Step 2: 运行定向测试并确认失败**：

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_input_safety.py -k "scan or member or size or changed" -v -p no:cacheprovider
```

- [ ] **Step 3: 用迭代式 `os.scandir()` 实现确定性扫描**；逐项计数、累计并早停，只返回计数事实。
- [ ] **Step 4: 实现安全直接子目录枚举**；权限和链接错误不得伪装为空目录。
- [ ] **Step 5: 运行完整 `test_input_safety.py` 并确认通过**。

### Task 4: 停用遗留上传 API

**Files:**
- Modify: `auto_tune/ui/app.py`
- Modify: `auto_tune/tests/test_upload_security.py`

**Interfaces:**
- Produces: `/api/dataset/upload` 与 `/api/training/analyze` 的 `410 LEGACY_UPLOAD_DISABLED` 契约。

- [ ] **Step 1: 写两个 410 API 测试**；断言统一错误码和目录提示，并以读取即抛错的伪上传对象证明未读取文件体。
- [ ] **Step 2: 运行并确认当前实现失败**：

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_upload_security.py -v -p no:cacheprovider
```

- [ ] **Step 3: 将两个路由改为固定 410**；不删除 `_safe_extract_zip()`、`_analyze_train_zip()`、旧模板或翻译。
- [ ] **Step 4: 重跑测试并确认通过**。

### Task 5: 目录浏览 API 接入

**Files:**
- Modify: `auto_tune/ui/app.py`
- Create: `auto_tune/tests/test_input_safety_api.py`

**Interfaces:**
- Consumes: 策略加载、路径校验和安全子目录枚举接口。
- Produces: `/api/browse-folder` 的稳定安全响应。

- [ ] **Step 1: 写失败测试**：允许根列表、合法进入、越界 403、链接 400、权限 403、非法策略 500、错误无堆栈。
- [ ] **Step 2: 运行并确认失败**：

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_input_safety_api.py -k browse -v -p no:cacheprovider
```

- [ ] **Step 3: 重构 `/api/browse-folder`**，移除直接 `os.listdir()` 及吞掉权限错误的行为。
- [ ] **Step 4: 重跑浏览 API 测试并确认通过**。

### Task 6: 数据集与训练结果目录门禁

**Files:**
- Modify: `auto_tune/ui/app.py`
- Modify: `auto_tune/tests/test_input_safety_api.py`
- Modify: `auto_tune/tests/test_dataset_snapshot_api.py`

**Interfaces:**
- Consumes: `validate_directory_path`、`scan_directory_bounded`。
- Produces: 正式目录入口的分析前门禁与 `input_scan` 响应事实。

- [ ] **Step 1: 写“分析前阻断”失败测试**：两个目录入口分别覆盖成员、容量、越界、链接和策略错误；分析桩必须保持零调用。
- [ ] **Step 2: 写状态不变测试**：失败时 `latest_dataset.json`、报告和统一历史字节不变。
- [ ] **Step 3: 运行并确认失败**：

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_input_safety_api.py auto_tune\tests\test_dataset_snapshot_api.py -v -p no:cacheprovider
```

- [ ] **Step 4: 在两个路由最前端接入预检**；必须早于 `rglob()`、文件读取、分析调用和状态写入。
- [ ] **Step 5: 成功响应加入 `member_count` 与 `total_bytes`，重跑 Task 6 测试**。

### Task 7: UI 错误反馈

**Files:**
- Modify: `auto_tune/ui/templates/single_page.html`
- Modify: `auto_tune/ui/i18n.py`
- Modify: `auto_tune/tests/test_ui_training_results.py`

**Interfaces:**
- Consumes: API 的稳定 `error` 与 `error_code`。

- [ ] **Step 1: 写模板失败测试**：安全错误使用 `textContent`，成功/失败后按钮恢复，页面不重新出现 ZIP 控件。
- [ ] **Step 2: 运行并确认失败**：

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_ui_training_results.py -k "input or directory or upload" -v -p no:cacheprovider
```

- [ ] **Step 3: 实现最小 UI 文案和错误处理**；不得用 `innerHTML` 显示服务端错误。
- [ ] **Step 4: 重跑 UI 定向测试并确认通过**。

### Task 8: Claude Code 交付验证

**Files:**
- No production changes.

- [ ] **Step 1: 运行 S1.4 定向套件**：

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests\test_input_safety.py auto_tune\tests\test_input_safety_api.py auto_tune\tests\test_upload_security.py auto_tune\tests\test_dataset_snapshot.py auto_tune\tests\test_dataset_snapshot_api.py auto_tune\tests\test_ui_training_results.py -v -p no:cacheprovider
```

- [ ] **Step 2: 运行完整套件**：

```powershell
& 'D:\Program Files\anaconda3\envs\auto_tune\python.exe' -m pytest auto_tune\tests -q -p no:cacheprovider
```

- [ ] **Step 3: 检查 `git status --short` 和 `git diff -- auto_tune`**，确认无文档修改、依赖新增、文件删除、凭据、本机业务路径或训练产物。
- [ ] **Step 4: 向 Codex 报告改动文件、每条测试结果、平台跳过项、偏离和风险**；不得提交或推送。

## Codex 独立验收门

Claude Code 交付后，Codex 独立审查差异、复跑定向与完整套件，并用 Chromium 验证正常目录、成员超限、容量超限、越界目录、按钮恢复和控制台。全部通过后才可宣布 S1.4 验收通过。
