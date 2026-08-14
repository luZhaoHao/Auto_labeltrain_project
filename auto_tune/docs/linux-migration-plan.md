# Linux 迁移方案

> 记录时间: 2026-07-28
> 目标: 将项目从 Windows 迁移到 Linux 环境运行

---

## 一、依赖层面

`environment.yml` 中的 pip 包基本都跨平台，唯二需要注意：

| 包 | 问题 |
|---|---|
| `pyreadline3==3.5.6` | **Windows 独有**（为 Python REPL 提供 readline 能力），Linux 上安装会失败或无用 |
| `torch==2.5.1` | 当前通过 pip 安装（默认 CPU 版），Linux 上如需 GPU 支持需指定 `--index-url https://download.pytorch.org/whl/cu121` |

### 对策

- 创建 `environment-linux.yml`，移除 `pyreadline3`，或者改为条件安装
- 如需要 GPU 训练，在 Linux 环境文件中调整 PyTorch 安装源

---

## 二、启动脚本（需重写为 .sh）

以下 `.bat` 文件在 Linux 上无法运行，需要创建等效的 Shell 脚本：

| 原 .bat 文件 | 新建 .sh 文件 | 说明 |
|---|---|---|
| `start_server.bat` | `start_server.sh` | `python -m auto_tune.main` 直接启动 |
| `start_app.bat` | `start_app.sh` | conda activate → python -m auto_tune.main |
| `setup.bat` | `setup.sh` | conda env create → 复制配置模板 → 验证导入 |
| `run_tests.bat` | `run_tests.sh` | conda run -n auto_tune pytest ... |
| `build_package.bat` | `build_package.sh` | conda-pack 封装脚本（按需创建） |

---

## 三、Python 核心代码修改

### 1. `auto_tune/ui/app.py` — 文件夹浏览器（约 30 行）

**位置（~L926-934）**：当路径为空或不存在时，Windows 上返回盘符列表。

当前逻辑：
```python
if not path or not os.path.isdir(path):
    if os.name == "nt":
        # 枚举 A-Z 盘符
```

Linux 对策：改为返回 `/` 作为根目录，或返回用户家目录作为起始目录。

**位置（~L951-952）**：盘符根目录无上级目录。

当前逻辑：
```python
if os.name == "nt" and len(path) == 3 and path[1:] == ":\\":
    parent = ""
```

Linux 对策：根目录 `/` 也需特殊处理（`dirname("/")` 返回 `/`，应返回 `parent = None`）。

建议修改为按平台分支处理：

```python
if not path or not os.path.isdir(path):
    if os.name == "nt":
        # Windows: 枚举盘符
        ...
    elif os.name == "posix":
        # Linux: 返回 / 和 ~ 作为可选项
        entries = [
            {"name": "/ (根目录)", "path": "/", "is_dir": True},
            {"name": f"~ ({os.path.expanduser('~')})", "path": os.path.expanduser("~"), "is_dir": True},
        ]
```

### 2. `auto_tune/modules/agent_engine/executor.py` — 无需修改

- `subprocess.Popen` 启动 `yolo train` 跨平台兼容
- 参数格式 `arg=value` 在 Ultralytics 所有平台上一致
- `rstrip("/\\")` 同时处理两种分隔符，已安全

### 3. `auto_tune/modules/train_analyzer/results_parser.py` — 无需修改

- `rstrip("/\\")` 已安全

---

## 四、辅助测试脚本（硬编码路径清理）

以下测试脚本中包含 Windows 硬编码的绝对路径，在 Linux 上会失败：

| 文件 | 问题 | 修改方案 |
|---|---|---|
| `_final_test.py` | `r"D:\Program Files\anaconda3\envs\auto_tune\python.exe"` | 改为使用 `sys.executable` 或通过 PATH 查找 |
| `_test_vision_models.py` | `r"e:\dataprocess_modeltrain\..."` 硬编码 | 改为相对路径 `os.path.join("detect", "trainXX", ...)` |
| `_test_qwen_keys.py` | 同上 | 同上 |
| `_test_llm_apis.py` | 硬编码路径 + GBK 编码注释 | 改为相对路径 |
| `_test_pipeline_full.py` | `msg.encode("gbk", errors="ignore")` | GBK 改为 UTF-8 或按系统编码动态处理 |

---

## 五、需要注意的事项

### 1. 文件路径大小写敏感

Linux 文件系统**区分大小写**，Windows 不区分。项目代码本身已使用正确的引用，但用户上传的数据集需注意：
- `.jpg` vs `.JPG` 扩展名差异
- 图片文件名与标签文件名的大小写对应关系

### 2. 路径分隔符

- `os.path.join()` 已安全（跨平台），无需修改
- `f"{d}:\\"` 盘符拼接仅在 `os.name == "nt"` 块内，Linux 不会执行

### 3. 编码问题

- 项目代码中文件读写均指定 `encoding="utf-8"`，Linux 兼容
- `_test_pipeline_full.py` 中的 GBK 编码处理需要移除

### 4. Conda 环境创建

Linux 上创建环境的命令与 Windows 相同：
```bash
conda env create -f environment.yml
```
但需注意：
- `defaults` channel 在 Linux 上仍然可用
- 建议为 Linux 创建独立的 `environment-linux.yml`

---

## 六、修改清单汇总

| 类别 | 文件 | 修改类型 | 工作量 |
|---|---|---|---|
| 依赖配置 | `environment-linux.yml`（新） | 新建 | 小 |
| 启动脚本 | `start_server.sh`（新） | 新建 | 小 |
| 启动脚本 | `start_app.sh`（新） | 新建 | 小 |
| 启动脚本 | `setup.sh`（新） | 新建 | 中 |
| 启动脚本 | `run_tests.sh`（新） | 新建 | 小 |
| 启动脚本 | `build_package.sh`（新） | 新建 | 中 |
| Python 核心 | `app.py`（文件夹浏览器） | 修改 ~30 行 | 小 |
| 测试脚本 | `_final_test.py` | 修改硬编码路径 | 小 |
| 测试脚本 | `_test_vision_models.py` | 修改硬编码路径 | 小 |
| 测试脚本 | `_test_qwen_keys.py` | 修改硬编码路径 | 小 |
| 测试脚本 | `_test_llm_apis.py` | 修改硬编码路径 | 小 |
| 测试脚本 | `_test_pipeline_full.py` | 移除 GBK 硬编码 | 小 |
| 文档 | `CLAUDE.md` | 补充 Linux 命令说明 | 小 |

**总体评估：** 核心代码改动很小（仅 `app.py` 文件夹浏览器处需要按平台分支处理），主要工作是创建 5 个 `.sh` 脚本替代 `.bat` 以及清理测试脚本中的硬编码路径。预计工作量约为**半天到一天**。
