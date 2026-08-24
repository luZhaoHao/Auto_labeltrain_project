# Studio S1.4 目录输入安全设计规格

> 完成状态（2026-08-24）：已按本规格实现并通过 Codex 独立验收。定向套件 176 passed，完整套件 460 passed、2 条既有 PCA warning、0 skipped；Chromium 目录安全场景通过。验收中补充修复了两个分析入口未展示稳定 `error_code` 的问题。

## 1. 结论与范围

S1.4 只处理正式目录入口的容量、成员数量和路径安全，不继续建设 ZIP 上传能力。现有数据集 ZIP 与训练结果 ZIP/JSON 上传 API 固定返回 HTTP `410 Gone`，提示改用目录选择；旧实现暂不永久删除，待确认无外部调用后另立清理批次。

本批不得进入 SQLite、多数据集、训练恢复、YOLOv8 Classify、YOLOv5、Research、Cloud 或安装交付，不新增第三方依赖。

## 2. 当前事实

- 当前页面统一渲染 `auto_tune/ui/templates/single_page.html`。
- 正式入口为 `/api/browse-folder`、`/api/dataset/analyze-folder` 和 `/api/training/analyze-folder`。
- `/api/dataset/upload`、`/api/training/analyze` 和 `_analyze_train_zip()` 仍在后端，但当前单页 UI 和测试都不调用。
- S1.2 已有 reparse point 拒绝、空间预检及快照校验；S1.4 复用安全语义，不修改快照身份、Schema 或不可变边界。

## 3. 目标与非目标

目标：

1. 在递归分析、哈希、复制或结果解析前执行统一目录安全预检。
2. 对成员数量和总字节数实施可配置硬上限，并在超限时提前停止。
3. 拒绝路径逃逸、非法路径类型、默认禁用的 UNC/设备路径、符号链接、junction 和其他 reparse point。
4. 目录浏览与最终分析共用允许根策略。
5. 安全失败不得创建报告、启动分析、创建快照或改写 `latest_dataset.json`。

非目标：

- 不永久删除 ZIP/JSON 上传函数、旧模板或翻译文案。
- 不实现 ZIP 流式上传、安全解压、压缩比或解压容量检查。
- 不增加数据库、用户认证、远程共享目录或网络文件系统支持。
- 不修改 Module A 算法、S1.2 Schema、训练命令、A0 审计或指标口径。

## 4. 配置契约

```yaml
input_safety:
  max_directory_members: 200000
  max_directory_bytes: 536870912000
  allowed_roots: []
  allow_unc_paths: false
```

- `max_directory_members` 必须是 `1..1000000` 的整数，布尔值无效。
- `max_directory_bytes` 必须是 `1..10995116277760` 的整数，布尔值无效。
- `allowed_roots` 是绝对目录列表；非空时输入必须位于其中一个规范化根内。空列表表示允许本机绝对路径，但仍应用全部路径和容量规则。
- `allow_unc_paths` 默认 `false`，本批不把 UNC/网络共享作为正式范围。
- 配置缺失使用上述默认值；字段存在但非法时返回配置错误，不静默转成无限制。

## 5. 模块与接口

新增 `auto_tune/modules/input_safety/`：

- `models.py`：冻结的 `InputSafetyPolicy`、`DirectoryScanResult`、`InputSafetyError` 及稳定子类。
- `service.py`：
  - `load_input_safety_policy(config: Mapping[str, Any]) -> InputSafetyPolicy`
  - `validate_directory_path(path: str | Path, policy: InputSafetyPolicy) -> Path`
  - `scan_directory_bounded(path: str | Path, policy: InputSafetyPolicy) -> DirectoryScanResult`
  - `list_safe_subdirectories(path: str | Path, policy: InputSafetyPolicy) -> tuple[Path, ...]`
- `__init__.py`：只导出公共接口。

FastAPI 层只负责调用公共接口及映射 HTTP 错误，不复制安全判断。

## 6. 路径与扫描规则

1. 输入必须是非空绝对路径；拒绝相对路径、NUL、Windows 设备命名空间及默认禁止的 UNC。
2. 允许根比较使用规范化路径组件，不使用字符串前缀；Windows 按大小写不敏感语义比较。
3. 目标必须是可读取目录；权限错误不能伪装为空目录。
4. 根目录及扫描中的每个成员都拒绝符号链接和 reparse point，不跟随链接。
5. 使用迭代式 `os.scandir()`；禁止先构造完整 `rglob()`、`list(os.walk())` 或全量路径列表。
6. 每发现目录项立即计数，普通文件立即累计不跟随链接的 stat 大小；任一超限立即失败。
7. 扫描顺序确定；成员在扫描期间消失、变型或失去权限时返回冲突，不继续分析部分目录。
8. 扫描结果只保存根路径、成员数和总字节数，不保存完整成员列表。

## 7. API 契约

### 7.1 `/api/browse-folder`

- 配置 `allowed_roots` 时根列表只展示允许根。
- 每次进入目录先验证路径，再安全枚举直接子目录。
- 越界、链接和权限错误返回稳定错误，不伪装为空目录。

### 7.2 `/api/dataset/analyze-folder`

- 在搜索 `data.yaml`、图片或调用 Module A 前完成验证和有界扫描。
- 失败时不得创建报告或改写 `latest_dataset.json`。
- 成功响应增加 `input_scan: {member_count, total_bytes}`。

### 7.3 `/api/training/analyze-folder`

- 在读取 `results.csv`、`args.yaml` 或调用 Module B 前完成预检。
- 失败时不得创建/覆盖报告或统一历史。

### 7.4 遗留上传 API

- `/api/dataset/upload` 和 `/api/training/analyze` 固定返回 `410`、`error_code=LEGACY_UPLOAD_DISABLED`。
- 处理器不得读取请求文件体，响应提示使用目录选择。
- 本批不再调用 `_safe_extract_zip()` 或 `_analyze_train_zip()`，但暂不永久删除。

## 8. 错误契约

| 场景 | HTTP 与错误码 |
|---|---|
| 成员数超限 | `413 INPUT_MEMBER_LIMIT_EXCEEDED` |
| 总容量超限 | `413 INPUT_SIZE_LIMIT_EXCEEDED` |
| 路径不在允许根 | `403 INPUT_PATH_NOT_ALLOWED` |
| 链接/reparse point | `400 INPUT_LINK_NOT_ALLOWED` |
| 权限不足 | `403 INPUT_PERMISSION_DENIED` |
| 扫描期间变化 | `409 INPUT_CHANGED_DURING_SCAN` |
| 安全配置非法 | `500 INPUT_POLICY_INVALID` |
| 遗留上传 API | `410 LEGACY_UPLOAD_DISABLED` |

旧 `latest_dataset.json`、S1.2 manifest、历史和审计保持可读；部分成功、审计 fatal、缺失值与真实零值等既有语义不变。

## 9. 验收标准

1. 单元测试覆盖配置边界、允许根、大小写、链接/reparse point、权限、成员数、容量、早停和扫描竞态。
2. API 测试证明三个正式目录入口共用策略，且预检发生在分析之前。
3. 410 测试证明上传请求体未被读取。
4. S1.2 快照定向测试和完整 `auto_tune/tests` 通过。
5. Chromium 验证正常目录、成员超限、容量超限、越界目录、按钮恢复及中文错误，无新增控制台错误。
6. 本批不触碰训练执行与指标解析时不要求新增真实短 epoch；若实际改动触及这些路径，Codex 追加最小真实训练。

## 10. Claude Code 边界

- 只修改业务代码和对应测试，不修改规格、计划、README、路线图、交接记录、DOCX 或发布说明。
- 不新增依赖、不删除文件、不提交、不推送。
- 严格测试先行；交付改动文件、命令结果、偏离与遗留风险。
- 不重新引入 ZIP 安全提取，也不提前永久删除遗留代码。
