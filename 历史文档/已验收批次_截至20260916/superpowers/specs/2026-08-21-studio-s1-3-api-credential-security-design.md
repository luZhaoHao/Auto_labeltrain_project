# Studio S1.3 API 凭据安全与客户配置设计规格

## 1. 结论

Studio S1.3 面向单人、单机、独立部署产品，将文本诊断、调参决策和视觉分析使用的 API Key 从明文 YAML 迁移到安全凭据解析层。Windows 默认使用当前系统用户的 Windows Credential Manager 持久化凭据，环境变量作为自动部署和 Linux 场景的只读高优先级来源；真实凭据不得进入 `config.yaml`、`APP_CONFIG`、日志、审计、历史、导出、URL、SSE 或 HTTP 响应。

Studio 保留客户自行填写、测试、替换和删除 API Key 的界面。默认文本诊断与调参决策使用 DeepSeek，默认视觉分析使用通义千问，同时允许配置 OpenAI-compatible 自定义端点。自定义端点默认禁止访问本机和内网；客户明确开启“允许本地/内网模型端点”后，才允许连接受约束的本机或私有网段 HTTP/HTTPS 服务。

本批遵循最小侵入原则：保留现有 `llm`、`vision`、`APP_CONFIG`、Module B/C、训练执行、快照、统一历史与审计主体架构，只增加薄的凭据存储、凭据解析、端点策略和设置 API/UI 适配层。

## 2. 背景与现状

当前 `auto_tune/config.template.yaml` 引导用户把文本和视觉 API Key 写入 `api_key` 字段；真实 `config.yaml` 也支持同一结构。以下三个联网调用路径直接从配置字典读取密钥并构造 `Authorization` 请求头：

1. `auto_tune/modules/train_analyzer/llm_analyzer.py`：训练结果文本诊断；
2. `auto_tune/modules/agent_engine/decision_agent.py`：调参决策；
3. `auto_tune/modules/train_analyzer/vision_analyzer.py`：视觉分析。

现有 `auto_tune/modules/agent_engine/audit.py` 能递归脱敏若干敏感字段，但只能保护经过该函数的审计对象，不能防止明文配置、供应商错误正文、UI/API 响应、日志或其他导出泄漏。因此 S1.3 必须把安全边界前移到凭据来源与网络调用入口。

## 3. 产品与威胁边界

### 3.1 产品边界

- Studio 当前是单人、单机、独立部署产品，不在 S1.3 引入账号、角色、租户或多人共享凭据。
- Studio 默认仅在本机回环地址提供 UI；非回环监听属于高级部署方式，不因此获得多人安全能力。
- Cloud 的多人凭据将来必须使用服务端 Secret Manager/KMS、租户隔离、权限控制和访问审计，不复用 Studio 的本机凭据实现。

### 3.2 保护目标

- 防止真实 API Key 被提交到公开仓库、进入项目备份、配置导出或普通日志。
- 防止浏览器、只读设置 API、异常响应或 SSE 获得密钥。
- 防止供应商错误正文意外回显请求头、凭据或其他敏感信息。
- 防止自定义 Endpoint 被默认用于访问本机、内网、链路本地或保留地址。
- 防止旧版明文 Key 在客户不知情的情况下被移动、删除或继续长期使用。

### 3.3 非防护承诺

- 无法防御已经取得当前 Windows 用户完整控制权的恶意程序或管理员。
- 不宣称环境变量对同权限进程绝对保密。
- 不使用自制加密算法或把密文和解密材料共同存放在项目目录。
- 不通过本批解决第三方供应商对上传数据的保存、训练或合规政策；产品只负责明确告知数据外发事实。

## 4. 目标

1. Windows 客户能够在 Studio UI 中安全录入、测试、替换和删除文本与视觉 API Key。
2. Windows Credential Manager 按当前系统用户保存持久凭据，其他 Windows 用户不能通过 Studio 读取。
3. 环境变量可作为只读、高优先级凭据来源，用于脚本、自动部署和 Linux。
4. 三个现有联网调用入口统一使用凭据解析器，不再读取 `config[section]["api_key"]`。
5. 凭据只在发送请求前短时存在于内存，不合并回共享 `APP_CONFIG`。
6. 默认支持 DeepSeek 文本/决策和通义千问视觉 API，并支持受策略约束的 OpenAI-compatible 自定义端点。
7. 未配置或删除 Key 时只禁用相应联网能力，本地数据分析、训练、历史和报告继续工作。
8. 旧版 YAML 明文 Key 必须经客户明确确认后迁移；迁移失败不修改原文件。
9. 配置、日志、审计、历史、导出、API、SSE 和测试产物不得包含真实 Key。

## 5. 非目标

- 不实现用户登录、RBAC、租户、共享凭据或 Cloud Secret Manager。
- 不实现 Linux Secret Service；Linux 在 S1.3 仅支持环境变量。
- 不引入第三方密码库或新的前端框架。
- 不支持客户上传自定义 Python 请求脚本、插件或任意协议适配器。
- 不改变大模型提示词、LLM 动作空间、Guardrails、训练参数 Schema 或 KPI 口径。
- 不重构现有 Module B/C、训练执行器、不可变数据集快照或统一训练收尾。
- 不提供查看、复制、导出或找回已保存 API Key 的能力。
- 不自动吊销供应商侧旧 Key；轮换必须由客户在供应商控制台完成。

## 6. 方案比较与选择

### 6.1 采用：Windows Credential Manager + 环境变量覆盖

Windows Credential Manager 提供与当前系统用户绑定的操作系统凭据保护，同时允许普通客户通过 UI 配置。环境变量适合自动部署与 Linux，并保持无额外依赖。

### 6.2 不采用：仅使用环境变量

仅使用环境变量虽然实现简单，但普通 Windows 客户配置和轮换体验差，且难以通过产品界面安全管理，因此只作为高级入口。

### 6.3 不采用：项目内加密文件

项目内加密文件需要额外解决主密钥保存、备份和设备绑定问题。若密文与解密材料共同分发，只形成伪安全；S1.3 不实现该降级路径。

### 6.4 不采用：继续在 YAML 中保存明文

即使 `config.yaml` 已被 `.gitignore` 排除，仍可能通过项目复制、支持包、日志或误提交泄漏，不满足客户交付要求。

## 7. 最小侵入模块边界

新增小型独立安全模块，避免把凭据逻辑散落到 UI 和三个模型调用文件：

```text
auto_tune/modules/security/
├─ __init__.py
├─ credentials.py
├─ endpoint_policy.py
└─ redaction.py
```

- `credentials.py`：定义凭据用途、来源、状态、Windows Credential Manager 读写、环境变量解析、最多五分钟内存缓存和缓存失效。
- `endpoint_policy.py`：规范化并校验默认、自定义公网以及显式允许的本地/内网 Endpoint。
- `redaction.py`：提供统一的结构化脱敏与安全错误分类；现有审计脱敏器应调用或复用此实现，避免形成两套不一致规则。
- `auto_tune/ui/app.py`：只承担设置 API、旧凭据迁移编排和把非敏感状态传给模板。
- `auto_tune/ui/templates/single_page.html`：增加现有页面风格下的两张 AI 服务配置卡，不承担凭据判断。
- 三个现有调用文件：只在发送请求前调用凭据解析和 Endpoint 校验，不调整其分析职责或提示词。

不得把真实密钥写入 `APP_CONFIG`。为控制改动范围，调用函数仍可接收现有 `config: dict`，但密钥通过 `purpose` 调用解析器获得。

## 8. 用途、供应商与默认值

S1.3 定义两个凭据用途，不按调用函数重复保存密钥：

| purpose | 使用范围 | 默认供应商 | 默认凭据目标名 | 环境变量 |
|---|---|---|---|---|
| `text` | 文本诊断、调参决策 | DeepSeek | `AutoTuneStudio/text/deepseek` | `AUTO_TUNE_TEXT_API_KEY` |
| `vision` | 混淆矩阵和错误样本视觉分析 | 通义千问 | `AutoTuneStudio/vision/qwen` | `AUTO_TUNE_VISION_API_KEY` |

文本诊断和调参决策共享 `text` 凭据，但继续使用各自现有提示词和输出契约。视觉分析使用独立 `vision` 凭据。

非敏感设置继续保存在 YAML：

```yaml
llm:
  enabled: true
  provider: deepseek
  credential_ref: AutoTuneStudio/text/deepseek
  model: deepseek-chat
  endpoint: https://api.deepseek.com/v1/chat/completions
  allow_private_endpoint: false

vision:
  enabled: true
  provider: qwen
  credential_ref: AutoTuneStudio/vision/qwen
  model: qwen-vl-plus
  endpoint: https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions
  allow_private_endpoint: false
```

`credential_ref` 只能引用应用定义的当前用途目标，不允许客户端传入任意 Credential Manager target 后读取其他系统凭据。

## 9. 凭据解析契约

### 9.1 数据对象

```python
CredentialPurpose = Literal["text", "vision"]
CredentialSource = Literal["environment", "windows_credential_manager", "missing"]

@dataclass(frozen=True)
class CredentialStatus:
    purpose: CredentialPurpose
    configured: bool
    source: CredentialSource
    writable: bool
    last_tested_at: str | None
    last_test_result: str | None
```

`CredentialStatus` 不得包含密钥、密钥长度、前缀、后缀或可用于验证猜测的摘要。

### 9.2 优先级

1. 对应用途的非空环境变量；
2. 当前 Windows 用户 Credential Manager 中的应用固定目标；
3. 未配置。

环境变量存在时，UI 显示“由环境变量管理”，`writable=false`；UI 不得覆盖、删除或迁移到该用途的有效凭据。

### 9.3 缓存

- 解析出的凭据最多缓存五分钟，使用单调时钟计算有效期。
- 缓存只存在于当前进程内，不落盘、不进入共享配置和序列化对象。
- 替换、删除、迁移成功、供应商切换或禁用服务时立即清除对应用途缓存。
- 测试不得断言或打印缓存中的真实值。

### 9.4 平台行为

- Windows：支持 Credential Manager 和环境变量。
- 非 Windows：只支持环境变量；UI 对持久化操作返回明确的“当前平台不支持本机凭据库”，不得回退到明文文件。
- Credential Manager 不可用或调用失败时，联网功能保持不可用并给出安全化错误，本地功能不受影响。

## 10. Windows Credential Manager 契约

- 使用 Python 标准库 `ctypes` 调用 Windows Credential API，不新增第三方依赖。
- 使用 Generic Credential，持久化范围不得扩大为企业漫游或跨用户共享。
- Target Name 只能由应用内部根据固定 purpose 映射生成。
- 写入采用覆盖指定 target 的语义，不扫描或枚举用户的其他凭据。
- 读取后应尽快释放 Windows 返回的凭据内存；Python 字符串无法保证物理擦除，不作无法兑现的安全承诺。
- 删除只针对当前用途的固定 target；目标不存在按幂等成功处理。
- API 层永远不返回 Credential Manager 的原始错误正文或 target 枚举。

## 11. Endpoint 策略

### 11.1 默认厂商端点

- DeepSeek 文本/决策使用项目内固定默认 HTTPS Endpoint。
- 通义千问视觉分析使用项目内固定默认 HTTPS Endpoint。
- 默认值可在非敏感设置中展示和恢复。

### 11.2 自定义公网端点

必须同时满足：

- 使用 `https://`；
- URL 不包含 username、password、query 或 fragment；
- hostname 非空，端口有效；
- 规范化后仍为 HTTP(S) URL；
- hostname 和 DNS 解析得到的全部地址均不是回环、私有、链路本地、组播、未指定或保留地址；
- 请求不自动跟随重定向；若未来需要重定向，必须对每一跳重新应用完整策略；
- 只调用固定的 OpenAI-compatible chat completions 请求，不执行供应商返回的 URL 或脚本。

### 11.3 本地/内网端点开关

`allow_private_endpoint` 默认 `false`。客户显式开启后：

- 允许本机、私有网段或企业内网的 `http://`/`https://` Endpoint；
- HTTP 页面显示“传输未加密”持续警告；
- 仍禁止 URL 内嵌凭据、query、fragment 和非 HTTP(S) 协议；
- 仍禁止自动重定向；
- 保存设置、测试连接和每次实际调用前均重新校验；
- 开关及 Endpoint 变化进入安全审计，但不记录密钥。

对 DNS 名称必须在调用前解析并校验所有返回地址，避免保存时与调用时解析结果变化形成绕过。连接层应尽量确保实际连接目标与已校验结果一致；如果当前 HTTP 库无法可靠绑定解析结果，至少必须在紧邻请求前复核并禁用重定向，同时将 DNS rebinding 记录为剩余风险。

## 12. 设置 API 契约

### 12.1 读取设置

`GET /api/ai-settings`

返回两个用途的非敏感配置和 `CredentialStatus`。响应及任何嵌套字段不得包含 `api_key`、`Authorization`、Key 局部值或可逆摘要。

### 12.2 更新非敏感设置

`PUT /api/ai-settings/{purpose}`

- `purpose` 只能是 `text` 或 `vision`；
- 只接受白名单字段：`enabled`、`provider`、`model`、`endpoint`、`allow_private_endpoint`；
- 保存前执行类型、长度和 Endpoint 策略校验；
- 不接受 `api_key`、`credential_ref` 或任意额外字段；
- 保存成功后使对应凭据缓存失效。

### 12.3 写入或替换凭据

`PUT /api/credentials/{purpose}`

- Key 只能放在 JSON 请求体固定字段，不允许 query、path 或 header 自定义转发；
- 限制请求总大小和 Key 长度，拒绝空值、控制字符和异常超长值；
- 环境变量生效时返回冲突，不覆盖系统来源；
- 支持 `test_before_replace=true|false`；
- 为 `true` 时先用内存中的新 Key 发送固定最小测试请求，成功后才写入并替换旧凭据；
- 为 `false` 时允许离线保存，但状态显示“未测试”；
- 响应只返回状态，不回显 Key。

### 12.4 删除凭据

`DELETE /api/credentials/{purpose}`

- 需要明确确认字段或确认令牌；
- 环境变量来源不可通过 UI 删除；
- 删除 Credential Manager 目标后立即清除缓存；
- 不影响非敏感模型设置和本地功能。

### 12.5 测试连接

`POST /api/credentials/{purpose}/test`

- 使用当前已解析凭据发送固定最小请求；
- 不发送客户训练数据、图片、报告或调优历史；
- 返回值仅允许：`success`、`authentication_failed`、`network_failed`、`endpoint_rejected`、`incompatible_response`、`rate_limited`、`provider_failed`；
- 不返回供应商原始响应正文。

### 12.6 迁移旧凭据

`POST /api/credentials/{purpose}/migrate`

- 只有检测到对应 YAML 中非空、非占位的旧 `api_key` 时可用；
- 必须由客户明确触发，不在启动时静默迁移；
- 环境变量已生效时拒绝迁移，避免来源含义不清；
- 先写入 Credential Manager并重新读取验证，再原子移除 YAML 中的明文字段；
- 任一步失败时不得删除或改写原 YAML；
- 成功后清除内存中的旧配置引用并重新加载非敏感配置；
- UI 必须提示：若旧 Key 曾被提交、分享或记录，应到供应商控制台吊销并生成新 Key。

## 13. 本地 Web 安全边界

- Studio 默认绑定回环地址；局域网监听不是 S1.3 默认客户模式。
- 所有修改凭据或设置的接口执行严格同源校验。
- 使用应用启动时生成、仅当前会话有效的 CSRF Token；修改类请求必须携带并验证。
- 不启用允许任意 Origin 的 CORS。
- 凭据请求体不得被访问日志、中间件、异常处理器或调试输出记录。
- 密钥禁止出现在 URL，因此浏览器历史、代理访问日志和 Referer 不应包含密钥。
- 非回环监听时 UI 持续显示安全警告；S1.3 不宣称该模式具备多人访问安全。

## 14. UI 行为

现有单页新增“AI 服务配置”区块，包含“文本与调参服务”和“视觉服务”两张卡。每张卡展示：

- 启用/禁用；
- 默认供应商或自定义 OpenAI-compatible；
- 模型和 Endpoint；
- “允许本地/内网模型端点”高级开关；
- 凭据状态、来源、可写状态、最近测试时间和安全结果；
- 填写/替换、测试、删除按钮。

交互要求：

- 密码输入框仅用于本次提交，提交或关闭后清空；不得由服务端预填。
- 已配置只显示状态，不显示 Key 的前缀、尾号、长度或摘要。
- 环境变量来源显示为只读，并禁用替换和删除按钮。
- 开启内网端点时显示风险确认；使用 HTTP 时持续显示未加密警告。
- 未配置凭据时允许禁用联网分析，页面明确说明本地训练仍可用。
- 首次启用外部厂商 API 时提示会发送的内容类别：文本诊断会发送结构化训练指标/摘要；视觉分析会发送所选分析图片；连接测试不发送客户业务数据。
- 旧明文 Key 只显示“检测到待迁移凭据”，绝不回显内容。

## 15. 网络调用改造

三个现有调用入口必须执行同一顺序：

1. 根据用途读取非敏感 provider/model/endpoint 设置；
2. 调用 Endpoint 策略校验；
3. 调用 `CredentialResolver` 获取临时 Key；
4. 在局部作用域构造 `Authorization` 请求头；
5. 发送禁用自动重定向、带明确连接/读取超时的请求；
6. 对响应执行结构校验；
7. 将失败转换为安全错误分类；
8. 离开调用作用域，不把请求头或 Key 附加到报告、异常或审计。

不得把解析出的 Key 注入 `config` 副本作为兼容捷径。测试可通过依赖注入或 monkeypatch 替换凭据解析器和 HTTP 调用，但不得依赖真实厂商或真实 Key。

## 16. 脱敏与错误契约

统一脱敏至少覆盖大小写及常见变体：

- `api_key`、`apikey`、`access_token`、`refresh_token`、`token`、`secret`、`password`；
- `Authorization`、`Proxy-Authorization`、Cookie 和 Set-Cookie；
- Bearer/Basic 等认证头格式；
- 已知当前进程解析过的真实凭据值，即使出现在自由文本中也必须替换。

结构化对象应按键名递归脱敏；自由文本应执行有限、可测试的认证模式和已知秘密值替换。脱敏不得改变源对象。

供应商原始错误正文不得直接进入：

- `RuntimeError` 文本；
- UI/API 响应；
- 训练报告、调优历史、统一历史或审计；
- 普通日志或完整训练日志。

安全错误可以包含供应商类别、HTTP 状态码、请求 ID（确认不敏感时）和内部错误码，但不包含请求头、请求体、完整响应正文或 Key。

## 17. 旧明文凭据迁移与轮换

### 17.1 检测

启动加载配置时只检测 `llm.api_key` 和 `vision.api_key` 是否为非空且不是项目定义的占位值。检测结果只进入内存状态，不打印值。

### 17.2 客户确认迁移

客户在 UI 中选择迁移后，系统按第 12.6 节执行。迁移成功前，旧文件保持原样；迁移成功后采用安全原子写入删除相应 `api_key` 字段。

### 17.3 迁移期间的使用策略

检测到旧明文 Key 后，对应联网能力进入 `migration_required` 状态，不能继续通过旧配置直接调用 API。客户可以迁移、填写新 Key或禁用该服务；本地功能继续可用。

### 17.4 轮换说明

项目文档必须明确：存储迁移不能撤销历史暴露。曾进入 Git、聊天、日志、截图、工单或共享文件的 Key，应在 DeepSeek/通义千问控制台吊销并重新生成，再将新 Key 保存到安全凭据库。产品不得声称能代替供应商侧轮换。

## 18. 配置模板与文档

- `auto_tune/config.template.yaml` 删除真实 Key 填写指导和 `api_key` 字段，改用非敏感 `credential_ref`、默认 Endpoint 和内网开关。
- 模块 README 中的 `sk-xxx` 明文配置示例改为 UI、环境变量和凭据状态说明。
- 用户文档说明默认供应商、客户自行承担的 API 费用、数据外发范围、离线禁用方式、凭据迁移和供应商侧轮换。
- 发布检查必须扫描真实 Key、本机绝对路径、配置文件、日志和测试产物。
- 文档不得建议通过命令行参数传入 Key，因为进程列表和 shell 历史可能泄漏。

## 19. 兼容边界

- 保留 `llm` 和 `vision` 非敏感配置段，现有 enabled/model/endpoint/temperature/max_tokens 语义不变。
- 文本诊断和调参决策继续共享文本服务设置，不拆成两个新配置体系。
- 不改变提示词、LLM JSON Schema、参数白名单、Guardrails、审计 fatal 策略或训练命令。
- 未配置凭据时，已有调用应返回明确的安全错误或跳过联网阶段，不得导致已经成功的训练被改记为失败。
- 训练成功、联网分析失败继续属于已有的部分成功语义。
- S1.2 快照校验和普通训练/调优数据绑定不得回归。
- 旧历史和旧审计继续可读；读取旧记录时仍执行脱敏后再返回 UI。

## 20. 失败策略

| 场景 | 行为 |
|---|---|
| 环境变量缺失且凭据库无 Key | 禁用对应联网调用，返回 `credential_missing`；本地功能继续 |
| 环境变量存在但 UI 尝试覆盖 | 返回冲突，说明由环境变量管理 |
| Credential Manager 写入失败 | 不删除旧 Key、不改 YAML，返回安全错误 |
| 新 Key 预测试失败 | 保留旧凭据，不替换 |
| Endpoint 策略拒绝 | 不发起网络请求 |
| DNS 解析失败 | 返回 `network_failed`，不回显内部堆栈 |
| 供应商认证失败 | 返回 `authentication_failed`，不回显正文 |
| 供应商限流 | 返回 `rate_limited`，可显示安全的重试提示 |
| 响应协议不兼容 | 返回 `incompatible_response`，不保存原始响应 |
| YAML 迁移写回失败 | 保持原 YAML 和已写凭据；标记迁移未完成并禁止声称成功，提供安全重试/人工处理指引 |
| 删除不存在的本机凭据 | 幂等成功并清缓存 |

迁移写入凭据库成功但 YAML 原子改写失败时，不得自动删除刚写入的安全凭据，因为这可能丢失客户唯一可用副本；系统应保持 `migration_required`，提示客户重试，并继续禁止使用 YAML 明文调用。

## 21. 测试设计

### 21.1 凭据服务单元测试

- 环境变量优先于 Windows Credential Manager；
- 环境变量状态只读且不可删除；
- 固定 purpose 只映射固定 target，不能读取任意 target；
- 写入、读取、替换、幂等删除；
- 五分钟缓存命中、过期和主动失效；
- 状态对象和异常不包含 Key；
- 非 Windows 不回退到文件；
- 模拟 Windows API 分配/释放和错误路径，不依赖开发机真实凭据。

### 21.2 Endpoint 策略测试

- 接受默认厂商 HTTPS Endpoint；
- 公网自定义 Endpoint 接受合法 HTTPS；
- 拒绝 URL 用户信息、query、fragment、非法端口和非 HTTP(S) 协议；
- 默认拒绝 localhost、IPv4/IPv6 回环、私有、链路本地、保留和未指定地址；
- 内网开关开启后接受受约束的 HTTP/HTTPS 私有地址；
- 公网 HTTP 即使打开内网开关仍不得被误当成安全公网 Endpoint；
- DNS 多地址中任一地址受禁即整体拒绝；
- 禁止重定向，保存与实际调用均复核策略。

### 21.3 API 测试

- GET 设置永不返回秘密字段或秘密片段；
- PUT 设置严格字段白名单；
- 写入请求大小、Key 长度、空值和控制字符限制；
- 测试成功后替换，失败保留旧凭据；
- 离线直接保存状态为未测试；
- 删除需确认且清缓存；
- 环境变量来源不可写；
- 同源和 CSRF 校验拒绝跨站修改；
- 连接测试不发送业务数据；
- 供应商错误正文包含测试秘密时，API/日志/审计仍无泄漏。

### 21.4 迁移测试

- 占位值不被识别为旧凭据；
- 启动不静默迁移；
- 未确认不改文件；
- Credential Manager 写入失败不改 YAML；
- 重新读取验证失败不改 YAML；
- YAML 原子改写成功才移除明文字段；
- YAML 写回失败保持 `migration_required` 且不丢失安全凭据；
- 迁移成功后内存配置和磁盘均不含 Key；
- 环境变量生效时拒绝迁移。

### 21.5 调用路径回归测试

- 文本诊断、调参决策和视觉分析均从 resolver 获取正确用途；
- 三条路径均不再读取 YAML `api_key`；
- 请求禁用自动重定向并设置明确超时；
- 认证、限流、网络和响应错误被安全分类；
- 报告、历史、审计和日志无真实测试秘密；
- 未配置 Key 时本地分析和训练不受影响；
- 现有决策 Schema、Guardrails、训练闭环和 S1.2 快照门禁回归通过。

### 21.6 UI 与真实环境验收

- Chromium 检查两张配置卡、密码框不预填、状态不泄漏、环境变量只读、风险确认和 HTTP 警告；
- 使用本机模拟 OpenAI-compatible 服务检查保存、测试、替换、删除、错误正文脱敏、禁止重定向和内网开关；
- 在受控 Windows 测试目标中验证凭据跨进程重启后可用，删除后不可用；测试结束仅删除明确创建的测试 target；
- 不把真实客户 Key 写入自动化测试、截图、日志或仓库。

## 22. 验收标准

S1.3 只有同时满足以下条件才能由 Codex 给出验收通过：

1. `config.yaml`、`config.template.yaml`、`APP_CONFIG` 和所有导出中无真实 API Key；
2. 三个联网调用入口统一通过 resolver 获取凭据；
3. Windows Credential Manager 持久化、环境变量覆盖、五分钟缓存和失效策略通过测试；
4. UI 支持填写、测试、替换、删除及明确确认迁移，且任何读取接口不回显 Key；
5. 默认 DeepSeek 文本/决策和通义千问视觉配置可用；
6. 自定义 OpenAI-compatible Endpoint 受策略保护，内网能力必须显式开启；
7. 供应商错误正文、日志、审计、历史、SSE 和 API 中注入的测试秘密均被移除；
8. 旧明文 Key 不被静默使用或迁移，失败路径不破坏原文件；
9. 未配置 Key 时本地训练与分析正常，训练成功状态不被联网分析失败覆盖；
10. S1.3 定向测试、关键兼容套件和完整 `auto_tune/tests` 全部通过；
11. Chromium 完成真实设置交互检查；
12. Codex 独立检查 Windows 测试凭据 target 已清理，且未触碰客户其他凭据；
13. S1.2 与 S1.3 在同一提交前重新执行敏感信息扫描、待提交清单检查和完整测试，经艾卡确认后才提交或推送。

不要求使用真实收费厂商 Key 完成自动化验收。若艾卡提供专用测试 Key 并明确授权，可另做一次受控真实连接测试，但 Key 不落盘、不进入命令历史和验收报告。

## 23. Claude Code 交付边界

- Claude Code 只实现业务代码和对应测试，不修改本规格、路线图、实施计划、README、DOCX 或发布说明。
- 开始编码前读取本规格和 Codex 后续生成的 S1.3 实施计划。
- 严格按测试先行顺序分批实施，不扩大到提示词优化、Cloud、Linux Secret Service 或其他架构重构。
- 不调用真实客户 API，不读取、打印、移动或删除现有真实 Key。
- 不自行提交或推送；交付时报告改动文件、测试命令与结果、偏离计划之处和遗留风险。
- Codex 独立审查、运行自动化测试、模拟服务、Windows 凭据与 Chromium 验收后，才可判定 S1.3 是否通过。

## 24. 文档与版本策略

- 本规格经艾卡确认后，Codex 再编写对应实施计划。
- S1.3 完成后由 Codex 更新交接、路线图、总实施计划、研发执行版 DOCX 和客户凭据轮换说明；按艾卡 2026-08-21 的决定，后续不再同步生成 PDF。
- 旧版研发执行文档按 `AGENTS.md` 移入根目录 `历史文档/`，不在 `docs/` 与新版并列。
- 按艾卡要求，S1.2 不单独提交；S1.3 验收完成后，S1.2 与 S1.3 一起进入提交前检查。本规格与实施计划当前也不提交、不推送。

## 25. 后续批次

- 两个大模型分析提示词优化在 S1.3 之后单独评审，不与凭据安全改造混合。
- Linux Secret Service 随 Studio Linux 正式适配批次设计。
- Cloud 多用户凭据使用 Secret Manager/KMS、租户隔离和权限审计，另立规格。
- S1.4 继续处理目录选择、成员数量、总容量及遗留 ZIP 解压后容量限制。

## 26. 实际实现与验收记录（2026-08-21）

### 26.1 完成结论

S1.3 已完成 Codex 独立代码审查、自动化测试、模拟供应商、Windows Credential Manager、Chromium 和艾卡真实 API 人工验收。代码与文档当前仍在本地未提交工作区，按艾卡要求等待 S1.2+S1.3 联合提交前检查。

### 26.2 实际实现

- 新增 `auto_tune/modules/security/`，收敛凭据解析、Windows 存储、Endpoint 策略和统一脱敏。
- 环境变量每次解析即时读取并保持最高优先级；最多五分钟缓存只用于 Windows 存储值。
- 文本诊断和调参决策共享 `text` 凭据，视觉分析使用 `vision` 凭据；三个调用入口均不再读取 YAML `api_key`。
- 设置 API/UI 支持非敏感配置、保存、测试、删除和旧明文迁移；已有安全凭据时迁移返回 409，避免覆盖。
- AI 配置模板属性显式 HTML 转义，CSRF Token 使用 Jinja `tojson` 注入，修复服务端首次渲染的存储型 XSS 风险。

### 26.3 验收证据

- S1.3 九文件定向套件：`180 passed`。
- 完整套件：`396 passed, 2 warnings, 0 skipped`；warning 为既有 sklearn PCA 常量数据数值告警。
- 模拟供应商：302 未跟随，500 错误正文和假密钥未进入公开错误，错误分类稳定。
- Windows 凭据：随机验收 target 创建、跨进程读取、删除和不存在复核全部成功，未触碰生产 target。
- Chromium：两张卡正常、密码框不预填、旧凭据提示和 HTTP 未加密警告正常、控制台无错误。
- 艾卡人工验收：轮换后的新 Key 保存成功、连接成功、重新大模型分析有效。
- 清理复核：`config.yaml` 已无 `llm.api_key`/`vision.api_key`，5 个历史明文测试文件已删除。

### 26.4 遗留风险

- DNS rebinding 理论窗口仍存在；当前在紧邻请求前复核 DNS、禁止自动重定向、默认拒绝私网。连接固定到已校验 IP 属后续独立加固，不阻断 S1.3。
- Studio 非回环监听仍不具备多人认证能力；多人凭据托管只在 Cloud 以 Secret Manager/KMS 和租户权限另行实现。

## 27. Linux 与 Docker 后续兼容设计

本节是已批准的后续兼容方向，不属于 S1.3 已实现验收范围。

### 27.1 部署矩阵

| 部署方式 | S1.3 当前能力 | 后续推荐 |
|---|---|---|
| Windows单机 | Windows Credential Manager、环境变量 | 保持 |
| Linux桌面单机 | 环境变量 | Secret Service/libsecret |
| Linux无桌面服务 | 环境变量 | systemd credentials或受控 Secret文件 |
| Docker/Compose | 环境变量 | Docker Secrets只读挂载 |
| Kubernetes/Cloud | 不属于 Studio | Secret Volume、Vault或云 KMS |

### 27.2 Secret文件接口

后续 S1.3.1 计划增加 `AUTO_TUNE_TEXT_API_KEY_FILE` 和 `AUTO_TUNE_VISION_API_KEY_FILE`。解析优先级为：直接环境变量 → Secret文件 → 操作系统凭据库 → 未配置。

Secret文件必须是绝对路径普通文件，禁止符号链接，限制文件大小和凭据长度；Linux推荐权限 `0600`。UI只返回 `configured/source/writable=false`，不得读取、覆盖、删除或导出外部 Secret。更新 Secret后应重启容器，或由受控接口立即失效对应缓存。

### 27.3 镜像安全边界

- Key不得进入 Dockerfile、镜像层、构建参数、提交到Git的 `.env`、Compose正文、健康检查URL或容器日志。
- 同一镜像更新不要求重新生成 Key；新宿主机或新部署实例必须重新注入 Secret。
- 推荐每个部署实例使用独立 Key，便于独立吊销、调用归因和费用审计。
- Linux Secret Service依赖桌面会话，不得作为容器或无桌面服务器的唯一实现。
