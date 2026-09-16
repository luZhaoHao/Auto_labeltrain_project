# Studio S1.5 运行状态与重连可靠性设计

## 1. 目标

把普通训练与自动调优的运行身份、进程事实和页面状态统一为同一份版本化契约。页面刷新、浏览器重连或服务重启后，只陈述能够由持久化记录和进程身份校验证明的事实，不把遗留状态文件或 PID 存在误报为“仍在运行”。

本批为一个统一交付批次，内部按状态契约、进程校验、API/SSE/UI 三个任务推进，统一验收和提交。S1.5 验收完成后才进入 S2。

## 2. 非目标

- 不接管服务重启前启动的训练子进程。
- 不实现断点续训、checkpoint 自动恢复或自动重启训练。
- 不引入任务队列、多用户并发、SQLite 或新依赖。
- 不修改训练参数、指标解析、数据集快照或历史记录语义。
- 不删除旧 `training_running.json` / `tuning_running.json` 的读取兼容。

## 3. 当前问题

- 普通训练和自动调优分别维护不同形状的 JSON，调优文件只有 `status=running`。
- 普通训练在内存对象丢失后，只要旧文件仍为 running，接口就会继续报告正在运行。
- 状态文件没有稳定运行身份、进程创建身份、阶段、最后事件序号和时间。
- SSE 断开后没有按运行身份和事件位置恢复“状态事实”的契约；页面只能重新猜测按钮状态。
- PID 被操作系统复用时，仅检查 PID 是否存在会误判另一进程为当前训练。

## 4. 统一状态契约

新模块：`auto_tune/modules/run_state/`。

持久化记录使用 Schema `1.0`，普通训练和自动调优共用以下字段：

```json
{
  "schema_version": "1.0",
  "run_id": "manual:550e8400-e29b-41d4-a716-446655440000",
  "run_kind": "manual",
  "status": "running",
  "phase": "training",
  "started_at": "2026-08-25T01:02:03.000000Z",
  "updated_at": "2026-08-25T01:03:04.000000Z",
  "pid": 1234,
  "process_create_token": "windows-filetime:133999999999999999",
  "last_event": {
    "seq": 12,
    "type": "training_log",
    "at": "2026-08-25T01:03:04.000000Z",
    "message": "Epoch 1/10"
  },
  "run_name": "train12",
  "terminal_reason": null
}
```

约束：

- `run_id` 每次启动唯一，格式为 `<run_kind>:<uuid4>`；`manual` 与 `tuning` 不共享命名空间。
- `status` 仅允许 `starting | running | completed | failed | cancelled | interrupted | unknown`。
- `phase` 仅陈述当前事实，允许 `preparing | launching | training | analyzing | finalizing | stopping | terminal`。
- `last_event.seq` 在单次运行内严格递增；刷新只需要读最新事件，不承诺重放完整日志。
- 每次写入使用同目录临时文件、flush、`os.fsync()`、`os.replace()`；禁止半写文件覆盖最后有效状态。
- 终态保留在状态文件中供刷新读取，不再以“删除文件”等价表达终态。
- 状态记录不得包含 API Key、完整命令、数据集业务绝对路径或训练日志全文。

## 5. 旧状态兼容

- 旧普通训练记录 `{train_name,status,start_time}` 和旧调优记录 `{status}` 必须可读。
- 旧记录没有 `run_id` 或进程创建身份，不能证明进程仍属于本次运行。
- 旧 `running` 记录统一投影为 `status=unknown`、`running=false`、`terminal_reason=legacy_identity_unverifiable`。
- 已有旧终态按其原值映射；无法识别或损坏的文件返回 `unknown`，不得抛出 500 或覆盖原文件。
- 新写入只使用 Schema 1.0，不继续产生旧格式。

## 6. PID 身份校验与状态复核

进程身份由 `(pid, process_create_token)` 共同确定：

- Windows 使用标准库 `ctypes` 调用 `OpenProcess` + `GetProcessTimes`，以创建时间 FILETIME 形成 token。
- Linux 使用 `/proc/<pid>/stat` 的 starttime 字段形成 token。
- 平台不支持、权限不足、进程消失或 token 不一致时，均不得判定为 running。

复核规则：

1. 内存中持有当前子进程且 `poll/returncode` 表明仍存活，同时身份匹配：`running=true`。
2. 服务重启后只有状态文件：身份匹配只证明进程仍存在，不代表服务能够继续消费输出或控制它；返回 `status=interrupted`、`running=false`、`terminal_reason=controller_lost`。
3. PID 不存在：`interrupted/process_missing`。
4. PID 存在但创建 token 不同：`interrupted/pid_reused`。
5. 无法校验：`unknown/process_identity_unverifiable`。

复核结果需要原子写回新 Schema，避免每次刷新重复误判。

## 7. 普通训练与自动调优接入

### 7.1 普通训练

- 请求通过校验后创建唯一 `run_id`，阶段从 `preparing` 开始。
- 子进程成功创建后记录 PID/token，进入 `training`。
- 结构化 SSE 事件同时更新 `last_event`；允许节流持久化，但终态必须同步落盘。
- 正常结束、失败、用户停止分别写 `completed`、`failed`、`cancelled`，阶段为 `terminal`。
- 流断开不得删除状态或把运行改为完成；子进程真实终态仍由服务器执行路径记录。

### 7.2 自动调优

- 调优会话本身使用唯一 `tuning:<uuid4>`，不要继续用毫秒时间戳充当唯一身份。
- 顶层阶段至少覆盖 `preparing`、`training`、`analyzing`、`finalizing`。
- 当内部 YOLO 子进程启动时，把实际 PID/token 绑定到调优状态；迭代切换时保留同一个调优 `run_id`。
- dry-run 没有训练 PID 时，以服务内存控制器为事实来源；服务重启后统一为 `interrupted/controller_lost`。
- 审计 `session_id` 与新 `run_id` 关联但保持旧审计格式可读，不重写历史审计文件。

## 8. API、SSE 与 UI

### 8.1 API

`GET /api/training/running` 与 `GET /api/tuning/status` 返回同一形状：

```json
{
  "running": false,
  "run_id": "manual:...",
  "run_kind": "manual",
  "status": "interrupted",
  "phase": "terminal",
  "last_event": {"seq": 12, "type": "training_log", "at": "...", "message": "..."},
  "terminal_reason": "controller_lost",
  "run_name": "train12"
}
```

无记录时返回 `status=unknown`、`running=false`、`run_id=null`。为兼容现有前端，保留顶层 `running` 与 `status`。

### 8.2 SSE

- 每条新 SSE 事件包含 `run_id`、`phase` 和 `event_seq`。
- 终态事件与状态文件的 `status/phase/event_seq` 一致。
- S1.5 不建设持久化完整事件日志或进程接管。服务进程存活期间，每个运行通过有界 EventBroker 保留最近 2000 条事件，浏览器读取状态 API 后可使用 `run_id + after_seq` 重新订阅；已完成控制器最多保留 20 个、TTL 30 分钟。缓冲缺口以非终态 `replay_truncated` 控制消息说明，超出保留期或服务重启后只返回持久化终态并明确重放不完整。
- 客户端收到与当前 `run_id` 不同的事件时忽略，避免旧连接污染新运行页面。

### 8.3 UI

- 训练监控页统一展示：运行中、已完成、失败、已取消、中断、未知。
- 只有 `running=true` 时显示停止按钮。
- `interrupted` 明确显示“运行控制已中断，无法确认或继续原进程”；不得显示“已恢复”。
- `unknown` 明确显示“状态无法确认”，不得显示仍在运行。
- 页面刷新和切换到监控页时同时查询普通训练与自动调优状态，按 `updated_at` 展示最新一个；不再只用两个接口结果分别猜按钮。
- 所有服务器消息通过 `textContent` 渲染。

## 9. 失败策略

- 首次状态原子写入失败：拒绝启动训练，返回稳定错误 `RUN_STATE_PERSIST_FAILED`。
- 训练已启动后的进度状态写入失败：训练继续，但 SSE/UI产生一次警告；终态写入再次尝试。
- 终态写入失败：训练事实不改写，接口回退为 `unknown` 并记录结构化错误；不得伪造 completed。
- 损坏状态文件保留原文件，不静默覆盖；API 返回 `unknown/state_corrupt`。

## 10. 验收标准

- 普通训练与自动调优新状态均符合 Schema 1.0，`run_id` 唯一，写入原子。
- 旧状态文件可读；旧 running 不再误报正在运行。
- PID 不存在、PID 复用、权限不足和服务控制器丢失均返回保守状态。
- API、SSE、UI 对 `running/completed/failed/cancelled/interrupted/unknown` 表述一致。
- 刷新、切页和断开连接不会把终态显示为运行中，也不会声称恢复训练。
- 相关定向测试、完整 pytest、真实 Chromium 刷新/重连验收通过。
- 使用最小合法数据集做一次短 epoch 普通训练；自动调优至少做 dry-run 状态链路，若改动实际调优子进程绑定则再做一次最短真实调优冒烟。
