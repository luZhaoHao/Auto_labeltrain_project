
可选架构设计方案
在 CVAT 中集成 YOLOv8 模型训练能力
目标检测第一阶段 · 同服务器多 GPU · 训练与 ONNX 导出
文档定位  本方案是供团队评估的候选技术路线，不代表已经决定在 CVAT 内建设训练平台。团队仍可选择“CVAT 负责标注、独立训练软件负责训练”的解耦路线。

项目
内容
方案版本
讨论稿 v1.0
目标读者
数据工程、AI 模型、后端、前端、运维及技术决策人员
当前优先级
非紧急；作为后续能力储备与方案比较依据
首期范围
Ultralytics YOLOv8 Detect；一个 Project 选择多个 Task
明确不含
YOLOv5/11/26、分类、分割、OBB、多节点训练、自动部署推理服务


1. 执行摘要
推荐结论  如果团队确需在 CVAT 内形成“标注—训练—模型交付”闭环，可采用常驻训练容器、单任务独占单卡、Redis/RQ 排队、数据快照与质量审查、训练完成自动导出 ONNX 的方案。若已有成熟训练软件，则优先采用解耦集成，避免重复建设。

当前 CVAT 已具备项目、任务、标注、数据集导出、组织权限、异步队列与自动标注推理能力，但没有模型训练管理模块。本方案在不把 CUDA/PyTorch 依赖塞入 CVAT API 进程的前提下，增加独立训练服务。
1.1 两条候选路线
维度
路线 A：CVAT 内集成训练
路线 B：标注与训练软件分离
适用条件
希望一站式闭环；训练流程相对标准
已有训练平台；模型类型和实验方式复杂
优势
操作入口统一；权限、数据版本和模型可追溯
职责清晰；复用成熟调度、实验跟踪和模型管理
代价
需要自行建设 GPU 调度、监控、存储和维护
需要做数据/身份/任务状态集成
推荐判断
仅在业务闭环价值明确时实施
若已有可用训练软件，优先选择

1.2 决策门槛
至少有一类高频、稳定的 YOLOv8 Detect 训练流程需要交给非算法人员使用。
现有训练软件不能以较低成本接收 CVAT 数据并回传模型。
团队愿意长期维护 GPU 调度、训练镜像、权重安全、磁盘容量和版本兼容。
首期范围能够严格限制，不演变为通用 MLOps 平台。
2. 目标、范围与非目标
2.1 第一阶段目标
在一个 CVAT Project 下选择多个 Task，生成固定、可追溯的数据集快照。
训练 Ultralytics YOLOv8 目标检测模型。
同服务器多 GPU 并存；每个任务独占一张 GPU，多任务并行。
训练前执行确定性数据质量审查；普通用户确认风险后可继续。
保存 best.pt、last.pt、训练日志、指标和 best.onnx。
复用 CVAT 的 Organization、全局管理员和 OPA 权限体系。
2.2 第一阶段非目标
单任务多卡 DDP、单卡多任务、跨服务器或 Kubernetes 分布式训练。
分类、实例分割、OBB；YOLOv5、YOLO11、YOLO26 等系列。
超参数搜索、实验编排、模型审批、模型注册中心。
自动构建或部署 Nuclio 推理函数；首期仅转换、验证和下载 ONNX。
允许用户执行任意 YAML、命令行参数或 Python 代码。
3. 推荐总体架构
核心原则是控制面与训练面分离：CVAT API 管理权限、任务和元数据；独立训练服务管理 GPU 与训练子进程；大文件保存到持久化训练目录。
组件
职责
主要技术
CVAT UI
创建任务、选择数据、质量确认、监控与下载
React、TypeScript、Ant Design
model_training Django App
训练记录、权限、参数校验、API 与文件索引
Django、DRF、OPA
训练队列
准备数据、调度与转换任务
现有 Redis/RQ
cvat_training_service
GPU 发现、排队、启动/停止子进程、状态采集
Python、NVIDIA Runtime
YOLOv8 训练适配器
数据校验、命令构建、指标解析、产物收集
Ultralytics YOLOv8、PyTorch
持久化目录
快照、日志、Checkpoint、ONNX
Docker Volume/宿主机受控目录
PostgreSQL
状态、配置、指标摘要、审计信息
现有 PostgreSQL

执行隔离  第一版采用“常驻训练容器 + 每任务独立子进程”。执行层保留 TrainingExecutor 接口，未来可替换为每任务独立 Docker 容器，而不重写 CVAT API 和页面。

4. 端到端业务流程
步骤
阶段
行为
1
选择数据
选择 Organization、Project，并勾选同一 Project 下多个 Task。
2
固化快照
提交后立即固化图片、标注、类别映射和来源版本。
3
质量审查
自动排除不可训练数据；展示警告并等待用户确认。
4
配置训练
选择模型规格、权重、epochs、imgsz、batch、workers。
5
等待 GPU
无空闲 GPU 时排队；GPU 空闲后分配唯一 GPU ID。
6
执行训练
独立子进程训练；按 Epoch 回写指标，5 秒采集 GPU 状态。
7
导出模型
保存 best.pt/last.pt；自动将 best.pt 转换为 best.onnx。
8
交付结果
展示指标、混淆矩阵、日志和模型下载；首期不自动部署。

4.1 状态机
创建中 → 准备数据 → 数据审查 →（有警告时）等待用户确认 → 等待 GPU → 训练中 → 导出 ONNX → 完成。异常状态包括：失败、已取消、中断；ONNX 失败不改变训练成功状态。
5. 数据选择、快照与划分
5.1 数据选择规则
一次训练只能选择一个 Project 下的多个 Task。
以 Project 标签体系建立统一类别索引，并保存 label_id、名称与 YOLO class index 映射。
提交与实际生成快照时各执行一次 Project/Task 读取权限检查。
仅接受矩形框；其他 shape 不隐式转换，并在审查结果中提示。
未标注图片默认保留为负样本。
5.2 Train/Validation 划分
优先使用 Task 已设置的 Subset（Train、Validation）；如果未配置，则允许按比例自动划分，默认 80%/20%，固定随机种子 42。必须检查相同图片不得同时进入 Train 和 Validation。
5.3 快照生命周期
用户提交后立即创建快照，使排队期间的标注修改不影响本次训练。快照默认保留 30 天；清理后仍保留训练记录、日志摘要、指标和模型产物。
6. 数据质量审查
原则：尽可能提示，不要求普通用户强制整改。无法参与训练的数据自动排除；排除后训练集/验证集仍有效时，用户确认即可继续。
检查类别
示例
首期处理
图像有效性
损坏、无法解码、宽高为零、格式不支持
自动排除并列明原因
标注几何
框越界、宽高≤0、无法解析
自动排除无效框；必要时排除图片
一致性
非当前 Project 标签、类别映射失败
无法建立映射时阻止提交
重复与泄漏
文件哈希重复、跨 Train/Validation 重复
泄漏必须消除；其他重复作为警告
分布风险
类别不均衡、样本过少、验证集缺类
警告，用户确认后继续
框统计
极小框、超大框、异常宽高比
警告并展示数量
基础统计
空标注、图像尺寸、每类图片/框数
信息展示

唯一硬性门槛  排除无效数据后，如果训练集为空、验证集为空或类别映射无法建立，则无法生成合法 YOLOv8 数据集，必须阻止提交。其他风险均允许普通用户确认后继续。

确认记录必须包含确认人、确认时间、问题摘要、排除数量、审查规则版本和数据快照 ID。
7. 训练配置与权重
7.1 页面开放参数
参数
支持范围/默认策略
说明
模型规格
yolov8n/s/m/l/x
首期仅 Detect
预训练权重
官方权重、用户上传、历史 best.pt/last.pt
用于迁移学习或继续训练
epochs
真实默认值预填
正整数并设置合理上限
imgsz
真实默认值预填
首期单一方形尺寸
batch
真实默认值预填
显存风险仅提示，不擅自改值
workers
真实默认值预填
受服务器 CPU/内存上限约束

优化器、学习率、早停、AMP、数据增强、验证与保存等参数全部采用固定 Ultralytics 版本的默认值，不在页面开放。系统仍保存 system_defaults、user_overrides 和 resolved_config 三份配置。
7.2 上传 .pt 的安全边界
限制扩展名、文件大小和组织存储配额，并计算 SHA-256。
上传文件只能在隔离训练容器中加载，禁止在 CVAT API 进程直接反序列化。
校验是否能由固定版本 Ultralytics 加载、是否为 YOLOv8 Detect，以及类别元数据。
保存来源、上传者、时间、文件哈希、解析结果与兼容性状态。
8. GPU 调度与训练执行
8.1 第一阶段调度规则
训练服务启动时通过 NVIDIA 管理接口发现 GPU 数量、型号和显存。
一个训练任务独占一张 GPU；通过 CUDA_VISIBLE_DEVICES 绑定。
多张 GPU 可并行多个任务；无空闲 GPU 的任务保持 queued。
接口预留 gpu_count 和 gpu_ids，但第一版强制 gpu_count=1。
第一版不允许多个任务共享同一张 GPU，也不支持单任务多卡。
8.2 调度一致性
GPU 分配必须使用数据库锁或等效原子租约，避免多个调度循环占用同一张卡。租约记录任务、GPU、进程 ID、服务实例和心跳；训练服务重启后对照数据库、进程与 Checkpoint 修复状态。
8.3 扩展路径
未来若支持单任务多卡，只需开放 gpu_count>1、实现成组资源分配并由 Ultralytics/PyTorch DDP 执行。若需要更强隔离，可新增 DockerTrainingExecutor。
9. ONNX 极简导出
产品原则  训练完成后默认自动把 best.pt 转换为 best.onnx。用户不需要理解 opset、dynamic、simplify 等参数。

输入尺寸继承训练 imgsz；固定 batch=1；固定输入尺寸。
opset、dynamic、simplify、half 等由系统固定并隐藏。
转换失败不影响训练成功，也不影响 best.pt 下载。
历史模型页面提供一个“转换为 ONNX”按钮，可单独重试。
导出后执行 ONNX 结构检查、ONNX Runtime 加载和一张样例图片的 PT/ONNX 对比。
页面仅展示转换状态、文件大小、输入尺寸、验证结果和下载按钮。
首期边界：只保证标准 YOLOv8 Detect 权重转换，不承担任意 PyTorch .pt 转换，也不自动部署到 Nuclio。
10. 权限模型
复用 CVAT 两层权限：全局 admin，以及 Organization 的 owner、maintainer、supervisor、worker。训练模块新增独立 OPA/Rego 策略，不在 View 中散布角色判断。
角色
建议权限
普通成员
创建任务；管理、停止、重试、下载自己的训练任务；必须具备所选 Project/Task 读取权限
Supervisor
查看本组织任务；停止异常任务；不删除共享模型和历史记录
Owner / Maintainer
管理本组织全部训练任务、权重和产物；查看本组织资源占用
全局 Admin
管理全平台任务、GPU、队列、优先级、模型、磁盘和训练服务配置

所有下载、停止、重试、删除和转换操作均再次经过 OPA 检查。训练任务必须保存 organization_id、owner_id、project_id 和 task_ids。
11. 存储、保留与清理
PostgreSQL 保存元数据；快照和产物保存到独立持久化目录。普通用户不得提交服务器路径。建议目录结构：organization-{id}/training-{id}/{dataset,logs,checkpoints,exports}。
数据
默认保留
清理规则
数据集快照
30 天
训练、续训或检查占用时禁止清理
训练记录与配置
长期
保留审计与复现信息
日志
长期
可将原始大日志归档，保留可查询摘要
best.pt / last.pt
长期
授权用户手动删除；引用中禁止删除
best.onnx
长期
与来源 PT 保持关联

管理员页面展示组织和全平台磁盘占用。
训练前检查剩余空间；不足时拒绝启动，避免占满 CVAT 服务器。
删除训练记录默认不立即删除模型文件；需要显式的产物清理操作。
12. 监控、指标与异常恢复
12.1 监控
状态：排队、准备数据、审查、等待确认、等待 GPU、训练、导出、完成、失败、取消、中断。
每个 Epoch 更新 box_loss、cls_loss、dfl_loss、Precision、Recall、mAP50、mAP50-95。
GPU 编号、型号、显存与利用率每 5 秒更新；不做 Batch 级曲线。
展示当前 Epoch、已用时间、预计剩余时间、实时日志、训练曲线和混淆矩阵。
12.2 异常恢复
场景
策略
数据准备/审查失败
允许从相应阶段重试
GPU OOM/训练进程退出
标记失败，不自动循环重启；保留日志和 Checkpoint
服务/服务器重启
标记中断；存在 last.pt 时提示用户创建续训任务
用户停止
先优雅停止并保存最近 Checkpoint，超时后终止子进程
ONNX 导出失败
训练保持成功；导出单独失败并允许重试
重试
复制原配置创建新运行，保存 parent_training_id，不覆盖原记录

13. 建议 API 与数据模型
13.1 核心数据实体
实体
关键字段
TrainingJob
id、organization、owner、project、task_ids、status、stage、gpu_count、gpu_ids、parent_id
DatasetSnapshot
source versions、class mapping、split config、path、expires_at、quality_rule_version
QualityReport
summary、warnings、excluded_items、confirmed_by、confirmed_at
TrainingConfig
system_defaults、user_overrides、resolved_config、trainer_image_digest
TrainingMetric
epoch、losses、precision、recall、mAP、timestamp
ModelArtifact
type(pt/onnx/log/plot)、path、size、sha256、validation_status
GpuLease
gpu_id、training_id、service_instance、pid、heartbeat、acquired_at

13.2 API 轮廓
GET/POST /api/training/jobs：按 OPA 过滤列表、创建任务。
GET /api/training/jobs/{id}：详情、阶段和指标摘要。
POST /confirm-quality、/stop、/retry、/resume：明确动作接口。
GET /logs、/metrics：增量读取。
GET /artifacts/{id}/download：鉴权下载。
POST /model-artifacts/{id}/export-onnx：独立转换或重试。
GET /training/gpus：普通用户返回简化容量；管理员返回完整状态。
14. UI 信息架构
建议新增“训练”一级入口，但保持页面简单。首期向导四步：选择数据 → 数据审查 → 配置训练 → 确认提交。
页面
首期内容
训练任务列表
状态、模型规格、Project、创建人、Epoch、GPU、创建时间
选择数据
Project、多个 Task、图片/框统计、Subset 与划分方式
数据审查
阻断项、排除项、警告、类别分布、确认继续
配置训练
仅六项开放参数和权重来源
训练详情
状态、Epoch、GPU、核心指标、日志、停止/重试
模型产物
best.pt、last.pt、best.onnx 的验证状态与下载
管理员资源页
GPU、排队任务、磁盘占用、失败任务和清理

15. 测试与验收
15.1 必要测试
权限矩阵：所有角色、跨组织访问、Project/Task 二次校验。
快照一致性：标注变更不影响已提交训练；类别映射稳定。
质量审查：损坏图、无效框、重复、泄漏、空集合和用户确认审计。
调度并发：多任务竞争、GPU 租约、服务重启、取消与孤儿进程。
训练适配器：参数解析、Epoch 指标、Checkpoint 和失败日志。
权重安全：不兼容、损坏、超限、非 YOLOv8 Detect 和恶意上传隔离。
ONNX：导出、加载、推理对比和失败重试。
存储清理：30 天策略、引用保护、磁盘不足和并发下载。
端到端：从多 Task 选择到模型下载的完整路径。
15.2 第一阶段验收标准
领域
通过标准
功能
YOLOv8 Detect 训练、监控、停止/重试、PT 与 ONNX 下载完整可用
并发
N 张可用 GPU 能并行 N 个单卡任务，额外任务稳定排队
可追溯
能从模型追溯到快照、Task、标注版本、配置、镜像和质量确认
安全
上传权重不在 API 进程加载；文件访问与组织权限隔离
恢复
异常与重启后状态可信、日志和 Checkpoint 不丢失
兼容
不影响 CVAT 标注、导入导出与现有 Worker 的稳定性

16. 分阶段路线与退出机制
阶段
范围
决策点
方案验证
用脚本/原型验证 CVAT 导出、YOLOv8 训练、指标解析、ONNX 导出
确认业务频率与现有训练软件差距
MVP
单组织试用；多 Task、质量审查、单卡任务、多 GPU 并行、基础监控
确认维护成本和用户价值
第一阶段正式版
权限、审计、存储清理、异常恢复和完整测试
是否继续在 CVAT 内建设
后续扩展
分类、分割、OBB；YOLO 其他版本；单任务多卡
按真实业务需求逐项立项

退出机制  所有训练输入和输出应保持标准格式：YOLO 数据集、结构化配置、PT/ONNX、JSON/CSV 指标。若团队后续改用独立训练软件，可以保留 CVAT 的数据选择/导出能力，并下线内置训练执行层，避免形成不可迁移的绑定。

17. 风险与缓解
风险
影响
缓解
重复建设训练平台
长期维护成本高
先评估现有训练软件；严格限制首期范围
训练抢占 CVAT 资源
标注服务变慢或磁盘耗尽
独立容器、GPU 独占、CPU/内存限制、磁盘预检
恶意或不可信 .pt
代码执行或服务受损
隔离容器解析、最小权限、文件限制与审计
版本不可复现
历史模型无法复训
固定训练镜像摘要、依赖版本、权重哈希和完整 resolved_config
权限泄漏
跨组织获取数据或模型
复用 OPA、查询过滤、下载二次鉴权
质量审查误导
用户把警告当作质量保证
明确审查范围和规则版本，不输出虚假的综合分数
需求膨胀
演变为通用 MLOps
每种新任务类型和框架单独立项、单独验收

18. 待实施前确认事项
确认最终采用路线 A（CVAT 内训练）还是路线 B（外部训练平台）。
盘点服务器 GPU、驱动、NVIDIA Container Toolkit、CPU、内存和磁盘。
选定并冻结 Ultralytics YOLOv8、PyTorch、CUDA、ONNX Runtime 版本。
确定组织级存储配额、上传权重大小限制和任务并发配额。
确定可接受的默认训练参数、参数上限和显存风险提示规则。
确定质量审查阈值及“自动排除”的最小粒度。
完成安全评审，特别是上传 .pt、容器权限和模型下载。
先制作技术原型并与现有训练软件做同一数据集的成本/体验对比。
最终建议  本方案可以作为 CVAT 内训练路线的完整设计基线，但不建议立即进入全面开发。优先做小型技术原型和方案对比；只有在一体化业务价值明显高于对接现有训练软件时，再启动正式实施。
