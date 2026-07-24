"""Internationalization for Auto-Tune UI.

Usage in templates:
    {{ _("Hello") }}  →  "你好" (zh) or "Hello" (en)
"""

TRANSLATIONS: dict[str, dict[str, str]] = {
    "zh": {
        # App header
        "YOLOv8 Auto-Tuning Agent": "YOLOv8 自动调优系统",
        "Dataset Analysis / Training Diagnosis / Hyperparameter Optimization":
            "数据集分析 / 训练诊断 / 超参数优化",

        # Navigation
        "Projects": "项目总览",
        "Dataset Report": "数据集报告",
        "Intelligent Analysis": "智能分析",
        "Training Monitor": "训练监控",
        "History": "历史记录",

        # Projects page
        "Current Project": "当前项目",
        "YOLOv8 detection project": "YOLOv8 检测项目",
        "Detection Target": "检测目标",
        "Data Type": "数据类型",
        "No project configuration found.": "未找到项目配置。",
        "Edit": "编辑",
        "Edit Project": "编辑项目",
        "Project Name": "项目名称",
        "Description": "项目描述",
        "Save": "保存",
        "Editing...": "保存中...",
        "Project saved": "项目已保存",
        "Save failed": "保存失败",
        "Dataset Analysis": "数据集分析",
        "Training Analysis": "训练分析",
        "Auto-Tuning": "自动调优",
        "annotations": "标注数",
        "Label Rate": "标注率",
        "Quality Score": "质量评分",
        "Key Issues": "关键问题",
        "None": "无",
        "No data": "暂无数据",
        "Run dataset analysis first": "请先运行数据集分析",
        "View Details": "查看详情",
        "Best mAP50": "最佳 mAP50",
        "Runs Analyzed": "已分析运行",
        "Avg mAP50": "平均 mAP50",
        "Runs with Issues": "有问题的运行",
        "Run training analysis first": "请先运行训练分析",
        "total iterations": "总迭代次数",
        "No tuning yet": "暂无调优",
        "Start auto-tuning from Intelligent Analysis": "从智能分析开始自动调优",
        "Start Tuning": "开始调优",
        "Failed": "失败",
        "Passed": "通过",
        "Retry": "重试",

        # Dataset Report page
        "Dataset Analysis Report": "数据集分析报告",
        "Analysis Parameters": "分析参数",
        "Save Config": "保存配置",
        "Config saved": "配置已保存",
        "Config save failed": "配置保存失败",
        "Total Images": "总图片数",
        "Label Rate (detailed)": "标注率（详细）",
        "labeled": "已标注",
        "unlabeled": "未标注",
        "Blur Ratio": "模糊率",
        "images below threshold": "张低于阈值",
        "Small Objects": "小目标",
        "Tiny / Small bboxes": "微小框 / 小框",
        "Class Distribution": "类别分布",
        "Image Quality": "图像质量",
        "Overexposure": "过曝",
        "Underexposure": "欠曝",
        "Low SNR": "低信噪比",
        "BBox Size Distribution": "边界框尺寸分布",
        "Tiny": "微小",
        "Small": "小",
        "Medium": "中",
        "Large": "大",
        "Spatial Distribution": "空间分布",
        "Center Concentration": "中心集中度",
        "Edge Ratio": "边缘占比",
        "Class Balance": "类别平衡",
        "Balanced": "平衡",
        "Yes": "是",
        "No": "否",
        "Imbalance Ratio": "不平衡比",
        "Long-tail Classes": "长尾类",
        "Key Issues (detailed)": "关键问题（详细）",
        "No key issues detected": "未检测到关键问题",
        "AI Summary": "AI 总结",
        "Analysis Thresholds & Parameters": "分析阈值与参数",
        "Parameter": "参数",
        "Value": "值",
        "Description": "描述",
        "Laplacian variance threshold for blur detection": "拉普拉斯方差模糊检测阈值",
        "Pixel value below this = underexposed": "低于此值的像素判定为欠曝",
        "Pixel value above this = overexposed": "高于此值的像素判定为过曝",
        "Relative area < this = tiny": "相对面积小于此值 = 微小",
        "Relative area < this = medium": "相对面积小于此值 = 中",
        "IoU above this = high overlap": "IoU 高于此值 = 高重叠",
        "Below avg_count x this = long-tail": "低于平均值 x 此值 = 长尾",
        "DBSCAN clustering radius": "DBSCAN 聚类半径",
        "DBSCAN min samples": "DBSCAN 最小样本数",
        "No Dataset Analysis Available": "暂无数据集分析结果",
        "Upload a dataset ZIP above to start analysis.": "上传数据集 ZIP 开始分析。",
        "Please run Module A dataset analysis first.": "请先运行数据集分析模块。",
        "Dataset quality score": "数据集质量评分",
        "Detected": "检测到",
        "issue(s)": "个问题",
        "No critical issues detected.": "未检测到关键问题。",
        "Pay attention to long-tail classes": "请注意长尾类",
        "Small object ratio is high": "小目标比例过高",
        "consider increasing imgsz or use P2/P3 model.": "建议增大 imgsz 或使用 P2/P3 模型。",
        "Selected": "已选择",
        "Uploading and analyzing...": "上传并分析中...",
        "Tuning complete": "调优完成",
        "Vision Analysis": "视觉分析",
        "Vision analysis requires confusion_matrix_normalized.png and val_batch*.jpg from the training output directory.":
            "视觉分析需要训练输出目录中的图片文件（confusion_matrix_normalized.png、val_batch*.jpg）。",
        "Upload a complete YOLO train directory ZIP, or ensure local detect/ directory has training results.":
            "请上传包含以上图片文件的完整 YOLO train 目录 ZIP，或确保本地 detect/ 目录下有训练结果。",
        "Connection error": "连接错误",
        "go to Intelligent Analysis page": "前往智能分析页面",
        "detection project": "检测项目",

        # Upload
        "Upload Dataset": "上传数据集",
        "Upload a ZIP file containing your YOLO dataset (images + labels + data.yaml)":
            "上传包含 YOLO 数据集的 ZIP 文件（图片 + 标注 + data.yaml）",
        "Drop ZIP file here or click to upload": "拖拽 ZIP 文件到此处或点击上传",
        "Upload & Analyze": "上传并分析",
        "Analyzing...": "分析中...",
        "Analysis complete!": "分析完成！",
        "Enter dataset folder path...": "输入数据集文件夹路径...",
        "Enter train directory path...": "输入训练目录路径...",
        "Browse...": "浏览...",
        "Analyze": "分析",
        "Analyzing dataset...": "数据集分析中...",
        "Analyzing training results...": "训练结果分析中...",
        "Enter the full path to your dataset folder (must contain images/ and labels/ or data.yaml)":
            "输入数据集文件夹的完整路径（需包含 images/ 和 labels/ 或 data.yaml）",
        "Enter the path to a YOLO train directory (must contain results.csv and args.yaml)":
            "输入 YOLO 训练目录的路径（需包含 results.csv 和 args.yaml）",
        "Select this folder": "选择此文件夹",

        # Intelligent Analysis page
        "Intelligent Analysis (Page)": "智能分析",
        "Upload Training Report": "上传训练报告",
        "Drop training report JSON here or click to upload": "拖拽训练报告 JSON 到此处或点击上传",
        "Upload & Analyze Training": "上传并分析",
        "Upload YOLO train directory (ZIP) or training report (JSON) for analysis":
            "上传 YOLO train 目录 (ZIP) 或训练报告 (JSON) 进行分析",
        "Uploading YOLO train directory...": "正在上传 YOLO train 目录...",
        "Run Name": "运行名称",
        "best epoch": "最佳轮次",
        "Analysis result": "分析结果",
        "Training Summary": "训练总结",
        "Hyperparameter Changes": "超参数修改",
        "Agent suggests": "智能体建议",
        "change(s)": "项修改",
        "Parameter (hp)": "参数",
        "Current": "当前值",
        "Suggested": "建议值",
        "Reason": "原因",
        "Suggested by tuning agent": "由调优智能体建议",
        "No hyperparameter changes suggested yet.": "暂无超参数修改建议。",
        "Start tuning to get recommendations.": "开始调优以获取建议。",
        "Probe mode (10 epochs first, then decide)": "探查模式（先跑 10 轮，再决定是否继续）",
        "Auto-analyze after training (Module B)": "训练后自动分析（模块 B）",
        "Auto-loop (retrain with new params)": "自动循环（使用新参数重新训练）",
        "Auto-Tuning Controls": "自动调优控制",
        "Reference Run": "参考运行",
        "Auto-detect": "自动检测",
        "Max Retries": "最大重试次数",
        "Mode": "模式",
        "Dry-Run (generate plan only)": "干运行（仅生成计划）",
        "Full Training": "完整训练",
        "Confirm & Start Training": "确认并开始训练",
        "Tuning Progress": "调优进度",
        "Tuning History": "调优历史",
        "Waiting to start...": "等待开始...",
        "Params:": "参数：",
        "No Training Data Available": "暂无训练数据",
        "Please run training analysis (Module B) first.": "请先运行训练分析模块（模块 B）。",
        "Running...": "运行中...",
        "Connecting...": "连接中...",
        "Initializing...": "初始化中...",

        # LLM & Vision analysis
        "LLM Analysis Report": "大模型分析报告",
        "Vision Analysis": "视觉分析",
        "Confusion Matrix": "混淆矩阵分析",
        "Error Crop Analysis": "错误裁剪分析",

        # Training Monitor page
        "Training Monitor (Page)": "训练监控",
        "Epochs": "训练轮数",
        "Configured": "已配置",
        "Best mAP50 (metric)": "最佳 mAP50",
        "Final metric": "最终指标",
        "mAP50-95": "mAP50-95",
        "COCO-style metric": "COCO 风格指标",
        "Box Loss": "边框损失",
        "Validation box loss": "验证集边框损失",
        "Loss Curves": "损失曲线",
        "Loss Chart": "损失图",
        "Blue: train_loss  Red: val_loss": "蓝：训练损失  红：验证损失",
        "mAP Curves": "mAP 曲线",
        "mAP Chart": "mAP 图",
        "Green: mAP50  Orange: mAP50-95": "绿：mAP50  橙：mAP50-95",
        "Training Log": "训练日志",
        "Latest run:": "最近运行：",
        "No training runs recorded yet.": "暂无训练记录。",
        "Start New Training": "开始新训练",
        "Idle": "空闲",
        "Running": "运行中",
        "training completed": "训练完成",

        # Probe
        "Probe Complete": "探查完成",
        "epochs": "轮",
        "Loss is decreasing steadily. No anomalies detected. Continue training?":
            "损失稳步下降，未检测到异常。是否继续训练？",
        "Continue Training": "继续训练",
        "Retune": "重新调优",

        # History page
        "Training History": "训练历史",
        "#": "#",
        "Time": "时间",
        "Diagnosis": "诊断",
        "Changes": "修改",
        "Status": "状态",
        "View": "查看",
        "Details": "详情",
        "Full Diagnosis": "完整诊断",
        "Guardrails": "防护检查",
        "All checks passed": "全部检查通过",
        "No guardrails data": "无防护检查数据",
        "No Tuning History": "暂无调优历史",
        "Start auto-tuning to see history here.": "开始自动调优以查看历史记录。",
        "Refresh": "刷新",
        "Export JSON": "导出 JSON",
        "Warnings:": "警告数：",
        "Errors:": "错误数：",
        "No hyperparameter changes": "无超参修改",

        # First-time training
        "Start Training": "开始训练",
        "Launch a new YOLOv8 training from scratch. Configure the dataset path and training parameters below.":
            "启动新的 YOLOv8 从头训练。请配置数据集路径和训练参数。",
        "(dataset path)": "（数据集路径）",
        "Model": "模型",

        # Generic
        "Loading...": "加载中...",
        "Error": "错误",
        "Success": "成功",
        "Close": "关闭",
        "Cancel": "取消",
        "Confirm": "确认",
        "Switch Language": "切换语言",
        "中文": "中文",
        "English": "English",

        # Best iteration tracking
        "Evaluation Mode": "评估模式",
        "Comprehensive (+Precision+Recall)": "全面模式（含 Precision+Recall）",
        "Quick (mAP50+mAP50-95)": "快速模式（仅 mAP50+mAP50-95）",
        "Best Iteration": "最佳迭代",
        "Best Iteration Full Analysis": "最佳迭代完整分析",
        "Iteration": "迭代",
        "Training directory": "训练目录",
        "View full analysis": "查看完整分析报告",
        "Best result across all tuning iterations": "所有调优迭代中的最佳结果",
        "mAP50 progression": "mAP50 变化趋势",
        "No mAP50 data": "暂无 mAP50 数据",
        "综合评分（快速）": "综合评分（快速）",
        "综合评分（全面）": "综合评分（全面）",
        "Tuning complete": "调优完成",
        "No analysis data available for this training run.": "该训练运行暂无分析数据。",
        "Save Report": "保存报告",
        "Saving...": "保存中...",
        "Saved": "已保存",
    },
    "en": {
        # Only include entries that differ from the key (no-op / identity entries can be omitted)
        "中文": "中文",
        "English": "English",
        "Switch Language": "Switch Language",
    },
}


def get_translations(lang: str) -> dict[str, str]:
    """Get translation dictionary for a language code."""
    return TRANSLATIONS.get(lang, TRANSLATIONS.get("zh", {}))


def translate(text: str, lang: str = "zh") -> str:
    """Translate a string to the target language."""
    translations = get_translations(lang)
    return translations.get(text, text)


def make_translator(lang: str):
    """Create a _() function bound to a language."""
    translations = get_translations(lang)

    def _(text: str) -> str:
        return translations.get(text, text)

    return _
