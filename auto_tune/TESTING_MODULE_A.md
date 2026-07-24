# Module A 数据集分析器 — 测试指南

## 环境准备

确保 `auto_tune` conda 环境已激活：

```bash
conda activate auto_tune
```

## 数据集目录结构

Module A 支持两种目录布局，自动检测无需配置。

### 方式一：标准 YOLO 结构（推荐）

```
dataset_root/
├── data.yaml              # 数据集配置文件
├── images/
│   └── train/
│       ├── img_001.jpg
│       ├── img_002.png
│       └── ...
└── labels/
    └── train/
        ├── img_001.txt
        ├── img_002.txt
        └── ...
```

### 方式二：平铺结构（图片和标签在同一文件夹）

```
dataset_root/
├── data.yaml              # 数据集配置文件
├── img_001.jpg
├── img_001.txt
├── img_002.jpg
├── img_002.txt            # 可为空（无标签）
├── img_003.jpg            # 可无对应txt（无标签图片）
└── ...
```

图片和 .txt 文件通过文件名前缀匹配（例如 `img_001.jpg` ↔ `img_001.txt`）。  
平铺模式下会自动读取每张图片的真实尺寸来解析 YOLO 归一化坐标。

**data.yaml 示例：**

```yaml
names:
  0: crack
  1: scratch
  2: dent
train: ./images/train
val: ./images/val
nc: 3
```

## 测试方法

### 方式一：命令行快速测试

```bash
conda run -n auto_tune python -c "
import json
import yaml
from auto_tune.modules.dataset_analyzer.analyzer import analyze_dataset

# 修改为你的数据集路径
dataset_dir = r'D:\path\to\your\dataset'
data_yaml_path = dataset_dir + '/data.yaml'

with open(data_yaml_path, 'r') as f:
    data_yaml = yaml.safe_load(f)

config = {
    'blur_threshold': 100.0,    # 拉普拉斯方差阈值（低于此值判定为模糊）
    'img_width': 640,            # 训练输入宽度
    'img_height': 640,           # 训练输入高度
    'high_iou_threshold': 0.7,   # 高 IoU 重叠阈值
    'dbscan_eps': 0.3,          # DBSCAN 聚类半径
    'dbscan_min_samples': 5,    # DBSCAN 最小样本数
}

result = analyze_dataset(dataset_dir, data_yaml, config)
print(json.dumps(result, indent=2, ensure_ascii=False))
"
```

### 方式二：Python 脚本文件

创建 `test_module_a.py`：

```python
import json
import yaml
from auto_tune.modules.dataset_analyzer.analyzer import analyze_dataset

dataset_dir = r'D:\path\to\your\dataset'

with open(f'{dataset_dir}/data.yaml', 'r') as f:
    data_yaml = yaml.safe_load(f)

config = {
    'blur_threshold': 100.0,
    'img_width': 640,
    'img_height': 640,
    'high_iou_threshold': 0.7,
    'dbscan_eps': 0.3,
    'dbscan_min_samples': 5,
}

result = analyze_dataset(dataset_dir, data_yaml, config)
print(json.dumps(result, indent=2, ensure_ascii=False))
```

运行：

```bash
conda run -n auto_tune python test_module_a.py
```

### 方式三：运行单元测试

验证所有基础功能正常：

```bash
cd e:\dataprocess_modeltrain\Auto_labeltrain_project
conda run -n auto_tune python -m pytest auto_tune/tests/ -v
```

预期输出：**19 passed**

## 输出 JSON 结构说明

```json
{
  "module": "dataset_analyzer",
  "version": "1.0",
  "analysis_timestamp": "2026-07-20T12:00:00Z",
  "dataset_path": "D:\\path\\to\\dataset",
  "total_images": 1000,
  "label_coverage": {
    "total_images": 1000,
    "with_labels": 850,
    "empty_labels": 50,
    "without_labels": 100,
    "label_rate": 0.85
  },
  "total_annotations": 3500,

  "class_distribution": { "0": 1500, "1": 1200, "2": 800 },
  "class_balance": {
    "is_balanced": true,
    "long_tail_classes": [],
    "imbalance_ratio": 0.53
  },

  "image_quality": {
    "blur_ratio": 0.02,
    "overexposure_ratio": 0.05,
    "underexposure_ratio": 0.03,
    "low_snr_ratio": 0.01
  },

  "bbox_analysis": {
    "tiny_bbox_ratio": 0.05,
    "small_bbox_ratio": 0.20,
    "medium_bbox_ratio": 0.45,
    "large_bbox_ratio": 0.30
  },

  "spatial_bias": {
    "center_concentration_score": 0.35
  },

  "overlap_analysis": {
    "high_iou_ratio": 0.02
  },

  "outlier_analysis": {
    "outlier_count": 3,
    "outlier_ratio": 0.015,
    "silhouette_score": 0.62
  },

  "summary": {
    "dataset_quality_score": 0.85,
    "key_issues": []
  }
}
```

### 关键字段解读

| 字段 | 说明 | 关注点 |
|------|------|--------|
| `label_coverage.label_rate` | 有标签的图片占比 | < 0.5 会标为 `low_label_coverage` 问题 |
| `image_quality.blur_ratio` | 模糊图像占比 | > 0.1 会标为 `high_blur_ratio` 问题 |
| `bbox_analysis.tiny_bbox_ratio` | 小目标占比（面积 < 图像 0.5%） | > 0.2 会标为 `tiny_bbox_high_ratio` 问题 |
| `spatial_bias.center_concentration_score` | 目标集中在画面中心的程度 | > 0.6 会标为 `center_spatial_bias` 问题 |
| `class_balance.is_balanced` | 类别是否平衡 | false 时会列出 `long_tail_classes` |
| `summary.dataset_quality_score` | 综合质量评分 [0, 1] | 越高越好 |
| `summary.key_issues` | 自动检测的问题列表 | 空数组表示无显著问题 |

## 实际测试结果示例

2026-07-20 对 `train_buchon513` 数据集的分析结果：

```json
{
  "total_images": 363,
  "label_coverage": {
    "total_images": 363,
    "with_labels": 163,
    "empty_labels": 0,
    "without_labels": 200,
    "label_rate": 0.449
  },
  "total_annotations": 163,
  "class_distribution": { "ng": { "count": 163, "ratio": 1.0 } },
  "image_quality": {
    "blur_ratio": 1.0,
    "overexposure_ratio": 0.0,
    "underexposure_ratio": 0.0,
    "low_snr_ratio": 1.0
  },
  "bbox_analysis": {
    "tiny_bbox_ratio": 0.006,
    "small_bbox_ratio": 0.288,
    "medium_bbox_ratio": 0.644,
    "large_bbox_ratio": 0.061
  },
  "spatial_bias": {
    "center_concentration_score": 0.0,
    "edge_distribution_ratio": 1.0
  },
  "summary": {
    "dataset_quality_score": 0.7,
    "key_issues": ["low_label_coverage", "high_blur_ratio"]
  }
}
```

### 各参数含义速查

| 参数 | 含义 | 你的数值 | 说明 |
|------|------|---------|------|
| `total_images` | 数据集中图片总数 | 363 | 含无标签图片 |
| `label_coverage.with_labels` | 有有效标签的图片数 | 163 | 有对应的 .txt 且内容非空 |
| `label_coverage.empty_labels` | 有 .txt 但内容为空的图片数 | 0 | |
| `label_coverage.without_labels` | 完全没有 .txt 的图片数 | 200 | |
| `label_coverage.label_rate` | 标签覆盖率 | 0.449 | < 0.5 触发 low_label_coverage 告警 |
| `total_annotations` | 所有 bbox 总数 | 163 | 每个 .txt 中的标注行总和 |
| `class_distribution` | 每个类别的实例数和占比 | ng: 163 (100%) | 单类别数据集 |
| `blur_ratio` | 模糊图片占比 | 1.0 | 所有采样图片均模糊（可能阈值偏严） |
| `overexposure_ratio` | 过曝图片占比 | 0.0 | |
| `underexposure_ratio` | 欠曝图片占比 | 0.0 | |
| `low_snr_ratio` | 低信噪比图片占比 | 1.0 | 所有采样图片 SNR < 10 |
| `tiny_bbox_ratio` | 微小目标占比 (< 图像 1%) | 0.006 | |
| `small_bbox_ratio` | 小目标占比 (1%-5%) | 0.288 | |
| `medium_bbox_ratio` | 中目标占比 (5%-20%) | 0.644 | 大部分为中目标 |
| `large_bbox_ratio` | 大目标占比 (> 20%) | 0.061 | |
| `center_concentration_score` | bbox 集中在画面中心的程度 | 0.0 | 全部 bbox 在边缘区域 |
| `edge_distribution_ratio` | bbox 在画面边缘的比例 | 1.0 | 与 center_concentration 对应 |
| `high_iou_ratio` | 高 IoU 重叠的 bbox 对比例 | 0.034 | 少量重叠 |
| `dataset_quality_score` | 综合质量分 [0, 1] | 0.7 | 受 blur_ratio 和 label_rate 拖累 |

## 可配置参数一览

所有分析阈值均通过 `config.yaml` 的 `dataset_analyzer` 段配置，每个参数都有默认值。修改后重新分析即可生效。

### 图像质量

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `blur_threshold` | 100.0 | Laplacian 方差阈值，低于此值判为模糊 |
| `under_exposure_pixel_threshold` | 50 | 像素值低于此判为欠曝 |
| `over_exposure_pixel_threshold` | 200 | 像素值高于此判为过曝 |
| `under_exposure_ratio_threshold` | 0.3 | 欠曝像素占比超过此值 → 欠曝图 |
| `over_exposure_ratio_threshold` | 0.3 | 过曝像素占比超过此值 → 过曝图 |
| `snr_threshold` | 10 | SNR 低于此值判为低信噪比 |
| `quality_sample_size` | 500 | 质量分析采样数 |

### BBox 几何

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `bbox_tiny_threshold` | 0.01 | 相对面积 < 此值 = tiny |
| `bbox_small_threshold` | 0.05 | 相对面积 < 此值 = small |
| `bbox_medium_threshold` | 0.2 | 相对面积 < 此值 = medium |
| `spatial_center_low` | 0.25 | 中心区域下界（归一化坐标） |
| `spatial_center_high` | 0.75 | 中心区域上界 |
| `spatial_edge_low` | 0.1 | 边缘区域外边界 |
| `spatial_edge_high` | 0.9 | 边缘区域外边界 |
| `high_iou_threshold` | 0.7 | IoU 高于此值标记为高重叠 |

### 类别分布

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `long_tail_ratio` | 0.3 | 类实例数 < 平均×此值 = 长尾类 |

### 质量分权重（总和 ≈ 1.0）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `quality_blur_weight` | 0.3 | 模糊率权重 |
| `quality_under_weight` | 0.2 | 欠曝率权重 |
| `quality_over_weight` | 0.2 | 过曝率权重 |
| `quality_imbalance_weight` | 0.3 | 类别不平衡权重 |

### 告警阈值（触发 key_issues）

| 参数 | 默认值 | 触发条件 |
|------|--------|---------|
| `warn_label_rate` | 0.5 | `label_rate < 0.5` → `low_label_coverage` |
| `warn_blur_ratio` | 0.1 | `blur_ratio > 0.1` → `high_blur_ratio` |
| `warn_tiny_bbox_ratio` | 0.2 | `tiny_bbox_ratio > 0.2` → `tiny_bbox_high_ratio` |
| `warn_center_concentration` | 0.6 | `center_concentration > 0.6` → `center_spatial_bias` |

### 特征聚类

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `cluster_sample_size` | 200 | 聚类采样数 |
| `dbscan_eps` | 0.3 | DBSCAN 聚类半径 |
| `dbscan_min_samples` | 5 | DBSCAN 最小样本数 |

## 测试完成后

如果确认 Module A 正常工作，告知我进入下一阶段（Module B — 训练结果分析器）的开发。
