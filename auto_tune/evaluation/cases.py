"""F1.1-B 阶段一 — 固定离线案例集（两档）。

每个案例固定六件事：数据集事实、参考运行参数、最终指标与趋势、检测到的
问题、允许使用的证据范围、合理的方向与幅度，以及是否应当 ``keep_params``。

案例以「Module A 报告 + Module B 报告 + args.yaml + results.csv」的源材料
形式给出；评估时由真实 ``build_perception`` / ``build_tuning_fact_package``
生成事实包，绝不手工拼装事实包，也不复用生产的调优审计产物。

两档刻意分开：

- ``core``：单主因、唯一合法动作。用于回归——它必然满分，**不能**用来证明
  建议质量。
- ``hard``：多个合法动作并存（多条规则族同时可用）。于是语义校验必然放行，
  「合法率」必然接近 100%，质量只能由 ``expectation`` 的
  ``counter_params`` / ``evidence_scope`` / ``incoherent_pairs`` /
  ``preferred_ratios`` 区分出来。困难案例必须满足这两点，由
  ``test_evaluation_cases.py`` 锁定。

``expectation`` 只描述「什么是一个可接受的决策」，用于离线人工审核矩阵与
客观质量指标；它不是第二个校验器，不参与生产路径。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# 参考运行 args.yaml 基线：YOLOv8 Detect 的常见默认值。各案例只在必要时覆盖，
# 差异本身即「参考运行参数事实」。
_BASE_ARGS: dict = {
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "box": 7.5,
    "cls": 0.5,
    "dfl": 1.5,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "dropout": 0.0,
    "batch": 16,
    "epochs": 100,
    "patience": 50,
    "imgsz": 640,
    "close_mosaic": 10,
    "optimizer": "auto",
    "cos_lr": False,
    "model": "yolov8n.pt",
}

_METRIC_COLUMNS = {
    "mAP50": "metrics/mAP50(B)",
    "mAP50_95": "metrics/mAP50-95(B)",
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
}


@dataclass(frozen=True)
class CaseExpectation:
    """场景的临床期望：这个场景下「称职的工程师会改什么」。

    只用于评估判定，不参与生产校验。它刻意**不是**硬性门槛：给定案例事实，
    语义注册表可能同时开放其它合法关系（例如 ``val_box_loss=rising`` 会同时
    开放 weight_decay 增大）。越出 ``primary_params`` 只说明建议与本场景的
    主要矛盾无关，属于质量信号；真正的安全门槛是注册表合法性，由
    :mod:`auto_tune.evaluation.metrics` 单独判定。

    「格式合法」与「建议质量」必须分开度量：语义校验是**逐参数 + 逐引用事实**
    判定的，只要每条证据对得上它支持的那个参数就通过，因此
    ``valid=True`` 只能证明建议**合法**，不能证明它**正确**——主因无关、
    证据掺杂、组合自相矛盾、幅度偏激的建议都可以合法通过。以下字段就是用来
    把这些质量维度与合法性分开的：

    - ``counter_params``：本场景下**合法但方向错**的参数（例如欠拟合时降 lr0：
      plateau 规则允许，但会加剧欠拟合）。
    - ``evidence_scope``：允许引用的 fact_id 前缀。事实包里大量真实事实
      （指标/比例/计数/参数当前值）结构上可引用但与该参数无关，引用它们即
      证据污染。
    - ``incoherent_pairs``：同时出现即自相矛盾的参数组合。
    - ``preferred_ratios``：幅度质量区间（参数 → (下界, 上界) 相对当前值的比值）。
      仍在注册表合法范围内但越出该区间的幅度记为「偏激」，属质量信号。
    """

    should_keep_params: bool
    primary_params: frozenset[str] = field(default_factory=frozenset)
    allowed_directions: dict = field(default_factory=dict)
    note: str = ""
    counter_params: frozenset[str] = field(default_factory=frozenset)
    evidence_scope: frozenset[str] = field(default_factory=frozenset)
    incoherent_pairs: tuple = ()
    preferred_ratios: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    title: str
    scenario: str
    reference_run: str
    dataset_report: dict
    training_report: dict
    args: dict
    final_metrics: dict
    expectation: CaseExpectation
    # "core"：单主因、唯一合法动作，用于回归；"hard"：多个合法动作并存，
    # 合法性容易满足而质量区分度高的困难案例。
    tier: str = "core"


def _module_a_report(
    *,
    total_images: int,
    total_annotations: int,
    label_rate: float | None,
    quality_score: float | None,
    key_issues: tuple = (),
    tiny_bbox_ratio: float | None = None,
    small_bbox_ratio: float | None = None,
    blur_ratio: float | None = None,
    overexposure_ratio: float | None = None,
    underexposure_ratio: float | None = None,
    is_balanced: bool = True,
    imbalance_ratio: float | None = 1.1,
) -> dict:
    """构造一个可通过 ``validate_module_a_report`` 的最小 Module A 报告。

    ``None`` 表示该事实缺失（会自然从事实包中消失），不是零值。
    """
    return {
        "module": "dataset_analyzer",
        "version": "1.0",
        "total_images": total_images,
        "total_annotations": total_annotations,
        "label_coverage": {"label_rate": label_rate},
        "class_balance": {
            "is_balanced": is_balanced,
            "long_tail_classes": [],
            "imbalance_ratio": imbalance_ratio,
        },
        "bbox_analysis": {
            "tiny_bbox_ratio": tiny_bbox_ratio,
            "small_bbox_ratio": small_bbox_ratio,
        },
        "image_quality": {
            "blur_ratio": blur_ratio,
            "overexposure_ratio": overexposure_ratio,
            "underexposure_ratio": underexposure_ratio,
        },
        "summary": {
            "dataset_quality_score": quality_score,
            "key_issues": list(key_issues),
        },
    }


def _run(
    name: str,
    args: dict,
    final_metrics: dict,
    issues: tuple = (),
    *,
    val_box_loss: str = "descending",
    val_cls_loss: str = "descending",
    mAP50: str = "",
) -> dict:
    """构造 Module B 报告中的一个 run 条目（args + 最终指标 + 问题 + 曲线趋势）。

    ``mAP50`` 默认为空串：``TrainAnalyzer`` 只把 ``analyze_loss_curves`` 的结果
    并入报告的 ``curve_analysis``，``analyze_metric_curves`` 产出的 mAP50 趋势
    从未进入报告（204/204 真实报告均无 ``mAP50`` 键）。空串表示该事实缺失，
    与生产事实包一致；案例不得凭空造出生产不会出现的事实。
    """
    final = {
        _METRIC_COLUMNS[key]: value
        for key, value in final_metrics.items()
        if value is not None
    }
    return {
        "name": name,
        "args": args,
        "results": {"final_metrics": final, "total_epochs": args.get("epochs", 100)},
        "issues": [{"type": t, "severity": s} for t, s in issues],
        "curve_analysis": {
            "val_box": {"trend": val_box_loss},
            "val_cls": {"trend": val_cls_loss},
            "mAP50": {"trend": mAP50},
        },
    }


def _module_b_report(runs: tuple, best_run: str, best_map50: float | None) -> dict:
    """构造一个可通过 ``validate_module_b_report`` 的 Module B 报告。"""
    run_dict = {r["name"]: r for r in runs}
    return {
        "module": "train_analyzer",
        "version": "1.0",
        "total_runs": len(run_dict),
        "runs": run_dict,
        "summary": {
            "best_overall_run": best_run,
            "best_mAP50": best_map50,
            "average_mAP50": best_map50,
            "runs_with_issues": sum(1 for r in runs if r["issues"]),
            "common_issues": [],
        },
        "project": {"name": "f11b-eval"},
    }


def _args(**overrides) -> dict:
    return {**_BASE_ARGS, **overrides}


def _case1_overfitting() -> EvalCase:
    """明显过拟合：训练损失下降而验证损失上升，mAP 已饱和。"""
    ref = "eval_overfit"
    metrics = {"mAP50": 0.7412, "mAP50_95": 0.4123, "precision": 0.8301, "recall": 0.6894}
    run = _run(
        ref,
        _args(weight_decay=0.0005, epochs=120, lr0=0.01),
        metrics,
        (("overfitting", "high"),),
        val_box_loss="rising",
        val_cls_loss="rising",
    )
    return EvalCase(
        case_id="overfitting",
        title="明显过拟合",
        scenario="训练损失持续下降但验证损失上升，mAP 已饱和，需抑制模型记忆。",
        reference_run=ref,
        dataset_report=_module_a_report(
            total_images=420, total_annotations=1180, label_rate=0.8643,
            quality_score=0.881, tiny_bbox_ratio=0.09, small_bbox_ratio=0.31,
            blur_ratio=0.06, overexposure_ratio=0.04, underexposure_ratio=0.11,
        ),
        training_report=_module_b_report((run,), ref, metrics["mAP50"]),
        args=_args(weight_decay=0.0005, epochs=120, lr0=0.01),
        final_metrics=metrics,
        expectation=CaseExpectation(
            should_keep_params=False,
            primary_params=frozenset({"weight_decay", "epochs"}),
            allowed_directions={"weight_decay": "increase", "epochs": "decrease"},
            note="只应增强正则或缩短训练；不应改 lr0 / imgsz / box / cls。",
        ),
    )


def _case2_underfitting() -> EvalCase:
    """欠拟合：全部指标偏低，mAP 仍在上升，早停过早。"""
    ref = "eval_underfit"
    metrics = {"mAP50": 0.2134, "mAP50_95": 0.0981, "precision": 0.4211, "recall": 0.3102}
    run = _run(
        ref,
        _args(weight_decay=0.0005, epochs=50, patience=50, batch=16),
        metrics,
        (("underfitting", "medium"), ("early_stop_too_soon", "medium")),
        val_box_loss="descending",
        val_cls_loss="descending",
    )
    return EvalCase(
        case_id="underfitting",
        title="欠拟合与过早早停",
        scenario="模型容量/训练量不足，mAP50 与 Recall 均低且曲线仍在改善。",
        reference_run=ref,
        dataset_report=_module_a_report(
            total_images=260, total_annotations=610, label_rate=0.7812,
            quality_score=0.824, tiny_bbox_ratio=0.07, small_bbox_ratio=0.24,
            blur_ratio=0.05, overexposure_ratio=0.03, underexposure_ratio=0.07,
        ),
        training_report=_module_b_report((run,), ref, metrics["mAP50"]),
        args=_args(weight_decay=0.0005, epochs=50, patience=50, batch=16),
        final_metrics=metrics,
        expectation=CaseExpectation(
            should_keep_params=False,
            primary_params=frozenset({"epochs", "weight_decay", "patience"}),
            allowed_directions={
                "epochs": "increase",
                "weight_decay": "decrease",
                "patience": "increase",
            },
            note="应放宽训练量并减弱正则；patience 只在 early_stop_too_soon 证据下允许。",
        ),
    )


def _case3_plateau() -> EvalCase:
    """指标平台期：mAP50 与两条 loss 均已停滞。"""
    ref = "eval_plateau"
    metrics = {"mAP50": 0.6518, "mAP50_95": 0.3442, "precision": 0.7903, "recall": 0.6105}
    run = _run(
        ref,
        _args(lr0=0.01, cos_lr=False, epochs=150),
        metrics,
        (("plateau", "medium"),),
        val_box_loss="plateaued",
        val_cls_loss="plateaued",
    )
    return EvalCase(
        case_id="plateau",
        title="指标平台期",
        scenario="mAP50 饱和、loss 不再下降，需要扰动学习率调度跳出局部最优。",
        reference_run=ref,
        dataset_report=_module_a_report(
            total_images=780, total_annotations=2400, label_rate=0.9211,
            quality_score=0.931, tiny_bbox_ratio=0.11, small_bbox_ratio=0.28,
            blur_ratio=0.03, overexposure_ratio=0.02, underexposure_ratio=0.05,
        ),
        training_report=_module_b_report((run,), ref, metrics["mAP50"]),
        args=_args(lr0=0.01, cos_lr=False, epochs=150),
        final_metrics=metrics,
        expectation=CaseExpectation(
            should_keep_params=False,
            primary_params=frozenset({"lr0", "cos_lr"}),
            allowed_directions={"lr0": "decrease", "cos_lr": "enable"},
            note="lr0 只允许降到当前值的 25%–80%（0.0025–0.008）；cos_lr 只允许 false→true。",
        ),
    )


def _case4_unstable() -> EvalCase:
    """训练不稳定：loss 上升，需降低学习率并延长 warmup。"""
    ref = "eval_unstable"
    metrics = {"mAP50": 0.4021, "mAP50_95": 0.1883, "precision": 0.6102, "recall": 0.4809}
    run = _run(
        ref,
        _args(lr0=0.02, warmup_epochs=1.0, batch=16),
        metrics,
        (("unstable_training", "high"),),
        val_box_loss="rising",
        val_cls_loss="rising",
    )
    return EvalCase(
        case_id="unstable_training",
        title="训练不稳定",
        scenario="验证 loss 上升、指标退化，属于优化过程不稳定而非数据问题。",
        reference_run=ref,
        dataset_report=_module_a_report(
            total_images=340, total_annotations=900, label_rate=0.8102,
            quality_score=0.790, tiny_bbox_ratio=0.13, small_bbox_ratio=0.33,
            blur_ratio=0.12, overexposure_ratio=0.09, underexposure_ratio=0.14,
        ),
        training_report=_module_b_report((run,), ref, metrics["mAP50"]),
        args=_args(lr0=0.02, warmup_epochs=1.0, batch=16),
        final_metrics=metrics,
        expectation=CaseExpectation(
            should_keep_params=False,
            primary_params=frozenset({"lr0", "warmup_epochs"}),
            allowed_directions={"lr0": "decrease", "warmup_epochs": "increase"},
            note="lr0 只允许 0.005–0.016；warmup_epochs 不超过当前值 +3（即 ≤4.0）。",
        ),
    )


def _case5_precision_recall_imbalance() -> EvalCase:
    """Precision 与 Recall 明显失衡：现行规则集没有任何可用关系。

    Precision 0.91 而 Recall 0.44，但两条曲线都在健康方向、没有任何
    training/dataset issue 事实。语义规则注册表没有覆盖这一现象的
    fact→parameter 关系，因此本案例下唯一合法动作是 ``keep_params``：
    宁可不动，也不允许未受证据支持或方向相反的修改。
    """
    ref = "eval_pr_imbalance"
    metrics = {"mAP50": 0.5983, "mAP50_95": 0.3011, "precision": 0.9112, "recall": 0.4418}
    run = _run(
        ref,
        _args(cls=0.5, box=7.5, epochs=100),
        metrics,
        (),
        val_box_loss="descending",
        val_cls_loss="descending",
    )
    return EvalCase(
        case_id="precision_recall_imbalance",
        title="Precision / Recall 失衡（无规则覆盖）",
        scenario="Precision 高而 Recall 低，但曲线健康且无 issue 事实；注册表未开放对应关系。",
        reference_run=ref,
        dataset_report=_module_a_report(
            total_images=520, total_annotations=1500, label_rate=0.8804,
            quality_score=0.902, tiny_bbox_ratio=0.10, small_bbox_ratio=0.29,
            blur_ratio=0.04, overexposure_ratio=0.03, underexposure_ratio=0.06,
        ),
        training_report=_module_b_report((run,), ref, metrics["mAP50"]),
        args=_args(cls=0.5, box=7.5, epochs=100),
        final_metrics=metrics,
        expectation=CaseExpectation(
            should_keep_params=True,
            note=(
                "本场景在现有语义规则下没有合法修改关系 → 期望 keep_params；"
                "若模型坚持 adjust，必须被语义校验拒绝且不得启动训练。"
            ),
        ),
    )


def _case6_healthy() -> EvalCase:
    """指标健康：无任何问题事实，不应改动参数。"""
    ref = "eval_healthy"
    metrics = {"mAP50": 0.8841, "mAP50_95": 0.6123, "precision": 0.9103, "recall": 0.8602}
    run = _run(
        ref,
        _args(epochs=200, lr0=0.01, weight_decay=0.0005),
        metrics,
        (),
        val_box_loss="descending",
        val_cls_loss="descending",
    )
    return EvalCase(
        case_id="healthy",
        title="指标健康，无需调整",
        scenario="无 issue、曲线健康、指标已高，继续调参只会引入无依据的改动。",
        reference_run=ref,
        dataset_report=_module_a_report(
            total_images=900, total_annotations=3100, label_rate=0.9502,
            quality_score=0.962, tiny_bbox_ratio=0.06, small_bbox_ratio=0.21,
            blur_ratio=0.02, overexposure_ratio=0.01, underexposure_ratio=0.03,
        ),
        training_report=_module_b_report((run,), ref, metrics["mAP50"]),
        args=_args(epochs=200, lr0=0.01, weight_decay=0.0005),
        final_metrics=metrics,
        expectation=CaseExpectation(
            should_keep_params=True,
            note="期望 keep_params；任何 adjust 都必须被语义校验拒绝。",
        ),
    )


def _case7_insufficient_facts() -> EvalCase:
    """事实不足：只有基础计数与一条 mAP50，没有 issue、没有曲线趋势。

    事实包会明显变窄（缺 label_rate / quality_score / 各 ratio / 曲线趋势），
    这种窄事实包下没有任何可被引用的支持关系，唯一合法动作是 keep_params。
    """
    ref = "eval_sparse_facts"
    metrics = {"mAP50": 0.4021, "mAP50_95": None, "precision": None, "recall": None}
    run = _run(
        ref,
        _args(epochs=40, lr0=0.01),
        metrics,
        (),
        val_box_loss="",
        val_cls_loss="",
    )
    return EvalCase(
        case_id="insufficient_facts",
        title="事实不足，应选择 keep_params",
        scenario="报告缺失指标与趋势，可引用证据极少；不得补造事实。",
        reference_run=ref,
        dataset_report=_module_a_report(
            total_images=60, total_annotations=40, label_rate=None, quality_score=None,
        ),
        training_report=_module_b_report((run,), ref, metrics["mAP50"]),
        args=_args(epochs=40, lr0=0.01),
        final_metrics=metrics,
        expectation=CaseExpectation(
            should_keep_params=True,
            note="期望 keep_params；缺失事实不得补造，任何 adjust 都必须被拒绝。",
        ),
    )


def _case8_reference_vs_history_conflict() -> EvalCase:
    """当前参考运行与历史最佳冲突：必须以当前参考运行为准。

    Module B 报告里存在一个指标更好的历史 run，且它是报告级别的
    ``best_overall_run``。事实包只能绑定参考运行自身的事实，决策与证据
    都必须建立在当前参考运行上，不得拿历史最佳的数字来解释本轮改动。
    """
    ref = "eval_conflict_ref"
    other = "eval_conflict_best"
    ref_metrics = {"mAP50": 0.5203, "mAP50_95": 0.2402, "precision": 0.6601, "recall": 0.5104}
    other_metrics = {"mAP50": 0.8712, "mAP50_95": 0.5904, "precision": 0.9002, "recall": 0.8401}

    ref_run = _run(
        ref,
        _args(weight_decay=0.0005, epochs=100, lr0=0.01),
        ref_metrics,
        (("overfitting", "high"),),
        val_box_loss="rising",
        val_cls_loss="rising",
    )
    other_run = _run(
        other,
        _args(weight_decay=0.002, epochs=80, lr0=0.005),
        other_metrics,
        (),
        val_box_loss="descending",
        val_cls_loss="descending",
        mAP50="improving",
    )
    return EvalCase(
        case_id="reference_vs_history_conflict",
        title="当前运行与历史信息冲突",
        scenario="报告级最佳属于另一个 run，参考运行本身过拟合；必须按当前参考运行决策。",
        reference_run=ref,
        dataset_report=_module_a_report(
            total_images=610, total_annotations=1780, label_rate=0.8901,
            quality_score=0.905, tiny_bbox_ratio=0.08, small_bbox_ratio=0.26,
            blur_ratio=0.05, overexposure_ratio=0.02, underexposure_ratio=0.08,
        ),
        training_report=_module_b_report((ref_run, other_run), other, other_metrics["mAP50"]),
        args=_args(weight_decay=0.0005, epochs=100, lr0=0.01),
        final_metrics=ref_metrics,
        expectation=CaseExpectation(
            should_keep_params=False,
            primary_params=frozenset({"weight_decay", "epochs"}),
            allowed_directions={"weight_decay": "increase", "epochs": "decrease"},
            note=(
                "证据必须来自参考运行的 overfitting 事实；不得引用另一个 run 的指标，"
                "也不得按历史最佳 run 的参数（weight_decay 0.002）去反推本轮改动。"
            ),
        ),
    )


def _case9_uncovered_dataset_issue() -> EvalCase:
    """数据集问题无对应关系：模糊率高，但自然解法（增强参数）未开放。

    模糊率高是 Module A 真实会产出的 key issue（18 份真实报告中 2 份含
    ``high_blur_ratio``），而语义注册表没有任何以 ``dataset.issue.high_blur_ratio``
    为事实的关系。工程师的第一反应通常是加大 hsv_*/mosaic 等增强，但这些参数
    在本批规则下不可自动修改，因此唯一合法动作是 ``keep_params``。

    本案例用于度量「提示词宣布可调但注册表未开放」这一死路：模型被要求从
    33 个可调参数中选，却只有 10 个有语义关系。
    """
    ref = "eval_blur_issue"
    metrics = {"mAP50": 0.6334, "mAP50_95": 0.3321, "precision": 0.7742, "recall": 0.6023}
    run = _run(
        ref,
        _args(epochs=110, lr0=0.01, weight_decay=0.0005, hsv_v=0.4, mosaic=1.0),
        metrics,
        (),
        val_box_loss="descending",
        val_cls_loss="descending",
    )
    return EvalCase(
        case_id="uncovered_dataset_issue",
        title="数据集问题无对应语义关系（高模糊率）",
        scenario="数据集模糊率高且曲线健康；自然解法是增强参数，但注册表未开放对应关系。",
        reference_run=ref,
        dataset_report=_module_a_report(
            total_images=470, total_annotations=1320, label_rate=0.8712,
            quality_score=0.712, key_issues=("high_blur_ratio",),
            tiny_bbox_ratio=0.07, small_bbox_ratio=0.25,
            blur_ratio=0.34, overexposure_ratio=0.06, underexposure_ratio=0.18,
        ),
        training_report=_module_b_report((run,), ref, metrics["mAP50"]),
        args=_args(epochs=110, lr0=0.01, weight_decay=0.0005, hsv_v=0.4, mosaic=1.0),
        final_metrics=metrics,
        expectation=CaseExpectation(
            should_keep_params=True,
            note=(
                "期望 keep_params；任何试图修改增强类参数的建议都必须被语义校验拒绝，"
                "不得因为「提示词说这些参数可调」而放行。"
            ),
        ),
    )


# ══════════════════════════════════════════════════════════════════════════
# 困难案例（tier="hard"）：先设计成「多个合法动作并存」，于是格式合法性容易
# 满足，而建议质量才真正分化。每个案例对应规格 §四 要求区分的一类质量问题。
# ══════════════════════════════════════════════════════════════════════════


def _case_h1_multi_issue_priority() -> EvalCase:
    """多个 issue 并存，但只有一个与主因直接相关。

    过拟合（high）与小目标占比高同时在事实包里，两条规则族都合法：既可以用
    weight_decay/epochs 处理过拟合，也可以用 imgsz/box 处理小目标。主因是过拟合
    ——验证损失在上升、指标已经很高，此时提高输入分辨率既不解决过拟合、还会
    放大记忆。测「能否识别主因」。
    """
    ref = "eval_h1_priority"
    metrics = {"mAP50": 0.7912, "mAP50_95": 0.4603, "precision": 0.8602, "recall": 0.7411}
    run = _run(
        ref,
        _args(weight_decay=0.0005, epochs=150, imgsz=640, box=7.5),
        metrics,
        (("overfitting", "high"),),
        val_box_loss="rising",
        val_cls_loss="rising",
    )
    return EvalCase(
        case_id="multi_issue_priority",
        title="多 issue 竞争：应以过拟合为主因",
        scenario="过拟合与小目标占比高并存；小目标规则也合法，但不解决主因。",
        reference_run=ref,
        dataset_report=_module_a_report(
            total_images=640, total_annotations=2100, label_rate=0.9012,
            quality_score=0.842, key_issues=("tiny_bbox_high_ratio",),
            tiny_bbox_ratio=0.42, small_bbox_ratio=0.55,
            blur_ratio=0.05, overexposure_ratio=0.03, underexposure_ratio=0.07,
        ),
        training_report=_module_b_report((run,), ref, metrics["mAP50"]),
        args=_args(weight_decay=0.0005, epochs=150, imgsz=640, box=7.5),
        final_metrics=metrics,
        expectation=CaseExpectation(
            should_keep_params=False,
            primary_params=frozenset({"weight_decay", "epochs"}),
            allowed_directions={"weight_decay": "increase", "epochs": "decrease"},
            evidence_scope=frozenset({
                "training.issue.", "training.curve."}),
            counter_params=frozenset({"imgsz", "box"}),
            note="主因是过拟合；提高分辨率属于「合法但不对症」。",
        ),
        tier="hard",
    )


def _case_h2_direction_trap() -> EvalCase:
    """方向陷阱：欠拟合与平台期并存，plateau 规则会诱使降低 lr0。

    指标很低且曲线仍在改善 → 欠拟合是主因，正确动作是加大训练量/减弱正则。
    但事实包同时含 plateau，于是「降低 lr0」也是合法的——对欠拟合模型降低学习
    率会进一步减慢收敛。测方向判断。
    """
    ref = "eval_h2_direction"
    metrics = {"mAP50": 0.2412, "mAP50_95": 0.1103, "precision": 0.4502, "recall": 0.3311}
    run = _run(
        ref,
        _args(lr0=0.01, cos_lr=False, epochs=60, weight_decay=0.0005),
        metrics,
        (("underfitting", "medium"), ("plateau", "low")),
        val_box_loss="descending",
        val_cls_loss="descending",
    )
    return EvalCase(
        case_id="direction_trap",
        title="方向陷阱：欠拟合主因下不应降低 lr0",
        scenario="欠拟合为主因，但 plateau 事实使降 lr0 也合法；降低学习率会加剧欠拟合。",
        reference_run=ref,
        dataset_report=_module_a_report(
            total_images=380, total_annotations=920, label_rate=0.8211,
            quality_score=0.836, tiny_bbox_ratio=0.09, small_bbox_ratio=0.27,
            blur_ratio=0.06, overexposure_ratio=0.04, underexposure_ratio=0.08,
        ),
        training_report=_module_b_report((run,), ref, metrics["mAP50"]),
        args=_args(lr0=0.01, cos_lr=False, epochs=60, weight_decay=0.0005),
        final_metrics=metrics,
        expectation=CaseExpectation(
            should_keep_params=False,
            primary_params=frozenset({"epochs", "weight_decay"}),
            allowed_directions={"epochs": "increase", "weight_decay": "decrease"},
            evidence_scope=frozenset({"training.issue."}),
            counter_params=frozenset({"lr0", "cos_lr"}),
            note="欠拟合主因；降 lr0 / 开 cos_lr 合法但方向错误。",
        ),
        tier="hard",
    )


def _case_h3_incoherent_combo() -> EvalCase:
    """组合自相矛盾：同时缩短总轮数并延长早停耐心。

    overfitting 与 early_stop_too_soon 并存时，weight_decay / epochs / patience
    三个参数都合法。但 epochs 减少与 patience 增加同时出现是自相矛盾的：既把
    总轮数砍掉，又要求容忍更多轮不改善。测组合合理性。
    """
    ref = "eval_h3_combo"
    metrics = {"mAP50": 0.7112, "mAP50_95": 0.3902, "precision": 0.8201, "recall": 0.6812}
    run = _run(
        ref,
        _args(epochs=120, patience=60, weight_decay=0.0005),
        metrics,
        (("overfitting", "high"), ("early_stop_too_soon", "medium")),
        val_box_loss="rising",
        val_cls_loss="rising",
    )
    return EvalCase(
        case_id="incoherent_combo",
        title="组合自相矛盾：缩短 epochs 同时增大 patience",
        scenario="三个参数都合法，但「减少总轮数 + 增大早停耐心」互相抵消。",
        reference_run=ref,
        dataset_report=_module_a_report(
            total_images=560, total_annotations=1680, label_rate=0.8811,
            quality_score=0.871, tiny_bbox_ratio=0.08, small_bbox_ratio=0.26,
            blur_ratio=0.04, overexposure_ratio=0.03, underexposure_ratio=0.06,
        ),
        training_report=_module_b_report((run,), ref, metrics["mAP50"]),
        args=_args(epochs=120, patience=60, weight_decay=0.0005),
        final_metrics=metrics,
        expectation=CaseExpectation(
            should_keep_params=False,
            primary_params=frozenset({"weight_decay", "epochs", "patience"}),
            allowed_directions={"weight_decay": "increase", "epochs": "decrease",
                                "patience": "increase"},
            evidence_scope=frozenset({"training.issue."}),
            incoherent_pairs=(("epochs", "patience"),),
            note="减少 epochs 与增大 patience 同时出现即自相矛盾。",
        ),
        tier="hard",
    )


def _case_h4_evidence_pollution() -> EvalCase:
    """证据污染：数据集量化事实丰富但全部无规则，只可作背景。

    仅 plateau 一个 issue，所有规则只认 ``training.issue.*``。数据集侧有意塞入
    tiny_bbox_ratio / blur_ratio / quality_score / total_images 等真实但无规则的
    事实：它们结构上可引用，却与 lr0/cos_lr 的实际因果关系无关。测证据相关性。
    """
    ref = "eval_h4_pollution"
    metrics = {"mAP50": 0.6418, "mAP50_95": 0.3412, "precision": 0.7803, "recall": 0.6015}
    run = _run(
        ref,
        _args(lr0=0.01, cos_lr=False, epochs=140),
        metrics,
        (("plateau", "medium"),),
        val_box_loss="plateaued",
        val_cls_loss="plateaued",
    )
    return EvalCase(
        case_id="evidence_pollution",
        title="证据污染：不得用无规则的量化事实当证据",
        scenario="只有 plateau 相关事实可取用；数据集比例/指标只是背景。",
        reference_run=ref,
        dataset_report=_module_a_report(
            total_images=640, total_annotations=1900, label_rate=0.8742,
            quality_score=0.741, tiny_bbox_ratio=0.28, small_bbox_ratio=0.44,
            blur_ratio=0.22, overexposure_ratio=0.15, underexposure_ratio=0.19,
        ),
        training_report=_module_b_report((run,), ref, metrics["mAP50"]),
        args=_args(lr0=0.01, cos_lr=False, epochs=140),
        final_metrics=metrics,
        expectation=CaseExpectation(
            should_keep_params=False,
            primary_params=frozenset({"lr0", "cos_lr"}),
            allowed_directions={"lr0": "decrease", "cos_lr": "enable"},
            evidence_scope=frozenset({"training.issue."}),
            preferred_ratios={"lr0": (0.4, 0.8)},
            note="证据只能来自 plateau；引用 dataset.*/metrics 即污染。",
        ),
        tier="hard",
    )


def _case_h5_amplitude_too_aggressive() -> EvalCase:
    """幅度偏激：合法但一次性砍到规则下界。

    plateau 下 lr0 的合法区间是当前值的 25%–80%。直接取 25%（0.0025）合法，
    但单轮 4 倍降幅过于激进；稳妥区间是 40%–80%。测幅度合理性——合法性查不出
    这个问题，只有质量维度能。
    """
    ref = "eval_h5_amplitude"
    metrics = {"mAP50": 0.6712, "mAP50_95": 0.3611, "precision": 0.7902, "recall": 0.6214}
    run = _run(
        ref,
        _args(lr0=0.01, cos_lr=False, epochs=160),
        metrics,
        (("plateau", "medium"),),
        val_box_loss="plateaued",
        val_cls_loss="plateaued",
    )
    return EvalCase(
        case_id="amplitude_aggressive",
        title="幅度偏激：合法但一次砍到下界",
        scenario="lr0 合法区间 25%–80%，期望落在较稳的 40%–80%。",
        reference_run=ref,
        dataset_report=_module_a_report(
            total_images=820, total_annotations=2450, label_rate=0.9102,
            quality_score=0.912, tiny_bbox_ratio=0.07, small_bbox_ratio=0.24,
            blur_ratio=0.03, overexposure_ratio=0.02, underexposure_ratio=0.05,
        ),
        training_report=_module_b_report((run,), ref, metrics["mAP50"]),
        args=_args(lr0=0.01, cos_lr=False, epochs=160),
        final_metrics=metrics,
        expectation=CaseExpectation(
            should_keep_params=False,
            primary_params=frozenset({"lr0", "cos_lr"}),
            allowed_directions={"lr0": "decrease", "cos_lr": "enable"},
            evidence_scope=frozenset({"training.issue."}),
            preferred_ratios={"lr0": (0.4, 0.8)},
            note="0.0025 合法但偏激；期望 0.004–0.008。",
        ),
        tier="hard",
    )


def _case_h6_archive_neutral_evidence() -> EvalCase:
    """真实归档复现：拿「参数当前值」当证据，既不对症也无效。

    取自真实审计（过拟合 + 训练不稳定并存、两条验证损失曲线上升）。当时模型给出
    ``epochs`` 变更却把 ``training.params.epochs``（参数自己的当前值）当证据——
    那既不能支持任何方向，也不解释为什么改。当前校验器会以
    NO_SUPPORTING_RULE 拒绝它，但「引用了合法证据却仍是无效建议」这类问题只有
    质量维度能看到。
    """
    ref = "eval_h6_archive"
    metrics = {"mAP50": 0.5883, "mAP50_95": 0.2911, "precision": 0.7204, "recall": 0.5612}
    run = _run(
        ref,
        _args(weight_decay=0.0005, lr0=0.01, warmup_epochs=1.0, epochs=100),
        metrics,
        (("overfitting", "high"), ("unstable_training", "high")),
        val_box_loss="rising",
        val_cls_loss="rising",
    )
    return EvalCase(
        case_id="archive_neutral_evidence",
        title="归档复现：不得拿参数当前值当证据",
        scenario="过拟合与训练不稳定并存；证据必须是 issue/曲线事实，不能是参数当前值。",
        reference_run=ref,
        dataset_report=_module_a_report(
            total_images=290, total_annotations=128, label_rate=0.4828,
            quality_score=0.85, tiny_bbox_ratio=0.11, small_bbox_ratio=0.3,
            blur_ratio=0.07, overexposure_ratio=0.05, underexposure_ratio=0.09,
        ),
        training_report=_module_b_report((run,), ref, metrics["mAP50"]),
        args=_args(weight_decay=0.0005, lr0=0.01, warmup_epochs=1.0, epochs=100),
        final_metrics=metrics,
        expectation=CaseExpectation(
            should_keep_params=False,
            primary_params=frozenset({"weight_decay", "epochs", "lr0", "warmup_epochs"}),
            allowed_directions={"weight_decay": "increase", "epochs": "decrease",
                                "lr0": "decrease", "warmup_epochs": "increase"},
            evidence_scope=frozenset({"training.issue.", "training.curve."}),
            note="两个 issue 都要处理；证据不得包含 training.params.*。",
        ),
        tier="hard",
    )


_CASE_BUILDERS = (
    _case1_overfitting,
    _case2_underfitting,
    _case3_plateau,
    _case4_unstable,
    _case5_precision_recall_imbalance,
    _case6_healthy,
    _case7_insufficient_facts,
    _case8_reference_vs_history_conflict,
    _case9_uncovered_dataset_issue,
    _case_h1_multi_issue_priority,
    _case_h2_direction_trap,
    _case_h3_incoherent_combo,
    _case_h4_evidence_pollution,
    _case_h5_amplitude_too_aggressive,
    _case_h6_archive_neutral_evidence,
)


def get_cases(tier: str | None = None) -> tuple[EvalCase, ...]:
    """返回固定顺序的离线案例（顺序稳定，便于逐轮对比）。

    ``tier="hard"`` 只取「多个合法动作并存」的困难案例，用于质量区分度评估；
    ``tier="core"`` 取单主因回归案例；``None`` 取全部。
    """
    cases = tuple(builder() for builder in _CASE_BUILDERS)
    if tier is None:
        return cases
    return tuple(case for case in cases if case.tier == tier)


def get_case(case_id: str) -> EvalCase:
    for case in get_cases():
        if case.case_id == case_id:
            return case
    raise KeyError(case_id)


def get_metric_columns() -> dict:
    return dict(_METRIC_COLUMNS)
