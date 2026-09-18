"""H1.2 真实 results.csv 指标提取测试 — 合成 CSV，无训练/LLM/网络。

目标列 metrics/mAP50-95(B) 有效有限值的最大值，同分取最早 epoch；epoch 为
1-based、严格递增且不重复。结构性非法直接判无可信指标（HPO_INVALID_METRICS），
不返回 0 伪造成功。
"""

import os
from pathlib import Path

import pytest

from auto_tune.modules.hpo.metrics import extract_objective, read_objective
from auto_tune.modules.hpo.models import HpoError


def _write(run_dir, text, name="results.csv"):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / name).write_text(text, encoding="utf-8")
    return run_dir / name


def test_objective_uses_best_value_and_earliest_tie(tmp_path):
    run_dir = tmp_path / "run"
    _write(run_dir, "epoch,metrics/mAP50-95(B)\n1,0.2\n2,0.8\n3,0.8\n4,0.1\n")
    result = extract_objective(run_dir, artifact_root=tmp_path,
                               run_id="tuning:test-run", epochs=4)
    assert result.state == "SUCCESS"
    assert result.value == 0.8
    assert result.evidence.epoch == 2
    assert result.evidence.artifact_relpath == "run/results.csv"
    assert result.evidence.run_id == "tuning:test-run"
    assert result.evidence.metric_key == "metrics/mAP50-95(B)"
    assert len(result.evidence.artifact_sha256) == 64


def test_objective_zero_is_valid(tmp_path):
    run_dir = tmp_path / "run"
    _write(run_dir, "epoch,metrics/mAP50-95(B)\n1,0.0\n2,0.5\n")
    result = extract_objective(run_dir, artifact_root=tmp_path,
                               run_id="tuning:r", epochs=2)
    assert result.value == 0.5
    zero = tmp_path / "zero"
    _write(zero, "epoch,metrics/mAP50-95(B)\n1,0.0\n")
    result_zero = extract_objective(zero, artifact_root=tmp_path,
                                    run_id="tuning:r", epochs=1)
    assert result_zero.value == 0.0
    assert result_zero.evidence.epoch == 1


def test_diagnostics_count_excluded_rows(tmp_path):
    run_dir = tmp_path / "run"
    _write(run_dir, "epoch,metrics/mAP50-95(B)\n"
                    "1,0.2\n2,\n3,1.5\n4,nan\n5,0.4\n6,abc\n")
    obj = read_objective(run_dir, artifact_root=tmp_path,
                         run_id="tuning:r", epochs=6)
    assert obj.value == 0.4
    assert obj.epoch == 5
    assert obj.diagnostics.total_rows == 6
    assert obj.diagnostics.excluded_rows == 4


@pytest.mark.parametrize("text,epochs", [
    # 缺列
    ("epoch\n1\n", 1),
    ("metrics/mAP50-95(B)\n0.5\n", 1),
    # 重复表头
    ("epoch,metrics/mAP50-95(B),metrics/mAP50-95(B)\n1,0.5,0.6\n", 1),
    # epoch 0-based / 超范围 / 非整数 / 递减 / 重复
    ("epoch,metrics/mAP50-95(B)\n0,0.5\n", 1),
    ("epoch,metrics/mAP50-95(B)\n2,0.5\n", 1),
    ("epoch,metrics/mAP50-95(B)\n1.5,0.5\n", 2),
    ("epoch,metrics/mAP50-95(B)\n2,0.5\n1,0.4\n", 2),
    ("epoch,metrics/mAP50-95(B)\n1,0.5\n1,0.6\n", 2),
    ("epoch,metrics/mAP50-95(B)\n1,0.5\n3,0.6\n", 2),
    # 全部目标无效
    ("epoch,metrics/mAP50-95(B)\n1,\n2,nan\n", 2),
    ("epoch,metrics/mAP50-95(B)\n1,1.5\n2,-0.2\n", 2),
])
def test_invalid_structure_or_no_valid_value_rejected(tmp_path, text, epochs):
    run_dir = tmp_path / "run"
    _write(run_dir, text)
    with pytest.raises(HpoError) as err:
        extract_objective(run_dir, artifact_root=tmp_path,
                          run_id="tuning:r", epochs=epochs)
    assert err.value.code == "HPO_INVALID_METRICS"


def test_missing_file_rejected(tmp_path):
    with pytest.raises(HpoError) as err:
        extract_objective(tmp_path / "nope", artifact_root=tmp_path,
                          run_id="tuning:r", epochs=1)
    assert err.value.code == "HPO_INVALID_METRICS"


def test_oversize_file_rejected(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    path = run_dir / "results.csv"
    with open(path, "wb") as fh:
        fh.write(b"epoch,metrics/mAP50-95(B)\n" + b"x" * (5 * 1024 * 1024))
    with pytest.raises(HpoError) as err:
        extract_objective(run_dir, artifact_root=tmp_path,
                          run_id="tuning:r", epochs=1)
    assert err.value.code == "HPO_INVALID_METRICS"


def test_results_csv_symlink_rejected(tmp_path):
    if os.name == "nt":
        pass
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    real = tmp_path / "real.csv"
    real.write_text("epoch,metrics/mAP50-95(B)\n1,0.5\n", encoding="utf-8")
    try:
        os.symlink(real, run_dir / "results.csv")
    except (OSError, NotImplementedError):
        pytest.skip("file symlink unsupported")
    with pytest.raises(HpoError) as err:
        extract_objective(run_dir, artifact_root=tmp_path,
                          run_id="tuning:r", epochs=1)
    assert err.value.code == "HPO_INVALID_METRICS"


def test_run_dir_outside_artifact_root_rejected(tmp_path):
    outside = tmp_path / "outside" / "run"
    _write(outside, "epoch,metrics/mAP50-95(B)\n1,0.5\n")
    root = tmp_path / "artifact_root"
    root.mkdir()
    with pytest.raises(HpoError) as err:
        extract_objective(outside, artifact_root=root,
                          run_id="tuning:r", epochs=1)
    assert err.value.code == "HPO_INVALID_METRICS"


# ── 第四轮：快速 / 全面评价模式（确定性单一目标） ────────────────────

_CSV_HEADER = ("epoch,metrics/precision(B),metrics/recall(B),"
               "metrics/mAP50(B),metrics/mAP50-95(B)")

# 四行刻意让两种模式的胜者不同：
#   quick          = 0.10*mAP50 + 0.90*mAP50-95
#   comprehensive  = 0.10*mAP50 + 0.50*mAP50-95 + 0.20*P + 0.20*R
_QUICK_ROWS = [
    (1, 0.90, 0.90, 0.20, 0.40),   # quick 0.38 / comp 0.58
    (2, 0.10, 0.10, 0.60, 0.60),   # quick 0.60 / comp 0.40
    (3, 0.10, 0.10, 1.00, 0.50),   # quick 0.55 / comp 0.39
    (4, 0.10, 0.10, 0.60, 0.60),   # quick 0.60（与第 2 轮同分，取更早）
]


def _metric_rows(rows):
    return "\n".join(f"{e},{p},{r},{m50},{m95}" for e, p, r, m50, m95 in rows)


def test_quick_mode_scores_two_metrics_and_picks_the_earliest_best_epoch(tmp_path):
    run_dir = tmp_path / "run"
    _write(run_dir, _CSV_HEADER + "\n" + _metric_rows(_QUICK_ROWS) + "\n")
    obj = read_objective(run_dir, artifact_root=tmp_path, run_id="tuning:r",
                         epochs=4, evaluation_mode="quick")
    assert obj.value == pytest.approx(0.60)
    assert obj.epoch == 2                     # 同分取最早 epoch
    assert obj.evaluation_mode == "quick"
    assert obj.objective == "quick_composite_best_epoch_v1"
    # 被选 epoch 的原始指标全部保存（快速模式仍保存可获得的四项）
    assert obj.metrics == {
        "metrics/precision(B)": 0.10, "metrics/recall(B)": 0.10,
        "metrics/mAP50(B)": 0.60, "metrics/mAP50-95(B)": 0.60,
    }
    assert obj.diagnostics.total_rows == 4
    assert obj.diagnostics.excluded_rows == 0


def test_comprehensive_mode_scores_four_metrics_and_picks_its_own_best(tmp_path):
    run_dir = tmp_path / "run"
    _write(run_dir, _CSV_HEADER + "\n" + _metric_rows(_QUICK_ROWS) + "\n")
    obj = read_objective(run_dir, artifact_root=tmp_path, run_id="tuning:r",
                         epochs=4, evaluation_mode="comprehensive")
    assert obj.value == pytest.approx(0.58)
    assert obj.epoch == 1                     # 与快速模式的胜者不同，证明公式真的换了
    assert obj.evaluation_mode == "comprehensive"
    assert obj.objective == "comprehensive_composite_best_epoch_v1"
    assert obj.metrics["metrics/precision(B)"] == 0.90
    assert obj.metrics["metrics/recall(B)"] == 0.90


def test_legacy_mode_keeps_the_old_map50_95_rule(tmp_path):
    run_dir = tmp_path / "run"
    _write(run_dir, _CSV_HEADER + "\n" + _metric_rows(_QUICK_ROWS) + "\n")
    obj = read_objective(run_dir, artifact_root=tmp_path, run_id="tuning:r",
                         epochs=4)
    assert obj.value == pytest.approx(0.60)   # 第 2 轮的 mAP50-95
    assert obj.epoch == 2
    assert obj.evaluation_mode == "legacy_map50_95"
    assert obj.objective == "val_map50_95_best_epoch_v1"


@pytest.mark.parametrize("row,bad_index", [
    ("2,0.10,0.10,0.60,0.60", None),      # 参照：完全合法
    ("2,0.10,,0.60,0.60", "recall"),
    ("2,0.10,abc,0.60,0.60", "recall"),
    ("2,0.10,nan,0.60,0.60", "recall"),
    ("2,0.10,1.5,0.60,0.60", "recall"),
])
def test_rows_with_a_missing_or_illegal_required_metric_are_excluded(tmp_path, row, bad_index):
    run_dir = tmp_path / "run"
    _write(run_dir, _CSV_HEADER + "\n" + row + "\n3,0.20,0.20,0.10,0.10\n")
    obj = read_objective(run_dir, artifact_root=tmp_path, run_id="tuning:r",
                         epochs=3, evaluation_mode="comprehensive")
    if bad_index is None:
        assert obj.epoch == 2
        assert obj.diagnostics.excluded_rows == 0
    else:
        # 非法行被排除而不是补 0；胜者来自另一行
        assert obj.epoch == 3
        assert obj.diagnostics.excluded_rows == 1
        assert obj.value == pytest.approx(0.10 * 0.10 + 0.50 * 0.10
                                          + 0.20 * 0.20 + 0.20 * 0.20)


def test_quick_mode_ignores_a_broken_precision_column_value(tmp_path):
    """快速模式只要求两项指标；precision/recall 非法不影响该行。"""
    run_dir = tmp_path / "run"
    _write(run_dir, _CSV_HEADER + "\n2,abc,,0.60,0.60\n")
    obj = read_objective(run_dir, artifact_root=tmp_path, run_id="tuning:r",
                         epochs=2, evaluation_mode="quick")
    assert obj.epoch == 2
    assert obj.value == pytest.approx(0.60)
    # 无法解析的原始指标不写入组成指标，绝不补 0
    assert obj.metrics == {"metrics/mAP50(B)": 0.60,
                           "metrics/mAP50-95(B)": 0.60}


@pytest.mark.parametrize("mode,header", [
    ("quick", "epoch,metrics/mAP50-95(B)\n1,0.5\n"),
    ("comprehensive", "epoch,metrics/mAP50(B),metrics/mAP50-95(B)\n1,0.5,0.5\n"),
])
def test_missing_required_column_is_a_structure_error(tmp_path, mode, header):
    run_dir = tmp_path / "run"
    _write(run_dir, header)
    with pytest.raises(HpoError) as err:
        read_objective(run_dir, artifact_root=tmp_path, run_id="tuning:r",
                       epochs=1, evaluation_mode=mode)
    assert err.value.code == "HPO_INVALID_METRICS"


def test_unknown_evaluation_mode_is_rejected(tmp_path):
    run_dir = tmp_path / "run"
    _write(run_dir, _CSV_HEADER + "\n1,0.5,0.5,0.5,0.5\n")
    with pytest.raises(HpoError) as err:
        read_objective(run_dir, artifact_root=tmp_path, run_id="tuning:r",
                       epochs=1, evaluation_mode="bogus")
    assert err.value.code == "HPO_INVALID_METRICS"


def test_extract_objective_carries_mode_objective_and_component_metrics(tmp_path):
    run_dir = tmp_path / "run"
    _write(run_dir, _CSV_HEADER + "\n" + _metric_rows(_QUICK_ROWS) + "\n")
    result = extract_objective(run_dir, artifact_root=tmp_path,
                               run_id="tuning:test-run", epochs=4,
                               evaluation_mode="comprehensive")
    evidence = result.evidence
    assert result.value == pytest.approx(0.58)
    assert evidence.epoch == 1
    assert evidence.evaluation_mode == "comprehensive"
    assert evidence.objective == "comprehensive_composite_best_epoch_v1"
    assert evidence.metrics["metrics/recall(B)"] == 0.90
    # 旧字段语义不变
    assert evidence.metric_key == "metrics/mAP50-95(B)"
    assert len(evidence.artifact_sha256) == 64
