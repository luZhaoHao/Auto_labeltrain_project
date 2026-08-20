"""Tests for the structured training-log contract (sanitize/classify/persist/payload)."""

from pathlib import Path

from auto_tune.modules.agent_engine.training_log import (
    TrainingLogEvent,
    append_training_log,
    build_training_sse_payload,
    classify_training_line,
    sanitize_training_line,
)


# ── Sanitize ──

def test_sanitize_removes_ansi_and_carriage_return():
    raw = "\x1b[32m  1/100  1.20G  1.234  0.456  0.789\x1b[0m\r"
    assert sanitize_training_line(raw) == "  1/100  1.20G  1.234  0.456  0.789"


def test_sanitize_keeps_chinese_path_colons_and_brackets():
    raw = "Results saved to D:\\数据\\检测\\train87 [OK]\r"
    assert sanitize_training_line(raw) == "Results saved to D:\\数据\\检测\\train87 [OK]"


def test_sanitize_strips_osc_and_non_display_control_chars():
    raw = "\x1b]0;title\x07normal\x08tail"
    assert sanitize_training_line(raw) == "normaltail"


# ── Classify ──

def test_classify_epoch_preserves_zero_values():
    event = classify_training_line("  1/100  1.20G  0  0.456  0.789")
    assert event.kind == "epoch"
    assert event.epoch == 1
    assert event.total_epochs == 100
    assert "box_loss=0" in event.summary


def test_classify_epoch_summary_contains_losses():
    event = classify_training_line("  1/100  1.20G  1.234  0.456  0.789")
    assert event.kind == "epoch"
    assert event.summary == "Epoch 1/100: box_loss=1.234 cls_loss=0.456 dfl_loss=0.789"
    assert event.raw == "  1/100  1.20G  1.234  0.456  0.789"


def test_classify_epoch_without_dfl_loss():
    event = classify_training_line("  1/100  1.20G  1.234  0.456")
    assert event.kind == "epoch"
    assert "dfl_loss" not in event.summary


def test_classify_epoch_with_trailing_tqdm_progress():
    # YOLO coalesces the epoch row and its tqdm redraws into one \r line when
    # stdout is piped; the leading loss tokens must still classify as epoch.
    line = ("        1/1     0.146G     0.9377      2.724     0.8697"
            "          1        320: 0% ━━━━━━━━ 0/4 0.4s")
    event = classify_training_line(line)
    assert event.kind == "epoch"
    assert event.epoch == 1
    assert event.total_epochs == 1
    assert "box_loss=0.9377" in event.summary
    assert "cls_loss=2.724" in event.summary
    assert "dfl_loss=0.8697" in event.summary


def test_classify_validation_preserves_zero_map():
    event = classify_training_line("all 10 50 0.123 0.456 0 0.111")
    assert event.kind == "validation"
    assert "mAP50=0" in event.summary


def test_classify_validation_distinguishes_missing_from_zero():
    event = classify_training_line("all 10 50 0.123 0.456 nan 0.111")
    assert event.kind == "validation"
    assert "mAP50=—" in event.summary
    assert "mAP50=0" not in event.summary


def test_traceback_and_cuda_oom_are_errors():
    assert classify_training_line("Traceback (most recent call last):").kind == "error"
    assert classify_training_line("torch.cuda.OutOfMemoryError: CUDA out of memory").kind == "error"
    assert classify_training_line("  File \"train.py\", line 42, in <module>").kind == "error"


def test_warning_keywords_are_warnings():
    event = classify_training_line("WARNING ⚠️ low GPU memory")
    assert event.kind == "warning"
    assert event.level == "warning"
    assert event.summary == event.raw


def test_lifecycle_results_saved():
    event = classify_training_line("Results saved to D:\\detect\\train87")
    assert event.kind == "lifecycle"
    assert event.summary == "Results saved to D:\\detect\\train87"


def test_tqdm_batch_progress_is_detail_only():
    event = classify_training_line("1/100 50%|█████| 5/10 [00:01<00:01, 4.5it/s]")
    assert event.kind == "detail"
    assert event.summary is None


def test_blank_line_is_detail():
    event = classify_training_line("   \r\n")
    assert event.kind == "detail"
    assert event.summary is None


def test_unknown_line_degrades_to_detail():
    event = classify_training_line("Ultralytics YOLOv8.2.103 🚀 Python-3.10.11 torch-2.2.2")
    assert event.kind == "detail"
    assert event.summary is None


# ── Persistence ──

def test_append_training_log_is_utf8_and_ordered(tmp_path):
    path = tmp_path / "training.log"
    append_training_log(path, "第一行")
    append_training_log(path, "second")
    assert path.read_text("utf-8") == "第一行\nsecond\n"


def test_append_training_log_creates_parent_missing_file(tmp_path):
    path = tmp_path / "training.log"
    append_training_log(path, "a")
    assert path.exists()
    assert path.read_text("utf-8") == "a\n"


# ── Payload ──

def test_build_payload_separates_summary_and_detail():
    event = classify_training_line("  1/100  1.20G  1.234  0.456  0.789")
    payload = build_training_sse_payload(event, "train87")
    assert payload["event"] == "training_log"
    assert payload["message"].startswith("Epoch 1/100:")
    assert payload["detail"] == event.raw
    assert payload["epoch"] == 1


def test_build_payload_fixed_contract_fields():
    event = TrainingLogEvent(
        kind="detail", level="info", raw="1/100 50%|█████| 5/10 [00:01<00:01, 4.5it/s]",
        summary=None,
    )
    payload = build_training_sse_payload(event, "train88")
    assert payload == {
        "status": "running",
        "event": "training_log",
        "train_name": "train88",
        "log_kind": "detail",
        "level": "info",
        "message": None,
        "detail": "1/100 50%|█████| 5/10 [00:01<00:01, 4.5it/s]",
        "epoch": None,
        "total_epochs": None,
    }


def test_build_payload_carries_validation_kind_and_success_level():
    event = classify_training_line("all 10 50 0.123 0.456 0 0.111")
    payload = build_training_sse_payload(event, "train87")
    assert payload["log_kind"] == "validation"
    assert payload["level"] == "success"
    assert payload["message"].startswith("val:")
