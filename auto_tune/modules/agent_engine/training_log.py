"""Structured training-log contract for the S1.1 layering batch.

Pure Python (no FastAPI / Jinja2 / browser / global-config dependency).
A single line of decoded subprocess output is sanitized, classified into a
``TrainingLogEvent``, appended to a UTF-8 ``training.log``, and turned into
the fixed SSE payload dict consumed by both the ordinary-training and the
auto-tuning UIs.

Contract boundaries (do not rename fields/functions in this module):
- ``TrainingLogEvent``: frozen dataclass with kind/level/raw/summary/epoch.
- ``sanitize_training_line``: strip ANSI CSI/OSC and non-display controls.
- ``classify_training_line``: never raises; unknown input degrades to ``detail``.
- ``append_training_log``: append-only, UTF-8, one line + ``\\n``.
- ``build_training_sse_payload``: fixed payload keys for the UI.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

LogKind = Literal["epoch", "validation", "lifecycle", "warning", "error", "detail"]
Level = Literal["info", "success", "warning", "error", "debug"]


@dataclass(frozen=True)
class TrainingLogEvent:
    kind: LogKind
    level: Level
    raw: str
    summary: str | None
    epoch: int | None = None
    total_epochs: int | None = None


# ── Sanitization ──

_ANSI_OSC_RE = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
_ANSI_CSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def sanitize_training_line(line: str) -> str:
    """Remove ANSI CSI/OSC sequences, carriage returns and non-display control chars.

    Keeps tabs/newlines and all normal characters (Chinese, paths, colons,
    brackets, emoji). A trailing ``\\r``/``\\n`` is stripped so a single line
    stays a single line.
    """
    if not line:
        return ""
    s = _ANSI_OSC_RE.sub("", line)
    s = _ANSI_CSI_RE.sub("", s)
    s = _CONTROL_RE.sub("", s)
    s = s.replace("\r", "")
    return s.strip("\r\n")


# ── Classification ──

_ERROR_RE = re.compile(
    r"(traceback|outofmemoryerror|cuda\s+out\s+of\s+memory|\berror\b|\bexception\b)",
    re.IGNORECASE,
)
_FILE_LINE_RE = re.compile(r'^\s*File ".+", line \d+')
_WARNING_RE = re.compile(r"\bWARNING\b|\bWARN\b", re.IGNORECASE)
_VALIDATION_RE = re.compile(
    r"^\s*all\s+(?P<images>\d+)\s+(?P<instances>\d+)\s+"
    r"(?P<P>[\d.]+)\s+(?P<R>[\d.]+)\s+"
    r"(?P<mAP50>[\d.]+|nan)\s+(?P<mAP5095>[\d.]+|nan)\s*$",
    re.IGNORECASE,
)
_EPOCH_RE = re.compile(
    r"^\s*(?P<epoch>\d+)/(?P<total>\d+)\s+"
    r"(?:(?P<size>[\d.]+[A-Za-z]+)\s+)?"
    r"(?P<box>[\d.]+)\s+(?P<cls>[\d.]+)(?:\s+(?P<dfl>[\d.]+))?"
)
_LIFECYCLE_RE = re.compile(
    r"(Results saved to|Starting training|启动训练|开始训练|训练完成|训练结束)",
    re.IGNORECASE,
)

_MISSING = "—"


def _fmt_metric(value: str | None) -> str:
    """Render a metric token; missing/nan becomes the missing marker, real 0 stays 0."""
    if value is None or value.lower() == "nan":
        return _MISSING
    return value


def classify_training_line(line: str) -> TrainingLogEvent:
    """Classify one sanitized output line into a ``TrainingLogEvent``.

    Never raises: anything unrecognized degrades to ``detail`` with a
    ``None`` summary so training output can never break the pipeline.
    """
    raw = sanitize_training_line(line)

    if _ERROR_RE.search(raw) or _FILE_LINE_RE.match(raw):
        return TrainingLogEvent(kind="error", level="error", raw=raw, summary=raw)

    if _WARNING_RE.search(raw):
        return TrainingLogEvent(kind="warning", level="warning", raw=raw, summary=raw)

    m = _VALIDATION_RE.match(raw)
    if m:
        summary = (
            f"val: P={m.group('P')} R={m.group('R')} "
            f"mAP50={_fmt_metric(m.group('mAP50'))} mAP50-95={_fmt_metric(m.group('mAP5095'))}"
        )
        return TrainingLogEvent(kind="validation", level="success", raw=raw, summary=summary)

    m = _EPOCH_RE.match(raw)
    if m:
        parts = [f"box_loss={m.group('box')}", f"cls_loss={m.group('cls')}"]
        if m.group("dfl") is not None:
            parts.append(f"dfl_loss={m.group('dfl')}")
        summary = f"Epoch {m.group('epoch')}/{m.group('total')}: " + " ".join(parts)
        return TrainingLogEvent(
            kind="epoch",
            level="info",
            raw=raw,
            summary=summary,
            epoch=int(m.group("epoch")),
            total_epochs=int(m.group("total")),
        )

    if _LIFECYCLE_RE.search(raw):
        return TrainingLogEvent(kind="lifecycle", level="info", raw=raw, summary=raw)

    return TrainingLogEvent(kind="detail", level="info", raw=raw, summary=None)


# ── Persistence ──

def append_training_log(path: Path, line: str) -> None:
    """Append one sanitized line to a UTF-8 training log (creates parents)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.write("\n")


# ── SSE payload ──

def build_training_sse_payload(event: TrainingLogEvent, train_name: str) -> dict:
    """Build the fixed SSE payload dict for a training_log event."""
    return {
        "status": "running",
        "event": "training_log",
        "train_name": train_name,
        "log_kind": event.kind,
        "level": event.level,
        "message": event.summary,
        "detail": event.raw,
        "epoch": event.epoch,
        "total_epochs": event.total_epochs,
    }
