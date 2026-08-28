"""Unified outbound projection for every local-index HTTP response.

Repository projections may keep full paths internally (they feed
reconciliation, diagnostics and the legacy-import pipeline); the API boundary
reduces every path-like value to a basename so a business path can never reach
the client. The redaction is recursive over ``params``/``artifacts``/
``decision``/``tuning`` and nested lists/dicts, path-typed keys are
force-redacted even when their value is not obviously path-shaped, and
path-shaped dict keys themselves are reduced to basenames.

``is_path_key`` is the single shared predicate: the comparison module uses it to
exclude path-typed parameters from the parameter diff, so ``projection.py`` and
``compare.py`` stay consistent about which keys carry paths.
"""

from __future__ import annotations

import os

# Path-typed keys whose string values are always reduced to a basename at the
# HTTP boundary, and whose parameters are excluded from comparison diffs. These
# cover the documented fields (data/path/report_path/run_dir/audit_path/
# canonical_path/data_yaml_path) plus YOLO argument names that carry paths.
# ``model`` is deliberately NOT here: a model identifier (``yolov8n.pt``) is a
# meaningful comparison dimension and is preserved; a path-shaped model value is
# still reduced to a basename by the value-shape check below.
_PATH_VALUE_KEYS = frozenset({
    "data",
    "data_yaml",
    "data_yaml_path",
    "path",
    "report_path",
    "run_dir",
    "audit_path",
    "canonical_path",
    "source_root",
    "dataset_path",
    "save_dir",
    "project",
    "weights",
    "config",
    "resume",
    "pretrained",
    "hyp",
    "cfg",
    "model_yaml",
    "existing_model",
    "labels",
    "images",
    "cache",
    "labels_path",
    "images_path",
    "cache_path",
    "source_path",
    "snapshot_path",
    "manifest_path",
    "data_path",
})


def is_path_key(key: str | None) -> bool:
    """Return True when ``key`` names a path-typed field."""
    return key in _PATH_VALUE_KEYS


def _looks_like_path(value: str) -> bool:
    return os.path.isabs(value) or "/" in value or "\\" in value


def _basename(value: str) -> str:
    base = os.path.basename(value)
    return base or "…"


def _redact_dict_key(key):
    """Reduce a path-shaped dict key to a basename (e.g. a path used as a key)."""
    if isinstance(key, str) and _looks_like_path(key):
        return _basename(key)
    return key


def project_outward(value, key: str | None = None):
    """Recursively redact path-like strings in an API payload.

    A string is reduced to its basename when it sits under a path-typed key, or
    when it is an absolute path / relative path containing a directory
    separator. Path-shaped dict keys are also reduced. Non-string scalars and
    non-path strings pass through unchanged.
    """
    if isinstance(value, dict):
        out: dict = {}
        for k, v in value.items():
            new_key = _redact_dict_key(k)
            out[new_key] = project_outward(v, new_key)
        return out
    if isinstance(value, (list, tuple)):
        return [project_outward(v, key) for v in value]
    if isinstance(value, str):
        if is_path_key(key) or _looks_like_path(value):
            return _basename(value)
        return value
    return value
