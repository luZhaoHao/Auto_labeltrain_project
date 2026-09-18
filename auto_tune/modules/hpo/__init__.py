"""H1.1 Detect HPO 搜索契约与持久化基础。

公共契约：严格输入/记录模型、候选生成、原子持久化与 HpoService 组合。
无真实训练、UI/API、LLM、产品排名或并行搜索。
"""

from .models import (
    EnvironmentSnapshot,
    Evidence,
    HpoError,
    ModelBinding,
    ResultInput,
    ResultPayload,
    SnapshotBinding,
    StudyConfig,
    StudyRecord,
    TrialRecord,
    utc_now_iso,
)
from .search_space import search_space_summary, suggest_candidate, validate_candidate
from .service import HpoService
from .execution_models import (
    ExecutionAttempt,
    ExecutionConfig,
    ExecutionEnvironment,
    ExecutionRecord,
    ExecutionRoots,
    MetricDiagnostics,
)
from .metrics import extract_objective, read_objective
from .ranking import rank_trials
from .execution import HpoRunner, execution_request_id

__all__ = [
    "HpoService",
    "StudyConfig",
    "StudyRecord",
    "TrialRecord",
    "ResultInput",
    "ResultPayload",
    "Evidence",
    "EnvironmentSnapshot",
    "ModelBinding",
    "SnapshotBinding",
    "HpoError",
    "ExecutionConfig",
    "ExecutionRecord",
    "ExecutionAttempt",
    "ExecutionEnvironment",
    "ExecutionRoots",
    "MetricDiagnostics",
    "HpoRunner",
    "execution_request_id",
    "extract_objective",
    "read_objective",
    "rank_trials",
    "search_space_summary",
    "suggest_candidate",
    "validate_candidate",
    "utc_now_iso",
]
