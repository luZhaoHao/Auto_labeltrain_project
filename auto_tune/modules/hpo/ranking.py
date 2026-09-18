"""H1.2 确定性排名：只对 SUCCESS 排序，key=(-value, trial.number)。

排名是可重算视图，不建第二份排名事实；不调用 Optuna best_trial、LLM 或使用
best.pt fitness 替代目标。返回深拷贝，不改调用方对象。
"""

from .models import HpoError, StudyRecord, TrialRecord
from .validation import validate_record


def rank_trials(record: StudyRecord) -> list[TrialRecord]:
    """严格重验证后返回 SUCCESS trial（value 降序、同分按 number 升序）。

    空列表表示无优胜 trial。非法/损坏记录抛 HPO_CORRUPT_STUDY。
    """
    checked = validate_record(record, expected_study_id=record.study_id)
    ordered = sorted(
        (trial for trial in checked.trials if trial.state == "SUCCESS"),
        key=lambda trial: (-trial.result.value, trial.number),
    )
    return [trial.model_copy(deep=True) for trial in ordered]
