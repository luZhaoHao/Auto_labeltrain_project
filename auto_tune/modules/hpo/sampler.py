"""H1.1 Optuna 内存 study 历史重建与候选生成。

``sampler_protocol='rebuild-per-trial-v1'``：每次分配编号 n 时按历史 0..n-1
重建 Optuna 内存 study，再以派生 seed 创建 TPE/RandomSampler。不写文件、不改
传入对象、不缓存 sampler RNG、不保存 pickle。
"""

import optuna
from pydantic import ValidationError

from .models import HpoError, StudyConfig, TrialRecord
from .search_space import suggest_candidate
from .validation import validate_history

SEED_MODULUS = 2147483648


def sample_next(config, history):
    """从 ``history``（连续终态 TrialRecord 列表）为下一个编号采样。

    返回 ``(sampled_params, candidate_params, distribution_json)``。只接受无
    PENDING 的连续编号终态历史；非法历史抛 ``HPO_INVALID_CONFIG``，不修补、
    不跳过。
    """
    try:
        data = config.model_dump(mode='python', warnings=False) if isinstance(config, StudyConfig) else config
        config = StudyConfig.model_validate(data)
    except (ValidationError, ValueError, TypeError) as exc:
        raise HpoError('HPO_INVALID_CONFIG', 'invalid sampler config') from exc
    if not isinstance(history, list):
        raise HpoError("HPO_INVALID_CONFIG", "history must be a list")
    for idx, saved in enumerate(history):
        if not isinstance(saved, TrialRecord):
            raise HpoError("HPO_INVALID_CONFIG",
                           "history items must be TrialRecord")
        if saved.state == "PENDING":
            raise HpoError("HPO_INVALID_CONFIG",
                           "history must not contain a PENDING trial")
        if saved.number != idx:
            raise HpoError("HPO_INVALID_CONFIG",
                           f"history must be contiguous from 0 (got "
                           f"number {saved.number} at index {idx})")

    history = validate_history(config, history)
    seed = (config.seed + len(history)) % SEED_MODULUS
    if config.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(
            seed=seed, n_startup_trials=5, n_ei_candidates=24,
            multivariate=False, constant_liar=False)
    else:
        sampler = optuna.samplers.RandomSampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler,
                                pruner=optuna.pruners.NopPruner())

    for saved in history:
        try:
            distributions = {
                key: optuna.distributions.json_to_distribution(value)
                for key, value in saved.distributions.items()
            }
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise HpoError("HPO_CORRUPT_STUDY",
                           f"malformed stored distribution: {exc}") from exc
        success = saved.state == "SUCCESS"
        try:
            study.add_trial(optuna.trial.create_trial(
                params=saved.sampled_params,
                distributions=distributions,
                state=(optuna.trial.TrialState.COMPLETE if success
                       else optuna.trial.TrialState.FAIL),
                value=saved.result.value if success else None))
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            raise HpoError('HPO_CORRUPT_STUDY', 'cannot rebuild stored trial') from exc

    trial = study.ask()
    sampled, candidate = suggest_candidate(trial, epochs=config.epochs)
    distributions_json = {
        key: optuna.distributions.distribution_to_json(value)
        for key, value in trial.distributions.items()
    }
    return sampled, candidate, distributions_json
