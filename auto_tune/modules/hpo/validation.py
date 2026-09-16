"""Validate HPO facts identically on load, write and sampler reconstruction."""

import json
import math

from pydantic import ValidationError

from .models import (
    ALLOWED_REASON_BY_STATE,
    HpoError,
    StudyRecord,
    TrialRecord,
    evaluation_score,
    required_metric_keys,
)
from .search_space import (
    LR0_RANGE, LRF_RANGE, MOMENTUM_RANGE, OPTIMIZER_CHOICES,
    WEIGHT_DECAY_RANGE, WARMUP_MAX_CAP, validate_candidate,
)


def check_success_evidence(evidence, value) -> None:
    """自描述成功证据的确定性复算。

    旧记录没有评价模式（``evaluation_mode is None``）→ 保持旧语义，不做复算。
    显式记录了评价模式的证据必须：组成指标齐全、目标版本与模式一致（模型层已
    校验），且综合分数可由这些原始指标按固定权重重算。不一致即非法事实。
    """
    mode = getattr(evidence, "evaluation_mode", None)
    if mode is None:
        return
    metrics = getattr(evidence, "metrics", None) or {}
    missing = [key for key in required_metric_keys(mode) if key not in metrics]
    if missing:
        raise ValueError("evidence is missing required component metrics")
    recomputed = evaluation_score(mode, metrics)
    if recomputed is None or not math.isclose(float(value), recomputed,
                                              rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError("composite score does not match its component metrics")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate distribution key')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError('non-finite distribution constant')


def _validate_distribution(raw, name, attrs):
    decoded = json.loads(raw, object_pairs_hook=_unique_pairs,
                         parse_constant=_reject_constant)
    if not isinstance(decoded, dict) or set(decoded) != {'name', 'attributes'}:
        raise ValueError('invalid distribution shape')
    actual = decoded['attributes']
    if decoded['name'] != name or not isinstance(actual, dict):
        raise ValueError('distribution type mismatch')
    # Optuna permits an omitted FloatDistribution step (default None).
    if name == 'FloatDistribution' and 'step' not in actual:
        actual['step'] = None
    if set(actual) != set(attrs):
        raise ValueError('distribution attributes mismatch')
    for key, expected in attrs.items():
        value = actual[key]
        if type(expected) is bool:
            valid_type = type(value) is bool
        elif type(expected) is int:
            valid_type = type(value) is int
        elif type(expected) is float:
            valid_type = type(value) in (int, float) and math.isfinite(value)
        elif isinstance(expected, list):
            valid_type = isinstance(value, list) and all(type(v) is str for v in value)
        else:
            valid_type = value is None
        if not valid_type or value != expected:
            raise ValueError(f'distribution attribute {key} violates search space')


def validate_trial(config, trial):
    """Revalidate mutable instances and all conditional/result invariants."""
    try:
        checked = TrialRecord.model_validate(trial.model_dump(mode='python', warnings=False))
        candidate = validate_candidate(checked.candidate_params, epochs=config.epochs)
        optimizer = candidate['optimizer']
        momentum_key = 'momentum_sgd' if optimizer == 'SGD' else 'beta1_adamw'
        expected_sampled = dict(candidate)
        expected_sampled[momentum_key] = expected_sampled.pop('momentum')
        if checked.sampled_params != expected_sampled:
            raise ValueError('sampled parameters do not map to candidate')
        if type(checked.sampled_params['warmup_epochs']) is not int:
            raise ValueError('sampled warmup_epochs must be an integer')
        expected = {
            'optimizer': ('CategoricalDistribution', {'choices': list(OPTIMIZER_CHOICES)}),
            'warmup_epochs': ('IntDistribution', {
                'low': 0, 'high': min(WARMUP_MAX_CAP, config.epochs - 1), 'step': 1, 'log': False}),
        }
        for key, bounds, logarithmic in (
            ('lr0', LR0_RANGE, True), ('lrf', LRF_RANGE, True),
            (momentum_key, MOMENTUM_RANGE[optimizer], False),
            ('weight_decay', WEIGHT_DECAY_RANGE, False),
        ):
            expected[key] = ('FloatDistribution', {
                'low': bounds[0], 'high': bounds[1], 'log': logarithmic, 'step': None})
        if set(checked.distributions) != set(expected):
            raise ValueError('conditional distribution keys mismatch')
        for key, (name, attrs) in expected.items():
            _validate_distribution(checked.distributions[key], name, attrs)
        if not checked.trial_id.endswith(f'_t{checked.number:04d}'):
            raise ValueError('trial_id and number mismatch')
        if checked.state == 'PENDING':
            if checked.finished_at is not None or checked.result is not None:
                raise ValueError('pending trial contains terminal fields')
        else:
            if checked.finished_at is None or checked.result is None:
                raise ValueError('terminal trial requires finish time and result')
            result = checked.result
            if checked.state == 'SUCCESS':
                if result.reason_code is not None or result.value is None or result.evidence is None:
                    raise ValueError('SUCCESS requires success evidence')
                if result.evidence.epoch > config.epochs:
                    raise ValueError('evidence epoch exceeds configured epochs')
                if result.evidence.evaluation_mode is not None \
                        and result.evidence.evaluation_mode != config.evaluation_mode:
                    raise ValueError('evidence evaluation mode differs from the study')
                check_success_evidence(result.evidence, result.value)
            elif (result.value is not None or result.evidence is not None
                  or result.reason_code not in ALLOWED_REASON_BY_STATE[checked.state]):
                raise ValueError('terminal state and failure reason mismatch')
        return checked
    except (ValidationError, ValueError, TypeError, KeyError, OverflowError, HpoError) as exc:
        raise HpoError('HPO_CORRUPT_STUDY', 'trial violates persisted HPO contract') from exc


def validate_history(config, history, *, allow_pending=False):
    if not isinstance(history, list) or len(history) > config.budget:
        raise HpoError('HPO_CORRUPT_STUDY', 'history exceeds budget or is not a list')
    checked = []
    requests = set()
    study_prefix = None
    for number, trial in enumerate(history):
        item = validate_trial(config, trial)
        prefix = item.trial_id.rsplit('_t', 1)[0]
        if item.number != number or (study_prefix is not None and prefix != study_prefix):
            raise HpoError('HPO_CORRUPT_STUDY', 'history order or identity mismatch')
        if item.request_id in requests:
            raise HpoError('HPO_CORRUPT_STUDY', 'duplicate request_id in history')
        if item.state == 'PENDING' and (not allow_pending or number != len(history) - 1):
            raise HpoError('HPO_CORRUPT_STUDY', 'outstanding trial must be unique and last')
        study_prefix = prefix
        requests.add(item.request_id)
        checked.append(item)
    return checked


def validate_record(record, *, expected_study_id):
    """Validate from plain data even when passed an already constructed model."""
    try:
        data = record.model_dump(mode='python', warnings=False) if isinstance(record, StudyRecord) else record
        checked = StudyRecord.model_validate(data)
    except (ValidationError, ValueError, TypeError, OverflowError) as exc:
        raise HpoError('HPO_CORRUPT_STUDY', 'study.json failed contract validation') from exc
    if checked.study_id != expected_study_id:
        raise HpoError('HPO_CORRUPT_STUDY', 'record identity differs from requested study')
    checked.trials = validate_history(checked.config, checked.trials, allow_pending=True)
    return checked
