"""Codex independent acceptance probes; synthetic files only, no training."""
import json
import os
import uuid
from pathlib import Path
import pytest
from auto_tune.tests.test_hpo_service import hpo_inputs
from auto_tune.modules.hpo import HpoService, StudyConfig, HpoError, ResultInput
from auto_tune.modules.hpo import Evidence
from auto_tune.modules.hpo.storage import StudyStore


@pytest.fixture
def setup_case(tmp_path):
    snapshot, model = hpo_inputs.__wrapped__(tmp_path)
    root = tmp_path / 'hpo'
    svc = HpoService(root)
    record = svc.create_study(StudyConfig(budget=3), snapshot_dir=snapshot, model_path=model)
    request = uuid.uuid4().hex
    trial = svc.ask(record.study_id, request_id=request)
    return svc, root, record, request, trial, snapshot, model


@pytest.mark.parametrize('damage', ['candidate', 'distribution', 'state_result', 'epoch', 'request_id'])
def test_corrupt_record_rejected_on_reload(setup_case, damage):
    svc, root, record, request, trial, *_ = setup_case
    target = root / record.study_id / 'study.json'
    data = json.loads(target.read_text(encoding='utf-8'))
    t = data['trials'][0]
    if damage == 'candidate':
        t['candidate_params']['lr0'] = 999.0
    elif damage == 'distribution':
        t['distributions']['lr0'] = '{}'
    elif damage == 'state_result':
        t['state'] = 'SUCCESS'
        t['finished_at'] = t['created_at']
        t['result'] = {'reason_code': 'oom', 'value': None, 'evidence': None}
    elif damage == 'epoch':
        t['state'] = 'SUCCESS'
        t['finished_at'] = t['created_at']
        t['result'] = {'reason_code': None, 'value': 0.5, 'evidence': {
            'run_id': 'synthetic', 'artifact_relpath': 'results.csv',
            'artifact_sha256': '0' * 64, 'metric_key': 'metrics/mAP50-95(B)',
            'epoch': 999}}
    elif damage == 'request_id':
        t['state'] = 'FAILED'
        t['finished_at'] = t['created_at']
        t['result'] = {'reason_code': 'oom', 'value': None, 'evidence': None}
        second = dict(t, number=1, trial_id=f'{record.study_id}_t0001')
        data['trials'].append(second)
    target.write_text(json.dumps(data), encoding='utf-8')
    before = target.read_bytes()
    with pytest.raises(HpoError) as err:
        svc.load_study(record.study_id)
    assert err.value.code == 'HPO_CORRUPT_STUDY'
    assert target.read_bytes() == before


def test_same_request_cannot_return_corrupt_candidate(setup_case):
    svc, root, record, request, *_ = setup_case
    target = root / record.study_id / 'study.json'
    data = json.loads(target.read_text(encoding='utf-8'))
    data['trials'][0]['candidate_params']['lr0'] = 999.0
    target.write_text(json.dumps(data), encoding='utf-8')
    with pytest.raises(HpoError) as err:
        svc.ask(record.study_id, request_id=request)
    assert err.value.code == 'HPO_CORRUPT_STUDY'


@pytest.mark.parametrize('with_trial', [False, True])
def test_record_identity_must_match_requested_directory(setup_case, with_trial):
    svc, root, record, request, trial, snapshot, model = setup_case
    other = svc.create_study(StudyConfig(), snapshot_dir=snapshot, model_path=model)
    if with_trial:
        svc.ask(other.study_id, request_id=request)
    original = root / record.study_id / 'study.json'
    original.write_bytes((root / other.study_id / 'study.json').read_bytes())
    with pytest.raises(HpoError) as err:
        svc.load_study(record.study_id)
    assert err.value.code == 'HPO_CORRUPT_STUDY'


def test_storage_root_symlink_rejected(tmp_path):
    snapshot, model = hpo_inputs.__wrapped__(tmp_path)
    destination = tmp_path / 'destination'
    destination.mkdir()
    root = tmp_path / 'root_link'
    os.symlink(destination, root, target_is_directory=True)
    with pytest.raises(HpoError):
        HpoService(root).create_study(StudyConfig(), snapshot_dir=snapshot, model_path=model)
    assert not list(destination.glob('hpo_*/study.json'))


@pytest.mark.parametrize('mode', ['assignment', 'copy'])
def test_mutated_config_revalidated_at_boundary(tmp_path, mode):
    snapshot, model = hpo_inputs.__wrapped__(tmp_path)
    config = StudyConfig()
    if mode == 'assignment':
        config.budget = 101
    else:
        config = config.model_copy(update={'budget': True})
    svc = HpoService(tmp_path / 'hpo')
    with pytest.raises(HpoError) as err:
        svc.create_study(config, snapshot_dir=snapshot, model_path=model)
    assert err.value.code == 'HPO_INVALID_CONFIG'
    assert not list((tmp_path / 'hpo').glob('*/study.json'))


@pytest.mark.parametrize('damage', [
    'mapping', 'branch', 'distribution_range', 'distribution_log',
    'distribution_choices', 'distribution_duplicate', 'distribution_bool',
    'pending_finished', 'terminal_unfinished', 'over_budget', 'two_pending',
    'pending_not_last', 'reason_mismatch',
])
def test_semantic_damage_never_returns_or_updates_record(setup_case, damage):
    svc, root, record, request, trial, *_ = setup_case
    target = root / record.study_id / 'study.json'
    data = json.loads(target.read_text(encoding='utf-8'))
    t = data['trials'][0]
    if damage == 'mapping':
        t['candidate_params']['lr0'] = 0.002 if t['sampled_params']['lr0'] != 0.002 else 0.003
    elif damage == 'branch':
        key = 'beta1_adamw' if t['sampled_params']['optimizer'] == 'SGD' else 'momentum_sgd'
        t['sampled_params'][key] = 0.9
        t['distributions'][key] = t['distributions']['lr0']
    elif damage.startswith('distribution_'):
        key = 'optimizer' if damage == 'distribution_choices' else 'lr0'
        dist = json.loads(t['distributions'][key])
        if damage == 'distribution_range':
            dist['attributes']['high'] = 1.0
        elif damage == 'distribution_log':
            dist['attributes']['log'] = False
        elif damage == 'distribution_choices':
            dist['attributes']['choices'].reverse()
        elif damage == 'distribution_bool':
            dist['attributes']['low'] = True
        if damage == 'distribution_duplicate':
            t['distributions'][key] = '{"name":"FloatDistribution","name":"FloatDistribution","attributes":{}}'
        else:
            t['distributions'][key] = json.dumps(dist)
    elif damage == 'pending_finished':
        t['finished_at'] = t['created_at']
    elif damage in ('terminal_unfinished', 'reason_mismatch'):
        t['state'] = 'FAILED'
        t['result'] = {'reason_code': 'oom' if damage == 'terminal_unfinished' else 'user_stopped'}
        if damage == 'reason_mismatch':
            t['finished_at'] = t['created_at']
    else:
        second = dict(t, number=1, trial_id=f'{record.study_id}_t0001', request_id=uuid.uuid4().hex)
        if damage in ('over_budget', 'pending_not_last'):
            second.update(state='FAILED', finished_at=t['created_at'], result={'reason_code': 'oom'})
        data['trials'].append(second)
        if damage == 'over_budget':
            data['config']['budget'] = 1
    target.write_text(json.dumps(data), encoding='utf-8')
    before = target.read_bytes()
    for call in (lambda: svc.load_study(record.study_id),
                 lambda: svc.ask(record.study_id, request_id=request),
                 lambda: svc.tell(record.study_id, 0, ResultInput(state='FAILED', reason_code='oom'))):
        with pytest.raises(HpoError) as err:
            call()
        assert err.value.code == 'HPO_CORRUPT_STUDY'
        assert target.read_bytes() == before


@pytest.mark.parametrize('mode', ['nested', 'copy', 'boolean'])
def test_mutable_result_boundary_has_stable_error_and_no_write(setup_case, mode):
    svc, root, record, request, trial, *_ = setup_case
    result = ResultInput(state='SUCCESS', value=0.5, evidence=Evidence(
        run_id='synthetic', artifact_relpath='results.csv', artifact_sha256='a' * 64, epoch=1))
    if mode == 'nested':
        result.evidence.epoch = '1'
    elif mode == 'copy':
        result = result.model_copy(update={'state': 'INVALID'})
    else:
        result.value = True
    target = root / record.study_id / 'study.json'
    before = target.read_bytes()
    with pytest.raises(HpoError) as err:
        svc.tell(record.study_id, 0, result)
    assert err.value.code == 'HPO_INVALID_RESULT'
    assert target.read_bytes() == before


def test_store_revalidates_mutated_record_before_replace(setup_case):
    svc, root, record, request, trial, *_ = setup_case
    current = svc.load_study(record.study_id)
    current.trials[0].candidate_params['lr0'] = 999.0
    target = root / record.study_id / 'study.json'
    before = target.read_bytes()
    store = StudyStore(root)
    with store.locked(record.study_id), pytest.raises(HpoError) as err:
        store.write(current)
    assert err.value.code == 'HPO_CORRUPT_STUDY'
    assert target.read_bytes() == before


def test_root_ancestor_link_rejected_before_mkdir(tmp_path):
    snapshot, model = hpo_inputs.__wrapped__(tmp_path)
    destination = tmp_path / 'destination'
    destination.mkdir()
    linked_parent = tmp_path / 'linked_parent'
    os.symlink(destination, linked_parent, target_is_directory=True)
    with pytest.raises(HpoError):
        HpoService(linked_parent / 'new' / 'hpo').create_study(
            StudyConfig(), snapshot_dir=snapshot, model_path=model)
    assert list(destination.iterdir()) == []


@pytest.mark.parametrize('name', ['.study.lock', 'study.json'])
def test_existing_link_targets_are_not_touched(setup_case, tmp_path, name):
    svc, root, record, *_ = setup_case
    target = root / record.study_id / name
    # Replace only the fixture file created by this test.
    target.unlink()
    destination = tmp_path / 'untouched'
    destination.write_bytes(b'')
    os.symlink(destination, target)
    with pytest.raises(HpoError) as err:
        svc.load_study(record.study_id)
    assert err.value.code == 'HPO_CORRUPT_STUDY'
    assert destination.read_bytes() == b''


def test_model_parent_link_rejected(tmp_path):
    snapshot, model = hpo_inputs.__wrapped__(tmp_path)
    parent = tmp_path / 'weights'
    parent.mkdir()
    real_model = parent / 'fixture.pt'
    real_model.write_bytes(b'synthetic')
    link = tmp_path / 'weights_link'
    os.symlink(parent, link, target_is_directory=True)
    with pytest.raises(HpoError) as err:
        HpoService(tmp_path / 'hpo').create_study(
            StudyConfig(), snapshot_dir=snapshot, model_path=link / 'fixture.pt')
    assert err.value.code == 'HPO_INVALID_CONFIG'


def test_store_does_not_overwrite_existing_corrupt_facts(setup_case):
    svc, root, record, *_ = setup_case
    valid = svc.load_study(record.study_id)
    target = root / record.study_id / 'study.json'
    target.write_text('{broken', encoding='utf-8')
    store = StudyStore(root)
    with store.locked(record.study_id), pytest.raises(HpoError) as err:
        store.write(valid)
    assert err.value.code == 'HPO_CORRUPT_STUDY'
    assert target.read_text(encoding='utf-8') == '{broken'


@pytest.mark.parametrize('damage', ['candidate', 'distribution'])
def test_sampler_revalidates_mutable_history(setup_case, damage):
    from auto_tune.modules.hpo.sampler import sample_next
    svc, root, record, *_ = setup_case
    svc.tell(record.study_id, 0, ResultInput(state='FAILED', reason_code='oom'))
    restored = svc.load_study(record.study_id)
    if damage == 'candidate':
        restored.trials[0].candidate_params['lr0'] = 999.0
    else:
        restored.trials[0].distributions['lr0'] = '{}'
    with pytest.raises(HpoError) as err:
        sample_next(restored.config, restored.trials)
    assert err.value.code == 'HPO_CORRUPT_STUDY'
