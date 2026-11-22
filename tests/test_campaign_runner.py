"""Recovery and registration checks must hold without launching GPU experiments."""
import json
from pathlib import Path
import pytest

from grokking_capacity.experiments.campaigns import runner
from grokking_capacity.experiments.campaigns.config import freeze_selection
from grokking_capacity.experiments.campaigns.status import status, read_trace
from grokking_capacity.experiments.registered import read, verify_sources, ARCHIVE, export_legacy


@pytest.fixture
def processes(monkeypatch):
    launched = []
    failures = set()
    class Process:
        def __init__(self, command, **kwargs):
            self.path = Path(command[command.index('--output') + 1])
            self.config = read(command[command.index('--config') + 1])
            self.pid = 999999
            self.done = False
            launched.append(self.path.name)
        def poll(self):
            if not self.done:
                self.path.mkdir()
                (self.path/'model.pt').write_bytes(b'checkpoint')
                if self.path.name not in failures:
                    (self.path/'summary.json').write_text(json.dumps(dict(config=self.config)))
                self.done = True
            return 1 if self.path.name in failures else 0
    monkeypatch.setattr(runner.subprocess, 'Popen', Process)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **k: 'test-commit\n')
    return launched, failures


def specs(*names):
    return [dict(name=name, config=dict(seed=i)) for i, name in enumerate(names)]


def execute(root, jobs, **kwargs):
    return runner.run_stage(root, 'test', jobs, ['cpu'], 'test.worker', poll_seconds=0, **kwargs)


def test_completed_training_reused_and_audit_retried(tmp_path, processes):
    launched, _ = processes
    def failed_audit(path):
        raise ValueError('audit failed')
    with pytest.raises(ValueError, match='audit failed'):
        execute(tmp_path, specs('a', 'b'), audit=failed_audit)
    audited = []
    execute(tmp_path, specs('a', 'b'), resume=True, audit=lambda p: audited.append(p.name))
    assert launched == ['a', 'b'] and audited == ['a', 'b']
    execute(tmp_path, specs('a', 'b'), resume=True, audit=lambda p: pytest.fail('Repeated audit'))
    assert read(tmp_path/'stages/test/controller.json')['phase'] == 'complete'


def test_failure_stops_pending_jobs_and_partial_outputs_are_never_overwritten(tmp_path, processes):
    launched, failures = processes
    failures.add('a')
    with pytest.raises(RuntimeError, match='failed'):
        execute(tmp_path, specs('a', 'b'))
    assert launched == ['a']
    assert read(tmp_path/'stages/test/controller.json')['not_launched_after_failure'] == ['b']
    with pytest.raises(RuntimeError, match='Partial output'):
        execute(tmp_path, specs('a', 'b'), resume=True)


def test_artifact_tampering_and_plan_changes_are_rejected(tmp_path, processes):
    execute(tmp_path, specs('a'))
    with pytest.raises(ValueError, match='Frozen inputs'):
        execute(tmp_path, [dict(name='a', config=dict(seed=400))], resume=True)
    (tmp_path/'jobs/a/model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Completed output changed'):
        execute(tmp_path, specs('a'), resume=True)
    assert processes[0] == ['a']


def test_source_guard_runs_on_resume_before_auditing(tmp_path, processes):
    execute(tmp_path, specs('a'))
    def changed():
        raise ValueError('source changed')
    with pytest.raises(ValueError, match='source changed'):
        execute(tmp_path, specs('a'), resume=True, source_check=changed)
    assert processes[0] == ['a']


def test_campaign_lock_blocks_overlapping_controllers(tmp_path, processes):
    with runner.exclusive(tmp_path/'controller.lock'):
        with pytest.raises(RuntimeError, match='Another controller'):
            execute(tmp_path, specs('a'))
    assert not processes[0]


def test_selection_replay_keeps_original_timestamp_and_hash(tmp_path):
    p = tmp_path/'selection.json'
    freeze_selection(p, dict(frozen_at=1, pairs=[1, 2]))
    digest = runner.sha(p)
    freeze_selection(p, dict(frozen_at=9, pairs=[1, 2]))
    assert runner.sha(p) == digest
    with pytest.raises(ValueError, match='selection changed'):
        freeze_selection(p, dict(frozen_at=9, pairs=[1, 3]))


def test_status_uses_registered_counts_and_real_steps(tmp_path):
    registry = {}
    for seed in range(3):
        name = f'job{seed}'
        config = dict(role='C', n_train=17, seed=seed)
        registry[name] = config
        path = tmp_path/'jobs'/name
        path.mkdir(parents=True)
        (path/'summary.json').write_text(json.dumps(dict(config=config, learning_gate_passed=True)))
        (path/'trace.jsonl').write_text(json.dumps(dict(epoch=2, step=6, val_acc=.99))+'\n{"unfinished":')
    (tmp_path/'job_registry.json').write_text(json.dumps(registry))
    result = status(tmp_path)
    assert result['groups'] == [dict(size=17, role='C', expected=3, completed=3, first99_count=3, mean_first99=6)]
    (tmp_path/'jobs/job2/trace.jsonl').write_text(json.dumps(dict(epoch=2, step=6, val_acc=.9)))
    assert status(tmp_path)['groups'][0]['mean_first99'] is None


def test_corrupt_interior_trace_not_silently_skipped(tmp_path):
    p = tmp_path/'trace.jsonl'
    p.write_text('{broken}\n{"step": 3}\n')
    with pytest.raises(json.JSONDecodeError):
        read_trace(p)


def test_archive_is_exact_and_never_presented_as_current(tmp_path):
    manifests = read(ARCHIVE/'manifest.json')
    expected = manifests['source_sets']['capacity_scaling']
    assert verify_sources(expected) == 'legacy-v1'
    with pytest.raises(ValueError):
        verify_sources(expected, allow_legacy=False)
    bad = {**expected, next(iter(expected)): '0'*64}
    with pytest.raises(ValueError):
        verify_sources(bad)
    exported = export_legacy(tmp_path/'original')
    assert verify_sources(expected, root=exported, allow_legacy=False) == 'current'


def test_source_sets_are_complete_files():
    from grokking_capacity.experiments.registered import ROOT, protocol_sources, CONTROLLERS
    for protocol in CONTROLLERS:
        paths = protocol_sources(protocol)
        assert paths and all((ROOT/p).is_file() for p in paths)
        assert 'src/grokking_capacity/experiments/arithmetic.py' in paths


def test_dataset_planner_resumes_with_custom_root_and_seed_count(tmp_path, processes, monkeypatch):
    from types import SimpleNamespace
    import yaml
    from grokking_capacity.experiments.campaigns import run_p97_dataset_scaling as campaign
    from grokking_capacity.experiments.registered import ROOT
    config = yaml.safe_load((ROOT/'configs/campaigns/dataset_scaling.yaml').read_text())
    constants = read(Path(__file__).parent/'fixtures/dataset_scaling_constants.json')['constants']
    evidence = tmp_path/'evidence.json'
    evidence.write_text(json.dumps(dict(constants=constants,evidence_sha256={})))
    config['models']['source'] = str(evidence)
    config['confirmation']['seeds'] = [2, 3, 4]
    config['dataset'].update(size_count=2, sizes=[928, 7449])
    path = tmp_path/'config.yaml'
    path.write_text(yaml.safe_dump(config))
    root = tmp_path/'custom-output'
    monkeypatch.setattr(campaign, 'RUN', root)
    args = SimpleNamespace(config=path,resume=False)
    reg, registry = campaign.initialize(args)
    assert len(registry) == 12
    jobs = [dict(name=n, config=c) for n,c in registry.items()]
    runner.run_stage(root,'confirmation',jobs,['cpu'],'worker',poll_seconds=0)
    before = runner.sha(root/'selection.json')
    args.resume = True
    _, resumed = campaign.initialize(args)
    assert resumed == registry and runner.sha(root/'selection.json') == before
    runner.run_stage(root,'confirmation',jobs,['cpu'],'worker',poll_seconds=0,resume=True)
    assert len(processes[0]) == 12


def test_grid_preserves_registered_capacity_jobs_and_cli_arguments():
    import yaml
    from grokking_capacity.experiments.campaigns.cli_grid import jobs, command
    from grokking_capacity.experiments.registered import ROOT
    config = yaml.safe_load((ROOT/'configs/campaigns/matched_capacity.yaml').read_text())
    specs = jobs(config)
    assert len(specs) == 96
    assert [(s['config']['options']['dim'],s['config']['options']['n_samples'],s['config']['options']['seed']) for s in specs] == [
        (d,n,s) for s in [20261103,20261104] for n in [1000,1374,1891,2601,3576,4917,6761,9300] for d in [10,12,14,16,18,22]]
    cmd = command(specs[0]['config'],Path('/tmp/job'), 'cpu')
    for flag,value in [('--p','113'),('--epochs','5000'),('--patience','100'),('--min-delta','0.0001'),('--dropout','0.2')]:
        assert cmd[cmd.index(flag)+1] == value
    assert '--save-model' in cmd
    followup = jobs(yaml.safe_load((ROOT/'configs/campaigns/dropout_followup.yaml').read_text()))
    assert len(followup) == 74
    arithmetic = [s for s in followup if s['config']['kind']=='groks']
    assert len(arithmetic) == 10
    assert '--ignore-memorisation' in command(arithmetic[0]['config'],Path('/tmp/job'),'cpu')
