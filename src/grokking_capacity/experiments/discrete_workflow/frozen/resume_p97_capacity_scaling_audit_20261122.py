#!/usr/bin/env python3
"""Resume after a completed calibration stage stopped during CPU-only audit.

Training is never restarted or modified. Execute the original frozen controller's
remaining AST statements, with checkpoint evaluation on the original GPU.
"""
import ast
import os
from pathlib import Path
import runpy
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    controller = ROOT / 'scripts/run_p97_capacity_scaling.py'
    ns = runpy.run_path(str(controller))
    run = ns['RUN']; read = ns['read']; atomic = ns['atomic_json']
    ns['torch'].set_num_threads(2)
    assert read(run / 'manifest.json')['source_hashes'] == ns['source_hashes']()
    assert not (run / 'selection.json').exists()
    assert not (run / 'campaign_complete.json').exists()
    assert read(run / 'campaign_error.json')['error'] == 'AssertionError()'
    for stage in ['C48_learning_s41', 'random_learning', 'capacity_grid']:
        state = read(run / 'stages' / stage / 'controller.json')
        assert state['phase'] == 'complete' and not state['errors']
        assert all(job['exit_code'] == 0 for job in state['completed'])
    recovery = run / 'audit_recovery'
    recovery.mkdir(exist_ok=False)
    atomic(recovery / 'manifest.json', dict(
        started_at=time.time(), pid=os.getpid(), recovery_source_sha256=ns['sha'](Path(__file__)),
        frozen_controller_sha256=ns['sha'](controller),
        reason='CPU capacity recomputation differed by 0.207 bits; training-GPU audit passed.',
        verification_device='cuda:1', training_jobs_restarted=0,
        failed_checkpoint='B_d48_capacity_n32768_s61',
        cpu_recomputed_bits=30846.104935328243, stored_bits=30846.311681846768,
        gpu_recomputed_bits=30846.310808192356,
        training_source_changes=False))

    def audit(path):
        result = ns['audit_job'](path, device='cuda:1')
        atomic(run / 'audits' / (path.name + '.json'), result)
        print('AUDITED_GPU ' + path.name, flush=True)
        return result

    ns['audit'] = audit
    reg = ns['yaml'].safe_load((run / 'registration.yaml').read_text())
    common = reg['common']; devices = reg['resources']['devices']
    old = ROOT / 'results/discrete/p97-dropout-match-v1/jobs'
    multi = ROOT / 'results/discrete/p97-multilevel-dropout-v1/jobs'
    calibration = {}
    for seed in reg['calibration']['seeds']:
        calibration['C', 32, seed] = [old / (f'A32f128_random_n2048_s{seed}' if n == 2048 else f'A32f128_capacity_n{n}_s{seed}')
            for n in reg['calibration']['existing_C32_grid']]
    calibration['B', 48, 61] = [multi / f'categorical_d48_f192_capacity_n{n}_s61' for n in [2048, 8192]]
    gates = {('B', 48): multi / 'categorical_d48_f192_learning_s41',
        ('B', 60): ROOT / 'results/discrete/p97-categorical-groks-v1/jobs/B_d60_f240_validation_s321',
        ('C', 48): run / 'jobs/C_d48_learning_s41'}
    for role, width in [('C', 48), ('B', 48), ('B', 60)]:
        for seed in reg['calibration']['seeds']:
            calibration.setdefault((role, width, seed), [])
    for stage in ['random_learning', 'capacity_grid']:
        for job in read(run / 'stages' / stage / 'jobs.json'):
            path = run / 'jobs' / job['name']; s = read(path / 'summary.json'); c = s['config']
            assert c == job['config'] == read(run / 'job_registry.json')[job['name']]
            if stage == 'random_learning': assert s['random_learning_passed']
            calibration[c['role'], c['dim'], c['seed']].append(path)
            audit_path = run / 'audits' / (path.name + '.json')
            if audit_path.exists():
                result = read(audit_path); assert result['all_passed']
                for name, digest in result['input_hashes'].items(): assert ns['sha'](path / name) == digest
            else:
                audit(path)
    assert sum(map(len, calibration.values())) == 32
    inherited = read(run / 'inherited.json')
    (run / 'campaign_error.json').rename(recovery / 'original_campaign_error.json')
    atomic(recovery / 'audits_completed.json', dict(completed_at=time.time(), all_passed=True))
    ns.update(reg=reg, common=common, devices=devices, calibration=calibration, gates=gates, inherited=inherited)
    # Preserve the exact frozen protocol after the already-completed grid stage.
    tree = ast.parse(controller.read_text())
    original_main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    original_try = next(n for n in original_main.body if isinstance(n, ast.Try))
    start = next(i for i, n in enumerate(original_try.body) if isinstance(n, ast.For)
        and isinstance(n.target, ast.Name) and n.target.id == 'extension')
    original_try.body = original_try.body[start:]
    tail = ast.Module(body=[original_try], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tail), str(controller), 'exec'), ns)


if __name__ == '__main__':
    main()
