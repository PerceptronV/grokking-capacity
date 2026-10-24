"""Bind completed p97 reports to immutable audited checkpoints; no training."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

from scripts.confirm_discrete import process_live
from scripts.report_discrete import validate
from scripts.report_fullwidth_development import export

root = Path.cwd().resolve()
confirmation = root/'results/discrete/p97-confirmation-v9-quantized-rms'
control = root/'results/discrete/p97-v9/capacity-C/C_baseline_d72_ff62_n8192_s61'
report = root/'reports/p97_quantized_v9'
development = root/'reports/p97_development'
def read(path): return json.loads(path.read_text())
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def relative(path): return str(path.resolve().relative_to(root))

assert read(root/'results/discrete/p97-v9/finalizer.json')['phase'] == 'complete'
rows, records, assessment, calibration, gates = validate(confirmation, [control])
export(root/'results/discrete', development, prefix='p97-')
artifacts = read(report/'artifact_hashes.json')
for path, expected in artifacts.items():
    assert digest(report/path) == expected, path

paths = {confirmation/row['name'] for row in rows}
paths.update(Path(row['path']) for group in calibration.values() for row in group)
paths.update([
    root/'results/discrete/p97-v6-learning/A_baseline_d26_ff104_s42',
    root/'results/discrete/p97-v9-learning/B_quantized_d72_ff62_q2_rms_s41',
    root/'results/discrete/p97-v9/random-learning/B_quantized_d72_ff62_q2_rms_random_learning_n2048_s61',
])
for path in list(paths):
    gate = read(path/'manifest.json').get('gate')
    if gate: paths.add(Path(gate['path']))
checks=[]
for path in sorted(paths):
    m=read(path/'manifest.json')
    c=torch.load(path/'checkpoint.pt', map_location='cpu', weights_only=True)
    trace=[json.loads(line) for line in (path/'trace.jsonl').read_text().splitlines()]
    assert c['trace']==trace and c['step']==trace[-1]['step'], path
    for key in ('config','source_hashes','dataset_sha256','gate'):
        assert c['scientific'][key]==m[key], (path,key)
    for source,expected in m['source_hashes'].items():
        assert digest(path/'source'/source)==expected, (path,source)
    kind='capacity' if m['config']['task']=='random' else 'arithmetic'
    audit_path=path/('audit_capacity.json' if kind=='capacity' else 'audit.json')
    a=read(audit_path)
    assert a['checkpoint_step']==c['step'] and a['param_count']==m['param_count']
    assert a['dataset_sha256']==m['dataset_sha256']
    if kind=='capacity': assert a['checkpoint_sha256']==digest(path/'checkpoint.pt')
    else:
        assert abs(a['train_acc']-trace[-1]['train_acc'])<1e-6
        assert abs(a['val_acc']-trace[-1]['val_acc'])<1e-6
    if m['gate']:
        gp=Path(m['gate']['path'])
        assert digest(gp/'summary.json')==m['gate']['summary_sha256']
        assert (gp/'audit.json').stat().st_mtime<m['start_time']
    checks.append(dict(path=relative(path),kind=kind,step=c['step'],
                       checkpoint_sha256=digest(path/'checkpoint.pt'),
                       audit_sha256=digest(audit_path),
                       saved_trace_matches_export=True,config_and_source_snapshots_verified=True))
    destination=development/'supporting_evidence'/path.parent.name/path.name
    # The main report already preserves all confirmation and calibration traces.
    if path.parent!=confirmation and not any(path==Path(r['path']) for g in calibration.values() for r in g):
        destination.mkdir(parents=True,exist_ok=True)
        for name in ('manifest.json','summary.json','trace.jsonl','audit.json','audit_capacity.json'):
            if (path/name).exists(): shutil.copyfile(path/name,destination/name)

first_confirmation=min(read(confirmation/row['name']/'manifest.json')['start_time'] for row in rows)
assert (confirmation/'selection.json').stat().st_mtime<first_confirmation
for group in ('A','B'):
    for row in calibration[group]:
        assert (Path(row['path'])/'audit_capacity.json').stat().st_mtime<first_confirmation
random_path=root/'results/discrete/p97-v9/random-learning/B_quantized_d72_ff62_q2_rms_random_learning_n2048_s61'
random_summary=read(random_path/'summary.json')
assert random_summary['final']['train_acc']>=.99 and random_summary['metrics']['train_99'] is not None
assert (random_path/'audit_capacity.json').stat().st_mtime < min(read(Path(row['path'])/'manifest.json')['start_time'] for row in calibration['B'])
workers=[]
for path in (confirmation/'orchestrator.json',root/'results/discrete/p97-v9/finalizer.json',root/'results/discrete/p97-v9/B_queue.json'):
    s=read(path)
    live=process_live(s['pid'],s['process_start_ticks'])
    assert s['phase']=='complete' and not live, path
    workers.append(dict(path=relative(path),phase=s['phase'],process_live=live))
result=dict(recorded_at=datetime.now(timezone.utc).isoformat(),
            script_sha256=digest(Path(__file__)),
            scientific_source_checkout='results/discrete/source-worktrees/p97-v9 @ aa70459708ff59b34b300d2b8c15ad0218b9617f',
            current_report_script_sha256=digest(root/'scripts/report_discrete.py'),
            report=dict(path=relative(report),verified_artifacts=len(artifacts),hash_manifest_sha256=digest(report/'artifact_hashes.json')),
            capacity_match_frozen_before_confirmation=True,
            all_capacity_audits_preceded_confirmation=True,
            arithmetic_audits_preceded_random_training=True,
            selected_random_validation_preceded_B_calibration=True,
            workers=workers,checkpoints=checks,
            counts=dict(confirmation=len(rows),calibration=sum(map(len,calibration.values())),checkpoints=len(checks)),
            selected_random_learning=dict(path=relative(random_path),final_accuracy=random_summary['final']['train_acc'],best_accuracy=max(json.loads(line)['train_acc'] for line in (random_path/'trace.jsonl').read_text().splitlines())))
(development/'completion_audit.json').write_text(json.dumps(result,indent=2)+'\n')
if Path(__file__).resolve() != (development/'completion_audit_script.py').resolve():
    shutil.copyfile(__file__,development/'completion_audit_script.py')
manifest={str(p.relative_to(development)):digest(p) for p in sorted(development.rglob('*')) if p.is_file() and p.name!='artifact_hashes.json'}
(development/'artifact_hashes.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(result['counts'],indent=2))
