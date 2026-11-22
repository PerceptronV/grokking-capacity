#!/usr/bin/env python3
"""Independent scalar/CSV verification and final provenance for the size sweep."""
from grokking_capacity.experiments.registered import verify_sources as verify_registered_sources

import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'reports/p97_capacity_scaling'
RUN=ROOT/'results/discrete/p97-capacity-scaling-v1'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def number(value):return int(value) if value else None


def main():
    summary=read(OUT/'summary.json');selection=read(OUT/'selection.json');done=read(OUT/'campaign_complete.json')
    records=list(csv.DictReader((OUT/'confirmation_runs.csv').open()));assert len(records)==200
    expected={}
    for row in records:
        path=Path(row['path']);s=read(path/'summary.json');c=s['config']
        trace=[json.loads(line) for line in (path/'trace.jsonl').read_text().splitlines()]
        def first(key,threshold):return next((x['step'] for x in trace if x[key]>=threshold),None)
        t=first('val_acc',.99);assert number(row['t_gen_updates'])==t
        if t is None:assert len(trace)==5000
        for out,key in [('paper_delay_updates','train_online_acc'),('eval_delay_updates','train_eval_acc')]:
            train=first(key,.99);val=first('val_acc',.98)
            delay=max(0,val-train) if train is not None and val is not None else None
            assert number(row[out])==delay
        stop=next((i+1 for i in range(2,len(trace)) if trace[i]['train_online_acc']>=.99
            and all(v['val_acc']>=.99 for v in trace[i-2:i+1])),None)
        assert len(trace)==(stop if stop is not None else 5000)
        assert (row['joint99_at_final']=='True')==(stop is not None)
        stable=any(all(v['val_acc']>=.99 for v in trace[i-2:i+1]) for i in range(2,len(trace)))
        learned=stable and trace[-1]['val_acc']>=.99 and trace[-1]['train_eval_acc']>=.99
        assert (row['learning_success']=='True')==learned
        expected[int(row['pair_index']),row['role'],int(row['seed'])]=t
        assert int(row['parameters'])==(32 if row['role']=='C' else 64)*int(row['dim'])**2+203*int(row['dim'])
    assert len(expected)==200
    for g in summary['groups']:
        values=[expected[g['pair_index'],g['role'],seed] for seed in range(401,411)]
        rows=[r for r in records if int(r['pair_index'])==g['pair_index'] and r['role']==g['role']]
        assert g['joint_stopping_successes']==sum(r['joint99_at_final']=='True' for r in rows)
        assert g['learning_successes']==sum(r['learning_success']=='True' for r in rows)
        obs=[v for v in values if v is not None];filled=[50000 if v is None else v for v in values]
        assert g['t_gen']['observed']==len(obs) and g['t_gen']['censored']==10-len(obs)
        assert g['t_gen']['mean_lower_bound']==statistics.mean(filled)
        assert g['t_gen']['mean']==(statistics.mean(obs) if len(obs)==10 else None)
        assert g['t_gen']['median']==(statistics.median(filled) if len(obs)>=6 else None)
        assert g['t_gen']['median_lower_bound']==statistics.median(filled)
    calibration_points=[]
    for key,paths in selection['calibration_paths'].items():
        cells=[read(Path(p)/'summary.json') for p in paths];best=max(cells,key=lambda s:s['final_bits'])
        calibration_points.append(dict(role=key.split('_')[0],parameters=best['param_count'],bits=best['final_bits']))
    constants={}
    for role in ['C','B']:
        pts=[p for p in calibration_points if p['role']==role];assert len(pts)==4
        k=sum(p['parameters']*p['bits'] for p in pts)/sum(p['parameters']**2 for p in pts)
        assert math.isclose(k,summary['constants'][role]['bits_per_parameter'],rel_tol=1e-12)
        constants[role]=k
    assert len(selection['pairs'])==10
    for pair in selection['pairs']:
        target=constants['C']*pair['C_parameters']
        best=min(range(24,161,2),key=lambda w:(abs(constants['B']*(64*w*w+203*w)/target-1),w))
        assert best==pair['B']['dim']
        assert math.isclose(pair['predicted_relative_mismatch'],constants['B']*pair['B_parameters']/target-1,abs_tol=1e-12)
        assert abs(pair['predicted_relative_mismatch'])<=.05
    independent_slopes={}
    for name in ['complete_range_scaling','capped_time_scaling']:
        fit=summary[name]
        if not fit['available']:continue
        indices=fit['pair_indices'];independent_slopes[name]={}
        for role in ['C','B']:
            values=[]
            for seed in range(401,411):
                x=[math.log(selection['pairs'][i][role+'_estimated_bits']) for i in indices]
                y=[math.log(expected[i,role,seed] if expected[i,role,seed] is not None else 50000) for i in indices]
                mx=statistics.mean(x);my=statistics.mean(y)
                values.append(sum((a-mx)*(b-my) for a,b in zip(x,y))/sum((a-mx)**2 for a in x))
            assert all(math.isclose(a,b,abs_tol=1e-10) for a,b in zip(values,fit['roles'][role]['seed_slopes']))
            independent_slopes[name][role]=statistics.mean(values)
    for name in ['mean_t_gen_scaling','mean_capped_t_gen_scaling']:
        fit=summary[name]
        if not fit['available']:continue
        independent_slopes[name]={}
        for role in ['C','B']:
            x=[math.log(selection['pairs'][i][role+'_estimated_bits']) for i in fit['pair_indices']]
            y=[math.log(statistics.mean(expected[i,role,seed] if expected[i,role,seed] is not None else 50000
                for seed in range(401,411))) for i in fit['pair_indices']]
            mx=statistics.mean(x);my=statistics.mean(y)
            slope=sum((a-mx)*(b-my) for a,b in zip(x,y))/sum((a-mx)**2 for a in x)
            assert math.isclose(slope,fit['roles'][role]['slope'],abs_tol=1e-10)
            independent_slopes[name][role]=slope
    extension=read(OUT/'analysis_plan_extension.json')
    assert extension['recorded_at']<selection['frozen_at']
    inputs=read(OUT/'input_hashes.json')
    for p,h in inputs.items():assert sha(Path(p))==h
    manifest=read(OUT/'manifest.json')
    verify_registered_sources(manifest['source_hashes'])
    audits=read(OUT/'checkpoint_audits.json');assert len(audits)==done['new_jobs']+done['inherited_jobs']
    assert all(a['all_passed'] for a in audits)
    for path,hashes in read(OUT/'inherited.json').items():
        for name,h in hashes.items():assert sha(Path(path)/name)==h
    verification=dict(all_passed=True,confirmation_traces_recomputed=200,first_stopping_epochs_verified=200,
        calibration_maxima_recomputed=len(calibration_points),constants_recomputed=constants,
        matching_choices_verified=10,independent_slopes=independent_slopes,input_hashes_verified=len(inputs),
        all_frozen_scientific_sources_unchanged=True,checkpoint_audits=len(audits))
    (OUT/'independent_validation.json').write_text(json.dumps(verification,indent=2)+'\n')
    recovery=read(OUT/'audit_recovery.json')
    assert recovery['manifest']['training_jobs_restarted']==0
    assert recovery['audits_completed']['all_passed']
    assert recovery['audits_completed']['completed_at']<selection['frozen_at']
    recovery_sources = [Path(__file__).with_name('resume_p97_capacity_scaling_audit.py'),
                        Path(__file__).with_name('frozen') / 'resume_p97_capacity_scaling_audit_20261122.py']
    assert recovery['manifest']['recovery_source_sha256'] in {sha(p) for p in recovery_sources}
    verify_registered_sources({'scripts/run_p97_capacity_scaling.py':recovery['manifest']['frozen_controller_sha256']})
    analysis=['src/grokking_capacity/experiments/discrete_workflow/report_p97_capacity_scaling.py','src/grokking_capacity/experiments/discrete_workflow/verify_p97_capacity_scaling.py',
        'src/grokking_capacity/experiments/discrete_workflow/status_p97_capacity_scaling.py','src/grokking_capacity/experiments/discrete_workflow/resume_p97_capacity_scaling_audit.py']
    diagnostic=ROOT/'results/discrete/p97-capacity-scaling-v1-registration-parse-failure/diagnostic.json'
    provenance=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),branch='idea/discrete',run_root=str(RUN),
        scientific_source_commit=manifest['source_commit'],new_jobs=done['new_jobs'],inherited_jobs=done['inherited_jobs'],
        confirmation_runs=200,all_requested_jobs_completed=True,all_frozen_scientific_sources_unchanged=True,
        independently_verified=verification,analysis_sources_sha256={p:sha(ROOT/p) for p in analysis},
        registration_startup_correction=dict(path=str(diagnostic),sha256=sha(diagnostic),jobs_before_correction=0),
        calibration_audit_recovery=recovery,
        report_artifacts_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='completion_provenance.json'})
    (OUT/'completion_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
