#!/usr/bin/env python3
"""Plot audited development evidence without implying a matched comparison."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path('reports/fullwidth_development')
a = json.loads((root/'a_calibration.json').read_text())
with (root/'capacity.csv').open() as f:
    rows = list(csv.DictReader(f))
points = [(f'A: seed {seed}', value/1000, False)
          for seed,value in a['per_seed_capacity_bits'].items()]
status = json.loads((root/'status.json').read_text())
for comparison in status.get('comparisons', []):
    if not comparison['capacity_match_accepted']:
        continue
    selected = json.loads((Path(comparison['path'])/'selection.json').read_text())
    family = next(run['model'] for run in selected['runs'] if run['name'].startswith('B_'))
    points.extend((f"B {family} (accepted): seed {row['seed']}", row['b_bits']/1000, False)
                  for row in selected['assessment']['estimates'])
for assessment_path in sorted(root.glob('*_capacity_assessment.json')):
    assessment = json.loads(assessment_path.read_text())
    if assessment['accepted'] or not assessment.get('estimates'):
        continue
    manifest = json.loads((Path(assessment['b_runs'][0])/'manifest.json').read_text())
    cfg = manifest['config']
    for estimate in assessment['estimates']:
        label = f"B {cfg['model']}: width {cfg['dim']}/FF{cfg.get('mlp_dim',4*cfg['dim'])}, seed {estimate['seed']}"
        points.append((label, estimate['b_bits']/1000, 'completed_failure'))
for row in rows:
    if row['above_target_rejection_verified'] != 'True':
        continue
    if row['family'] == 'categorical':
        label = f"B categorical: width {row['dim']}, alphabet {row['alphabet']}"
    else:
        label = f"B scalar: width {row['dim']}, FF {row['mlp_dim']}, {row['levels']} levels"
        if row.get('quantizer_surrogate') == 'rms_clipped':
            label += ', RMS normalized'
    points.append((label, float(row['audited_checkpoint_bits'])/1000, True))
fig, ax = plt.subplots(figsize=(11, max(5.7, .4 * len(points) + 1.8)))
target = a['target_bits']/1000
ax.axvspan(.9*target,1.1*target,color='#daf0df',label='Fixed dataset target ±10%')
ax.axvline(target,color='#3c6548',ls='--',lw=1)
for i,(label,value,lower_bound) in enumerate(points):
    ax.scatter(value,i,marker=('x' if lower_bound == 'completed_failure' else '>' if lower_bound else 'o'),s=65,
               color='#c66b28' if lower_bound else '#3069a2',zorder=3)
    ax.annotate(f'{value:.2f}',(value,i),xytext=(7,-3),textcoords='offset points',fontsize=9)
ax.set_yticks(range(len(points)),[p[0] for p in points])
ax.invert_yaxis();ax.set_xlim(65,125);ax.grid(axis='x',alpha=.15)
ax.set_xlabel('Measured label information (thousand bits)')
ax.set_title('Capacity calibration against the fixed dataset target',pad=13)
ax.legend(loc='upper right',frameon=False)
fig.text(.03,.02,'Circles/crosses: recorded maxima across two sizes at 120k updates; crosses failed the capacity assessment.\nTriangles: independently audited checkpoint lower bounds. All B recipes shown passed arithmetic.',fontsize=9)
fig.tight_layout(rect=(0,.1,1,1))
for ext in ['png','pdf']:fig.savefig(root/f'capacity_development.{ext}',dpi=180)
plt.close(fig)
with (root/'arithmetic.csv').open() as file:
    qualified = [Path(row['path']) for row in csv.DictReader(file)
                 if row['family'] == 'categorical'
                 and row['arithmetic_gate_passed'] == 'True'
                 and row['audit_present'] == 'True']
path = max(qualified, key=lambda p: json.loads((p/'manifest.json').read_text())['start_time'])
manifest = json.loads((path/'manifest.json').read_text())
cfg = manifest['config']
audit = json.loads((path/'audit.json').read_text())
trace=[json.loads(line) for line in (path/'trace.jsonl').read_text().splitlines()]
fig,ax=plt.subplots(figsize=(8,4.5))
for key,label,color,style in [('val_mode_acc','Held out: hard argmax','#3069a2','-'),
                            ('val_acc','Held out: sampled','#c66b28','-'),
                            ('train_mode_acc','Train: hard argmax','#3069a2','--'),
                            ('train_acc','Train: sampled','#c66b28','--')]:
    ax.plot([r['step'] for r in trace],[100*r[key] for r in trace],label=label,color=color,ls=style,lw=1.4)
ax.axhline(99,color='grey',ls=':',lw=1)
ax.set(xlabel='Optimizer updates',ylabel='Accuracy (%)',ylim=(0,101),xlim=(0,trace[-1]['step']),
       title=f"Categorical arithmetic qualification: p{cfg['p']} division, seed {cfg['seed']}")
ax.grid(alpha=.15);ax.legend(loc='lower right',frameon=False)
fig.text(.03,.02,
         f"Latest audited development recipe: width {cfg['dim']}, FF {cfg.get('mlp_dim') or 4*cfg['dim']}, alphabet {cfg['alphabet']}; sampled training.\n"
         f"Independent audit: {100*audit['cpu_mode_val_acc']:.2f}% hard-mode held out; {100*audit['sampled_val_acc']:.2f}% sampled held out.",
         fontsize=9)
fig.tight_layout(rect=(0,.12,1,1))
for ext in ['png','pdf']:fig.savefig(root/f'categorical_qualification.{ext}',dpi=180)
plt.close(fig)
print('Wrote capacity and categorical qualification figures (PNG/PDF).')
