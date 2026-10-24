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
for row in rows:
    if row['above_target_rejection_verified'] != 'True':
        continue
    if row['family'] == 'categorical':
        label = f"B categorical: width {row['dim']}, alphabet {row['alphabet']}"
    else:
        label = f"B scalar: width {row['dim']}, FF {row['mlp_dim']}, {row['levels']} levels"
    points.append((label, float(row['audited_checkpoint_bits'])/1000, True))
fig, ax = plt.subplots(figsize=(10,5.7))
target = a['target_bits']/1000
ax.axvspan(.9*target,1.1*target,color='#daf0df',label='Fixed dataset target ±10%')
ax.axvline(target,color='#3c6548',ls='--',lw=1)
for i,(label,value,lower_bound) in enumerate(points):
    ax.scatter(value,i,marker='>' if lower_bound else 'o',s=65,
               color='#c66b28' if lower_bound else '#3069a2',zorder=3)
    ax.annotate(f'{value:.2f}',(value,i),xytext=(7,-3),textcoords='offset points',fontsize=9)
ax.set_yticks(range(len(points)),[p[0] for p in points])
ax.invert_yaxis();ax.set_xlim(65,125);ax.grid(axis='x',alpha=.15)
ax.set_xlabel('Measured label information (thousand bits)')
ax.set_title('Capacity calibration: A matches; audited B candidates exceed the target',pad=13)
ax.legend(loc='upper right',frameon=False)
fig.text(.03,.02,'A: maxima across two dataset sizes at 120k updates. B: audited checkpoint lower bounds, not saturated capacities.\nOnly arithmetic-qualified B recipes are shown. No accepted capacity-matched A/B/C comparison yet.',fontsize=9)
fig.tight_layout(rect=(0,.1,1,1))
for ext in ['png','pdf']:fig.savefig(root/f'capacity_development.{ext}',dpi=180)
plt.close(fig)
path=Path('results/discrete/fullwidth-learning-v5/B_categorical_d58_k16_rms_t01_mode_s42')
trace=[json.loads(line) for line in (path/'trace.jsonl').read_text().splitlines()]
fig,ax=plt.subplots(figsize=(8,4.5))
for key,label,color,style in [('val_mode_acc','Held out: hard argmax','#3069a2','-'),
                            ('val_acc','Held out: sampled','#c66b28','-'),
                            ('train_mode_acc','Train: hard argmax','#3069a2','--'),
                            ('train_acc','Train: sampled','#c66b28','--')]:
    ax.plot([r['step'] for r in trace],[100*r[key] for r in trace],label=label,color=color,ls=style,lw=1.4)
ax.axhline(99,color='grey',ls=':',lw=1)
ax.set(xlabel='Optimizer updates',ylabel='Accuracy (%)',ylim=(0,101),xlim=(0,13500),
       title='Categorical arithmetic qualification: p149 division, seed 42')
ax.grid(alpha=.15);ax.legend(loc='lower right',frameon=False)
fig.text(.03,.02,'Sampled training throughout. Width 58, alphabet 16; hard-mode primary evaluation.\nIndependent audit: 99.03% hard-mode held out; fresh sampled evaluations average 96.86%.',fontsize=9)
fig.tight_layout(rect=(0,.12,1,1))
for ext in ['png','pdf']:fig.savefig(root/f'categorical_qualification.{ext}',dpi=180)
plt.close(fig)
print('Wrote capacity and categorical qualification figures (PNG/PDF).')
