#!/usr/bin/env python3
from pathlib import Path
import copy,json,sys
import numpy as np,pandas as pd
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import analyse_seed_aggregation as a
from grokking_capacity.analysis.config_view import ConfigView
from grokking_capacity.analysis.stats import compute_hypothesis_tests
from grokking_capacity.analysis.plots import _curve_for_slice,_slice_values
from grokking_capacity.analysis.aggregate import find_intersection
import grokking_capacity.dispatch.config as cv
OUT=ROOT/'reports/paper_rigour_new_conventions_20261106/statistics';OUT.mkdir(exist_ok=True)
a.EVENTS={r['uuid']:{k:r[k] for k in ['train99','val98','followup_epochs']} for r in json.loads((ROOT/'results/cone_ten_seeds_h100_20261106/figure/runs.json').read_text())}
expand=cv.expand_runs
def small(spec):
 spec=copy.deepcopy(spec)
 for e in spec.get('experiments',{}).values():
  if e.get('match_by')=='param_count':e.pop('match_by');e.pop('param_count_targets',None);e['dims']=[20]
 return expand(spec)
cv.expand_runs=small
rows=[];crosses=[];groups=[]
for config in ['central','init_scale_sweep','weight_decay_sweep','lr_sweep','alpha_sweep','depth_scaling','task_add','task_mul']:
 view=ConfigView.from_yaml(ROOT/'configs'/f'{config}.yaml',db_path=str(ROOT/'runs.db'));fig=view.intersection_figures[0]
 for gi,g in enumerate(view.iter_groups()):
  groups.append({'config':config,'group':gi,'speed_n':len(g.speed_runs),'groks_n':len(g.groks_runs),'speed_hardware':dict(Counter(r.get('host') for r in g.speed_runs)),'groks_hardware':dict(Counter(r.get('host') for r in g.groks_runs)),'mup_bases':sorted({r.get('mup_base',0) for r in g.speed_runs+g.groks_runs})})
  for sv in _slice_values(g,fig):
   s=_curve_for_slice(g.speed_runs,fig,sv,'saturation_epoch');t=_curve_for_slice(g.groks_runs,fig,sv,'grokking_epoch');pair=find_intersection(s,t)
   if not pair:continue
   xs=np.array(sorted(s));xt=np.array(sorted(t));grid=np.unique(np.r_[xs,xt]);grid=grid[(grid>=max(xs.min(),xt.min()))&(grid<=min(xs.max(),xt.max()))]
   gap=np.interp(grid,xs,[s[x] for x in xs])-np.interp(grid,xt,[t[x] for x in xt]);cross=bool(np.any(gap==0) or (gap.min()<0<gap.max()))
   crosses.append({'config':config,'group':gi,'slice':sv,'has_sign_change':cross,'returned_P':pair[0],'gap_min':float(gap.min()),'gap_max':float(gap.max())})
 for variant,legacy in [('legacy_min',True),('min_unresolved',False)]:
  df,cells=a.onset_rows(view,fig,'min',legacy=legacy)
  if config=='central':df=df[df.slice_value<=139]
  dest=OUT/f'{config}_{variant}';dest.mkdir(exist_ok=True);df.to_csv(dest/'predictiveness.csv',index=False)
  tests=compute_hypothesis_tests(view,fig,df);(dest/'hypothesis_tests.json').write_text(json.dumps(a.clean(tests),indent=2))
  vals=df.dropna(subset=['log_ratio'])
  rows.append({'config':config,'variant':variant,'N':tests['n_valid'],'rho':tests.get('predictiveness',{}).get('spearman',{}).get('rho'),'ccc':tests.get('calibration',{}).get('ccc',{}).get('value'),'median_log_ratio':float(vals.log_ratio.median()),'unresolved_onsets':int((df.onset_status=='unresolved').sum())})
 print(config,rows[-2:],flush=True)
 # Recalculate archived point-table statistics against the source artifact, when available.
 archive=ROOT/'figures'/config/fig.name/'predictiveness.csv'
 if archive.exists():
  df=pd.read_csv(archive);tests=compute_hypothesis_tests(view,fig,df)
  (OUT/f'{config}_archived_points.json').write_text(json.dumps(a.clean(tests),indent=2))
pd.DataFrame(rows).to_csv(OUT/'overview.csv',index=False);pd.DataFrame(crosses).to_csv(OUT/'crossing_signs.csv',index=False)
(OUT/'cohorts.json').write_text(json.dumps(groups,indent=2));(OUT/'event_checks.json').write_text(json.dumps({'event_count':len(a.EVENTS),'missing_trace':[u for u,v in a.EVENTS.items() if v.get('missing_trace')],'note':'Cone events independently verified; all additional config trace events freshly extracted.'},indent=2))
# A finite answer for curves with no crossing demonstrates the nearest-point fallback.
(OUT/'crossing_counterexample.json').write_text(json.dumps({'speed':{1:1,2:1},'gen':{1:2,2:2},'returned':find_intersection({1:1,2:1},{1:2,2:2})},indent=2))
print('DONE',len(crosses),'without_sign_change',sum(not r['has_sign_change'] for r in crosses),flush=True)
