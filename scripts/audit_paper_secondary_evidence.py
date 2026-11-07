#!/usr/bin/env python3
from pathlib import Path
import json,math,hashlib
import numpy as np
from scipy.stats import wilcoxon
R=Path(__file__).resolve().parents[1];O=R/'reports/paper_rigour_new_conventions_20261106';T=R/'results/theory'
used=[]
def read(p):
 p=Path(p);used.append({'path':str(p.relative_to(R)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()});return json.loads(p.read_text())
C=read(R/'results/capacity_h100_matched_20261103/summary.json')['C'];scale=C/2.16
batch=read(T/'gamma_by_batch.json');full=read(T/'gamma_fullbatch.json')
result={'batch':{'capacity_scale':scale,'old_window':[.08,1],'current_window':[.08/scale,1/scale],'gamma_inf':batch['sqrtB_form']['gamma_inf']*scale,'c_prime':batch['sqrtB_form']['c_prime']*scale,'max_abs_residual':max(abs(x) for x in batch['sqrtB_form']['resid'])*scale,'tobit_power':batch['gamma_vs_B_loglog_slope_tobit'],'censored_counts':{k:(v['tobit_epochs']['n_obs'],v['tobit_epochs']['n_cens']) for k,v in batch['per_batch'].items()},'fullbatch_gamma':full['per_batch']['-1']['tobit_epochs']['gamma']*scale,'fullbatch_CI':[v*scale for v in full['per_batch']['-1']['tobit_epochs']['ci95']]}}
ab=read(T/'character_ablation.json')['runs'];result['ablation']={'runs':len(ab),'winner_range':[min(r['n_winners'] for r in ab),max(r['n_winners'] for r in ab)],'median_base':float(np.median([r['acc_base'] for r in ab])),'median_excluded':float(np.median([r['acc_excluded'] for r in ab])),'median_restricted':float(np.median([r['acc_restricted'] for r in ab])),'mean_random_loss':float(np.mean([r['acc_base']-r['acc_ctrl_mean'] for r in ab])),'median_random_loss':float(np.median([r['acc_base']-r['acc_ctrl_mean'] for r in ab])),'scope':'Recalculated from intervention result rows, not rerun checkpoint ablations.'}
sel=read(T/'init_overlap_selection_mult.json');x=[r['log_ratio'] for r in sel];result['selection']={'n':len(x),'positive':sum(v>0 for v in x),'mean_share_ratio':float(np.exp(np.mean(x))),'wilcoxon':float(wilcoxon(x).pvalue),'mean_rank':float(np.mean([r['mean_rank'] for r in sel]))}
nu=read(T/'init_overlap_null.json')['cells'];nr=[]
for r in nu:
 pred=r['ev_prediction_1p_sqrt'];nr.append({'p':r['p'],'dim':r['d'],'real_vs_null_fraction':r['fmax_real_mean']/r['fmax_null_mean']-1,'formula_vs_real_fraction':pred/r['enhancement_real']-1,'formula_vs_null_fraction':pred/r['enhancement_null']-1})
result['init_null']={'cells':nr,'max_abs_real_vs_null':max(abs(r['real_vs_null_fraction']) for r in nr),'max_abs_formula_vs_real':max(abs(r['formula_vs_real_fraction']) for r in nr),'max_abs_formula_vs_null':max(abs(r['formula_vs_null_fraction']) for r in nr)}
g=read(T/'character_growth_lambda_sweep.json');lams=[.1,.3,1.];rates=[g[str(l)]['windows']['grok+[0,300]']['amplitude']['winner'] for l in lams];result['winner_lambda']={'lambda':lams,'rates':rates,'slope':float(np.polyfit(np.log(lams),np.log(rates),1)[0]),'window':'grok+[0,300] AFTER grokking','takeoff':[g[str(l)]['takeoff_1p5x'] for l in lams],'grok':[g[str(l)]['grok_epoch'] for l in lams]}
al=read(T/'character_alpha_width.json');result['character_alpha_width']={k:{m:v for m,v in d.items() if m.startswith('exponent')} for k,d in al.items()}
wd=read(T/'wd_shift_test.json');result['wd_other_exponents']=[{'lambda':r['lambda'],'gamma_current':r['gamma']*scale,'beta':r['beta']} for r in wd if 'gamma' in r]
result['sources']=used
(O/'secondary_numbers.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['sources','init_null','wd_other_exponents']},indent=2));print('INIT NULL maxima',{k:v for k,v in result['init_null'].items() if k!='cells'})
