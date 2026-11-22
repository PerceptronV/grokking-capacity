"""Onset-dependent consequences of the fixed empirical timescale laws."""
from pathlib import Path
import math
import sqlite3
import numpy as np
import pandas as pd
from .paths import ROOT, RESULTS
from grokking_capacity.analysis.aggregate import onset_summary, summarize_delays
from grokking_capacity.utils.artifacts import save_json, sha
from .laws import branches, agreement


def current_model_onsets(law, out, statistics=RESULTS/'statistics'):
    source=ROOT/'results/cone_cellmeans_min_20261106/min_delay_cells.csv'
    cells=pd.read_csv(source);summaries={};rows=[]
    for domain,frame in [('d_le_256',cells[cells.dim<=256]),('full_grid',cells)]:
        for p,group in frame.groupby('p'):
            on=onset_summary({int(r.param_count):None if pd.isna(r.aggregate_delay) else float(r.aggregate_delay) for r in group.itertuples()})
            lo,hi=branches(p,law)
            rows.append({'p':int(p),'domain':domain,'lower_crossing_P':lo,'upper_crossing_P':hi,**on,
                         'first_confirmed_dim':None if on['first_confirmed'] is None else int(group.loc[group.param_count==on['first_confirmed'],'dim'].iloc[0])})
            if domain=='d_le_256':summaries[int(p)]=on
    pd.DataFrame(rows).to_csv(out/'onset_predictions.csv',index=False)
    observed=pd.read_csv(statistics/'central/predictiveness.csv')
    for r in observed.itertuples():assert summaries[r.p]['value']==r.empirical_onset_x
    x,y=np.log10(observed.p.to_numpy()),np.log10(observed.empirical_onset_x.to_numpy())
    coef=np.polyfit(x,y,1);rng=np.random.default_rng(20261109);draws=[]
    for _ in range(2000):
        ix=rng.integers(0,len(x),len(x))
        if np.unique(x[ix]).size>1:
            fit=np.polyfit(x[ix],y[ix],1)
            draws.append([np.polyval(fit,np.log10(p)) for p in [149,197,251]])
    null={'slope':float(coef[0]),'intercept':float(coef[1]),
          'method':'Retrospective refit to identified first-model ten-seed onsets. 2,000 prime-bootstrap draws; fit uncertainty only, without seed-population or censoring uncertainty.'}
    pd.DataFrame(draws,columns=['p149','p197','p251']).to_csv(out/'onset_powerlaw_bootstrap.csv',index=False)
    for i,p in enumerate([149,197,251]):
        null[f'p{p}']={'log10_P_onset':float(np.polyval(coef,np.log10(p))),
                       'fit_ci95_log10':np.quantile(np.array(draws)[:,i],[.025,.975]).tolist()}
    return {'central_onset_agreement':agreement(law,summaries),
            'onset_definition':'First positive minimum, unresolved if any smaller minimum could still qualify.',
            'onset_predictions':rows,'onset_powerlaw':null,'onset_input_sha256':sha(source)}


def alpha_followups(out):
    """Recompute the five-seed p=197 training-fraction follow-ups from raw traces."""
    runs=[]
    for filename in ['p197_alpha_runs.db','p197_alpha025_runs.db']:
        db=Path.home()/filename;con=sqlite3.connect(db.as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
        records=[dict(r) for r in con.execute('select * from runs where status="completed" and experiment_type="groks"')];con.close()
        for r in records:
            with np.load(r['npz_path'],allow_pickle=False) as z:
                def first(a,t):
                    hits=np.flatnonzero(a>=t)
                    return int(hits[0]+1) if len(hits) else None
                train,val=first(z['train_acc'],99),first(z['val_acc'],98)
            delay=None if val is None else max(0,val-train) if train is not None else 0
            runs.append({**{k:r[k] for k in ['uuid','p','train_fraction','dim','param_count','seed']},
                         'train99':train,'val98':val,'delay':delay,'trace_path':r['npz_path'],
                         'trace_sha256':sha(r['npz_path']),'database':str(db)})
    save_json(out/'alpha_followup_runs.json',runs)
    frames=pd.DataFrame(runs);rows=[];summaries=[]
    for a,frame in frames.groupby('train_fraction'):
        curve={}
        for (P,d),group in frame.groupby(['param_count','dim']):
            assert len(group)==5 and set(group.seed)==set(range(42,47))
            summary=summarize_delays([None if pd.isna(v) else float(v) for v in group.delay],'min')
            curve[int(P)]=summary['value'];rows.append({'alpha':a,'P':int(P),'dim':int(d),**summary})
        on=onset_summary(curve)
        summaries.append({'alpha':float(a),**on,'log10P':None if on['value'] is None else math.log10(on['value'])})
    save_json(out/'alpha_followup_cells.json',rows);save_json(out/'alpha_followup_onsets.json',summaries)
    return summaries
