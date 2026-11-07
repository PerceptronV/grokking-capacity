#!/usr/bin/env python3
"""Live cone preview; incomplete seed cohorts never appear as completed positive cells."""
from pathlib import Path
import argparse,json,shutil,sqlite3,time
from cone_data import aggregate,score,write_json
from plot_cone_seedmin import plot
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/cone_p199_h100_20261105';LIVE=OUT/'live'
def render():
    LIVE.mkdir(exist_ok=True)
    if (OUT/'validation.json').exists():
        for ext in ['png','pdf']:shutil.copy2(OUT/'figure'/f'cone_seedmin_fixed.{ext}',LIVE/f'cone_seedmin_fixed.{ext}')
        write_json(LIVE/'status.json',{'status':'complete_and_verified','validation':str(OUT/'validation.json'),'training_runs':190,'planned_runs':190});return True
    m=json.loads((OUT/'manifest.json').read_text());base=ROOT/m['base_cohort'];old=json.loads((base/'cells.json').read_text());summary=json.loads((base/'summary.json').read_text())
    con=sqlite3.connect((OUT/'runs.db').as_uri()+'?mode=ro',uri=True);con.row_factory=sqlite3.Row
    fresh=[score(dict(r)) for r in con.execute("select * from runs where status='completed'")];con.close()
    cells=aggregate(fresh);pending=[]
    for c in cells:
        c['planned_seeds']=10
        if c['n_seeds']<10:
            pending.append(c['dim'])
            # An observed zero fixes the final minimum irrespective of later seeds.
            # Otherwise a partially sampled cell cannot yet be classified positive.
            if c['state']!='zero':c['state']='pending';c['mean_delay']=None;c['minimum_delay']=None
    checks=len(list((OUT/'checkpoint_checks').glob('*.json')))
    title=f'p=199 in progress: {len(fresh)}/190 runs complete; {checks}/190 independently checked\nTen seeds planned per new width; incomplete positive cells remain unclassified'
    plot(old+cells,summary,LIVE,fixed_only=True,title=title)
    write_json(LIVE/'status.json',{'status':'training_or_verification_in_progress','training_runs':len(fresh),'planned_runs':190,'independent_CPU_checks':checks,'widths_with_incomplete_seed_sets':pending,'new_cells_shown':len(cells),'classification_rule':'Any observed zero determines minimum zero; positive requires all10 planned seeds observed positive. Otherwise incomplete cohorts use an awaiting-seeds marker. Historical columns unchanged.'})
    print(json.dumps({'completed':len(fresh),'CPU_checks':checks}),flush=True);return False
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--watch',action='store_true');args=ap.parse_args()
    while True:
        done=render()
        if done or not args.watch:break
        time.sleep(60)
