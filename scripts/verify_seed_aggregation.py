#!/usr/bin/env python3
"""Independent arithmetic and artifact-preservation checks for the sensitivity audit."""
from pathlib import Path
from collections import defaultdict,Counter
import csv,hashlib,json,math,statistics,shutil
import pandas as pd
import pymupdf
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/seed_aggregation_sensitivity_20261106'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    frozen=ROOT/'results/cone_ten_seeds_h100_20261106/figure/runs.json'
    runs=json.loads(frozen.read_text());by=defaultdict(list)
    for r in runs:by[r['p'],r['dim']].append(r)
    assert len(runs)==17360 and len(by)==1736
    assert all(len(v)==10 and {r['seed'] for r in v}==set(range(42,52)) for v in by.values())
    assert sum(r['delay'] is None for r in runs)==370
    results={}
    fns={'min':min,'median':statistics.median,'mean':statistics.mean}
    for metric,fn in fns.items():
        with (OUT/f'cone/cells_{metric}.csv').open() as f:cells=list(csv.DictReader(f))
        states=Counter()
        for c in cells:
            rr=by[int(c['p']),int(c['dim'])];vals=[r['delay'] for r in rr]
            lo=fn([0 if v is None else v for v in vals]);hi=fn([float('inf') if v is None else v for v in vals])
            value=lo if lo==hi and math.isfinite(hi) else None
            stored=float(c['aggregate_delay']) if c['aggregate_delay'] else None
            assert stored==value or stored is not None and value is not None and math.isclose(stored,value,abs_tol=1e-12)
            state='unresolved' if value is None else 'positive' if value>0 else 'zero'
            assert c['state']==state
            assert int(c['unresolved_seeds'])==sum(v is None for v in vals)
            states[state]+=1
        results[metric]=dict(states)
    for rr in by.values():
        vals=[r['delay'] for r in rr]
        if None not in vals:assert (min(vals)>0)<=(statistics.median(vals)>0)<=(statistics.mean(vals)>0)
    main=pd.read_csv(OUT/'statistics/central_main_ten/overview.csv').set_index('variant')
    assert main.loc['min','n']==10 and abs(main.loc['min','rho']-.976)<.0005
    assert abs(main.loc['min','ccc']-.74)<.005 and abs(main.loc['min','slope']-.985)<.0005
    for metric in ['min','median','mean']:
        df=pd.read_csv(OUT/f'statistics/central_main_ten/{metric}/predictiveness.csv')
        assert len(df)==10 and set(df.slice_value)=={97,101,103,107,109,113,127,131,137,139}
    # Check the fold condition and actual fitted-law equality independently.
    laws=json.loads((OUT/'laws/comparison.json').read_text())['unified_cell_reducers']
    for name,x in laws.items():
        m,g=x['mem'],x['gen'];p=x['fold'];K=p*(p-1)/2*math.log2(p+2);q=m['gamma_original']/2.16
        P=q*K/g['beta']
        gap=m['ln_b']+q*K/P-(g['lnA']+g['a']*math.log(p)-g['beta']*math.log(P))
        assert abs(gap)<1e-9,(name,gap)
        for p in [149,197]:
            P=x[f'p{p}']
            if P is None:continue
            K=p*(p-1)/2*math.log2(p+2)
            gap=m['ln_b']+q*K/P-(g['lnA']+g['a']*math.log(p)-g['beta']*math.log(P))
            assert abs(gap)<1e-8
    before=json.loads((OUT/'protected_before.json').read_text())
    for rel,digest in before.items():assert sha(ROOT/rel)==digest,rel
    assert sha(OUT/'default_render_verification/cone_seedmin_fixed.png')==sha(ROOT/'results/cone_ten_seeds_h100_20261106/figure/cone_seedmin_fixed.png')
    pdfs=list(OUT.glob('*.pdf'))+list((OUT/'median_preview').glob('*.pdf'))+list((OUT/'crossing_cli_median').glob('*.pdf'))
    for p in pdfs:
        assert p.with_suffix('.png').exists()
        doc=pymupdf.open(p)
        for page in doc:
            for b in page.get_text('blocks'):
                box=pymupdf.Rect(b[:4]);assert box.x0>=-.1 and box.y0>=-.1 and box.x1<=page.rect.width+.1 and box.y1<=page.rect.height+.1,(p,b)
    changed=[]
    for old in (OUT/'source_before').rglob('*.py'):
        rel=old.relative_to(OUT/'source_before');now=ROOT/rel
        if now.exists() and sha(old)!=sha(now):changed.append(str(rel))
    new=['scripts/analyse_seed_aggregation.py','scripts/analyse_seed_aggregation_crossings.py','scripts/summarise_seed_aggregation.py','scripts/report_seed_aggregation.py','scripts/verify_seed_aggregation.py','scripts/plot_cone_aggregate.py','tests/test_seed_aggregation.py','tests/test_figures.py','tests/test_intersection_axes.py','src/grokking_capacity/analysis/AGGREGATION.txt']
    source_hashes={}
    for rel in sorted(set(changed+new)):
        p=ROOT/rel;q=OUT/'source'/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);source_hashes[rel]=sha(p)
    input_files=[frozen,ROOT/'results/law_complete_ten_seed_sensitivity_20261106/selected_runs.csv',ROOT/'results/a100_h100_grokking_agreement_20261106/paired_runs.csv',*ROOT.glob('configs/*.yaml')]
    result={'independent_1736_cell_rescoring':results,'seed_count_and_identity_verified':True,
        'unresolved_seeds_retained':370,'main_test_ten_prime_baseline_reproduced':True,'law_fold_and_crossing_equations_checked':True,
        'default_cone_png_byte_identical':True,'protected_publication_files_unchanged':len(before),
        'figure_PDF_and_PNG_pairs_checked':len(pdfs),'PDF_text_not_clipped':True,'source_hashes':source_hashes,
        'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in input_files},
        'tests_log':'tests.log','known_preexisting_test_failure':'preexisting_test_failure.json'}
    (OUT/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_hashes','input_hashes']},indent=2))
if __name__=='__main__':main()
