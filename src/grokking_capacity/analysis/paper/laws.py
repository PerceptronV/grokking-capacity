"""Registered law-fit cohort and comparisons used by the paper and sensitivities."""
from collections import defaultdict
import math
import sqlite3
import numpy as np
from scipy.stats import spearmanr
from ..config_view import ConfigView
from ..forecast_onset import _pick_group, CENTRAL_PRIMES
from ..plots import _passes_filters
from ..matching import compute_n_equiv
from ..aggregate import onset_summary, reduce_values
from ..law_fitting import crossing, fit, fold, branches as _branches
from .paths import ROOT, CENTRAL, C_OLD

EXPECTED = set(range(42, 52))


def branches(p, law):
    """Roots for the registered laws, whose original capacity scale is 2.16."""
    return _branches(p, law, C_OLD)


def extract():
    view=ConfigView.from_yaml(ROOT/'configs/central.yaml',db_path=str(ROOT/'runs.db'))
    group=_pick_group(view);figure=view.intersection_figures[0];C=float(group.capacity_constant)
    con=sqlite3.connect((ROOT/'runs.db').as_uri()+'?mode=ro',uri=True)
    hardware={u:(h,g) for u,h,g in con.execute('select uuid,host,gpu_type from runs')};con.close()
    out={};details=[]
    for kind,rows in [('mem',group.speed_runs),('gen',group.groks_runs)]:
        cells=defaultdict(list)
        for r in rows:
            if r['p'] not in CENTRAL_PRIMES or not _passes_filters(r,figure):continue
            bits=(r.get('dataset_bits') or r['n_samples']*math.log2(r['p']+2)) if kind=='mem' else compute_n_equiv(r['p'],group.key.operation,float(group.key.train_fraction))[1]
            f=bits/(C*r['param_count'])
            if not .08<=f<=1.:continue
            host,gpu=hardware[r['uuid']]
            assert host=='147-224-250-73' and gpu=='A100'
            event=r.get('saturation_epoch' if kind=='mem' else 'grokking_epoch')
            observed=event is not None and r.get('saturated') is not False
            cells[r['p'],r['param_count']].append({'uuid':r['uuid'],'kind':kind,'p':r['p'],
                'dim':r['dim'],'P':r['param_count'],'seed':r['seed'],'host':host,'gpu_type':gpu,
                'f_original':f,'event_epoch':event,'observed':observed})
        current=[];complete=[];summaries=[]
        for (p,P),rr in sorted(cells.items()):
            assert len(rr)==10 and {r['seed'] for r in rr}==EXPECTED
            observed=[r for r in rr if r['observed']];all_ten=len(observed)==10
            summaries.append({'p':p,'dim':rr[0]['dim'],'P':P,'observed_seeds':len(observed),
                              'observed_seed_ids':[r['seed'] for r in observed],'included_complete_ten':all_ten})
            for r in rr:details.append({**r,'included_current_fit':r['observed'],'included_complete_ten_fit':all_ten})
            if kind=='mem':points=[{'p':p,'P':P,'f':r['f_original'],'t':r['event_epoch']} for r in observed]
            else:points=[] if not observed else [{'p':p,'P':P,'f':rr[0]['f_original'],'t':float(np.mean([r['event_epoch'] for r in observed]))}]
            current.extend(points)
            if all_ten:complete.extend(points)
        out[kind]={'current':current,'complete':complete,'cells':summaries}
    return C,out,details

def fit_clock_pair(data, metric, cohort):
    """Same reducer, selection and cell weights for both clocks; no cap values."""
    points = {}
    for kind in ('mem', 'gen'):
        chosen = []
        for (p, P), group in data[data.kind == kind].groupby(['p', 'P']):
            assert len(group) == 10 and set(group.seed) == set(range(42, 52))
            complete = bool(group.observed.all())
            assert bool(group.included_complete_ten_fit.all()) == complete
            if cohort == 'complete' and not complete:
                continue
            observed = group[group.observed]
            if observed.empty:
                continue
            chosen.append({'p': int(p), 'P': int(P),
                           'f': float(group.f_original.iloc[0]),
                           't': reduce_values(observed.event_epoch, metric),
                           'n_observed': len(observed)})
        points[kind] = chosen
    mem, gen = fit('mem', points['mem']), fit('gen', points['gen'])
    return {'cohort': cohort, 'metric': metric, 'mem': mem, 'gen': gen,
            'fold': fold(mem, gen, C_OLD),
            'f_original_at_fold': gen['b'] / mem['gamma_original']}, points

def get_onsets(cells, limit=256):
    result = {}
    for p in sorted({c['p'] for c in cells}):
        curve = {c['param_count']: c['aggregate_delay'] for c in cells
                 if c['p'] == p and (limit is None or c['dim'] <= limit)}
        result[p] = onset_summary(curve)
    return result

def agreement(law, onsets):
    observed = np.array([onsets[p]['value'] for p in CENTRAL], dtype=float)
    predicted = np.array([crossing(p, law['mem'], law['gen'], C_OLD) for p in CENTRAL])
    assert np.isfinite(observed).all() and np.isfinite(predicted).all()
    x, y = np.log10(predicted), np.log10(observed)
    r = y - x
    delta = float(np.median(r))
    covariance = np.mean((x - x.mean()) * (y - y.mean()))
    ccc = 2 * covariance / (x.var() + y.var() + (x.mean() - y.mean()) ** 2)
    return {'n': len(CENTRAL), 'spearman_rho': float(spearmanr(x, y).statistic),
            'raw_CCC': float(ccc), 'raw_MAE_dex': float(np.mean(abs(r))),
            'calibration_offset_dex': delta,
            'in_sample_calibrated_MAE_dex': float(np.mean(abs(r - delta))),
            'offset_sd_dex': float(np.std(r, ddof=1))}

def verify_curve_domains(fig):
    fitted = extrapolated = 0
    for ax in fig.axes:
        assert ax.get_title() == ''
        for line in ax.lines:
            if line.get_gid() == 'fitted_prime_curve':
                assert max(line.get_xdata()) <= max(CENTRAL)
                fitted += 1
            elif line.get_gid() == 'extrapolated_prime_curve':
                assert min(line.get_xdata()) == max(CENTRAL)
                assert line.get_linestyle() == ':'
                extrapolated += 1
    assert fitted and extrapolated
