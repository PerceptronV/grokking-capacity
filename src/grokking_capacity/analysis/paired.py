"""Censored time summaries and paired comparisons, independent of a campaign."""
import math
import numpy as np


def summarize(values,limit):
    observed=[v for v in values if v is not None]
    capped=[limit if v is None else v for v in values]
    return dict(n=len(values),observed=len(observed),censored=len(values)-len(observed),
        mean=float(np.mean(capped)) if len(observed)==len(values) else None,
        capped_mean=float(np.mean(capped)),mean_lower_bound=float(np.mean(capped)),
        median=float(np.median(capped)) if len(observed)>len(values)//2 else None,
        median_lower_bound=float(np.median(capped)),min_observed=min(observed) if observed else None)


def crossing_brackets(sizes,differences,complete):
    result=[]
    for i in range(1,len(sizes)):
        a,b=differences[i-1:i+1]
        if complete[i-1] and complete[i] and a*b<0:
            fraction=-a/(b-a)
            result.append(dict(lower_n=int(sizes[i-1]),upper_n=int(sizes[i]),
                log_interpolated_n=float(np.exp(np.log(sizes[i-1])+fraction*np.log(sizes[i]/sizes[i-1]))),
                direction='C faster to B faster' if a>0 else 'B faster to C faster'))
    return result


def analyze(rows, sizes, seeds, limit, *, label_count, resamples=10000, bootstrap_seed=9002):
    """Paired seed bootstrap across dataset sizes, retaining censored runs.

    Both roles must have exactly one observation for every size/seed cell.
    Resampling a seed keeps its entire trajectory and both roles together.
    """
    if label_count < 2 or not np.isfinite(limit) or limit <= 0 or resamples < 1:
        raise ValueError('Require at least two labels and positive limit/resample count')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Require distinct seeds')
    if not sizes or any(n <= 0 for n in sizes) or any(a >= b for a,b in zip(sizes,sizes[1:])):
        raise ValueError('Require positive increasing dataset sizes')
    expected = {(role, seed, n) for role in ('C','B') for seed in seeds for n in sizes}
    indexed = {}
    for row in rows:
        key = row['role'], row['seed'], row['n_train']
        if key in indexed:
            raise ValueError(f'Duplicate observation: {key}')
        value = row['t_gen_updates']
        if value is not None and (not np.isfinite(value) or not 0 < value <= limit):
            raise ValueError(f'Observed time outside the training budget: {key}')
        indexed[key] = row
    if set(indexed) != expected:
        raise ValueError('Rows must cover exactly the requested roles, seeds, and sizes')
    matrix = {role: np.array([[limit if indexed[role,seed,n]['t_gen_updates'] is None
                              else indexed[role,seed,n]['t_gen_updates']
                              for n in sizes] for seed in seeds], float) for role in ('C','B')}
    rng=np.random.default_rng(bootstrap_seed)
    draws=rng.integers(0,len(seeds),size=(resamples,len(seeds)))
    diff=matrix['B']-matrix['C']
    estimate=diff.mean(axis=0)
    bootstrap=diff[draws].mean(axis=1)
    interval=np.percentile(bootstrap,[2.5,97.5],axis=0).T
    radius=float(np.percentile(np.max(abs(bootstrap-estimate),axis=1),95))
    boots={r:matrix[r][draws].mean(axis=1) for r in ['C','B']}
    ratio=matrix['B'].mean(axis=0)/matrix['C'].mean(axis=0)
    ratio_ci=np.percentile(boots['B']/boots['C'],[2.5,97.5],axis=0).T
    complete=[];table=[]
    for i,n in enumerate(sizes):
        row=dict(n_train=n,dataset_bits=float(n*math.log2(label_count)))
        for role in ['C','B']:
            rr=[indexed[role,seed,n] for seed in seeds]
            summary=summarize([r['t_gen_updates'] for r in rr],limit)
            row.update({role+'_'+k:v for k,v in summary.items()})
            row[role+'_final_learning']=sum(r['final_learning'] for r in rr)
            row[role+'_joint_stop']=sum(r['joint_stop'] for r in rr)
            row[role+'_positive_paper_delay']=sum(r['paper_delay_updates'] is not None and r['paper_delay_updates']>0 for r in rr)
            row[role+'_undefined_paper_delay']=sum(r['paper_delay_updates'] is None for r in rr)
            delays=[r['paper_delay_updates'] for r in rr]
            row[role+'_min_paper_delay']=min(delays) if all(v is not None for v in delays) else None
            row[role+'_capped_mean_ci_low'],row[role+'_capped_mean_ci_high']=np.percentile(boots[role][:,i],[2.5,97.5]).tolist()
        complete.append(row['C_censored']==row['B_censored']==0)
        row.update(mean_difference=float(estimate[i]) if complete[-1] else None,
            capped_mean_difference=float(estimate[i]),difference_ci_low=float(interval[i,0]),difference_ci_high=float(interval[i,1]),
            simultaneous_low=float(estimate[i]-radius),simultaneous_high=float(estimate[i]+radius),
            mean_ratio=float(ratio[i]) if complete[-1] else None,capped_mean_ratio=float(ratio[i]),
            ratio_ci_low=float(ratio_ci[i,0]),ratio_ci_high=float(ratio_ci[i,1]))
        table.append(row)
    observed=crossing_brackets(sizes,estimate,complete)
    capped=crossing_brackets(sizes,estimate,[True]*len(sizes))
    positive=[sizes[i] for i in range(len(sizes)) if complete[i] and estimate[i]-radius>0]
    negative=[sizes[i] for i in range(len(sizes)) if complete[i] and estimate[i]+radius<0]
    return dict(sizes=table,observed_mean_crossing_brackets=observed,capped_mean_crossing_brackets=capped,
        resolved_crossing=bool(positive and negative),simultaneously_C_faster_sizes=positive,
        simultaneously_B_faster_sizes=negative,simultaneous_band_radius=radius,
        confidence=f'Paired bootstrap of {len(seeds)} seeds, {resamples} draws, seed {bootstrap_seed}. Pointwise 95% intervals; centered maximum absolute deviation across {len(sizes)} capped contrasts gives simultaneous 95% bands. Conditional on fixed architectures and calibrated constants.',
        interpretation='Capped curves are not uncensored mean generalisation times where failures occur. Crossing interpolation only describes adjacent fully observed points, not a fitted physical threshold.')
