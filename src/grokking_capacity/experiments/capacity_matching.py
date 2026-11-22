"""Capacity calibration and matching for the registered depth-two model families."""
import numpy as np


def model_spec(role,width):
    if width<2 or width%2:raise ValueError('Use positive even widths for rotary attention')
    if role not in ['B','C']:raise ValueError(role)
    result=dict(model='baseline' if role=='C' else 'categorical',dim=width,mlp_dim=4*width)
    if role=='B':result.update(alphabet=8,categorical_temperature=1.,
        categorical_normalization='slot_layernorm',categorical_training_sampling=False)
    return result


def parameter_count(role,width,p=97):
    # Depth2, heads1, FF4d: 32d^2 continuous, plus32d^2 categorical; embeddings/norms=(2(p+2)+5)d.
    return (32 if role=='C' else 64)*width**2+(2*(p+2)+5)*width


def fit_constant(points):
    if len({r['parameters'] for r in points})<2:raise ValueError('Need at least two architecture sizes')
    x=np.array([r['parameters'] for r in points],float);y=np.array([r['capacity_bits'] for r in points],float)
    if np.any(y<=0):raise ValueError('Capacity must be positive')
    k=float(x@y/(x@x));a,b=np.polyfit(x,y,1);pred=k*x
    return dict(bits_per_parameter=k,relative_residuals=(y/pred-1).tolist(),
        max_absolute_relative_residual=float(np.max(np.abs(y/pred-1))),
        affine_diagnostic=dict(slope=float(a),intercept=float(b)),
        parameter_range=[int(x.min()),int(x.max())],capacity_range=[float(y.min()),float(y.max())],
        per_seed_constants={str(seed):float(sum(r['parameters']*r['capacity_bits'] for r in points if r['seed']==seed)/
            sum(r['parameters']**2 for r in points if r['seed']==seed)) for seed in sorted({r['seed'] for r in points})},points=points)


def match_pairs(constants,widths,candidates,tolerance=.05):
    pairs=[]
    for index,cw in enumerate(widths):
        cp=parameter_count('C',cw);target=constants['C']*cp
        bw=min(candidates,key=lambda w:(abs(constants['B']*parameter_count('B',w)/target-1),w))
        bp=parameter_count('B',bw);error=constants['B']*bp/target-1
        if abs(error)>tolerance:raise ValueError('Cannot match predicted capacity within tolerance')
        pairs.append(dict(index=index,C=model_spec('C',cw),B=model_spec('B',bw),C_parameters=cp,B_parameters=bp,
            C_estimated_bits=target,B_estimated_bits=constants['B']*bp,predicted_relative_mismatch=error))
    if any(a['B']['dim']>=b['B']['dim'] for a,b in zip(pairs,pairs[1:])):raise ValueError('Require distinct increasing B sizes')
    return pairs
