"""Log-space timescale fits and intersections of exponential and power laws.

``gamma_original`` is measured in the capacity convention passed to the solver.
These routines do not select runs or write manuscript artifacts.
"""
import math
import numpy as np
from scipy.optimize import brentq
from .numerics import ols as _ols


def K(p):return p*(p-1)/2*math.log2(p+2)

def fit(kind,points,prime_sample=None):
    if prime_sample is not None:points=[r for p in prime_sample for r in points if r['p']==p]
    if kind=='mem':X=np.array([[1,r['f']] for r in points]);keys=['ln_B','gamma_original']
    else:X=np.array([[1,math.log(r['p']),-math.log(r['P'])] for r in points]);keys=['lnA','a','b']
    assert np.linalg.matrix_rank(X)==len(keys)
    coeff,r2,rss=_ols(X,np.log([r['t'] for r in points]))
    return {**dict(zip(keys,map(float,coeff))),'r2':float(r2),'n_observations':len(points)}

def crossing(p,mem,gen,C):
    b=gen['b'];q=mem['gamma_original']/C
    rhs=gen['lnA']-mem['ln_B']+gen['a']*math.log(p)-b*math.log(q*K(p))
    if b<=0 or q<=0 or rhs<b*(1-math.log(b)):return None
    hi=max(100.,b*2)
    while hi-b*math.log(hi)<rhs:hi*=2
    x=brentq(lambda x:x-b*math.log(x)-rhs,b,hi)
    return q*K(p)/x

def fold(mem,gen,C):
    b=gen['b'];q=mem['gamma_original']/C
    if b<=0 or q<=0:return None
    margin=lambda p:gen['lnA']-mem['ln_B']+gen['a']*math.log(p)-b*math.log(q*K(p))-b*(1-math.log(b))
    if margin(2)*margin(10000)>=0:return None
    return float(brentq(margin,2,10000))

def branches(p, law, capacity_constant):
    """Solve both roots in log-parameter space, including a double root at fold."""
    m, g = law['mem'], law['gen']
    qk = m['gamma_original'] / capacity_constant * K(p)
    b = g['b']
    if b <= 0 or qk <= 0:
        return None, None
    pivot = math.log(qk / b)
    def gap(logP):
        return g['lnA'] + g['a'] * math.log(p) - b * logP - m['ln_B'] - qk * math.exp(-logP)
    maximum = gap(pivot)
    if maximum < -1e-10:
        return None, None
    if abs(maximum) <= 1e-10:
        return math.exp(pivot), math.exp(pivot)
    lower = math.exp(brentq(gap, pivot - 50, pivot, xtol=1e-13))
    upper = math.exp(brentq(gap, pivot, pivot + 500, xtol=1e-13))
    return lower, upper
