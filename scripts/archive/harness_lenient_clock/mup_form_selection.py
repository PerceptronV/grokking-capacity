#!/usr/bin/env python3
"""muP functional-form selection (recorded 2026-09-21; candidates were fixed
before fitting, see mup-apex-locate.md 'New functional forms').
  T_mem(f):   M1 b e^{g f} | M2 T0 + b e^{g f} | M3 b (1+f/f0)^k
  T_gen(p,P): G1 A p^a P^-b | G2 A p^a P^-b + B p^c P^+w | G3 A p^a P^-b + B p^c
Fits on the muP corpus *excluding* the farApex rows (so the far-apex test
stays out of sample), then prints the predicted window (T_gen - T_mem) at
the registered far-apex cells p in {449, 907}, d in {880, 1400, 2000}.
"""
import json, math, sys
import numpy as np
from scipy.optimize import least_squares

rows = [json.loads(l) for l in open('results/speedrun/runs.jsonl')]
C = 2.16
K = lambda p: 0.5*p*(p-1)*math.log2(p+2)
Pc = lambda p, d: 32*d*d + 2*(p+2)*d
MTAGS = {'mupM', 'mupLawM'}
GTAGS = {'mupG', 'mupLawG', 'mupPair197', 'mupPair223', 'reentrance', 'openTop',
         'openTopX', 'apexSearch', 'apexLocate'}
M = [(K(r['p'])/(C*Pc(r['p'], r['dim'])), r['t_train']) for r in rows if r.get('tag') in MTAGS and r['t_train']]
G = [(r['p'], float(Pc(r['p'], r['dim'])), r['t_gen']) for r in rows
     if r.get('tag') in GTAGS and r.get('mup_base') and r['t_gen']]
f = np.array([x for x, _ in M]); tm = np.array([t for _, t in M], float)
p_ = np.array([a for a, _, _ in G], float); P_ = np.array([b for _, b, _ in G]); tg = np.array([c for _, _, c in G], float)
print('n_mem=%d (f %.3f-%.2f), n_gen=%d (p %d-%d, P %.0e-%.0e)' % (len(f), f.min(), f.max(), len(tg), p_.min(), p_.max(), P_.min(), P_.max()))

def aic(res, n, k): return n*math.log(max(res, 1e-12)/n) + 2*k
def fit(fun, x0, n):
    r = least_squares(fun, x0, max_nfev=40000)
    return r.x, aic(float(2*r.cost), n, len(x0))

n = len(f); ln = np.log(tm)
x1, a1 = fit(lambda th: th[0] + th[1]*f - ln, [1, 20], n)
x2, a2 = fit(lambda th: np.log(np.exp(th[0]) + np.exp(th[1])*np.exp(th[2]*f)) - ln, [2, -1, 25], n)
x3, a3 = fit(lambda th: np.log(np.exp(th[0])*(1+f/math.exp(th[1]))**th[2]) - ln, [1, -3, 3], n)
print('T_mem: M1 exp AIC %.0f (g=%.1f) | M2 floor+exp AIC %.0f (T0=%.1f, g=%.1f) | M3 power AIC %.0f (k=%.1f)'
      % (a1, x1[1], a2, math.exp(x2[0]), x2[2], a3, x3[2]))

n = len(tg); ln = np.log(tg); lp, lP = np.log(p_), np.log(P_)
g1, b1 = fit(lambda th: th[0] + th[1]*lp - th[2]*lP - ln, [20, -0.5, 1.1], n)
g2, b2 = fit(lambda th: np.log(np.exp(th[0]+th[1]*lp-th[2]*lP) + np.exp(th[3]+th[4]*lp+th[5]*lP)) - ln,
             [20, 0.9, 1.0, -10, 0.5, 0.4], n)
g3, b3 = fit(lambda th: np.log(np.exp(th[0]+th[1]*lp-th[2]*lP) + np.exp(th[3]+th[4]*lp)) - ln,
             [20, 0.9, 1.0, 0, 0.5], n)
print('T_gen: G1 power AIC %.0f | G2 +wall AIC %.0f (a=%.2f beta=%.2f wall: p^%.2f P^+%.2f) | G3 +floor AIC %.0f (floor ~ p^%.2f)'
      % (b1, b2, g2[1], g2[2], g2[4], g2[5], b3, g3[4]))

memf = lambda ff: math.exp(x2[0]) + math.exp(x2[1])*math.exp(x2[2]*ff)
genf = lambda pp, PP: math.exp(g2[0]+g2[1]*math.log(pp)-g2[2]*math.log(PP)) + math.exp(g2[3]+g2[4]*math.log(pp)+g2[5]*math.log(PP))

print('\nmax over P of (T_gen - T_mem):')
for pp in (197, 251, 311, 383, 449, 600, 907):
    Ps = np.logspace(5.2, 9.5, 400)
    h = [genf(pp, PP) - memf(K(pp)/(C*PP)) for PP in Ps]; i = int(np.argmax(h))
    print('  p=%4d: max gap %+7.1f ep at P=%.1e' % (pp, h[i], Ps[i]))

print('\npredicted at the registered far-apex cells (M2 x G2 winners):')
print('   p     d        P      T_mem   T_gen   T_gen-T_mem')
for pp in (449, 907):
    for d in (880, 1400, 2000):
        PP = Pc(pp, d); a = memf(K(pp)/(C*PP)); b = genf(pp, PP)
        print('  %3d  %4d  %.2e  %7.1f %7.1f   %+7.1f' % (pp, d, PP, a, b, b - a))

if len(sys.argv) > 1 and sys.argv[1] == '--measured':
    from collections import defaultdict
    cells = defaultdict(list)
    for r in rows:
        if r.get('tag') == 'farApex':
            cells[(r['p'], r['dim'])].append(r)
    print('\nmeasured farApex cells: p d n  mean t_train  mean t_gen  mean delay (se)  frac grokked')
    for k in sorted(cells):
        g = cells[k]; dl = [(r['t_gen'] or 1500) - (r['t_train'] or 1500) for r in g]
        print('  %3d %4d %2d  %6.1f  %6.1f  %+5.2f (%.2f)  %.2f' % (
            k[0], k[1], len(g), np.mean([r['t_train'] or 1500 for r in g]), np.mean([r['t_gen'] or 1500 for r in g]),
            np.mean(dl), np.std(dl, ddof=1)/math.sqrt(len(dl)) if len(dl) > 1 else float('nan'),
            np.mean([r['t_gen'] is not None for r in g])))
