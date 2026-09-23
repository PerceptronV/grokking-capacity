#!/usr/bin/env python3
"""Crossing programme (registered 2026-09-23): per prime, ten log-spaced
widths straddling the coarse crossing of T_mem and T_gen seen in the
registry race maps; gc-speed and gc-groks at every width, five seeds,
cap 2500. Writes configs/crossing_fixed.yaml and configs/crossing_mup.yaml
and prints the registration table: for the fixed-rate protocol, the
law-derived crossing per prime (onset root) from the calibration-window
constants in results/theory/forecast_onset.json; for muP, the grid only
(no muP law is fitted on the canonical clock).
"""
import json, math, yaml, numpy as np
from scipy.optimize import brentq
from grokking_capacity.models import TransformerTorch

R = json.load(open("results/theory/forecast_onset.json"))
g, lnb = R["mem_law"]["gamma"], R["mem_law"]["ln_b"]
lnA, a, beta = R["gen_law"]["lnA"], R["gen_law"]["a"], R["gen_law"]["beta"]
C = 2.16
K = lambda p: (p * (p - 1) // 2) * math.log2(p + 2)
_pc = {}
def P(p, d):
    if (p, d) not in _pc:
        m = TransformerTorch(depth=2, dim=d, heads=1, n_tokens=p + 2, seq_len=4, dropout=0.2, init_scale=1.0)
        _pc[(p, d)] = sum(x.numel() for x in m.parameters())
    return _pc[(p, d)]
def law_gap(p, d):  # ln T_mem - ln T_gen at width d
    PP = P(p, d); return (lnb + g * K(p) / (C * PP)) - (lnA + a * math.log(p) - beta * math.log(PP))
def law_crossing(p):
    ds = np.arange(40, 1001, 4); gaps = [law_gap(p, int(d)) for d in ds]
    for i in range(len(ds) - 1):
        if gaps[i] > 0 and gaps[i + 1] <= 0:
            return brentq(lambda d: law_gap(p, int(round(d))), ds[i], ds[i + 1])
    return None

# coarse measured crossings from the registry race maps (2026-09-22), interpolated in log d
fixed_centre = {149: 160, 151: 196, 157: 196, 163: 196, 167: 250, 173: 304, 179: 330, 181: 350, 191: 476, 193: 520, 197: 600}
mup_centre = {97: 92, 113: 105, 139: 145, 151: 200, 197: 300, 223: 360, 251: 480, 281: 600, 311: 700}
def grid(c, lo=64, hi=1000):
    ds = np.geomspace(max(lo, c / 2.2), min(hi, c * 2.2), 10)
    return sorted({int(4 * round(d / 4)) for d in ds})

def config(name, centres, mup_base, desc):
    defaults = dict(operation="/", train_fraction=0.5, split_type="random", depth=2, heads=1, weight_decay=1.0, dropout=0.2,
                    lr=0.001, beta1=0.9, beta2=0.98, batch_size=512, max_epochs=2500, mup_base=mup_base, seeds=[42, 43, 44, 45, 46])
    exps = {}
    for p, c in centres.items():
        exps[f"mem_p{p}"] = dict(type="speed", primes=[p], dims=grid(c), n_samples="auto")
        exps[f"gen_p{p}"] = dict(type="groks", primes=[p], dims=grid(c))
    yaml.safe_dump(dict(name=name, description=desc, defaults=defaults, experiments=exps,
                        analysis=dict(use_default_capacity_constant=True, intersection_figures=[])),
                   open(f"configs/{name}.yaml", "w"), sort_keys=False)
    return exps

fx = config("crossing_fixed", fixed_centre, 0,
            "Crossing programme, fixed rate (registered 2026-09-23): ten widths straddling the coarse crossing per prime, "
            "gc-speed and gc-groks at every width, five seeds, cap 2500. See theory/registrations.md.")
mu = config("crossing_mup", mup_centre, 236,
            "Crossing programme, per-layer scaling base 236 (registered 2026-09-23): as crossing_fixed, muP primes 97..311.")
print("| p | grid widths | law crossing d | law crossing log10 P |")
print("|---|---|---|---|")
for p, c in fixed_centre.items():
    dc = law_crossing(p)
    print(f"| {p} | {grid(c)[0]} to {grid(c)[-1]} | {'none in [40, 1000]' if dc is None else round(dc)} | {'' if dc is None else round(math.log10(P(p, int(round(dc)))), 2)} |")
print("\nmuP grids:", {p: (grid(c)[0], grid(c)[-1]) for p, c in mup_centre.items()})
n_fx = sum(len(e['dims']) for e in fx.values()) * 5; n_mu = sum(len(e['dims']) for e in mu.values()) * 5
print(f"runs: fixed {n_fx}, muP {n_mu}")
