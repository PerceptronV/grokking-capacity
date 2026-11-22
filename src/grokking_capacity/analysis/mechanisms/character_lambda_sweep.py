#!/usr/bin/env python3
"""λ-sweep analysis of character traces (todo item 7, 2026-07-29).

Reads results/theory/character_growth_abs_wd{0.1,0.3,1}.npz (produced by
src/grokking_capacity/analysis/mechanisms/character_growth.py --wd ... with absolute spectral power) and
tests the two-stage mechanism's λ-predictions:

  1. takeoff timing of the winning character cohort is λ-independent
     (selection is init/task-structure-driven, not decay-driven);
  2. loser pruning: rate of loser decay vs λ — in SHARE space (power
     fractions, renormalised) and in AMPLITUDE space (absolute spectral
     power, the quantity weight decay acts on). The naive free-decay
     prediction is amplitude rate ∝ λ.

Windows are grok-aligned (val-99 epoch + [0, 300] etc.) so different-λ
runs are compared in the same dynamical phase.

Output: results/theory/character_growth_lambda_sweep.json
"""
import json
from pathlib import Path

import numpy as np

from grokking_capacity.analysis.characters import CharacterTrace, log_slopes, sweep_options


def slopes(ep, mat, idx, mask):
    return log_slopes(ep,mat,idx,mask,min_points=5)


def analyse(fn, windows=((0,300),(300,800),(800,2000))):
    trace = CharacterTrace.load(fn)
    ep, pw, share = trace.epochs, trace.power, trace.share
    winners, losers = trace.cohorts()
    grok = trace.first_epoch(trace.validation,.99)
    gi = np.flatnonzero(ep == grok)[0] if grok is not None else None
    res = {
        "grok_epoch": grok,
        "train_sat_epoch": trace.first_epoch(trace.train,.99),
        "takeoff_1p5x": trace.takeoff(winners,1.5),
        "takeoff_2x": trace.takeoff(winners,2.),
        "n_winners": int(len(winners)),
        "effn": {"init": float(1 / (share[0] ** 2).sum()),
                 "grok": float(1 / (share[gi] ** 2).sum()) if gi is not None else None,
                 "final": float(1 / (share[-1] ** 2).sum())},
        "windows": {},
    }
    if grok is None:
        return res
    for w0, w1 in windows:
        m = (ep >= grok + w0) & (ep <= grok + w1)
        if m.sum() < 5:
            continue
        res["windows"][f"grok+[{w0},{w1}]"] = {
            "share": {"winner": slopes(ep, share, winners, m),
                      "loser": slopes(ep, share, losers, m)},
            "amplitude": {"winner": slopes(ep, pw, winners, m),
                          "loser": slopes(ep, pw, losers, m),
                          "total": float(np.polyfit(
                              ep[m], np.log(pw[m].sum(1)), 1)[0])},
        }
    return res


def main(argv=None):
    inputs,ofn,config=sweep_options("weight_decay",argv)
    out = {}
    for wd, name in config["files"].items():
        fn = inputs / name
        if not fn.exists():
            print(f"λ={wd}: {name} missing, skipped")
            continue
        out[str(wd)] = analyse(fn,config["windows"])
        r = out[str(wd)]
        w = r["windows"].get("grok+[0,300]", {})
        print(f"λ={wd}: grok={r['grok_epoch']} "
              f"takeoff(1.5x/2x)={r['takeoff_1p5x']}/{r['takeoff_2x']} "
              f"winners={r['n_winners']} "
              f"effn {r['effn']['init']:.0f}->{r['effn']['grok']}"
              f"->{r['effn']['final']:.0f}")
        if w and all(w[space][kind] is not None for space in ("share","amplitude") for kind in ("winner","loser")):
            print(f"   grok+[0,300]/ep: share w {w['share']['winner']:+.2e} "
                  f"l {w['share']['loser']:+.2e} | amp w "
                  f"{w['amplitude']['winner']:+.2e} "
                  f"l {w['amplitude']['loser']:+.2e} "
                  f"tot {w['amplitude']['total']:+.2e}")

    # Fit only observed positive loser rates; absent events stay unresolved.
    for space in ('share','amplitude'):
        pairs = sorted((float(k),abs(v['windows']['grok+[0,300]'][space]['loser']))
                       for k,v in out.items() if isinstance(v,dict)
                       and 'grok+[0,300]' in v['windows']
                       and v['windows']['grok+[0,300]'][space]['loser'] not in (None,0))
        if len(pairs)>=2:
            x,y=zip(*pairs)
            out[f'loser_{space}_rate_vs_lambda_exponent']=float(np.polyfit(np.log(x),np.log(y),1)[0])
    ofn.write_text(json.dumps(out,indent=1,allow_nan=False)+'\n')
    print("saved", ofn)


if __name__ == "__main__":
    main()
