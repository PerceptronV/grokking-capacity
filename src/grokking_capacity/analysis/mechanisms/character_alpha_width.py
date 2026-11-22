#!/usr/bin/env python3
"""Winner-rate vs α and vs width — the drive-vs-SNR discriminator
(principled/gen-snr-escape proposal; also attacks b at the rate level for
gen-solvable-circuit).

Discriminator: the drive-side channel predicts winner amplitude growth
rate ∝ α^{~3.7} and strongly width-dependent; pure SNR predicts α^{0.5–1}
and ~width-free.

Reads the character_growth trace family at p=113, λ=1:
  α arm:  character_growth_a0.3.npz / (α=0.5: character_growth_abs_wd1.npz)
          / character_growth_a0.7.npz            (d=128)
  d arm:  character_growth_d64.npz / (d=128 as above) /
          character_growth_d256.npz / character_growth_d512.npz  (α=0.5)

For each trace: grok epoch (val≥0.99), takeoff epochs, winner cohort
(final share > 2/m), winner amplitude log-slope on [takeoff, grok] (the
escape phase) and on grok+[0,300] (early cleanup), loser share slope.
Then power-law exponents of the winner rate in α and in width.

Output: results/theory/character_alpha_width.json
"""
import json
import math
from pathlib import Path

import numpy as np

from grokking_capacity.analysis.characters import CharacterTrace, log_slopes, sweep_options


def slopes(ep, mat, idx, mask):
    return log_slopes(ep,mat,idx,mask,min_points=4)


def analyse(fn):
    trace = CharacterTrace.load(fn)
    ep,pw,share=trace.epochs,trace.power,trace.share
    winners,losers=trace.cohorts()
    grok=trace.first_epoch(trace.validation,.99)
    if grok is None:
        return {"grok": None, "note": "no grok within budget",
                "final_val": float(trace.validation[-1]), "n_winners": int(len(winners))}
    take=trace.takeoff(winners,1.5)
    esc=trace.window(take,grok)
    post=trace.window(grok,grok+300)
    return {
        "grok": grok, "takeoff_1p5x": take,
        "train_sat": trace.first_epoch(trace.train,.99),
        "n_winners": int(len(winners)),
        "winner_amp_rate_escape": slopes(ep, pw, winners, esc),
        "winner_amp_rate_post": slopes(ep, pw, winners, post),
        "loser_share_rate_post": slopes(ep, share, losers, post),
        "effn_final": float(1 / (share[-1] ** 2).sum()),
    }


def powerfit(pairs, key):
    pts = [(x, r[key]) for x, r in pairs
           if r.get(key) is not None and r[key] > 0]
    if len(pts) < 2:
        return None
    return float(np.polyfit(np.log([x for x, _ in pts]),
                            np.log([y for _, y in pts]), 1)[0])


def main(argv=None):
    inputs,ofn,config=sweep_options("alpha_width",argv)
    out = {arm:{} for arm in config["arms"]}
    for arm,files in config["arms"].items():
        xlab=arm.removesuffix("_arm")
        pairs = []
        for x, name in sorted(files.items()):
            fn = inputs / name
            if not fn.exists():
                print(f"[{arm}] {name} missing, skipped")
                continue
            r = analyse(fn)
            out[arm][str(x)] = r
            if r.get("grok") is not None:
                pairs.append((x, r))
            print(f"[{arm}] {xlab}={x}: grok={r.get('grok')} "
                  f"take={r.get('takeoff_1p5x')} "
                  f"esc-rate={r.get('winner_amp_rate_escape')} "
                  f"post-rate={r.get('winner_amp_rate_post')}")
        for key in ("winner_amp_rate_escape", "winner_amp_rate_post"):
            e = powerfit(pairs, key)
            if e is not None:
                out[arm][f"exponent_{key}"] = e
                print(f"[{arm}] {key} ∝ {xlab}^{e:+.2f}")
        tg = powerfit(pairs, "grok") if pairs else None
        if tg is not None:
            out[arm]["exponent_grok_epoch"] = tg
            print(f"[{arm}] grok epoch ∝ {xlab}^{tg:+.2f}")
    ofn.write_text(json.dumps(out, indent=1, allow_nan=False))
    print("saved", ofn)


if __name__ == "__main__":
    main()
