#!/usr/bin/env python3
"""Winner-rate vs α and vs width — the drive-vs-SNR discriminator
(principled/gen-snr-escape proposal; also attacks β at the rate level for
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

REPO = Path(__file__).resolve().parents[1]
RES = REPO / "results/theory"

ALPHA_ARM = {0.3: "character_growth_a0.3.npz",
             0.5: "character_growth_abs_wd1.npz",
             0.7: "character_growth_a0.7.npz"}
WIDTH_ARM = {64: "character_growth_d64.npz",
             128: "character_growth_abs_wd1.npz",
             256: "character_growth_d256.npz",
             512: "character_growth_d512.npz"}


def slopes(ep, mat, idx, mask):
    if mask.sum() < 4:
        return None
    return float(np.mean([
        np.polyfit(ep[mask], np.log(mat[mask, i] + 1e-300), 1)[0]
        for i in idx]))


def analyse(fn):
    d = np.load(fn)
    ep = d["epochs"]
    pw = d["out_pow"]
    share = pw / pw.sum(1, keepdims=True)
    tr, va = d["train_acc"], d["val_acc"]
    n = pw.shape[1]
    final = share[-1]
    winners = np.where(final > 2 / n)[0]
    losers = np.where(final <= 2 / n)[0]
    if not (va >= 0.99).any():
        return {"grok": None, "note": "no grok within budget",
                "final_val": float(va[-1]), "n_winners": int(len(winners))}
    gi = int(np.argmax(va >= 0.99))
    grok = int(ep[gi])
    wshare = share[:, winners].mean(1) * n
    take = int(ep[np.argmax(wshare > 1.5)])
    esc = (ep >= take) & (ep <= grok)
    post = (ep >= grok) & (ep <= grok + 300)
    return {
        "grok": grok, "takeoff_1p5x": take,
        "train_sat": int(ep[np.argmax(tr >= 0.99)]) if (tr >= .99).any() else None,
        "n_winners": int(len(winners)),
        "winner_amp_rate_escape": slopes(ep, pw, winners, esc),
        "winner_amp_rate_post": slopes(ep, pw, winners, post),
        "loser_share_rate_post": slopes(ep, share, losers, post),
        "effn_final": float(1 / (final ** 2).sum()),
    }


def powerfit(pairs, key):
    pts = [(x, r[key]) for x, r in pairs
           if r.get(key) is not None and r[key] > 0]
    if len(pts) < 2:
        return None
    return float(np.polyfit(np.log([x for x, _ in pts]),
                            np.log([y for _, y in pts]), 1)[0])


def main():
    out = {"alpha_arm": {}, "width_arm": {}}
    for arm, files, xlab in (("alpha_arm", ALPHA_ARM, "alpha"),
                             ("width_arm", WIDTH_ARM, "dim")):
        pairs = []
        for x, name in sorted(files.items()):
            fn = RES / name
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
    ofn = RES / "character_alpha_width.json"
    ofn.write_text(json.dumps(out, indent=1))
    print("saved", ofn)


if __name__ == "__main__":
    main()
