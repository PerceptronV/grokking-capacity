#!/usr/bin/env python3
"""λ-sweep analysis of character traces (todo item 7, 2026-07-29).

Reads results/theory/character_growth_abs_wd{0.1,0.3,1}.npz (produced by
scripts/character_growth.py --wd ... with absolute spectral power) and
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

REPO = Path(__file__).resolve().parents[1]
FILES = {0.1: "character_growth_abs_wd0.1.npz",
         0.3: "character_growth_abs_wd0.3.npz",
         1.0: "character_growth_abs_wd1.npz"}
WINDOWS = [(0, 300), (300, 800), (800, 2000)]


def slopes(ep, mat, idx, mask):
    """Mean per-character log slope (per epoch) over mask, for columns idx."""
    return float(np.mean([
        np.polyfit(ep[mask], np.log(mat[mask, i] + 1e-300), 1)[0]
        for i in idx]))


def analyse(fn):
    d = np.load(fn)
    ep = d["epochs"]
    pw = d["out_pow"]                      # absolute per-character power
    share = pw / pw.sum(1, keepdims=True)
    tr, va = d["train_acc"], d["val_acc"]
    n = pw.shape[1]
    final = share[-1]
    winners = np.where(final > 2 / n)[0]
    losers = np.where(final <= 2 / n)[0]
    grok = int(ep[np.argmax(va >= 0.99)])
    wshare = share[:, winners].mean(1) * n
    res = {
        "grok_epoch": grok,
        "train_sat_epoch": int(ep[np.argmax(tr >= 0.99)]),
        "takeoff_1p5x": int(ep[np.argmax(wshare > 1.5)]),
        "takeoff_2x": int(ep[np.argmax(wshare > 2.0)]),
        "n_winners": int(len(winners)),
        "effn": {"init": float(1 / (share[0] ** 2).sum()),
                 "grok": float(1 / (share[np.argmax(va >= 0.99)] ** 2).sum()),
                 "final": float(1 / (share[-1] ** 2).sum())},
        "windows": {},
    }
    for w0, w1 in WINDOWS:
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


def main():
    out = {}
    for wd, name in FILES.items():
        fn = REPO / "results/theory" / name
        if not fn.exists():
            print(f"λ={wd}: {name} missing, skipped")
            continue
        out[str(wd)] = analyse(fn)
        r = out[str(wd)]
        w = r["windows"].get("grok+[0,300]", {})
        print(f"λ={wd}: grok={r['grok_epoch']} "
              f"takeoff(1.5x/2x)={r['takeoff_1p5x']}/{r['takeoff_2x']} "
              f"winners={r['n_winners']} "
              f"effn {r['effn']['init']:.0f}->{r['effn']['grok']:.0f}"
              f"->{r['effn']['final']:.0f}")
        if w:
            print(f"   grok+[0,300]/ep: share w {w['share']['winner']:+.2e} "
                  f"l {w['share']['loser']:+.2e} | amp w "
                  f"{w['amplitude']['winner']:+.2e} "
                  f"l {w['amplitude']['loser']:+.2e} "
                  f"tot {w['amplitude']['total']:+.2e}")

    # scaling of the loser rates with λ (early-cleanup window)
    lams = [float(k) for k in out
            if "grok+[0,300]" in out[k]["windows"]]
    if len(lams) >= 2:
        lams.sort()
        for space in ("share", "amplitude"):
            r = [abs(out[str(l)]["windows"]["grok+[0,300]"][space]["loser"])
                 for l in lams]
            s = np.polyfit(np.log(lams), np.log(r), 1)[0]
            out[f"loser_{space}_rate_vs_lambda_exponent"] = float(s)
            print(f"loser {space} rate ∝ λ^{s:+.2f}  "
                  f"(free-decay prediction for amplitude: +1)")
    ofn = REPO / "results/theory/character_growth_lambda_sweep.json"
    json.dump(out, open(ofn, "w"), indent=1)
    print("saved", ofn)


if __name__ == "__main__":
    main()
