#!/usr/bin/env python3
"""Track multiplicative-character amplitudes through training (p=113, d=128,
seed 42, central protocol, no early stop, 2500 epochs). Records per-character
power fractions of the unembedding and embedding every 5 epochs, plus
accuracies — measures the per-character amplification rates (the 'gap') that
the T_gen mechanism needs. Output: results/theory/character_growth.npz
(or character_growth_wd<λ>.npz when --wd is given).
"""
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from grokking_capacity.data import grokking_data_torch
from grokking_capacity.models import TransformerTorch

ap = argparse.ArgumentParser()
ap.add_argument("--wd", type=float, default=1.0)
ap.add_argument("--alpha", type=float, default=0.5, help="train fraction")
ap.add_argument("--dim", type=int, default=128)
ap.add_argument("--epochs", type=int, default=2500)
ap.add_argument("--out", type=str, default=None)
ARGS = ap.parse_args()

P, DIM, SEED, EPOCHS, EVERY, BATCH = 113, ARGS.dim, 42, ARGS.epochs, 5, 512
DEV = "cuda"
if ARGS.out:
    OUT = ARGS.out
else:
    tags = ""
    if ARGS.wd != 1.0:
        tags += f"_wd{ARGS.wd:g}"
    if ARGS.alpha != 0.5:
        tags += f"_a{ARGS.alpha:g}"
    if ARGS.dim != 128:
        tags += f"_d{ARGS.dim}"
    OUT = f"results/theory/character_growth{tags or ''}.npz" if tags \
        else "results/theory/character_growth.npz"

def primitive_root(p):
    fac, n, d = [], p - 1, 2
    while d * d <= n:
        if n % d == 0:
            fac.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        fac.append(n)
    return next(g for g in range(2, p)
                if all(pow(g, (p - 1) // q, p) != 1 for q in fac))

G = primitive_root(P)
ORDER = [pow(G, m, P) for m in range(P - 1)]

def mult_pow(W):
    """Absolute per-character spectral power (unnormalised) — share is
    pw/pw.sum(); keeping pw lets amplitude decay be separated from
    renormalisation."""
    X = W[ORDER, :].astype(np.float64)
    X -= X.mean(0, keepdims=True)
    C = np.fft.rfft(X, axis=0)
    return (np.abs(C[1:(P - 1) // 2 + 1, :]) ** 2).sum(1)

np.random.seed(SEED)
torch.manual_seed(SEED)
Xtr, Ttr, Xva, Tva = grokking_data_torch(P, op="/", split_type="random",
                                         train_fraction=ARGS.alpha, device="cpu")
model = TransformerTorch(depth=2, dim=DIM, heads=1, n_tokens=P + 2,
                         seq_len=4, dropout=0.2, init_scale=1.0).to(DEV)
opt = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98),
                  weight_decay=ARGS.wd)
Xtr, Ttr, Xva, Tva = Xtr.to(DEV), Ttr.to(DEV), Xva.to(DEV), Tva.to(DEV)

epochs_rec, out_spec, emb_spec, tr_acc, va_acc = [], [], [], [], []
n = Xtr.shape[0]
for ep in range(EPOCHS):
    model.train()
    perm = torch.randperm(n, device=DEV)
    for i in range(0, n, BATCH):
        idx = perm[i:i + BATCH]
        opt.zero_grad()
        loss = F.cross_entropy(model(Xtr[idx]), Ttr[idx])
        loss.backward()
        opt.step()
    if ep % EVERY == 0 or ep == EPOCHS - 1:
        model.eval()
        with torch.no_grad():
            ta = (model(Xtr).argmax(-1) == Ttr).float().mean().item()
            va = (model(Xva).argmax(-1) == Tva).float().mean().item()
            sd = model.state_dict()
            out_spec.append(mult_pow(sd["out.weight"].cpu().numpy()))
            emb_spec.append(mult_pow(sd["embedding.weight"].cpu().numpy()))
        epochs_rec.append(ep)
        tr_acc.append(ta)
        va_acc.append(va)
        if ep % 100 == 0:
            sh = out_spec[-1] / out_spec[-1].sum()
            print(f"ep {ep}: train {ta:.3f} val {va:.3f} "
                  f"top-char {sh.max()*len(sh):.1f}x", flush=True)

out_pow, emb_pow = np.array(out_spec), np.array(emb_spec)
np.savez(OUT,
         epochs=np.array(epochs_rec),
         out_spec=out_pow / out_pow.sum(1, keepdims=True),
         emb_spec=emb_pow / emb_pow.sum(1, keepdims=True),
         out_pow=out_pow, emb_pow=emb_pow,
         train_acc=np.array(tr_acc), val_acc=np.array(va_acc),
         weight_decay=ARGS.wd, alpha=ARGS.alpha, dim=DIM)
print(f"saved {OUT}")
