#!/usr/bin/env python3
"""Speedrun trainer: central protocol + three levers (see speedrun/README.md).
--char-boost K  : multiply amplitude of 8 random mult. characters by sqrt(K)
                  in embedding+unembedding at init ('oracle' = keep only them)
--wd-schedule   : "1.0" or "1.0:30,0.3" (wd until epoch 30, then 0.3)
--logit-scale S : logits *= S at loss time (S<1 = richer/less lazy)
Appends one JSON row to results/speedrun/runs.jsonl.
"""
import argparse, json, math, time
import numpy as np, torch, torch.nn.functional as F, torch.optim as optim
from grokking_capacity.data import grokking_data_torch
from grokking_capacity.models import TransformerTorch

ap = argparse.ArgumentParser()
ap.add_argument("--p", type=int, default=197)
ap.add_argument("--dim", type=int, default=236)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--char-boost", default="1")
ap.add_argument("--wd-schedule", default="1.0")
ap.add_argument("--logit-scale", type=float, default=1.0)
ap.add_argument("--mup-base", type=int, default=0,
                help="per-layer muP-Adam lr: embedding+norms at base lr, all "
                     "matrix params at lr*(mup_base/dim). 0 = off. Init is "
                     "left standard (lr-only muP).")
ap.add_argument("--epochs", type=int, default=1500)
ap.add_argument("--tag", default="")
a = ap.parse_args()
DEV = "cuda"

def primroot(p):
    f, n, d = [], p - 1, 2
    while d * d <= n:
        if n % d == 0:
            f.append(d)
            while n % d == 0: n //= d
        d += 1
    if n > 1: f.append(n)
    return next(g for g in range(2, p) if all(pow(g, (p-1)//q, p) != 1 for q in f))

P = a.p
G = primroot(P); ORDER = [pow(G, m, P) for m in range(P - 1)]

def colour(W, boost, keep):
    X = W[ORDER, :].clone()
    Cf = torch.fft.rfft(X - X.mean(0, keepdim=True), dim=0)
    mask = torch.ones(Cf.shape[0], 1, device=W.device)
    if boost == "oracle":
        mask[:] = 0.0; mask[0] = 1.0
        for k in keep: mask[k] = 1.0
    else:
        for k in keep: mask[k] = math.sqrt(float(boost))
    Y = torch.fft.irfft(Cf * mask, n=P - 1, dim=0)
    Y *= X.norm() / Y.norm()
    W2 = W.clone(); W2[ORDER, :] = Y + X.mean(0, keepdim=True)
    return W2

np.random.seed(a.seed); torch.manual_seed(a.seed)
Xtr, Ttr, Xva, Tva = grokking_data_torch(P, op="/", split_type="random",
                                         train_fraction=0.5, device="cpu")
model = TransformerTorch(depth=2, dim=a.dim, heads=1, n_tokens=P + 2,
                         seq_len=4, dropout=0.2, init_scale=1.0).to(DEV)
if a.char_boost != "1":
    rng = np.random.default_rng(a.seed)
    keep = (1 + rng.choice((P - 1)//2 - 1, 8, replace=False)).tolist()
    with torch.no_grad():
        model.embedding.weight.copy_(colour(model.embedding.weight, a.char_boost, keep))
        model.out.weight.copy_(colour(model.out.weight, a.char_boost, keep))

sched = [(0, float(a.wd_schedule.split(":")[0]))]
if ":" in a.wd_schedule:
    head, tail = a.wd_schedule.split(":")[1].split(",")
    sched.append((int(head), float(tail)))
if a.mup_base > 0:
    vec, mat = [], []
    for name, prm in model.named_parameters():
        (vec if ("embedding" in name or "norm" in name) else mat).append(prm)
    opt = optim.AdamW(
        [{"params": vec, "lr": a.lr},
         {"params": mat, "lr": a.lr * a.mup_base / a.dim}],
        betas=(0.9, 0.98), weight_decay=sched[0][1])
else:
    opt = optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.98),
                      weight_decay=sched[0][1])
Xtr, Ttr, Xva, Tva = Xtr.to(DEV), Ttr.to(DEV), Xva.to(DEV), Tva.to(DEV)
n = Xtr.shape[0]; B = 512
t_train = t_gen = None; steps = 0; t0 = time.time()
for ep in range(a.epochs):
    if len(sched) > 1 and ep == sched[1][0]:
        for g in opt.param_groups: g["weight_decay"] = sched[1][1]
    model.train()
    perm = torch.randperm(n, device=DEV)
    for i in range(0, n, B):
        idx = perm[i:i + B]
        opt.zero_grad()
        F.cross_entropy(model(Xtr[idx]) * a.logit_scale, Ttr[idx]).backward()
        opt.step(); steps += 1
    model.eval()
    with torch.no_grad():
        ta = (model(Xtr).argmax(-1) == Ttr).float().mean().item()
        va = (model(Xva).argmax(-1) == Tva).float().mean().item()
    if t_train is None and ta >= 0.99: t_train = ep + 1
    if t_gen is None and va >= 0.99: t_gen = ep + 1
    if t_gen is not None and ep >= t_gen + 10: break
row = dict(p=P, dim=a.dim, seed=a.seed, lr=a.lr, char_boost=a.char_boost,
           wd_schedule=a.wd_schedule, logit_scale=a.logit_scale,
           mup_base=a.mup_base, tag=a.tag,
           t_train=t_train, t_gen=t_gen,
           steps_gen=None if t_gen is None else t_gen * math.ceil(n / B),
           final_val=va, wall_s=round(time.time() - t0, 1))
with open("results/speedrun/runs.jsonl", "a") as fh:
    fh.write(json.dumps(row) + "\n")
print(row)
