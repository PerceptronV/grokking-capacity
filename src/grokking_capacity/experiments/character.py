"""Track character amplitudes with the original fixed-horizon training protocol."""
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from ..data import grokking_data_torch
from ..models import TransformerTorch
from ..analysis.characters import spectral_power, primitive_root


def parser():
    result=argparse.ArgumentParser(description=__doc__)
    result.add_argument('--p',type=int,default=113)
    result.add_argument('--seed',type=int,default=42)
    result.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    result.add_argument('--wd',type=float,default=1.)
    result.add_argument('--alpha',type=float,default=.5)
    result.add_argument('--dim',type=int,default=128)
    result.add_argument('--epochs',type=int,default=2500)
    result.add_argument('--every',type=int,default=5)
    result.add_argument('--batch-size',type=int,default=512)
    result.add_argument('--out',type=Path)
    return result


def output_path(args):
    if args.out is not None:
        return args.out
    defaults={'p':113,'seed':42,'wd':1.,'alpha':.5,'dim':128}
    labels={'p':'p','seed':'s','wd':'wd','alpha':'a','dim':'d'}
    tags=''.join(f'_{labels[key]}{getattr(args,key):g}' for key,value in defaults.items() if getattr(args,key)!=value)
    return Path(f'results/theory/character_growth{tags}.npz')


def run(args):
    if args.epochs < 1 or args.every < 1 or args.batch_size < 1:
        raise ValueError('Epochs, recording cadence, and batch size must be positive')
    primitive_root(args.p)
    P,DIM,SEED,EPOCHS,EVERY,BATCH=args.p,args.dim,args.seed,args.epochs,args.every,args.batch_size
    DEV=args.device
    OUT=output_path(args)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    Xtr, Ttr, Xva, Tva = grokking_data_torch(P, op="/", split_type="random",
                                             train_fraction=args.alpha, device="cpu")
    model = TransformerTorch(depth=2, dim=DIM, heads=1, n_tokens=P + 2,
                             seq_len=4, dropout=0.2, init_scale=1.0).to(DEV)
    opt = optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.98),
                      weight_decay=args.wd)
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
                out_spec.append(spectral_power(sd["out.weight"].cpu().numpy(),P))
                emb_spec.append(spectral_power(sd["embedding.weight"].cpu().numpy(),P))
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
             weight_decay=args.wd, alpha=args.alpha, dim=DIM, prime=P, seed=SEED, batch_size=BATCH)
    print(f"saved {OUT}")
    return OUT


def main(argv=None):
    run(parser().parse_args(argv))


if __name__ == '__main__':
    main()
