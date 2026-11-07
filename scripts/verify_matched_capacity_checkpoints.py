#!/usr/bin/env python3
"""Independent CPU evaluation of the twelve selected capacity checkpoints."""
from pathlib import Path
from collections import defaultdict
import hashlib,json,math,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from grokking_capacity.experiments.capacity import _get_dataset
from grokking_capacity.models import TransformerTorch,count_parameters
OUT=ROOT/'results/capacity_h100_matched_20261103'
torch.set_num_threads(1)
rows=json.loads((OUT/'cells.json').read_text());groups=defaultdict(list)
for r in rows:groups[r['dim'],r['seed']].append(r)
checks=[]
for (dim,seed),items in sorted(groups.items()):
 r=max(items,key=lambda x:x['bits']);np.random.seed(seed);torch.manual_seed(seed)
 X,Y=_get_dataset(r['n'],113,'random',seq_len=4,device='cpu')
 model=TransformerTorch(depth=2,dim=dim,heads=1,n_tokens=115,seq_len=4,dropout=.2,init_scale=1.)
 path=Path(r['npz_path']).parent/'model.pt';state=torch.load(path,map_location='cpu',weights_only=True)
 model.load_state_dict(state['model_state_dict']);model.eval();loss=0.
 with torch.no_grad():
  for start in range(0,len(Y),512):loss+=torch.nn.functional.cross_entropy(model(X[start:start+512]),Y[start:start+512],reduction='sum').item()
 bits=len(Y)*math.log2(115)-loss/math.log(2);err=abs(bits-r['bits'])
 assert count_parameters(model)==r['P'];assert err<.05,(r,err)
 checks.append(dict(dim=dim,seed=seed,n=r['n'],bits_saved=r['bits'],bits_independent_cpu=bits,error_bits=err,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),dataset_sha256=hashlib.sha256(X.numpy().tobytes()+Y.numpy().tobytes()).hexdigest()))
result=dict(checks=checks,max_error_bits=max(x['error_bits'] for x in checks),all_passed=True)
(OUT/'checkpoint_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
