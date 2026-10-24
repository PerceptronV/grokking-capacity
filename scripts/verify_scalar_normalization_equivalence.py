#!/usr/bin/env python3
"""Prove that adding scalar RMS quantization leaves baseline training unchanged."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
from scripts.verify_discrete_source_equivalence import same

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'results/discrete/source-worktrees/fullwidth-v5'
FULLWIDTH = 'src/grokking_capacity/models/fullwidth_discrete.py'


def unaffected_ast(path):
    tree = ast.parse(path.read_text())
    changed = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FullWidthQuantizer']
    assert len(changed) == 1
    assert all(isinstance(n, ast.FunctionDef) for n in changed[0].body)
    tree.body = [n for n in tree.body if n is not changed[0]]
    return ast.dump(tree, include_attributes=False)


def main():
    assert unaffected_ast(OLD/FULLWIDTH) == unaffected_ast(ROOT/FULLWIDTH)
    unchanged = ['src/grokking_capacity/experiments/discrete.py',
                 'src/grokking_capacity/models/discrete.py',
                 'src/grokking_capacity/models/transformer.py',
                 'src/grokking_capacity/data/modular.py']
    assert all((OLD/p).read_bytes() == (ROOT/p).read_bytes() for p in unchanged)
    output = ROOT/'results/discrete/protocol-equivalence-v5-v6'
    output.mkdir(exist_ok=False)
    cfg = dict(task='modular', model='baseline', p=149, operation='/', train_fraction=.5,
               dim=44, mlp_dim=176, depth=2, heads=1, dropout=0., seed=61,
               lr=.001, weight_decay=1., beta1=.9, beta2=.98, batch_size=512,
               max_steps=64, eval_every=16, checkpoint_every=64,
               sustained_evals=3, stop_on_success=False)
    program = """import json,sys,torch
from pathlib import Path
from grokking_capacity.experiments.discrete import run
torch.set_num_threads(2)
run(json.loads(sys.argv[1]),Path(sys.argv[2]),torch.device('cuda:0'))
"""
    roots = {'v5': OLD, 'v6': ROOT}
    for name, root in roots.items():
        with (output/(name+'.log')).open('w') as log:
            subprocess.run([sys.executable, '-c', program, json.dumps(cfg), str(output/name)],
                           cwd=ROOT, env={**os.environ, 'PYTHONPATH':str(root/'src')},
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    saved = {name:torch.load(output/name/'checkpoint.pt',map_location='cpu',weights_only=False) for name in roots}
    keys=['model','optimizer','step','examples','order','cursor','batch_rng','torch_rng','cuda_rng']
    checks={key:same(saved['v5'][key],saved['v6'][key]) for key in keys}
    checks['observations_excluding_wall_time']=same(
        [{k:v for k,v in row.items() if k!='wall_seconds'} for row in saved['v5']['trace']],
        [{k:v for k,v in row.items() if k!='wall_seconds'} for row in saved['v6']['trace']])
    assert all(checks.values()), checks
    result=dict(scope='Baseline equivalence only. Baseline does not instantiate FullWidthQuantizer; no discrete gate is transferred to a new surrogate.',
                changed_class='FullWidthQuantizer', unaffected_module_ast_equal=True,
                unchanged_source_files=unchanged,
                baseline_dependency_hashes={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in unchanged if 'experiments/discrete.py' not in name},
                fullwidth_sha256={name:hashlib.sha256((root/FULLWIDTH).read_bytes()).hexdigest() for name,root in roots.items()},
                validation_config=cfg, exact_checkpoint_checks=checks,
                checkpoint_sha256={name:hashlib.sha256((output/name/'checkpoint.pt').read_bytes()).hexdigest() for name in roots})
    (ROOT/'reports/fullwidth_development/scalar_normalization_source_equivalence.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
