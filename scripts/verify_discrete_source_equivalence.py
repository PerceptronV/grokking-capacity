#!/usr/bin/env python3
"""Verify the narrowly scoped v4/v5 equivalence of baseline random training."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
RUNNER = 'src/grokking_capacity/experiments/discrete.py'


class RandomTraining(ast.NodeTransformer):
    def visit_Expr(self, node):
        # Pure policy validation added in v5, before initialization. It does
        # not read or change RNG, data, parameters, or optimizer state.
        if (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                and node.value.func.id == 'primary_accuracy_keys'):
            return None
        return self.generic_visit(node)

    def visit_If(self, node):
        # This branch is unreachable for task=random in both versions.
        if ast.unparse(node.test) == "config['stop_on_success'] and config['task'] == 'modular'":
            return None
        return self.generic_visit(node)


def training_program(source):
    tree = RandomTraining().visit(ast.parse(source))
    names = {'dataset', 'evaluate', 'run', 'sync', 'tensor_digest'}
    selected = {n.name: ast.dump(n, include_attributes=False) for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name in names}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DEFAULTS' for t in node.targets):
            selected['DEFAULTS'] = ast.dump(node, include_attributes=False)
    assert set(selected) == names | {'DEFAULTS'}
    return selected


def same(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def main():
    old_root = ROOT / 'results/discrete/source-worktrees/fullwidth-v4'
    roots = {'v4': old_root, 'v5': ROOT}
    gates = {
        'v4': ROOT / 'results/discrete/fullwidth-learning-v4/B_quantized_d72_ff576_q4_s42',
        'v5': ROOT / 'results/discrete/fullwidth-learning-v5/B_categorical_d58_k16_rms_t01_mode_s42'}
    sources = {name: (root / RUNNER).read_text() for name, root in roots.items()}
    assert training_program(sources['v4']) == training_program(sources['v5'])
    scientific = ['src/grokking_capacity/models/discrete.py',
                  'src/grokking_capacity/models/fullwidth_discrete.py',
                  'src/grokking_capacity/models/transformer.py',
                  'src/grokking_capacity/data/modular.py']
    for filename in scientific:
        assert (old_root / filename).read_bytes() == (ROOT / filename).read_bytes()
    output = ROOT / 'results/discrete/protocol-equivalence-v4-v5'
    output.mkdir(exist_ok=False)
    config = dict(task='random', model='baseline', p=149, operation='/', train_fraction=.5,
                  dim=44, mlp_dim=176, depth=2, heads=1, dropout=0., seed=61,
                  n_samples=2048, lr=.001, weight_decay=1., beta1=.9, beta2=.98,
                  batch_size=512, max_steps=64, eval_every=16, checkpoint_every=64,
                  sustained_evals=3, stop_on_success=False)
    program = """import json,sys,torch
from pathlib import Path
from grokking_capacity.experiments.discrete import run
torch.set_num_threads(2)
run(json.loads(sys.argv[1]),Path(sys.argv[2]),torch.device('cuda:0'),gate=Path(sys.argv[3]))
"""
    for name, root in roots.items():
        audit = json.loads((gates[name] / 'audit.json').read_text())
        assert audit['arithmetic_gate_passed']
        with (output / (name + '.log')).open('w') as log:
            subprocess.run([sys.executable, '-c', program, json.dumps(config), str(output/name), str(gates[name])],
                           cwd=ROOT, env={**os.environ, 'PYTHONPATH': str(root/'src')},
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    saved = {name: torch.load(output/name/'checkpoint.pt', map_location='cpu', weights_only=False) for name in roots}
    keys = ['model', 'optimizer', 'step', 'examples', 'order', 'cursor', 'batch_rng', 'torch_rng', 'cuda_rng']
    checks = {key: same(saved['v4'][key], saved['v5'][key]) for key in keys}
    checks['observations_excluding_wall_time'] = same(
        [{k:v for k,v in row.items() if k != 'wall_seconds'} for row in saved['v4']['trace']],
        [{k:v for k,v in row.items() if k != 'wall_seconds'} for row in saved['v5']['trace']])
    assert all(checks.values()), checks
    result = dict(scope='Baseline random-label training only; does not equate modular stopping or categorical evaluation policies.',
                  runner_sha256={name: hashlib.sha256(source.encode()).hexdigest() for name,source in sources.items()},
                  unchanged_model_and_data_files=scientific,
                  random_training_ast_equal=True,
                  normalization='Remove pure policy validation and an unreachable task=modular stopping branch; all other random-training AST is identical.',
                  validation_config=config, exact_checkpoint_checks=checks,
                  checkpoint_sha256={name:hashlib.sha256((output/name/'checkpoint.pt').read_bytes()).hexdigest() for name in roots},
                  validation_runs={name:str(output/name) for name in roots})
    target = ROOT / 'reports/fullwidth_development/source_equivalence.json'
    target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
