"""One verified equivalence for reusing baseline random-label calibration.

The v4/v5 runner change affects modular evaluation policy and stopping only.
Evidence: reports/fullwidth_development/source_equivalence.json, including
identical normalized random-training AST and exact model/optimizer/RNG replay.
Model and data hashes must still match exactly. No other revision is admitted.
"""
RUNNER = 'src/grokking_capacity/experiments/discrete.py'
VERIFIED_RANDOM_RUNNERS = frozenset({
    'f7667e4bbdee0d16df9975b2c15e4a025276c6a2c3768b8d076243dee4f905ac',
    '6e3016ef2a6d104415966b0ce6ef8350eb1a4d7247c25647382d58264d0a7669',
})


def baseline_random_sources_compatible(left, right):
    if left == right:
        return True
    return (left.keys() == right.keys() and RUNNER in left
            and frozenset({left[RUNNER], right[RUNNER]}) == VERIFIED_RANDOM_RUNNERS
            and all(left[key] == right[key] for key in left if key != RUNNER))
