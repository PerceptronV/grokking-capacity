"""One verified equivalence for reusing baseline random-label calibration.

The v4/v5 runner change affects modular evaluation policy and stopping only.
Evidence: reports/fullwidth_development/source_equivalence.json, including
identical normalized random-training AST and exact model/optimizer/RNG replay.
All used baseline model/data hashes still match exactly. The second audited
equivalence changes only the unused FullWidthQuantizer class; see
scalar_normalization_source_equivalence.json. No other revision is admitted.
"""
RUNNER = 'src/grokking_capacity/experiments/discrete.py'
VERIFIED_RANDOM_RUNNERS = frozenset({
    'f7667e4bbdee0d16df9975b2c15e4a025276c6a2c3768b8d076243dee4f905ac',
    '6e3016ef2a6d104415966b0ce6ef8350eb1a4d7247c25647382d58264d0a7669',
})


FULLWIDTH = 'src/grokking_capacity/models/fullwidth_discrete.py'
VERIFIED_UNUSED_QUANTIZER_MODULES = frozenset({'b147a59bc2c4edf6657ec576f0035ec925dc5a254b97516d86c7cbf46dd35f33', '9497d94ee93d04e28100f0dd4d8b07fbdd92a4c21d0d09ba7bed017264be1ff9'})


def baseline_random_sources_compatible(left, right):
    if left == right:
        return True
    if left.keys() != right.keys():
        return False
    for key in left:
        if left[key] == right[key]:
            continue
        pair = frozenset({left[key], right[key]})
        if key == RUNNER and pair == VERIFIED_RANDOM_RUNNERS:
            continue
        if key == FULLWIDTH and pair == VERIFIED_UNUSED_QUANTIZER_MODULES:
            continue
        return False
    return True
