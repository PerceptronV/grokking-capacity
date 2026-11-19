"""Coefficient names for T_gen(P,p) = A p^a P^-b and T_mem(P,p) = B exp(gamma f).

New fits use ``gen['b']`` for the parameter-count exponent and ``mem['B']``
(or ``mem['ln_B']``) for the memorisation prefactor. Older saved law artifacts
remain immutable; normalise their names at the loading boundary only.
"""
from __future__ import annotations

import re
from typing import Any


def normalise_timescale_laws(value: Any) -> Any:
    """Return a copy of a saved timescale-law artifact in the current notation.

    This is for law artifacts, not arbitrary regression or optimiser records.
    It accepts old and current names, including nested bootstrap/sensitivity
    summaries. All numerical values are preserved. Conflicting aliases fail
    explicitly rather than silently selecting a coefficient.
    """
    if isinstance(value, list):
        return [normalise_timescale_laws(item) for item in value]
    if not isinstance(value, dict):
        return value
    old_prefactor = 'ln_b' in value or (
        'b' in value and any(key.startswith('gamma') for key in value)
        and 'lnA' not in value and 'a' not in value
    ) or ('b' in value and 'beta' in value)
    out = {}
    for key, item in value.items():
        name = key
        if isinstance(name, str):
            name = name.replace('ln_b', 'ln_B')
            name = re.sub(r'(?<![A-Za-z])beta(?![A-Za-z0-9])', 'b', name)
            if key == 'b' and old_prefactor:
                name = 'B'
        item = normalise_timescale_laws(item)
        if name in out and out[name] != item:
            raise ValueError(f'Conflicting timescale coefficients for {name!r}')
        out[name] = item
    return out
