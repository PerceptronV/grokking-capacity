import math

import numpy as np
import pytest

from grokking_capacity.analysis.law_fitting import K, branches, fit


def test_both_roots_satisfy_original_timescale_equation():
    law = {"mem": {"ln_B": 0., "gamma_original": 1.},
           "gen": {"lnA": 12., "a": 0., "b": 1.}}
    p, capacity = 97, 2.16
    lower, upper = branches(p, law, capacity)
    assert 0 < lower < upper
    for parameters in (lower, upper):
        assert K(p) / (capacity * parameters) == pytest.approx(12 - math.log(parameters))
    pivot = K(p) / capacity
    law["gen"]["lnA"] = 1 + math.log(pivot)
    assert branches(p, law, capacity) == pytest.approx((pivot, pivot))
    law["gen"]["lnA"] -= .1
    assert branches(p, law, capacity) == (None, None)


def test_fits_recover_known_laws():
    points = [{"p": p, "P": P, "f": K(p) / (2.16 * P)}
              for p in (97, 113, 139) for P in (1e4, 4e4, 1e5)]
    mem = fit("mem", [{**r, "t": np.exp(.5 + 2 * r["f"])} for r in points])
    gen = fit("gen", [{**r, "t": np.exp(20) * r["p"] ** -.4 * r["P"] ** -1.1}
                      for r in points])
    assert [mem["ln_B"], mem["gamma_original"]] == pytest.approx([.5, 2])
    assert [gen["lnA"], gen["a"], gen["b"]] == pytest.approx([20, -.4, 1.1])
