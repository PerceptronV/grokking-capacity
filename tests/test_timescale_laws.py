"""Prefactors and exponents remain distinct across fitting and saved-law loading."""
import copy
import math

import numpy as np
import pytest

from grokking_capacity.analysis.forecast_onset import (
    fit_gen_law, fit_gen_law_fixed_b, fit_mem_law, solve_crossing,
)
from grokking_capacity.analysis.threshold_grid import _fit_log_linear
from grokking_capacity.analysis.timescale_laws import normalise_timescale_laws


def test_old_law_names_are_translated_without_changing_values_or_input():
    old = {'mem': {'ln_b': 1.1, 'b': 3., 'gamma': 15.},
           'gen': {'lnA': 20., 'a': -.4, 'beta': 1.1,
                   'per_prime_beta': {'97': {'beta': 1.2}},
                   'steps_unit_check': {'beta_steps': 1.1}},
           'prime_bootstrap': {'beta_ci95': [1., 1.2]},
           'optimiser': {'beta1': .9, 'beta2': .98}}
    before = copy.deepcopy(old)
    new = normalise_timescale_laws(old)
    assert new['mem'] == {'ln_B': 1.1, 'B': 3., 'gamma': 15.}
    assert new['gen']['b'] == 1.1
    assert new['gen']['per_prime_b']['97']['b'] == 1.2
    assert new['gen']['steps_unit_check']['b_steps'] == 1.1
    assert new['prime_bootstrap']['b_ci95'] == [1., 1.2]
    assert new['optimiser'] == old['optimiser']
    assert old == before
    assert normalise_timescale_laws(new) == new


def test_flat_record_renames_both_old_symbols_simultaneously():
    old = {'b': 3., 'beta': 1.1, 'a': -.4, 'gamma': 15.}
    assert normalise_timescale_laws(old) == {'B': 3., 'b': 1.1, 'a': -.4, 'gamma': 15.}


def test_conflicting_saved_aliases_are_rejected():
    with pytest.raises(ValueError, match='Conflicting'):
        normalise_timescale_laws({'ln_b': 1., 'ln_B': 2.})


def test_memorisation_fits_report_B_and_gamma():
    fractions = np.linspace(.1, .8, 9)
    times = 3. * np.exp(4. * fractions)
    points = [{'p': 97, 'f': float(f), 't_mem': float(t)}
              for f, t in zip(fractions, times)]
    fitted = fit_mem_law(points, [97], light=True)
    np.testing.assert_allclose([fitted['B'], fitted['ln_B'], fitted['gamma']],
                               [3., math.log(3.), 4.], rtol=1e-12)
    assert 'b' not in fitted and 'ln_b' not in fitted
    descriptive = _fit_log_linear(fractions, times)
    np.testing.assert_allclose([descriptive['B'], descriptive['gamma']], [3., 4.])


def test_generalisation_fit_and_fixed_exponent_use_b():
    primes = [97, 113, 139]
    cells = [{'p': p, 'param_count': P, 'in_window': True,
              't_gen': 1e8 * p**-.4 * P**-1.2}
             for p in primes for P in [1e4, 3e4, 1e5, 3e5]]
    fitted = fit_gen_law(cells, primes)
    np.testing.assert_allclose([fitted['lnA'], fitted['a'], fitted['b']],
                               [math.log(1e8), -.4, 1.2], rtol=1e-11)
    fixed = fit_gen_law_fixed_b(cells, primes, 1.2)
    np.testing.assert_allclose([fixed['lnA'], fixed['a'], fixed['b']],
                               [math.log(1e8), -.4, 1.2], rtol=1e-11)
    assert 'beta' not in fitted


def test_lower_intersection_uses_prefactor_B_and_exponent_b():
    # Construct an exact lower intersection at P=1e5, with gamma*f > b.
    p, K, C, P = 113, 1e5, 1., 1e5
    mem = {'ln_B': math.log(3.), 'gamma': 4.}
    gen = {'lnA': math.log(3.) + 4. + 1.2*math.log(P) + .4*math.log(p),
           'a': -.4, 'b': 1.2}
    result = solve_crossing(p, K, C, mem, gen)
    assert result == pytest.approx(math.log10(P), abs=1e-10)
