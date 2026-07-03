"""The log-space cross section must reproduce the direct one where both work.

The ln_* variants exist because K1(xi) underflows float64 for
xi = R_eff/lamb >~ 700 (e.g. lamb = 0.2 um). Where the direct machinery is
healthy (xi <= 10 across its tabulated q_tilde range) the two must agree.
"""

import numpy as np
import pytest

from luhdm import config, cross_section, units

R_EFF = config.R_EFF
Q_TH = config.Q_THRESH
ALPHA = 1e-4 * config.N_NEUTRONS


@pytest.mark.parametrize("lamb", [2e-3, 2e-4, 2e-5])
def test_dsigma_dq_ln_matches_direct(lamb):
    direct = cross_section.make_dsigma_dq_interpolant(1e-25, R_EFF, lamb,
                                                      N_points=400)
    ln_tab = cross_section.make_ln_dsigma_dq_interpolant(R_EFF, lamb)
    qs = np.geomspace(Q_TH, 10 * Q_TH, 8)
    for v in (3e-4, 1e-3):
        d = cross_section.dsigma_dq(qs, ALPHA, lamb, R_EFF, v, direct)
        ln = cross_section.dsigma_dq_ln(qs, ALPHA, lamb, R_EFF, v, ln_tab)
        both = (d > 0) & (ln > 0)
        assert (d > 0).sum() == (ln > 0).sum()
        if both.any():
            np.testing.assert_allclose(ln[both], d[both], rtol=0.03)


@pytest.mark.parametrize("lamb", [2e-3, 2e-4, 2e-5, 2e-6])
def test_impact_parameter_max_ln_matches_direct(lamb):
    vs = np.geomspace(1e-6, config.VESC, 40)
    for alpha_n in (1e-6, 1e-3, 1.0):
        a = alpha_n * config.N_NEUTRONS
        bd = cross_section.impact_parameter_max(Q_TH, a, lamb, R_EFF, vs)
        bl = cross_section.impact_parameter_max_ln(Q_TH, a, lamb, R_EFF, vs)
        both = (bd > 0) & (bl > 0)
        if both.any():
            np.testing.assert_allclose(bl[both], bd[both], rtol=1e-3)


def test_ln_path_finite_at_xi_1000():
    """lamb = 0.2 um: the direct K1(xi) underflows; the ln path must not."""
    lamb = 2e-7
    assert cross_section.kn(1, R_EFF / lamb) == 0.0  # the problem being solved
    a = 1.0 * config.N_NEUTRONS
    ln_tab = cross_section.make_ln_dsigma_dq_interpolant(R_EFF, lamb)
    ds = cross_section.dsigma_dq_ln(np.array([Q_TH]), a, lamb, R_EFF, 1e-3,
                                    ln_tab)
    b = cross_section.impact_parameter_max_ln(Q_TH, a, lamb, R_EFF,
                                              np.array([1e-3]))
    assert np.isfinite(ds).all() and (ds > 0).all()
    assert np.isfinite(b).all() and (b > 0).all()
    # b is measured from the sensor center: reach = R_eff + a few lambda
    assert R_EFF < b[0] < R_EFF + 20 * lamb
