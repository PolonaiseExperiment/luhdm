"""Fast smoke tests for the luhdm package."""

import numpy as np
import pytest
from scipy.special import kn

from luhdm import config, cross_section, halo, limits, units


def test_shm_normalization():
    v = np.linspace(0, config.VESC, 4000)
    assert np.trapezoid(halo.standard_halo_model(v), v) == pytest.approx(1.0, abs=1e-3)


def test_k1_inverse_roundtrip():
    for beta in (0.01, 0.5, 3.0, 10.0):
        k1 = kn(1, beta)
        assert cross_section.interpolant_k1_inverse(k1) == pytest.approx(beta, rel=1e-2)


def test_shape_factor_limits():
    assert cross_section.shape_factor(1e-8) == pytest.approx(1.0, abs=1e-6)
    assert cross_section.shape_factor(0.1) == pytest.approx(1.001, abs=1e-3)


def test_conversions():
    # 1 m = 5.0679e15 GeV^-1
    assert units.conv_m2pGeV(1.0) == pytest.approx(5.0679e15, rel=1e-4)


def test_extremeness_zero_events_matches_poisson():
    # With no events, the 95% optimum-interval limit is mu = -ln(0.05) ~ 3.0:
    # extremeness must cross 0.95 there.
    table = limits.new_table(seed=1)
    qs = np.geomspace(1.0, 10.0, 50)
    flat = np.ones_like(qs)  # arbitrary shape; only mu matters with no events
    p_low, mu_low = limits.extremeness_and_mu(
        table, np.array([]), qs, flat / np.trapezoid(flat, qs) * 2.0, t_obs=1.0, n_mc=20000
    )
    p_high, mu_high = limits.extremeness_and_mu(
        table, np.array([]), qs, flat / np.trapezoid(flat, qs) * 4.0, t_obs=1.0, n_mc=20000
    )
    assert mu_low == pytest.approx(2.0, rel=1e-6) and mu_high == pytest.approx(4.0, rel=1e-6)
    assert p_low < 0.95 < p_high


def test_excluded_band_edges():
    alphas = np.geomspace(1e-9, 1e-2, 15)
    ps = np.where((alphas > 1e-7) & (alphas < 1e-4), 1.0, 0.0)
    lo, hi = limits.excluded_band(alphas, ps)
    assert 1e-8 < lo < 1e-6 and 1e-5 < hi < 1e-3
