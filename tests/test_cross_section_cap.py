"""Impact-parameter cap on the cross section dsigma/dq.

The cap ``b_constrained_max`` (metres) truncates the impact-parameter integral
at min(b_constrained_max, b_max(q)). It is a byte-for-byte no-op when unset
(None) for both the finite-lambda K1 machinery and the massless Coulomb closed
form, and, when set, removes the reach beyond the cap. The massless capped form
is the analytic closed form; it must agree with the general finite-lambda path
in the Coulomb regime (large lambda). Guards forbid a cap below the sensor
radius R_eff (which would invert the integration limits -> negative rate).
"""

import numpy as np
import pytest

from luhdm import config, cross_section, rate, units

R_EFF = config.R_EFF
LAMB_DIRECT = 2e-3          # xi = R_eff/lamb = 0.1 -> direct K1 path is healthy


def retained(r):
    """Reference retained fraction (verbatim capped-Coulomb closed form)."""
    r = np.asarray(r, dtype=float)
    bites = r > 1.0
    r2 = np.where(bites, r * r, 1.0)
    s = np.sqrt(r2 - 1.0)
    return np.where(bites, 1.0 - (2.0 / np.pi) * (s / r2 + np.arctan(s)), 1.0)


# --------------------------------------------------------------------------- #
# NO-OP: b_constrained_max=None reproduces the pre-cap tabulation byte-for-byte
# --------------------------------------------------------------------------- #
def test_noop_direct_interpolant():
    a = cross_section.make_dsigma_dq_interpolant(1e-25, R_EFF, LAMB_DIRECT,
                                                 N_points=60)
    b = cross_section.make_dsigma_dq_interpolant(1e-25, R_EFF, LAMB_DIRECT,
                                                 N_points=60,
                                                 b_constrained_max=None)
    assert np.array_equal(a.x, b.x)
    assert np.array_equal(a.y, b.y)


def test_noop_ln_interpolant():
    a = cross_section.make_ln_dsigma_dq_interpolant(R_EFF, LAMB_DIRECT,
                                                    N_points=80)
    b = cross_section.make_ln_dsigma_dq_interpolant(R_EFF, LAMB_DIRECT,
                                                    N_points=80,
                                                    b_constrained_max=None)
    assert np.array_equal(a.x, b.x)
    assert np.array_equal(a.y, b.y)


def test_noop_massless_via_rate():
    """dsigma_dq_any on an uncapped massless xs == plain Rutherford, byte-wise."""
    xs = rate.make_xsec(None)                       # b_constrained_max defaults None
    assert xs["b_constrained_max"] is None
    alpha = 1e-4 * config.N_NEUTRONS
    q = 3.2e2
    vs = np.geomspace(1e-4, config.VESC, 37)
    got = rate.dsigma_dq_any(q, alpha, vs, xs)
    ref = cross_section.cross_section_rutherford_projection(q, alpha, vs)
    assert np.array_equal(got, ref)


# --------------------------------------------------------------------------- #
# CAPPED-COULOMB closed-form properties
# --------------------------------------------------------------------------- #
def test_retained_edge_and_limit():
    assert retained(1.0) == 1.0                     # no suppression at the edge
    assert retained(1e6) == pytest.approx(0.0, abs=1e-6)   # full suppression
    assert 0.0 < float(retained(1.0000001)) < 1.0


def test_retained_monotone_decreasing():
    # strictly decreasing where the verbatim formula is numerically sound
    # (retained ~ 1/(pi r^3); the 1 - (...) cancellation only breaks past r~1e4)
    rr = np.geomspace(1.0000001, 1e4, 400)
    v = retained(rr)
    assert np.all(np.diff(v) < 0.0)


def test_capped_equals_uncapped_when_cap_huge():
    """r < 1 everywhere (huge cap): capped == uncapped closed form."""
    alpha = 1e-4 * config.N_NEUTRONS
    q = 5e2
    vs = np.geomspace(1e-4, config.VESC, 25)
    huge = 1e12                                     # metres; b_max << huge
    capped = cross_section.cross_section_rutherford_projection_capped(
        q, alpha, vs, huge)
    unc = cross_section.cross_section_rutherford_projection(q, alpha, vs)
    # cap does not bite anywhere
    b_max = 2 * alpha / (q * vs) / units.conv_m2pGeV(1.0)
    assert np.all(b_max < huge)
    np.testing.assert_allclose(capped, unc, rtol=0, atol=0)


# --------------------------------------------------------------------------- #
# CAP BITES: massless closed form below the uncapped Rutherford by retained(r)
# --------------------------------------------------------------------------- #
def test_cap_bites_massless():
    alpha = 1.0
    v = 1e-3
    b_cap = 0.1
    # pick q so b_max ~ 0.5 m (r = 5, cap clearly bites)
    q = 2 * alpha / (v * 0.5 * units.conv_m2pGeV(1.0))
    qa = np.array([q])
    va = np.array([v])
    b_max = 2 * alpha / (q * v) / units.conv_m2pGeV(1.0)
    r = b_max / b_cap
    assert r > 1.0                                  # cap bites
    capped = cross_section.cross_section_rutherford_projection_capped(
        qa, alpha, va, b_cap)
    unc = cross_section.cross_section_rutherford_projection(qa, alpha, va)
    assert np.all(capped < unc)
    np.testing.assert_allclose(capped, unc * retained(r), rtol=1e-12)


# --------------------------------------------------------------------------- #
# GUARDS: a cap below R_eff / below the inner cutoff must raise (never assert)
# --------------------------------------------------------------------------- #
def test_guard_interpolant_cap_below_reff():
    with pytest.raises(ValueError):
        cross_section.make_dsigma_dq_interpolant(
            1e-25, R_EFF, LAMB_DIRECT, N_points=20, b_constrained_max=R_EFF / 2)
    with pytest.raises(ValueError):
        cross_section.make_ln_dsigma_dq_interpolant(
            R_EFF, LAMB_DIRECT, N_points=20, b_constrained_max=R_EFF / 2)


def test_guard_core_xi_cap_below_xi():
    xi = 0.1                                        # K1(0.1) ~ 9.85
    q_tilde = 0.5                                   # < K1(xi): passes early return
    with pytest.raises(ValueError):
        cross_section.dsigma_dq_tilde(
            q_tilde, xi, cross_section.interpolant_k1_inverse, xi_cap=xi / 2)


def test_guard_core_ln_cap_below_cutoff():
    # ln_k1_cap > ln_q_tilde_max (K1 decreasing) <=> xi_cap < xi
    ln_q_tilde_max = float(cross_section.ln_k1(0.1))
    ln_k1_cap = float(cross_section.ln_k1(0.05))    # > ln_q_tilde_max
    assert ln_k1_cap > ln_q_tilde_max
    with pytest.raises(ValueError):
        cross_section.ln_dsigma_dq_tilde(2.0, ln_q_tilde_max,
                                         ln_k1_cap=ln_k1_cap)


# --------------------------------------------------------------------------- #
# CONSISTENCY: capped massless closed form vs the finite-lambda K1 machinery
# at a large regulator lambda (Coulomb regime), where the cap bites.
# --------------------------------------------------------------------------- #
def test_capped_closed_form_matches_finite_lambda():
    lamb = 100.0                                    # xi = R_eff/lamb = 2e-6
    b_cap = 1.0                                     # >> R_eff; cap in the reach
    alpha = 1.0
    v = 1e-3
    interp = cross_section.make_dsigma_dq_interpolant(
        1e-25, R_EFF, lamb, N_points=400, b_constrained_max=b_cap)
    # q range with q_tilde in [0.05, 20], all below K1(xi_cap) ~ 100 (cap bites)
    G2 = cross_section.shape_factor(R_EFF / lamb)
    q_tilde = np.geomspace(0.05, 20.0, 10)
    qs = q_tilde * 2 * alpha * G2 / (units.conv_m2pGeV(lamb) * v)
    d_fin = cross_section.dsigma_dq(qs, alpha, lamb, R_EFF, v, interp)
    d_mless = cross_section.cross_section_rutherford_projection_capped(
        qs, alpha, np.full_like(qs, v), b_cap)
    d_unc = cross_section.cross_section_rutherford_projection(
        qs, alpha, np.full_like(qs, v))
    assert np.all(d_mless < d_unc * 0.9999)         # cap bites over the range
    np.testing.assert_allclose(d_fin, d_mless, rtol=0.01)


# --------------------------------------------------------------------------- #
# TRANSIT REACH: the cap also clips impact_parameter_max_any, so the transit
# diagnostics (n_transit, halo bmax) stay consistent with the capped dsigma/dq.
# --------------------------------------------------------------------------- #
def _reach_handle(lamb, cap=None):
    """Minimal xs for impact_parameter_max_any, skipping make_xsec's interpolant
    build (the reach dispatch never touches it)."""
    xi = None if lamb is None else config.R_EFF / lamb
    return dict(lamb=lamb, use_ln=(xi is not None and xi > 30),
                interp=None, R_EFF=config.R_EFF, b_constrained_max=cap)


def test_make_xsec_threads_cap():
    """make_xsec stores the cap on the handle for every branch."""
    assert rate.make_xsec(None, b_constrained_max=0.1)["b_constrained_max"] == 0.1
    assert rate.make_xsec(None)["b_constrained_max"] is None


def test_transit_reach_capped_and_noop():
    q = config.Q_THRESH
    vs = np.geomspace(1e-4, config.VESC, 20)
    alpha = 1.0 * config.N_NEUTRONS                  # strong coupling
    cap = 0.1
    # Cap BITES where the reach exceeds it: massless/Coulomb and long-range
    # (direct K1) reaches both run past 10 cm at this coupling.
    for lamb in (None, 2.0):
        b0 = rate.impact_parameter_max_any(q, alpha, vs, _reach_handle(lamb))
        bc = rate.impact_parameter_max_any(q, alpha, vs, _reach_handle(lamb, cap))
        assert np.any(b0 > cap)                          # cap actually bites here
        assert np.all(bc <= cap + 1e-12)                 # reach clipped at the cap
        assert np.array_equal(bc, np.minimum(b0, cap))   # exactly min(b_max, cap)
    # Cap IDLE where the reach stays below it: short-range reach is unchanged
    # (the Yukawa cutoff keeps b_max ~ R_eff << 10 cm).
    b0s = rate.impact_parameter_max_any(q, alpha, vs, _reach_handle(2e-6))
    bcs = rate.impact_parameter_max_any(q, alpha, vs, _reach_handle(2e-6, cap))
    assert np.all(b0s < cap) and np.array_equal(bcs, b0s)
