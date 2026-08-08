"""R_eff inner cutoff on the massless (Coulomb) projected cross section.

The impact-parameter integral runs over the ANNULUS R_eff <= b <= b_cap: a flyby
cannot approach closer than the sensor radius, so the impulse saturates at

    q_max = 2 alpha / (v R_eff) / conv_m2pGeV(1)

and dsigma/dq is identically zero above it. This is the massless counterpart of
the finite-range cutoff q_tilde < K1(R_eff/lamb) that the K1 machinery has always
applied; before the fix the massless branch continued the bare q^-3 Rutherford
tail past q_max, which no trajectory can populate.

By linearity of the b-integral the annulus is the difference of two discs,

    dsigma_annulus(q) = disc(q, b_cap) - disc(q, R_eff),

each disc being the uncapped Coulomb projection times the shell fraction
F(b_max/b_outer) (:func:`cross_section.rutherford_shell_fraction`). The tests
below check that identity against a brute-force Monte Carlo, the hard zero above
q_max, continuity below it, the R_eff -> 0 limit, and that the finite-lambda path
is untouched (and now agrees with the massless closed form as lamb -> infinity).
"""

import numpy as np
import pytest
from scipy.special import kn

from luhdm import config, cross_section, rate, units

R_EFF = config.R_EFF
CONV = units.conv_m2pGeV(1.0)          # GeV^-1 per metre
B_CAP = 0.1                            # the campaign's 10 cm cap


def bare_rutherford(q, alpha, v):
    """Uncapped, uncut Coulomb projection 2 pi alpha^2 / (v^2 q^3)."""
    return 2 * np.pi * alpha**2 / (v**2 * q**3)


# --------------------------------------------------------------------------- #
# MC: the annulus identity against a brute-force flyby simulation
# --------------------------------------------------------------------------- #
def _mc_dsigma(q_edges, alpha, v, b_lo_m, b_hi_m, n, seed=20260805):
    """MC dsigma/dq_z [GeV^-3] for b uniform-by-area on [b_lo, b_hi].

    Samples the annulus with the area weight, azimuth uniform, projects
    q_z = q_perp cos(phi) with q_perp = 2 alpha / (b v), and histograms the
    positive half (the sign convention of the closed form).
    """
    rng = np.random.default_rng(seed)
    b_lo, b_hi = b_lo_m * CONV, b_hi_m * CONV          # GeV^-1
    area = np.pi * (b_hi**2 - b_lo**2)                 # GeV^-2
    u = rng.random(n)
    b = np.sqrt(b_lo**2 + u * (b_hi**2 - b_lo**2))
    phi = rng.uniform(0.0, 2 * np.pi, n)
    qz = (2 * alpha / (b * v)) * np.cos(phi)
    counts, _ = np.histogram(qz, bins=q_edges)
    widths = np.diff(q_edges)
    val = area * counts / (n * widths)
    err = area * np.sqrt(np.maximum(counts, 1.0)) / (n * widths)
    return val, err


def _bin_average(fn, lo, hi, npts=2001):
    g = np.linspace(lo, hi, npts)
    return float(np.trapezoid(fn(g), g) / (hi - lo))


def test_mc_matches_annulus_closed_form():
    """Brute-force MC over the annulus == capped(b_cap) - capped(R_eff)."""
    alpha = 1.961e-9 * config.N_NEUTRONS               # headline-cell coupling
    v = 1e-3
    q_max = cross_section.coulomb_q_max(alpha, v)
    # a band well inside the reach, where the outer cap no longer bites
    edges = np.linspace(0.02 * q_max, 0.05 * q_max, 5)
    mc, err = _mc_dsigma(edges, alpha, v, R_EFF, B_CAP, n=20_000_000)
    for i in range(len(edges) - 1):
        ref = _bin_average(
            lambda g: cross_section.cross_section_rutherford_projection_capped(
                g, alpha, v, B_CAP), edges[i], edges[i + 1])
        assert abs(mc[i] - ref) < 5.0 * err[i], (edges[i], mc[i], ref, err[i])
        assert abs(mc[i] / ref - 1.0) < 0.05


def test_mc_finds_nothing_above_qmax():
    """No sampled flyby in the annulus delivers q > q_max."""
    alpha = 1.961e-9 * config.N_NEUTRONS
    v = 1e-3
    q_max = cross_section.coulomb_q_max(alpha, v)
    edges = np.linspace(q_max, 4.0 * q_max, 4)
    mc, _ = _mc_dsigma(edges, alpha, v, R_EFF, B_CAP, n=4_000_000)
    assert np.all(mc == 0.0)


def test_mc_cap_dominated_band():
    """MC also reproduces the closed form where the OUTER cap dominates."""
    alpha = 1.961e-9 * config.N_NEUTRONS
    v = 1e-3
    q_cap = 2 * alpha / (v * B_CAP * CONV)             # q at b = b_cap
    edges = np.linspace(0.3 * q_cap, 0.6 * q_cap, 4)
    mc, err = _mc_dsigma(edges, alpha, v, R_EFF, B_CAP, n=20_000_000)
    for i in range(len(edges) - 1):
        ref = _bin_average(
            lambda g: cross_section.cross_section_rutherford_projection_capped(
                g, alpha, v, B_CAP), edges[i], edges[i + 1])
        assert abs(mc[i] - ref) < 5.0 * err[i]
        assert abs(mc[i] / ref - 1.0) < 0.02


# --------------------------------------------------------------------------- #
# Hard zero above q_max (uncapped AND capped, closed form and rate dispatch)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cap", [None, B_CAP])
def test_zero_above_qmax(cap):
    alpha = 1.961e-9 * config.N_NEUTRONS
    v = 1e-3
    q_max = cross_section.coulomb_q_max(alpha, v)
    qs = q_max * np.array([1.0, 1.0 + 1e-9, 1.01, 2.0, 10.0, 1e4])
    d = cross_section.cross_section_rutherford_projection_capped(
        qs, alpha, v, cap)
    assert np.all(d == 0.0)
    # and through the pipeline entry point
    xs = rate.make_xsec(None, b_constrained_max=cap)
    assert np.all(rate.dsigma_dq_any(qs, alpha, v, xs) == 0.0)
    # the bare power law would have been wildly non-zero there
    assert np.all(bare_rutherford(qs, alpha, v) > 0.0)


def test_qmax_formula_and_headline_value():
    """q_max = 2 alpha/(v R_eff)/conv; ~317 GeV at the v3 headline cell."""
    alpha = 1.961e-9 * config.N_NEUTRONS
    v = 1e-3
    q_max = cross_section.coulomb_q_max(alpha, v)
    assert q_max == pytest.approx(2 * alpha / (v * R_EFF * CONV), rel=1e-15)
    assert q_max == pytest.approx(317.256, rel=1e-4)
    # the reach at q_max is exactly the sensor radius
    assert cross_section.coulomb_reach(q_max, alpha, v) == pytest.approx(
        R_EFF, rel=1e-12)


def test_qmax_scales_inversely_with_v_and_R_eff():
    alpha = 1.0
    assert cross_section.coulomb_q_max(alpha, 2e-3) == pytest.approx(
        0.5 * cross_section.coulomb_q_max(alpha, 1e-3), rel=1e-14)
    assert cross_section.coulomb_q_max(alpha, 1e-3, R_eff=2 * R_EFF) == \
        pytest.approx(0.5 * cross_section.coulomb_q_max(alpha, 1e-3), rel=1e-14)


# --------------------------------------------------------------------------- #
# Continuity / shape below q_max
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cap", [None, B_CAP])
def test_continuous_and_vanishing_at_qmax(cap):
    """dsigma/dq is positive, decreasing, and -> 0 continuously at q_max."""
    alpha = 1.961e-9 * config.N_NEUTRONS
    v = 1e-3
    q_max = cross_section.coulomb_q_max(alpha, v)
    frac = 1.0 - np.geomspace(1e-6, 0.5, 60)           # approach q_max from below
    qs = np.sort(frac) * q_max
    d = cross_section.cross_section_rutherford_projection_capped(
        qs, alpha, v, cap)
    assert np.all(d > 0.0)                             # strictly inside the reach
    assert np.all(np.diff(d) < 0.0)                    # monotone decreasing
    # approaching q_max the annulus vanishes as (1 - q/q_max)^(3/2): at
    # 1 - q/q_max = 1e-6 it is already ~2e-3 of the bare power law there
    ratio = float(d[-1] / bare_rutherford(qs[-1], alpha, v))
    assert ratio < 3e-3
    assert ratio == pytest.approx((4 / np.pi) * np.sqrt(2e-6), rel=0.05)
    # right-continuity onto the hard zero: already 4 orders down from mid-reach
    assert d[-1] < 1e-3 * d[0]


def test_far_below_qmax_recovers_the_power_law():
    """Deep inside the reach the cutoff is a negligible constant offset."""
    alpha = 1.961e-9 * config.N_NEUTRONS
    v = 1e-3
    q_max = cross_section.coulomb_q_max(alpha, v)
    qs = q_max * np.geomspace(1e-8, 1e-4, 20)
    d = cross_section.cross_section_rutherford_projection(qs, alpha, v)
    bare = bare_rutherford(qs, alpha, v)
    np.testing.assert_allclose(d, bare, rtol=1e-8)
    assert np.all(d <= bare)                           # never adds anything


def test_inner_shell_is_q_independent():
    """The removed inner disc is a q-independent constant, R^3 v/(3 alpha)."""
    alpha = 1.0
    v = 1e-3
    q_max = cross_section.coulomb_q_max(alpha, v)
    # inside the reach but not so far that the (tiny) removal underflows the
    # float64 difference: q/q_max in [1e-3, 1e-2] leaves ~7 significant digits
    qs = q_max * np.geomspace(1e-3, 1e-2, 8)
    removed = (bare_rutherford(qs, alpha, v)
               - cross_section.cross_section_rutherford_projection(qs, alpha, v))
    expect = units.conv_m2pGeV(R_EFF)**3 * v / (3.0 * alpha)
    np.testing.assert_allclose(removed, expect, rtol=1e-4)


# --------------------------------------------------------------------------- #
# R_eff -> 0 limit: reproduces the pre-fix (bare / retained-fraction) functions
# --------------------------------------------------------------------------- #
def _retained_legacy(r):
    """Verbatim pre-fix retained fraction (kept for the R_eff -> 0 limit)."""
    r = np.asarray(r, dtype=float)
    bites = r > 1.0
    r2 = np.where(bites, r * r, 1.0)
    s = np.sqrt(r2 - 1.0)
    return np.where(bites, 1.0 - (2.0 / np.pi) * (s / r2 + np.arctan(s)), 1.0)


def test_reff_zero_reproduces_bare_rutherford():
    alpha = 1.961e-9 * config.N_NEUTRONS
    v = np.geomspace(1e-4, config.VESC, 17)
    qs = 1e3
    got = cross_section.cross_section_rutherford_projection(qs, alpha, v,
                                                            R_eff=0.0)
    np.testing.assert_array_equal(got, bare_rutherford(qs, alpha, v))
    assert np.array_equal(
        cross_section.cross_section_rutherford_projection(qs, alpha, v,
                                                          R_eff=None), got)


def test_reff_zero_reproduces_legacy_capped():
    alpha = 1.0
    v = 1e-3
    # choose q so the reach spans 1 cm .. 2 m, i.e. straddles the 10 cm cap
    # (staying at r = b_max/b_cap <= 20, where the legacy 1 - (...) form is
    # still numerically sound -- the new one is accurate to arbitrary r)
    b_target = np.geomspace(1e-2, 2.0, 25)
    qs = 2 * alpha / (v * b_target * CONV)
    b_max = cross_section.coulomb_reach(qs, alpha, v)
    got = cross_section.cross_section_rutherford_projection_capped(
        qs, alpha, v, B_CAP, R_eff=0.0)
    ref = bare_rutherford(qs, alpha, v) * _retained_legacy(b_max / B_CAP)
    assert np.any(b_max > B_CAP)                       # the cap really bites
    np.testing.assert_allclose(got, ref, rtol=1e-9)


def test_reff_to_zero_is_continuous():
    """Shrinking R_eff converges monotonically onto the R_eff = 0 curve."""
    alpha = 1.961e-9 * config.N_NEUTRONS
    v = 1e-3
    q = 50.0
    bare = float(bare_rutherford(q, alpha, v))
    prev = 0.0
    for R in (R_EFF, R_EFF / 10, R_EFF / 100, R_EFF / 1e4):
        val = float(cross_section.cross_section_rutherford_projection(
            q, alpha, v, R_eff=R))
        assert val > prev                              # monotone in 1/R_eff
        prev = val
    assert prev == pytest.approx(bare, rel=1e-9)


def test_shell_fraction_endpoints():
    F = cross_section.rutherford_shell_fraction
    assert float(F(0.0)) == 1.0                        # b_outer = infinity
    assert float(F(np.inf)) == 0.0                     # b_outer = 0
    assert float(F(1.0)) == 1.0                        # b_outer = b_max
    rr = np.geomspace(1.0000001, 1e6, 500)
    assert np.all(np.diff(F(rr)) < 0.0)                # monotone in r
    assert np.all(F(rr) > 0.0)                         # never negative/zero
    # matches the legacy retained fraction where that form is still sound
    # (r <~ 50; past that its 1 - (...) cancellation dominates -- at r = 1e5 the
    # legacy value is wrong by ~1e-5 relative and eventually clips to 0)
    r = np.geomspace(1.05, 50.0, 60)
    np.testing.assert_allclose(F(r), _retained_legacy(r), rtol=1e-10)
    assert _retained_legacy(1e6) == 0.0                # legacy underflowed
    assert 0.0 < float(F(1e6)) == pytest.approx(4 / (3 * np.pi) * 1e-18,
                                                rel=1e-6)
    # series / closed-form branches join smoothly at x = 1/r = 1e-2
    lo, hi = float(F(1.0 / 0.0100001)), float(F(1.0 / 0.0099999))
    assert lo == pytest.approx(hi, rel=1e-4)


# --------------------------------------------------------------------------- #
# The finite-lambda path is unchanged (and is what the massless limit now meets)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("lamb", [2e-3, 2e-6])
def test_finite_lambda_dispatch_bitwise_unchanged(lamb):
    """rate.dsigma_dq_any on a finite-lambda handle is byte-for-byte the
    untouched cross_section.dsigma_dq / dsigma_dq_ln call."""
    alpha = 1e-4 * config.N_NEUTRONS
    vs = np.geomspace(1e-4, config.VESC, 23)
    q = 3.2e2
    for cap in (None, B_CAP):
        xs = rate.make_xsec(lamb, b_constrained_max=cap)
        got = rate.dsigma_dq_any(q, alpha, vs, xs)
        fn = (cross_section.dsigma_dq_ln if xs["use_ln"]
              else cross_section.dsigma_dq)
        ref = fn(q, alpha, lamb, R_EFF, vs, xs["interp"])
        np.testing.assert_array_equal(got, ref)


def test_finite_lambda_zero_above_its_own_qmax():
    """The K1 path always had the inner cutoff; confirm it still bites."""
    lamb = 2e-3
    alpha = 1e-4 * config.N_NEUTRONS
    v = 1e-3
    interp = cross_section.make_dsigma_dq_interpolant(1e-25, R_EFF, lamb,
                                                      N_points=200)
    G2 = cross_section.shape_factor(R_EFF / lamb)
    q_max = 2 * alpha * G2 * kn(1, R_EFF / lamb) / (units.conv_m2pGeV(lamb) * v)
    qs = q_max * np.array([1.001, 1.1, 10.0])
    assert np.all(cross_section.dsigma_dq(qs, alpha, lamb, R_EFF, v, interp)
                  == 0.0)


def test_massless_is_the_large_lambda_limit():
    """The fixed massless closed form == the K1 machinery at lamb = 100 m,
    including the shared hard zero above q_max (this is what the missing
    cutoff broke: the massless floor sat a factor ~2 below the lamb = 2 m one)."""
    lamb = 100.0                                       # xi = R_eff/lamb = 2.6e-6
    alpha = 1.0
    v = 1e-3
    interp = cross_section.make_dsigma_dq_interpolant(1e-25, R_EFF, lamb,
                                                      N_points=1200)
    q_max = cross_section.coulomb_q_max(alpha, v)
    qs = q_max * np.geomspace(1e-3, 0.9, 12)
    d_fin = cross_section.dsigma_dq(qs, alpha, lamb, R_EFF, v, interp)
    d_ml = cross_section.cross_section_rutherford_projection(qs, alpha, v)
    # 2% band: the residual is the finite-lambda tabulation's linear
    # interpolation error (it shrinks to <1e-3 at N_points=4000)
    np.testing.assert_allclose(d_fin, d_ml, rtol=2e-2)
    # both vanish above the (common) q_max
    q_hi = q_max * np.array([1.01, 2.0])
    assert np.all(cross_section.dsigma_dq(q_hi, alpha, lamb, R_EFF, v, interp)
                  == 0.0)
    assert np.all(cross_section.cross_section_rutherford_projection(
        q_hi, alpha, v) == 0.0)


# --------------------------------------------------------------------------- #
# Transit reach stays consistent with the new dsigma/dq
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cap", [None, B_CAP])
def test_reach_zero_exactly_where_dsigma_is_zero(cap):
    alpha = 1.961e-9 * config.N_NEUTRONS
    vs = np.geomspace(1e-4, config.VESC, 40)
    xs = rate.make_xsec(None, b_constrained_max=cap)
    for q in (1.0, 50.0, 200.0, 317.0, 400.0, 5e3):
        d = np.atleast_1d(rate.dsigma_dq_any(q, alpha, vs, xs))
        b = np.atleast_1d(rate.impact_parameter_max_any(q, alpha, vs, xs))
        assert np.array_equal(d == 0.0, b == 0.0), q
        assert np.all(b[b > 0] > R_EFF)


def test_expected_transits_drops_when_reach_is_sub_sensor():
    """A coupling so weak that b_max < R_eff at threshold yields no transits."""
    xs = rate.make_xsec(None)
    m = 1e10
    strong = rate.expected_transits(1e-9, m, __import__(
        "luhdm.halo", fromlist=["x"]).standard_halo_model, xs, 1e6)
    weak = rate.expected_transits(1e-30, m, __import__(
        "luhdm.halo", fromlist=["x"]).standard_halo_model, xs, 1e6)
    assert strong > 0.0
    assert weak == 0.0
