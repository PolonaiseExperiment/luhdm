"""Detector-rate pipeline shared by the notebook and the remote-node scan scripts.

One home for the physics that turns (coupling, mass, arrival distribution)
into a momentum-kick spectrum, so notebooks/limit_contour.ipynb and
scripts/scan_grid.py / scripts/make_maps.py cannot drift apart. Notation
follows Dorian's original modules (mu = m_DM inside rate formulas).

The mediator range enters through an "xs" handle from make_xsec():
lamb in meters for a finite range (log-space tabulation picked automatically
once xi = R_eff/lamb > 30, where the direct Bessels degrade), or lamb=None
for a massless mediator (analytic Rutherford projection, Coulomb reach).
"""

import numpy as np

from luhdm import config, cross_section, halo, units

Q_THRESH = config.Q_THRESH
R_EFF = config.R_EFF


def make_xsec(lamb, R_eff=R_EFF, N_points=600, force_ln=False,
              b_constrained_max=None):
    """Build the cross-section handle for one mediator range.

    lamb : mediator range in meters, or None for massless (analytic).
    force_ln : use the (fast, trapz-based) log-space tabulation even at small
    xi, e.g. when tabulating many ranges; validated against the direct path
    in tests/test_cross_section_ln.py.
    b_constrained_max : optional impact-parameter cap [m]. Clips the outer edge
    of the b-integral to min(b_constrained_max, b_max(q)) in BOTH the cross
    section dsigma/dq and the geometric transit reach (impact_parameter_max_any),
    so n_transit / halo bmax stay consistent. None = uncapped, a byte-for-byte
    no-op relative to the pre-cap pipeline for finite and massless mediators.
    Returns a dict consumed by dsigma_dq_any / differential_rate_trapz /
    expected_transits / transit_count_halo.
    """
    if lamb is None:
        return dict(lamb=None, use_ln=False, interp=None,
                    b_constrained_max=b_constrained_max)
    xi = R_eff / lamb
    use_ln = force_ln or xi > 30
    if use_ln:
        interp = cross_section.make_ln_dsigma_dq_interpolant(
            R_eff, lamb, b_constrained_max=b_constrained_max)
    else:
        interp = cross_section.make_dsigma_dq_interpolant(
            1e-25, R_eff, lamb, N_points=N_points,
            b_constrained_max=b_constrained_max)
    return dict(lamb=lamb, use_ln=use_ln, interp=interp, R_eff=R_eff,
                b_constrained_max=b_constrained_max)


def dsigma_dq_any(q, alpha, vs, xs, R_eff=R_EFF):
    """Projected dsigma/dq in GeV^-3, dispatching on the mediator range."""
    if xs["lamb"] is None:
        if xs.get("b_constrained_max") is not None:
            return cross_section.cross_section_rutherford_projection_capped(
                q, alpha, vs, xs["b_constrained_max"])
        return cross_section.cross_section_rutherford_projection(q, alpha, vs)
    # For finite lamb the cap is already baked into xs["interp"].
    fn = cross_section.dsigma_dq_ln if xs["use_ln"] else cross_section.dsigma_dq
    return fn(q, alpha, xs["lamb"], R_eff, vs, xs["interp"])


def impact_parameter_max_any(q, alpha, vs, xs, R_eff=R_EFF):
    """Threshold reach b_max [m], dispatching on the mediator range.

    The b_constrained_max cap (when set on ``xs``) clips the reach to
    min(b_max, b_constrained_max) — the same geometric cutoff the cap applies to
    the cross section — so the transit diagnostics (expected_transits n_transit,
    transit_count_halo nt/bmax) stay consistent with dsigma/dq.
    """
    if xs["lamb"] is None:
        b = 2 * alpha / (q * vs) / units.conv_m2pGeV(1.0)  # Coulomb
    else:
        fn = (cross_section.impact_parameter_max_ln if xs["use_ln"]
              else cross_section.impact_parameter_max)
        b = fn(q, alpha, xs["lamb"], R_eff, vs)
    cap = xs.get("b_constrained_max")
    return b if cap is None else np.minimum(b, cap)


def differential_rate_trapz(qs, alpha_n, mu, f_v_f, xs, R_eff=R_EFF, eff=None):
    """dR/dq in s^-1 GeV^-1 via trapz (mu = m_DM in the original notation).

    The rate carries the DM-fraction scaling f_X = config.F_X: this species
    makes up only that fraction of the local density.

    ``eff`` optionally folds in the detection efficiency: a callable
    epsilon(q_GeV) in [0, 1] (see :mod:`luhdm.efficiency`) applied to the
    returned dR/dq, so downstream the optimum interval sees the *detected*
    momentum-kick rate. ``eff=None`` (default) is the raw rate, byte-identical
    to the pre-efficiency pipeline. Efficiency applies ONLY here, not to the
    geometric transit diagnostics.
    """
    alpha = alpha_n * config.N_NEUTRONS
    n_dm = config.F_X * halo.number_density_dm(mu)

    # Precompute f_vf on a fixed grid spanning all v_mins
    v_min_global = qs.min() / mu
    vs_global = np.geomspace(v_min_global, config.VESC, 500)
    f_vf_grid = f_v_f(vs_global)  # KDE evaluated once

    results = []
    for q in qs:
        mask = vs_global >= q / mu
        vs = vs_global[mask]
        integrand = n_dm * f_vf_grid[mask] * vs * dsigma_dq_any(
            q, alpha, vs, xs, R_eff)
        results.append(np.trapezoid(integrand, vs) * units.CONV2RATE)

    out = np.array(results)
    if eff is not None:
        out = out * np.asarray(eff(qs), dtype=float)  # detected rate = eps(q)*dR/dq
    return np.maximum(out, 0)


def expected_transits(alpha_n, mu, f_v_f, xs, t_total, R_eff=R_EFF):
    """Expected flybys within the threshold reach during the exposure."""
    alpha = alpha_n * config.N_NEUTRONS
    vs = np.geomspace(max(Q_THRESH / mu, 1e-8), config.VESC, 300)
    b = impact_parameter_max_any(Q_THRESH, alpha, vs, xs, R_eff)
    n_m3 = config.F_X * 0.3 / mu * 1e6  # f_X x (0.3 GeV/cm^3) -> 1/m^3
    return t_total * float(np.trapezoid(
        f_v_f(vs) * n_m3 * (vs * units.C_M_S) * np.pi * b**2, vs))


def transit_count_halo(m, alpha_n, xs, t_total, R_eff=R_EFF):
    """(N_t, flux-averaged pi*b_max^2 [m^2]) for the unattenuated halo flux."""
    vs = np.geomspace(max(Q_THRESH / m, 1e-8), config.VESC, 200)
    if vs.size < 2 or vs[0] >= config.VESC:
        return 0.0, 0.0
    alpha = alpha_n * config.N_NEUTRONS
    b = impact_parameter_max_any(Q_THRESH, alpha, vs, xs, R_eff)
    flux_w = halo.standard_halo_model(vs) * (vs * units.C_M_S)
    area = np.pi * b**2
    n_m3 = config.F_X * 0.3 / m * 1e6
    nt = t_total * n_m3 * np.trapezoid(flux_w * area, vs)
    a_eff = np.trapezoid(flux_w * area, vs) / np.trapezoid(flux_w, vs)
    return nt, a_eff


def transit_maps(ms_map, alphas_map, xs, t_total, R_eff=R_EFF):
    """N_t and flux-averaged reach grids over (mass, coupling).

    Returns (NT, B) with shape (alphas, masses); B is sqrt(<pi b^2>/pi) in m.
    """
    out = np.array([[transit_count_halo(m, a, xs, t_total, R_eff)
                     for m in ms_map] for a in alphas_map])
    return out[:, :, 0], np.sqrt(out[:, :, 1] / np.pi)
