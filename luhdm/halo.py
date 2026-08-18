"""Module containing cross section class and functions."""

import numpy as np
from scipy import special
from luhdm import config


def maxwell_boltzmann_tilde(v_tilde):
    """Return MB halo velocity distribution."""

    norm = 4 / np.sqrt(np.pi)

    return norm * v_tilde**2 * np.exp(-v_tilde**2)


def maxwell_boltzmann(v):
    """Return MB halo velocity distribution."""

    norm = 4 * np.pi / (np.pi * config.V0**2)**1.5

    return norm * v**2 * np.exp(-v**2 / config.V0**2)


def _shm_galactic_rest_frame(v):
    """Truncated Maxwellian in the Galactic rest frame (no Earth motion)."""

    sigma_v = config.V0 / np.sqrt(2)

    n_esc = special.erf(config.VESC / config.V0) - \
            np.sqrt(2 / np.pi) * config.VESC / sigma_v * np.exp(-config.VESC**2 / (2 * sigma_v**2))

    return 4 * np.pi * v**2 * (1 / ((2 * np.pi * sigma_v**2)**1.5 * n_esc)) * np.exp(-v**2 / (2 * sigma_v**2)) * np.heaviside(config.VESC - v, 0)


def standard_halo_model(v, v_E=None):
    """Return the SHM speed distribution f(v), normalised on [0, VESC + v_E].

    ``v_E`` is the observer's (Earth's) speed through the halo in natural units;
    ``None`` takes config.V_E, which is 0 unless LUHDM_V_EARTH says otherwise.

    v_E = 0 is the Galactic rest frame: the truncated Maxwellian this module has
    always returned, evaluated by the very same expression so the default
    pipeline is bit-identical.

    v_E > 0 is the LAB frame -- the isotropic Maxwellian boosted into the
    detector frame and integrated over arrival direction (Lewin & Smith 1996;
    the form used by Monteiro 2020 arXiv:2007.12067 and Tseng 2025
    arXiv:2508.00815 Eq. "shm_f_v", the two results this analysis overlays):

        f(v) = (pi v0^2 / (N0 v_E)) v [ exp(-(v - v_E)^2 / v0^2)
                                        - exp(-w_+^2 / v0^2) ],
        w_+   = min(v + v_E, VESC),      f(v > VESC + v_E) = 0,
        N0    = (pi v0^2)^{3/2} [ erf(x) - (2/sqrt(pi)) x exp(-x^2) ],  x = VESC/v0.

    The min() collapses Lewin & Smith's two non-zero branches into one
    expression: below v = VESC - v_E the full boosted shell contributes, above
    it the shell is clipped by the escape speed, and the two agree at the seam.
    The support ends at VESC + v_E, where the bracket vanishes continuously.

    As v_E -> 0 the bracket -> exp(-v^2/v0^2) 4 v v_E / v0^2 and the 1/v_E
    prefactor cancels, recovering the rest-frame form exactly -- but the
    expression is 0/0 there, so v_E = 0 is dispatched to it explicitly rather
    than approached numerically.
    """
    if v_E is None:
        v_E = config.V_E
    if v_E == 0.0:
        return _shm_galactic_rest_frame(v)

    v = np.asarray(v, dtype=float)
    v0 = config.V0
    x_esc = config.VESC / v0
    n0 = (np.pi * v0**2)**1.5 * (
        special.erf(x_esc) - 2 / np.sqrt(np.pi) * x_esc * np.exp(-x_esc**2))

    w_plus = np.minimum(v + v_E, config.VESC)
    bracket = np.exp(-(v - v_E)**2 / v0**2) - np.exp(-w_plus**2 / v0**2)
    f = np.pi * v0**2 / (n0 * v_E) * v * bracket

    # Beyond VESC + v_E no halo particle can reach the detector; the bracket
    # would go negative there, so the support is cut explicitly.
    return np.where(v < config.VESC + v_E, f, 0.0)


def number_density_dm(m_dm):
    """Return DM number GeV^3 with m_dm in GeV."""
    
    return config.RHO_DM / m_dm
