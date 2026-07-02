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


def standard_halo_model(v):
    """Return SHM halo velocity distribution from SHM++, for example."""

    sigma_v = config.V0 / np.sqrt(2)
    
    n_esc = special.erf(config.VESC / config.V0) - \
            np.sqrt(2 / np.pi) * config.VESC / sigma_v * np.exp(-config.VESC**2 / (2 * sigma_v**2))

    return 4 * np.pi * v**2 * (1 / ((2 * np.pi * sigma_v**2)**1.5 * n_esc)) * np.exp(-v**2 / (2 * sigma_v**2)) * np.heaviside(config.VESC - v, 0)

def number_density_dm(m_dm):
    """Return DM number GeV^3 with m_dm in GeV."""
    
    return config.RHO_DM / m_dm
