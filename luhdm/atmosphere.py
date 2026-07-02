"""Module containing atmospheric attenuation functions."""

import numpy as np

from scipy import constants as sconstants
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde

from luhdm import config
from luhdm import halo

N0 = 2.4e25  # Number density at sea-level
H = 8.5e3  # Characteristic decay length
U_GeV = 0.9315  # Atomic mass unit in GeV
N_neu_mol = 2 * 7  # Number of neutrons in N2
N_nuc_mol = 2 * 14  # Number of nucleons in N2
M_MOL = N_nuc_mol * U_GeV

# ─── Utilities ───
def conv_m2pGeV(meter):
    return 5.0679e6 * meter * 1e9

# ─── Atmosphere Model ───
def compute_n_atm(z):
    """Return atmospheric decay number density."""
    
    return N0 * np.exp(-z / H)
    
# ─── Model Functions ───
def compute_beta(alpha_n, lamb, m_mol, v):
    """Return the coupling strength parameter."""
    alpha = alpha_n * N_neu_mol 
    return alpha / (conv_m2pGeV(lamb) * m_mol * v**2)

def compute_sigma_T(alpha_n, lamb, m_mol, v):
    """Return transverse momentum cross section."""
    beta = compute_beta(alpha_n, lamb, m_mol, v)
    alpha = alpha_n * N_neu_mol
    return 2 * np.pi * (alpha / (m_mol * v**2))**2 * np.log(1 + beta**(-2))

# ─── ODE ───
def dvdz(z, v, alpha_n, lamb, m_dm, m_mol):
    """Return ODE."""

    sigma_T = compute_sigma_T(alpha_n, lamb, m_mol, v[0]/sconstants.c) / conv_m2pGeV(1)**2
    n_atm = compute_n_atm(z)

    return m_mol / m_dm * n_atm * sigma_T * (v[0])


def solve_ode(v_i, alpha_n, lamb, m_dm, m_mol, v_min, h_max=None, z_eval=None): 
    """
    Integrate the ODE from z=H (top of atmosphere) to z=0 (ground).
 
    Parameters
    ----------
    v_i       : initial DM speed at top of atmosphere
    alpha_mol : effective molecular coupling  = N_n * alpha_n
    mu_r      : reduced mass of DM-molecule system
    M_X       : DM mass
    lam       : mediator range (take large for massless mediator)
    H         : atmosphere height to integrate from (default 5*h)
    v_min     : detector threshold velocity; integration stops if v drops below
    n_points  : number of z points for output
 
    Returns
    -------
    z   : array of heights
    v   : array of speeds
    v_f : final speed at ground (or v_min if terminated early)
    """

    if h_max is None:
        h_max = 10 * H
 
    # Termination event: particle slows below v_min
    def below_threshold(z, v, *args):
        return (v[0] - v_min)
    
    below_threshold.terminal  = True
    below_threshold.direction = -1
 
    sol = solve_ivp(
        dvdz,
        t_span=(h_max, 0),
        y0=[v_i*sconstants.c],
        args=(alpha_n, lamb, m_dm, m_mol),
        events=below_threshold,
        t_eval=z_eval,
        method='RK45',
        rtol=1e-8,
        atol=1e-12,
    )
 
    z = sol.t
    v = sol.y[0]
    v_f = v[-1]
 
    return z, v, v_f


def compute_v_f_interpolant(v_is, alpha_n, lamb, m_dm, v_min):
    """Return v_i to v_f mapping interpolant."""
    v_fs = []
    for v_i in v_is:
        _, _, v_f = solve_ode(v_i, alpha_n, lamb, m_dm, M_MOL, v_min*sconstants.c, h_max = 20*H)
        v_fs.append(v_f / sconstants.c)

    return interp1d(v_is, v_fs, bounds_error=False, fill_value=0.)


def sample_shm(n_samples):
    """Sample speeds from the standard halo model via rejection sampling."""
    
    # f(v) <= M * g(v) where g is the unnormalised MB
    # Find M = max(f(v)/g(v)) numerically
    v_test = np.linspace(0, config.VESC, 10000)
    f_vals = halo.standard_halo_model(v_test)
    M = np.max(f_vals) * 1.1  # small buffer
    
    samples = []
    while len(samples) < n_samples:
        # Propose from uniform on [0, v_esc]
        v_prop = np.random.uniform(0, config.VESC, n_samples)
        f_prop = halo.standard_halo_model(v_prop)
        
        # Accept/reject
        u = np.random.uniform(0, M, n_samples)
        accepted = v_prop[u < f_prop]
        samples.extend(accepted)
    
    return np.array(samples[:n_samples])

    
def compute_v_f_distribution(alpha_n, lamb, m_dm, v_i_samples, v_min=1e-7, n_samples=int(1e6), n_grid=500):
    """
    Sample v_i from the SHM and push each through the atmospheric ODE.
    
    Parameters
    ----------
    alpha     : effective molecular coupling (= N_n * alpha_n)
    lamb      : mediator range [GeV^-1]
    m_dm      : DM mass [GeV]
    m_mol     : molecule mass [GeV]
    n_samples : number of MC samples
    
    Returns
    -------
    v_f_samples : array of final velocities at ground level [dimensionless, v/c]
    """
    v_i_grid = np.geomspace(v_i_samples.min(), config.VESC*1.1, n_grid)
    
    interpolant = compute_v_f_interpolant(v_i_grid, alpha_n, lamb, m_dm, v_min)
    v_f_samples = interpolant(v_i_samples)
    
    return v_f_samples


def compute_f_vf(v_f_samples, v_floor=1e-7):
    """
    Build the attenuated velocity distribution from ODE output samples.
    
    Parameters
    ----------
    v_f_samples : array of final velocities [dimensionless, v/c]
    v_floor     : numerical floor below which particles are considered stopped
    
    Returns
    -------
    f_vf       : callable, attenuated distribution evaluated at v
    F_survive  : survival fraction
    """
# Split into stopped and surviving
    stopped = v_f_samples[v_f_samples <= v_floor]
    surviving = v_f_samples[v_f_samples > v_floor]
    F_survive = len(surviving) / len(v_f_samples)
    
    # Edge cases: all particles stopped or only one survived
    if len(surviving) < 2:
        def f_v_f(v):
            return np.zeros_like(np.atleast_1d(np.asarray(v, dtype=float)))
        return f_v_f, F_survive
        
    kde = gaussian_kde(surviving)
    
    def f_v_f(v):
        v = np.atleast_1d(np.asarray(v, dtype=float))
        result = np.zeros_like(v)
        above = v > v_floor
        result[above] = F_survive * kde(v[above])
        return result
    
    return f_v_f, F_survive
    
    