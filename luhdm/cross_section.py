"""Module containing cross section class and functions."""

import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from tqdm import tqdm

# ============================================================================
# Dimensionless Mappings
# ============================================================================


def epsilon_map(alpha, lamb, energy):
    """Map from physical variables to dimensionless epsilon."""

    return lamb * energy / alpha


def rho_map(r, lamb):
    """Map from physical variables to dimensionless rho."""

    return r / lamb


def beta_map(b, lamb):
    """Map from physical variables to dimensionless beta."""

    return b / lamb


def q_tilde2q(q_tilde, p):
    """Return dimensionful momentum."""
    
    return q_tilde * 2 * p


# ============================================================================
# Orbital Angle
# ============================================================================

def orbital_objective(rho, beta, epsilon):
    """Objective function for orbital root finding."""
    return 1 - beta**2 / rho**2 - 1 / epsilon * np.exp(-rho) / rho


def solve_orbital_objective(beta, epsilon):
    """Solve the orbital objective function for given beta and epsilon."""
    def func(rho):
        return orbital_objective(rho, beta, epsilon)

    rho_solution = fsolve(func, beta)

    return rho_solution[0]


def integrand_radical(beta, rho_min, epsilon, u):
    """Radical in integrand"""
    radical = 1 - beta**2 / rho_min**2 * u**2 - \
        1 / epsilon * np.exp(-rho_min / u) / rho_min * u
    return radical


def orbital_integrand(u, beta, epsilon):
    """Integrand for the orbital angle calculation."""

    rho_min = solve_orbital_objective(beta, epsilon)
    radical = integrand_radical(beta, rho_min, epsilon, u)
    return radical**(-0.5)


def orbital_angle(beta, epsilon):
    """Calculate the orbital angle for given beta and epsilon."""

    rho_min = solve_orbital_objective(beta, epsilon)
    integral, _ = quad(orbital_integrand, 0, 1, args=(beta, epsilon))

    return np.pi - 2 * beta / rho_min * integral


# ============================================================================
# Cross Section
# ============================================================================ 
          
class YukawaPointCrossSection:
    """Dimensionless cross section for a Yukawa point-like potential
    at fixed epsilon.
    """

    def __init__(self,
                 epsilon,
                 beta_min=0.0001,
                 beta_max=10.0,
                 n_points=500):

        self.epsilon = epsilon

        self.beta_grid = np.geomspace(beta_min, beta_max, n_points)

        print("Calculating orbital angles for beta grid...")
        self.theta_grid = np.array(
            [orbital_angle(beta, self.epsilon) for beta in 
             tqdm(self.beta_grid)])

        if not np.all(np.diff(self.theta_grid) < 0):
            raise ValueError("theta_grid is not strictly decreasing; \
                cannot safely invert.")

        self.theta_of_beta_spline = \
            CubicSpline(self.beta_grid, self.theta_grid)
        self.beta_of_theta_spline = \
            CubicSpline(self.theta_grid[::-1], self.beta_grid[::-1])

        self.deriv_theta_beta_spline = \
            self.theta_of_beta_spline.derivative(1)

    def diff_cross_section_omega_dimensionless(self, theta):
        beta = self.beta_of_theta_spline(theta)
        return beta / np.sin(theta) / \
            np.abs(self.deriv_theta_beta_spline(beta))

    def diff_cross_section_q_dimensionless(self, q):
        """Return differential scattering cross section in solid angle."""
        theta = self.theta_of_momentum_transfer(q)
        prefactor = 2 * np.pi * q / self.momentum**2
        return prefactor * self.diff_cross_section_omega_dimensionless(theta)

 
class YukawaPointCrossSectionPhysical:
    """Class representing the cross section for
    a Yukawa point-like potential.
    """

    def __init__(self,
                 alpha,
                 lamb,
                 momentum,
                 reduced_mass,
                 beta_min=None,
                 beta_max=10.0,
                 n_points=1000):
        
        self.alpha = alpha
        self.lamb = lamb
        self.momentum = momentum
        self.reduced_mass = reduced_mass
        self._energy = self.momentum**2 / (2 * self.reduced_mass)
        self._epsilon = epsilon_map(self.alpha, self.lamb, self._energy)
        
        # Automatic beta_min: scale with 1/epsilon to capture backscattering
        if beta_min is None:
            beta_min = min(0.001, 0.01 / self._epsilon)  # ~0.1/epsilon
            print(f"  Auto beta_min: {beta_min:.4f} (based on \
                ε={self._epsilon:.2f})")
            
        self.beta_grid = np.geomspace(beta_min, beta_max, n_points)
        
        print("Calculating orbital angles for beta grid...")
        self.theta_grid = np.array(
            [orbital_angle(beta, self._epsilon) for beta in tqdm(
                self.beta_grid)])

        self.theta_of_beta_spline = CubicSpline(self.beta_grid,
                                                self.theta_grid)

        self.beta_of_theta_spline = CubicSpline(self.theta_grid[::-1],
                                                self.beta_grid[::-1])

        self.deriv_theta_beta_spline = \
            self.theta_of_beta_spline.derivative(1)
        self.deriv_beta_of_theta_spline = \
            self.beta_of_theta_spline.derivative(1)

        print(f"  θ range: [{self.theta_grid.min():.3f}, \
              {self.theta_grid.max():.3f}] rad")
        print(f"  β range: [{self.beta_grid.min():.3f}, \
            {self.beta_grid.max():.3f}]")
        print("Computation complete.\n")
    
    def momentum_transfer_of_theta(self, theta):
        """Return momentum transfer as a function of orbital angle."""
        return 2 * self.momentum * np.sin(theta/2)
    
    def theta_of_momentum_transfer(self, q):
        """Return orital angle as a function of momentum transfer."""
        return 2 * np.arcsin(q / (2 * self.momentum))

    def diff_cross_section_omega_dimensionless(self, theta):
        """Return dimensionless differential scattering cross section 
        in solid angle."""
        beta = self.beta_of_theta_spline(theta)
        return beta / (np.sin(theta)) * (
            np.abs(self.deriv_beta_of_theta_spline(theta)))

    def diff_cross_section_omega(self, theta):
        """Return differential scattering cross section in solid angle."""
        return self.lamb**2 * \
            self.diff_cross_section_omega_dimensionless(theta)

    def diff_cross_section_q_dimensionless(self, q):
        """Return differential scattering cross section in solid angle."""
        theta = self.theta_of_momentum_transfer(q)
        prefactor = 2 * np.pi * q / self.momentum**2
        return prefactor * self.diff_cross_section_omega_dimensionless(theta)
   
    def diff_cross_section_q(self, q):
        """Return differential scattering cross section in solid angle."""
        return self.lamb**2 * self.diff_cross_section_q_dimensionless(q)
     

# ============================================================================
# Straight-Line Impulse (K1) Cross Section
# ============================================================================
# Machinery from the limit notebook (12_limit_atmos): impulse along a
# straight-line trajectory, q(b) = 2 alpha / (lamb v) * G2(R_eff/lamb)
# * K1(b/lamb), with the measured q the projection on the sensitive axis.

from scipy.interpolate import interp1d
from scipy.special import kn

from luhdm import units


def shape_factor(x):
    """Finite-size sensor form factor G2(x)."""
    x = np.asarray(x, dtype=float)
    small = x < 1e-6
    x_safe = np.where(small, 1.0, x)  # avoid division by zero
    full = 3.0 * (x_safe * np.cosh(x_safe) - np.sinh(x_safe)) / x_safe**3
    taylor = 1.0 + x**2 / 10.0
    return np.where(small, taylor, full)


def q_analytical(alpha, lamb, b, R_eff, v):
    """Impulse at impact parameter b (all lengths in meters, q in GeV)."""
    return 2 * alpha / (units.conv_m2pGeV(lamb) * v) * \
        shape_factor(R_eff / lamb) * kn(1, b / lamb)


# K1 inverse, built once at import (as in the notebook)
_xs = np.geomspace(1e-3, 500, 10000)
_scipy_interpolant_k1_inverse = interp1d(kn(1, _xs), _xs, kind='linear')


def interpolant_k1_inverse(k1_values):
    """Invert K1: return beta such that K1(beta) = k1_values."""
    k1_values = np.asarray(k1_values, dtype=float)
    conditions = [k1_values > 100., k1_values <= 100.]
    fns = [lambda k1: 1 / k1,
           lambda k1: _scipy_interpolant_k1_inverse(k1)]
    return np.piecewise(k1_values, conditions, fns)


def cross_section_rutherford_projection(q, alpha, v):
    """Projected massless-mediator (Coulomb) limit: dsigma/dq = 2 pi a^2/(v^2 q^3).

    This is the lamb -> infinity limit of the K1 machinery below (K1(x) -> 1/x
    turns the arccosh integral into pi/4, collapsing dsigma/dq_tilde to
    (pi/4)/q_tilde^3).
    """
    dsigma_rutherford = 2 * np.pi * alpha**2 / (v**2 * q**3)
    return dsigma_rutherford


def q_tilde_map(q, alpha, lamb, R_eff, v):
    """Map physical momentum transfer to dimensionless q_tilde."""
    F_factor = shape_factor(R_eff / lamb)
    return q * units.conv_m2pGeV(lamb) * v / (2 * alpha * F_factor)


def impact_parameter_max(q, alpha, lamb, R_eff, v):
    """Largest impact parameter [m] whose flyby delivers impulse |q(b)| >= q.

    Inverts q(b) = 2 alpha G2 K1(b/lamb) / (lamb v): b_max = lamb *
    K1^-1(q_tilde). Returns 0 where the momentum-transfer cap makes q
    unreachable (q_tilde >= K1(R_eff/lamb)). In the massless limit this tends
    to the Coulomb reach b_max = 2 alpha G2 / (q v) (in natural units).
    """
    q_tilde = q_tilde_map(q, alpha, lamb, R_eff, v)
    q_tilde = np.atleast_1d(np.asarray(q_tilde, dtype=float))
    q_tilde_max = kn(1, R_eff / lamb)
    b = np.zeros_like(q_tilde)
    mask = q_tilde < q_tilde_max
    if np.any(mask):
        b[mask] = lamb * interpolant_k1_inverse(q_tilde[mask])
    return b


def dsigma_dq_tilde(q_tilde, xi, K1_inv):
    """Dimensionless projected differential cross section."""
    K1_xi = kn(1, xi)
    if q_tilde >= K1_xi:
        return 0.0
    t_max = np.arccosh(K1_xi / q_tilde)

    def integrand(t):
        kappa = q_tilde * np.cosh(t)
        beta = K1_inv(kappa)
        dK1_dbeta = -0.5 * (kn(0, beta) + kn(2, beta))
        return beta / np.abs(dK1_dbeta)

    result, _ = quad(integrand, 0, t_max, limit=100)
    return result


def make_dsigma_dq_interpolant(q_tilde_min,
                               R_eff,
                               lamb,
                               N_points=1000):
    """Tabulate dsigma/dq_tilde once; returns a linear interpolant."""
    xi = R_eff / lamb
    q_tilde_max = kn(1, xi)
    q_tilde_values = np.geomspace(q_tilde_min, q_tilde_max, N_points)
    K1_inv = interpolant_k1_inverse
    dsigma_values = np.array(
        [dsigma_dq_tilde(q_tilde, xi, K1_inv)
         for q_tilde in tqdm(q_tilde_values)])
    return interp1d(q_tilde_values, dsigma_values, kind='linear')


def dsigma_interpolant(q_tilde, R_eff, lamb, interpolant):
    """Evaluate the tabulated dsigma/dq_tilde, zero beyond the endpoint."""
    q_tilde = np.atleast_1d(np.asarray(q_tilde, dtype=float))
    q_tilde_max = kn(1, R_eff / lamb)

    result = np.zeros_like(q_tilde)
    mask = q_tilde < q_tilde_max
    result[mask] = interpolant(q_tilde[mask])

    return result


def dsigma_dq(q, alpha, lamb, R_eff, v, interpolant):
    """Physical dsigma/dq in GeV^-3 (q in GeV, lengths in m, v in c units)."""
    q_tilde = q_tilde_map(q, alpha, lamb, R_eff, v)
    prefactor = units.conv_m2pGeV(lamb)**3 * v / \
        (alpha * shape_factor(R_eff / lamb))
    return dsigma_interpolant(q_tilde, R_eff, lamb, interpolant) * prefactor
