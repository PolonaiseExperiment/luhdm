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

from luhdm import config, units

R_EFF = config.R_EFF


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


def coulomb_reach(q, alpha, v):
    """Massless (Coulomb) impact parameter [m] delivering the impulse q.

    b_max(q) = 2 alpha / (q v) / conv_m2pGeV(1): the largest b whose flyby still
    transfers |q| (exactly the massless branch of ``impact_parameter_max_any``).
    """
    return 2 * alpha / (q * v) / units.conv_m2pGeV(1.0)


def coulomb_q_max(alpha, v, R_eff=R_EFF):
    """Largest momentum transfer [GeV] a massless-mediator flyby can deliver.

    No trajectory approaches closer than the sensor radius, so the b-integral
    starts at b = R_eff and the impulse saturates at

        q_max = 2 alpha / (v R_eff) / conv_m2pGeV(1)   [GeV]

    (the lamb -> infinity limit of the finite-range cutoff q_tilde < K1(R_eff/lamb),
    since K1(x) -> 1/x and G2(x) -> 1). dsigma/dq vanishes identically above it.
    """
    return 2 * alpha / (v * units.conv_m2pGeV(R_eff))


# ── projection-kernel convention ──
# "planar-signed" is the shipped kernel: the signed-projection density under a
# planar-arrival geometry (coefficient 2 pi, arcsine shell fraction).
# "isotropic-folded" is the absolute one-axis projection under the pipeline's
# isotropic-arrival model (coefficient 8 pi / 3, shell fraction x^3; for finite
# lambda the dimensionless integrand becomes pi * int beta dbeta / K1(beta)).
# The default reproduces the shipped pipeline byte-for-byte; the choice is a
# recorded convention (see the projection-kernel verdict memo), not a fit.
KERNEL_PLANAR_SIGNED = "planar-signed"
KERNEL_ISOTROPIC_FOLDED = "isotropic-folded"
KERNEL_DEFAULT = KERNEL_PLANAR_SIGNED
_KERNELS = (KERNEL_PLANAR_SIGNED, KERNEL_ISOTROPIC_FOLDED)


def _check_kernel(kernel):
    if kernel not in _KERNELS:
        raise ValueError(f"unknown projection kernel {kernel!r}; "
                         f"expected one of {_KERNELS}")
    return kernel


def rutherford_shell_fraction(r):
    """Fraction of the uncapped Coulomb projection contributed by b <= b_outer.

    ``r = b_max(q) / b_outer``. Writing x = 1/r = b_outer/b_max, the projected
    b-integral over the disc b <= b_outer is the uncapped result times

        F(x) = (2/pi) (arcsin x - x sqrt(1 - x^2))     for x < 1  (r > 1)
        F    = 1                                       for x >= 1 (r <= 1),

    i.e. the shell already contains the whole reach. F(0) = 0, F(1) = 1, and F is
    monotonically increasing in x. Equivalent to the previous
    ``1 - (2/pi)(sqrt(r^2-1)/r^2 + arctan sqrt(r^2-1))`` retained fraction, but
    evaluated in the form that keeps full precision for x << 1 (small discs),
    where the series F = (4/(3 pi)) x^3 (1 + 3x^2/10 + 9x^4/56 + ...) is used.
    Vectorized; r may be 0 (b_outer -> infinity, F = 1) or inf (b_outer = 0, F = 0).
    """
    r = np.asarray(r, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(r > 1.0, 1.0 / r, 1.0)      # x = b_outer/b_max, clipped to 1
    x = np.where(np.isfinite(x), x, 0.0)         # r = inf (b_outer = 0) -> x = 0
    small = x < 1e-2
    x_s = np.where(small, x, 0.5)                # dummy where unused
    x_d = np.where(small, 0.5, x)
    series = (4.0 / (3.0 * np.pi)) * x_s**3 * (
        1.0 + 0.3 * x_s**2 + (9.0 / 56.0) * x_s**4)
    direct = (2.0 / np.pi) * (
        np.arcsin(x_d) - x_d * np.sqrt(np.maximum(1.0 - x_d * x_d, 0.0)))
    return np.where(small, series, direct)


def rutherford_shell_fraction_iso(r):
    """Isotropic-folded analogue of :func:`rutherford_shell_fraction`.

    Under the absolute one-axis projection with isotropic arrivals the b <= b_outer
    disc contributes the fraction F(x) = x^3 of the uncapped result
    (x = b_outer/b_max clipped to 1): dsigma/d|q| = (pi v/(3 alpha))
    (b_out^3 - b_in^3) in natural units — closed form, no special functions.
    Same edge conventions as the planar-signed version: r may be 0
    (b_outer -> infinity, F = 1) or inf (b_outer = 0, F = 0).
    """
    r = np.asarray(r, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(r > 1.0, 1.0 / r, 1.0)      # x = b_outer/b_max, clipped
    x = np.where(np.isfinite(x), x, 0.0)         # r = inf (b_outer = 0) -> 0
    return x ** 3


def cross_section_rutherford_projection_capped(q, alpha, v, b_constrained_max,
                                               R_eff=R_EFF,
                                               kernel=KERNEL_DEFAULT):
    """Massless-mediator (Coulomb) projected dsigma/dq in GeV^-3.

    The impact-parameter integral runs over the ANNULUS
    ``R_eff <= b <= b_constrained_max``:

    * the inner edge is the sensor radius — a flyby cannot approach closer, so
      the impulse saturates at ``q_max = coulomb_q_max(alpha, v, R_eff)`` and
      dsigma/dq is identically zero above it. This is the massless counterpart of
      the ``q_tilde < K1(R_eff/lamb)`` cutoff the finite-range path applies;
    * the outer edge ``b_constrained_max`` (metres, or None for no outer cap)
      removes flybys reaching beyond the cap.

    Both edges enter through :func:`rutherford_shell_fraction`, the fraction of
    the uncapped Coulomb projection ``2 pi alpha^2/(v^2 q^3)`` coming from
    b <= b_outer, so

        dsigma/dq = uncapped(q) * [F(b_max/b_cap) - F(b_max/R_eff)] .

    Above q_max both shells equal the full reach (F = 1) and the difference is
    exactly 0; ``R_eff = 0`` recovers the pure disc integral (no inner cutoff).
    Vectorized over q and v.
    """
    if R_eff is None:
        R_eff = 0.0
    if b_constrained_max is not None and b_constrained_max < R_eff:
        raise ValueError(
            f"b_constrained_max ({b_constrained_max} m) below sensor "
            f"radius R_eff ({R_eff} m)")
    if _check_kernel(kernel) == KERNEL_ISOTROPIC_FOLDED:
        uncapped = (8.0 * np.pi / 3.0) * alpha**2 / (v**2 * q**3)
        frac = rutherford_shell_fraction_iso
    else:
        uncapped = 2 * np.pi * alpha**2 / (v**2 * q**3)
        frac = rutherford_shell_fraction
    b_max = coulomb_reach(q, alpha, v)
    outer = (1.0 if b_constrained_max is None
             else frac(b_max / b_constrained_max))
    inner = (0.0 if R_eff == 0.0
             else frac(b_max / R_eff))
    return np.maximum(uncapped * (outer - inner), 0.0)


def cross_section_rutherford_projection(q, alpha, v, R_eff=R_EFF):
    """Massless-mediator (Coulomb) projected dsigma/dq in GeV^-3, no outer cap.

    Shorthand for :func:`cross_section_rutherford_projection_capped` with
    ``b_constrained_max=None``: the b-integral still starts at the sensor radius
    ``R_eff``, so dsigma/dq vanishes above ``q_max = coulomb_q_max(alpha, v,
    R_eff)``. Only with ``R_eff = 0`` does this reduce to the bare
    ``2 pi alpha^2/(v^2 q^3)`` power law (the lamb -> infinity limit of the K1
    machinery below, whose arccosh integral collapses to pi/4).
    """
    return cross_section_rutherford_projection_capped(q, alpha, v, None,
                                                      R_eff=R_eff)


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


def dsigma_dq_tilde(q_tilde, xi, K1_inv, xi_cap=None, kernel=KERNEL_DEFAULT):
    """Dimensionless projected differential cross section.

    ``xi_cap`` (= b_constrained_max/lamb) optionally caps the impact parameter:
    the outer b-limit becomes min(b_constrained_max, b_max(q)). In the hyperbolic
    substitution t=0 is the OUTER edge b=b_max(q) and t=t_max is the INNER edge
    b=R_eff, so the cap raises the LOWER limit to t_min and only bites when
    b_constrained_max < b_max(q), i.e. q_tilde < K1(xi_cap). ``xi_cap=None``
    (default) leaves the integral from t=0 unchanged.

    ``kernel``: "planar-signed" (default, the shipped arccosh machinery) or
    "isotropic-folded": dsigma/dq_tilde = pi * int beta dbeta / K1(beta) over
    beta in [xi, min(K1^-1(q_tilde), xi_cap)] — the absolute one-axis
    projection under isotropic arrivals, same prefactor chain in dsigma_dq.
    """
    if xi_cap is not None and xi_cap < xi:
        raise ValueError(
            f"xi_cap ({xi_cap}) < xi ({xi}): cap below the inner cutoff")
    K1_xi = kn(1, xi)
    if q_tilde >= K1_xi:
        return 0.0

    if _check_kernel(kernel) == KERNEL_ISOTROPIC_FOLDED:
        beta_hi = float(np.atleast_1d(K1_inv(q_tilde))[0])
        if xi_cap is not None:
            beta_hi = min(beta_hi, xi_cap)
        if beta_hi <= xi:
            return 0.0
        # integrand beta/K1(beta) grows ~ e^beta: carry it in logs on a dense
        # grid (kve = K1 e^beta) and trapz; deterministic and float-safe out to
        # the K1-underflow range the interpolants cover.
        bs = np.linspace(xi, beta_hi, 20001)
        ln_integrand = np.log(bs) + bs - np.log(kve(1, bs))
        peak = ln_integrand.max()
        return float(np.pi * np.exp(peak)
                     * np.trapezoid(np.exp(ln_integrand - peak), bs))

    t_max = np.arccosh(K1_xi / q_tilde)

    t_min = 0.0
    if xi_cap is not None:
        K1_cap = kn(1, xi_cap)
        if q_tilde < K1_cap:            # cap inside the reach: b_c < b_max(q)
            t_min = np.arccosh(K1_cap / q_tilde)

    def integrand(t):
        kappa = q_tilde * np.cosh(t)
        beta = K1_inv(kappa)
        dK1_dbeta = -0.5 * (kn(0, beta) + kn(2, beta))
        return beta / np.abs(dK1_dbeta)

    result, _ = quad(integrand, t_min, t_max, limit=100)
    return result


def make_dsigma_dq_interpolant(q_tilde_min,
                               R_eff,
                               lamb,
                               N_points=1000,
                               b_constrained_max=None,
                               kernel=KERNEL_DEFAULT):
    """Tabulate dsigma/dq_tilde once; returns a linear interpolant.

    ``b_constrained_max`` (metres, or None) optionally caps the outer edge of
    the impact-parameter integral at min(b_constrained_max, b_max(q)); None is a
    byte-for-byte no-op. ``kernel`` selects the projection convention (see
    :func:`dsigma_dq_tilde`); the default reproduces the shipped tabulation.
    """
    xi = R_eff / lamb
    q_tilde_max = kn(1, xi)
    q_tilde_values = np.geomspace(q_tilde_min, q_tilde_max, N_points)
    if _check_kernel(kernel) == KERNEL_ISOTROPIC_FOLDED:
        # The iso kernel vanishes LINEARLY in q_tilde at the endpoint, where
        # the log-spaced grid is coarsest (~10% steps at N=600 spanning 25
        # decades) — linear interpolation there is ~1% off where the planar
        # kernel's arccosh shape stays ~0.2%. A linear tail over the top
        # octave restores endpoint parity; the planar grid is untouched.
        tail = np.linspace(0.5 * q_tilde_max, q_tilde_max, N_points // 2)
        q_tilde_values = np.unique(np.concatenate([q_tilde_values, tail]))
    K1_inv = interpolant_k1_inverse
    if b_constrained_max is None:
        xi_cap = None
    else:
        if b_constrained_max < R_eff:
            raise ValueError(
                f"b_constrained_max ({b_constrained_max} m) below sensor "
                f"radius R_eff ({R_eff} m)")
        xi_cap = b_constrained_max / lamb
    dsigma_values = np.array(
        [dsigma_dq_tilde(q_tilde, xi, K1_inv, xi_cap=xi_cap, kernel=kernel)
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


# ============================================================================
# Log-space variants (for xi = R_eff/lamb >~ 700, where K1(xi) underflows and
# G2(xi) overflows float64; e.g. lamb = 0.2 um with R_eff = 200 um). Same
# physics as the functions above, carried in logs via the exponentially
# scaled Bessels kve(n, x) = K_n(x) e^x. Validated against the direct
# functions at the sub-percent level for xi in [0.01, 100].
# ============================================================================

from scipy.special import kve


def ln_shape_factor(x):
    """ln G2(x), stable at large x (G2 ~ 1.5 e^x (x - 1) / x^3)."""
    if x < 30:
        return float(np.log(shape_factor(x)))
    return float(np.log(1.5 * (x - 1)) + x - 3 * np.log(x))


def ln_k1(beta):
    """ln K1(beta), stable at large beta."""
    return np.log(kve(1, beta)) - beta


# invert ln K1 on a grid wide enough for xi = 1000 plus the reach beyond it
_ln_beta_grid = np.geomspace(1e-4, 2000.0, 20000)
_ln_k1_grid = ln_k1(_ln_beta_grid)
_scipy_interpolant_ln_k1_inverse = interp1d(_ln_k1_grid[::-1],
                                            _ln_beta_grid[::-1])


def interpolant_ln_k1_inverse(ln_k1_values):
    """Return beta such that ln K1(beta) = ln_k1_values (1/K1 above grid)."""
    ln_k1_values = np.asarray(ln_k1_values, dtype=float)
    hi = ln_k1_values > _ln_k1_grid[0]
    out = np.empty_like(ln_k1_values)
    out[hi] = np.exp(-ln_k1_values[hi])
    out[~hi] = _scipy_interpolant_ln_k1_inverse(
        np.clip(ln_k1_values[~hi], _ln_k1_grid[-1], _ln_k1_grid[0]))
    return out


def ln_dsigma_dq_tilde(delta, ln_q_tilde_max, ln_k1_cap=None,
                       kernel=KERNEL_DEFAULT):
    """ln of the dimensionless dsigma/dq_tilde at Delta = ln(qt_max/qt).

    ``ln_k1_cap`` (= ln K1(b_constrained_max/lamb), or None) optionally caps the
    impact parameter: the cap raises the lower integration limit to t_min and
    bites only when b_constrained_max < b_max(q), i.e. when
    c = ln(K1(xi_cap)/q_tilde) > 0. ``ln_k1_cap=None`` is a byte-for-byte no-op.

    ``kernel``: "planar-signed" (default, shipped) or "isotropic-folded"
    (pi * int beta dbeta / K1(beta), beta in [xi, min(K1^-1(qt), xi_cap)]),
    carried in logs via kve exactly like the shipped branch.
    """
    # xi_cap < xi <=> ln K1(xi_cap) > ln K1(xi) = ln_q_tilde_max (K1 decreasing)
    if ln_k1_cap is not None and ln_k1_cap > ln_q_tilde_max:
        raise ValueError("cap below the inner cutoff")

    if _check_kernel(kernel) == KERNEL_ISOTROPIC_FOLDED:
        xi = float(interpolant_ln_k1_inverse(np.array([ln_q_tilde_max]))[0])
        ln_qt = ln_q_tilde_max - delta
        beta_hi = float(interpolant_ln_k1_inverse(np.array([ln_qt]))[0])
        if ln_k1_cap is not None:
            beta_cap = float(
                interpolant_ln_k1_inverse(np.array([ln_k1_cap]))[0])
            beta_hi = min(beta_hi, beta_cap)
        if beta_hi <= xi:
            return -np.inf
        bs = np.linspace(xi, beta_hi, 20001)
        ln_integrand = np.log(bs) + bs - np.log(kve(1, bs))
        peak = ln_integrand.max()
        return float(np.log(np.pi) + peak
                     + np.log(np.trapezoid(np.exp(ln_integrand - peak), bs)))

    t_max = np.arccosh(np.exp(delta)) if delta < 30 else delta + np.log(2.0)
    t_min = 0.0
    if ln_k1_cap is not None:
        c = ln_k1_cap - ln_q_tilde_max + delta   # = ln(K1(xi_cap)/q_tilde)
        if c > 0.0:
            t_min = np.arccosh(np.exp(c))
    ts = np.linspace(t_min, t_max, 800)
    ln_kappa = (ln_q_tilde_max - delta) + np.log(np.cosh(ts))
    beta = interpolant_ln_k1_inverse(ln_kappa)
    # |dK1/dbeta| = (K0 + K2)/2; the e^-beta of kve carried in the log
    ln_integrand = (np.log(beta)
                    - np.log(0.5 * (kve(0, beta) + kve(2, beta))) + beta)
    peak = ln_integrand.max()
    return peak + np.log(np.trapezoid(np.exp(ln_integrand - peak), ts))


def make_ln_dsigma_dq_interpolant(R_eff, lamb, delta_max=32.0, N_points=400,
                                  b_constrained_max=None,
                                  kernel=KERNEL_DEFAULT):
    """Tabulate ln(dsigma/dq_tilde) vs Delta = ln(qt_max/qt) once.

    ``b_constrained_max`` (metres, or None) optionally caps the outer edge of
    the impact-parameter integral at min(b_constrained_max, b_max(q)); None is a
    byte-for-byte no-op. ``kernel`` selects the projection convention (see
    :func:`ln_dsigma_dq_tilde`); the default reproduces the shipped tabulation.
    """
    xi = R_eff / lamb
    ln_q_tilde_max = float(ln_k1(xi))
    if b_constrained_max is None:
        ln_k1_cap = None
    else:
        if b_constrained_max < R_eff:
            raise ValueError(
                f"b_constrained_max ({b_constrained_max} m) below sensor "
                f"radius R_eff ({R_eff} m)")
        ln_k1_cap = float(ln_k1(b_constrained_max / lamb))
    deltas = np.linspace(1e-4, delta_max, N_points)
    ln_values = [ln_dsigma_dq_tilde(d, ln_q_tilde_max, ln_k1_cap=ln_k1_cap,
                                    kernel=kernel)
                 for d in deltas]
    return interp1d(deltas, ln_values, bounds_error=False, fill_value=-np.inf)


def dsigma_dq_ln(q, alpha, lamb, R_eff, v, ln_interpolant):
    """Physical dsigma/dq in GeV^-3 via the log-space tabulation.

    Same contract as dsigma_dq (q in GeV, lengths in m, v in c units,
    vectorized in v); use with make_ln_dsigma_dq_interpolant.
    """
    xi = R_eff / lamb
    ln_G2 = ln_shape_factor(xi)
    lamb_gev = units.conv_m2pGeV(lamb)
    # Delta = ln q_tilde_max - ln q_tilde, with ln q_tilde built term by term
    delta = (float(ln_k1(xi)) + ln_G2 + np.log(2 * alpha)
             - np.log(q * lamb_gev * v))
    ln_prefactor = 3 * np.log(lamb_gev) + np.log(v) - np.log(alpha) - ln_G2
    return np.exp(ln_interpolant(delta) + ln_prefactor)


def impact_parameter_max_ln(q, alpha, lamb, R_eff, v):
    """Largest impact parameter [m] with |q(b)| >= q, via log-space K1 inverse.

    Same contract as impact_parameter_max; needed when K1(R_eff/lamb)
    underflows. Returns 0 where the momentum-transfer cap makes q unreachable.
    """
    xi = R_eff / lamb
    ln_G2 = ln_shape_factor(xi)
    ln_q_tilde = np.log(q * units.conv_m2pGeV(lamb) * v) \
        - np.log(2 * alpha) - ln_G2
    ln_q_tilde = np.atleast_1d(np.asarray(ln_q_tilde, dtype=float))
    ln_q_tilde_max = float(ln_k1(xi))
    b = np.zeros_like(ln_q_tilde)
    mask = ln_q_tilde < ln_q_tilde_max
    if np.any(mask):
        b[mask] = lamb * interpolant_ln_k1_inverse(ln_q_tilde[mask])
    return b
