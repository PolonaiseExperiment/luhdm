"""Bridge to the ``optimum_interval`` statistics package.

Everything here is statistics plumbing: it consumes a precomputed differential
rate ``(qs, rate)`` and an event list, and returns the optimum-interval
extremeness of the data for that spectrum. The physics (halo, attenuation,
cross section, the rate formula itself) stays in the notebook, where it gets
tweaked.

Because for our signals the spectrum shape depends on the coupling being
limited (finite-range cross section; attenuation), ``upper_limit`` does not
apply directly. The pattern instead: scan the coupling(s), call
:func:`extremeness_and_mu` at each grid point, and take the confidence-level
set of the resulting surface (:func:`excluded_band` for a 1-D scan, a contour
for a 2-D grid).
"""

import numpy as np
from optimum_interval import OptimumIntervalTable, spectrum_cdf_from_samples

__all__ = [
    "extremeness_and_mu",
    "excluded_band",
    "new_table",
    "round_log",
    "spectrum_from_rate",
]


def new_table(seed=0):
    """A seeded Monte-Carlo calibration table (reusable across the whole scan)."""
    return OptimumIntervalTable(rng=np.random.default_rng(seed))


def round_log(x, dex=0.02):
    """Round onto a log grid so calibration tables are shared between points."""
    return 10 ** (np.round(np.log10(x) / dex) * dex)


def spectrum_from_rate(qs, rate, t_obs):
    """Expected counts and normalized spectrum CDF from a differential rate.

    Parameters
    ----------
    qs, rate : arrays
        Differential rate dR/dq [1/s/GeV] on the grid ``qs`` [GeV].
    t_obs : float
        Livetime in seconds.

    Returns
    -------
    (mu, cdf, q_lo, q_hi), or ``None`` if the rate has no support.
    """
    rate = np.maximum(np.asarray(rate, dtype=float), 0.0)
    if not np.any(rate > 0):
        return None
    hi = np.max(np.where(rate > 0)[0])
    qs, rate = qs[: hi + 1], rate[: hi + 1]
    mu = float(np.trapezoid(rate, qs)) * t_obs
    cum = np.concatenate(
        [[0.0], np.cumsum(0.5 * (rate[1:] + rate[:-1]) * np.diff(qs))])
    if cum[-1] <= 0:
        return None
    cdf = spectrum_cdf_from_samples(qs, cum / cum[-1])
    return mu, cdf, qs[0], qs[-1]


def extremeness_and_mu(table, events, qs, rate, t_obs,
                       n_mc=2500, mu_floor=0.2, mu_cap=40.0, mu_dex=0.02):
    """Optimum-interval extremeness of ``events`` for this spectrum.

    Returns ``(p, mu)`` where ``p`` is the probability that a background-free
    experiment looks *less* extreme than the data (the limit is where ``p``
    crosses the confidence level), and ``mu`` the expected counts.

    Shortcuts: ``mu < mu_floor`` returns ``p = 0`` (nothing expected);
    ``mu > mu_cap`` returns ``p = 1`` (with few observed events the exclusion
    is overwhelming, no Monte Carlo needed). ``mu`` is rounded onto a
    ``mu_dex`` log grid so tables are reused across the scan.
    """
    spec = spectrum_from_rate(qs, rate, t_obs)
    if spec is None:
        return 0.0, 0.0
    mu, cdf, q_lo, q_hi = spec
    if mu < mu_floor:
        return 0.0, mu
    if mu > mu_cap:
        return 1.0, mu
    inside = np.asarray(events, dtype=float)
    inside = inside[(inside > q_lo) & (inside < q_hi)]
    mu_r = round_log(mu, mu_dex)
    table.generate(mu_r, n_mc)
    stat = table.optimum_interval_statistic(inside, mu_r, spectrum_cdf=cdf)
    return table.extremeness_of_opt_itv_stat(stat, mu_r), mu


def excluded_band(alphas, ps, level=0.95):
    """(low, high) edges of the excluded band along a 1-D coupling scan.

    ``ps`` is the extremeness at each ``alphas``; edges are log-interpolated
    crossings of ``level``. Returns NaNs when nothing is excluded; the upper
    edge saturates at ``alphas[-1]`` if the ceiling lies beyond the scan.
    """
    ps = np.asarray(ps, dtype=float)
    above = ps >= level
    if not above.any():
        return np.nan, np.nan
    idx = np.where(above)[0]
    if idx[0] > 0:
        lo = np.interp(level, ps[idx[0] - 1: idx[0] + 1],
                       np.log10(alphas[idx[0] - 1: idx[0] + 1]))
    else:
        lo = np.log10(alphas[0])
    if idx[-1] < len(alphas) - 1:
        hi = np.interp(-level, -ps[idx[-1]: idx[-1] + 2],
                       np.log10(alphas[idx[-1]: idx[-1] + 2]))
    else:
        hi = np.log10(alphas[-1])
    return 10 ** lo, 10 ** hi
