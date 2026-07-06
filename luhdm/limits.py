"""Bridge to the ``optimum_interval`` statistics package.

The scan pattern itself (round-onto-a-log-grid table sharing, extremeness at
a grid point, level crossings) now lives upstream in
``optimum_interval.scanning``, since nothing about it is UHDM-specific. This
module keeps the names and signatures the luhdm notebooks and scripts have
always used, delegating to the package.

Because for our signals the spectrum shape depends on the coupling being
limited (finite-range cross section; attenuation), ``upper_limit`` does not
apply directly. The pattern instead: scan the coupling(s), call
:func:`extremeness_and_mu` at each grid point, and take the confidence-level
set of the resulting surface (:func:`excluded_band` for a 1-D scan, a contour
for a 2-D grid).
"""

from optimum_interval import scanning

__all__ = [
    "extremeness_and_mu",
    "excluded_band",
    "new_table",
    "round_log",
    "spectrum_from_rate",
]

new_table = scanning.new_table
round_log = scanning.round_log
spectrum_from_rate = scanning.spectrum_from_rate
excluded_band = scanning.excluded_interval


def extremeness_and_mu(table, events, qs, rate, t_obs,
                       n_mc=2500, mu_floor=0.2, mu_cap=40.0, mu_dex=0.02):
    """Optimum-interval extremeness of ``events`` for this spectrum.

    Thin wrapper over :func:`optimum_interval.scan_extremeness`, keeping the
    historical luhdm signature (``n_mc`` etc.). Returns ``(p, mu)``.
    """
    return scanning.scan_extremeness(
        table, events, qs, rate, t_obs,
        n=n_mc, mu_floor=mu_floor, mu_cap=mu_cap, mu_dex=mu_dex)
