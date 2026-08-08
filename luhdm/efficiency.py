"""Measured detection efficiency epsilon(q) per sensor mode.

The table is built by ``scripts/export_efficiency.py`` from the analysis product
and committed as ``reference_data/efficiency_curves.npz`` so it is available on
every host (no data/ dir needed). ``make_efficiency`` returns a vectorised
callable epsilon(q_GeV) in [0, 1] for use as the ``eff`` argument of
``rate.differential_rate_trapz``.
"""
import os
from pathlib import Path

import numpy as np

# LUHDM_EFFICIENCY_NPZ overrides the table for veto-variant scans (re-averaged efficiency).
_TABLE = Path(os.environ.get(
    "LUHDM_EFFICIENCY_NPZ",
    str(Path(__file__).resolve().parent / "reference_data" / "efficiency_curves.npz")))


def table_path():
    """Path of the efficiency table actually in use (LUHDM_EFFICIENCY_NPZ aware).

    Provenance writers record this (and its sha256) so an env-overridden table
    is visible in the run/assembly metadata instead of being silently assumed.
    """
    return _TABLE


def load_curve(mode, df=3):
    """(q_GeV, efficiency) arrays for one mode and dof hypothesis (df=2 or 3)."""
    with np.load(_TABLE) as d:
        return d[f"q_gev_{mode}"].copy(), d[f"eff_{mode}_df{df}"].copy()


def make_efficiency(mode, df=3):
    """Detection efficiency epsilon(q_GeV) as a vectorised callable in [0, 1].

    Below the smallest calibrated momentum epsilon = 0 (kick undetectable);
    above the largest, epsilon is held at the measured saturated value (~1).
    """
    q, e = load_curve(mode, df)
    e_hi = float(e[-1])
    return lambda qq: np.interp(qq, q, e, left=0.0, right=e_hi)
