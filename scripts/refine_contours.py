#!/usr/bin/env python3
"""Refine the 95% CL exclusion-island boundary by bracketed log-bisection.

The release cube samples the optimum-interval extremeness p on a coarse
(alpha_n x mass) grid (0.233 x 0.119 dex cells), and the figure's island
outline is the log-interpolated level crossing of that grid -- so the drawn
boundary carries up-to-one-cell discretization stairs (left wall) and
one-cell risers (atmospheric ceiling). This script replaces the grid
crossing with a root-found crossing of the *same statistic*:

* Per mass column of the cube, the coarse-grid cells bracketing each edge
  (floor and ceiling) of the excluded interval start a log-bisection of
  p(alpha) >= level down to ``--tol-alpha`` (default 0.005 dex).
* Where adjacent refined columns' edges jump by more than ``--tol-wall``
  (default 0.05 dex), new mass columns are inserted at the geometric
  midpoint and refined recursively down to ``--mass-res`` (0.01 dex) --
  contour tracing specialized to the (floor, ceiling) band representation
  the figure draws (cost follows the walls and tips, not the area).
* At each island end the tip is localized by bisection in mass on the
  predicate "any alpha excluded", the predicate evaluated by a 5-point
  pre-scan (unimodality guard) plus golden-section refinement of
  max_alpha p -- the fold-point parameterization switch of continuation
  methods (in alpha the two roots merge at the tip; in mass the curve is
  well-conditioned there).

The oracle is the exact cell body of ``scripts/build_release.py::
_process_chunk`` (attenuation ODE -> dR/dq -> ``limits.extremeness_and_mu``,
or the bare SHM when the surface carries no atmosphere) with the same pinned
constants read back from the cube being refined (seed, fidelity incl. the
optional two-tier MC rule, q grid, efficiency, events, impact-parameter cap and
massless ODE regulator). The projection kernel and the massless kinematic
endpoint convention are read back too, but this build of ``luhdm`` carries no
switch for either, so they are *asserted* to be the shipped ones rather than
applied -- a cube that asks for anything else stops the run. MC calibration
goes through the same :class:`PerMuTable` (fresh seed-20260702 table per
0.02-dex-rounded mu), so p is a pure, bit-reproducible function of
(alpha, m, surface) -- independent of worker count, evaluation order, and of
*which* alphas the refinement visits. At the cube's own grid points the oracle
reproduces the released values bit-for-bit in the cube's float32 storage
(spot-checked by ``--spot``), which is what ties the refined boundary to the
release: figure contour = root-found crossing of the same statistic; the
released grid reproduces it to its stated cell size.

Open-topped columns
-------------------
A surface whose excluded region reaches the top of the alpha axis (the
no-atmosphere plane is open at strong coupling: with no overburden there is
no ceiling to come back down to) has no ceiling to refine. Such columns are
refined on the floor only and their ``ceiling_alpha_n`` entry is JSON
``null``; the count is reported per surface as ``n_open_top_columns``. The
same thing happens along the mass axis: an island still excluded at the last
mass the cube carries has no tip to localize on that side, and the sidecar's
``tips`` entry says ``open_at_mass_axis_edge`` instead of a bisected end.

The mass cut
------------
A halo of mass-m particles delivers N(m) = f_DM (rho_0/m) <v> T_obs pi b_cap^2
of them within the impact-parameter cap during the exposure, so above
m_cut = f_DM rho_0 <v> T_obs pi b_cap^2 / N_req fewer than N_req cross the
aperture and the excluded region cannot be read as a limit. The cut is a flux
argument applied after the fact: the cube's stored surfaces stay uncapped, but
the cube records the line (``m_cut_<cap>_f<f_DM>_gev``, with
``m_cut_n_transits_required`` and ``m_cut_b_cap_m``), and the RELEASED contour
is truncated at it -- m_cut is the right edge. Mass columns above m_cut are
dropped before any phase runs (nothing is refined that the release drops), one
exact column at m = m_cut is refined as the polyline's last vertex, and the
right tip is not traced: the island does not end inside the cube, the cut ends
it (the sidecar's ``tips`` entry says ``cut_at_m_cut`` and names N_req and
b_cap). A cube that records no such attribute is refined untruncated -- the cut
is read back, never invented -- and ``--no-m-cut`` disables the truncation for
audit runs.

Nothing in luhdm/ or the cube builder changes; the output is a standalone
sidecar (JSON) with the refined (mass, floor, ceiling) polyline per surface
plus full provenance (cube sha256 + version tag, tolerances, seed policy,
git SHA, per-vertex coarse brackets). An optional NPZ per surface records
every oracle evaluation for audit/debug.

Determinism guarantees inherited from the cube's scheme
-------------------------------------------------------
``limits.extremeness_and_mu`` rounds mu onto a 0.02-dex log grid and
:class:`build_release.PerMuTable` gives every rounded mu its own freshly
seeded table, so the MC extremeness is a pure function of (rounded mu, seed,
n_mc). Bisection queries therefore see a deterministic (slightly rough)
p(alpha); no stochastic root-finding machinery is needed, and a re-run of
this script reproduces every refined vertex exactly.

Usage
-----
  # validate on a few columns (single worker, bit-exact endpoint check):
  python scripts/refine_contours.py --surfaces massless_f1 --columns 40,60,80 \\
      --spot 6 --workers 1 --no-insert --no-tips --out /tmp/demo.json

  # one full surface, all phases:
  python scripts/refine_contours.py --surfaces massless_f1 \\
      --out /tmp/contours_massless.json --evals-dir /tmp/evals

  # the figure surfaces this cube carries (production):
  python scripts/refine_contours.py --surfaces all --out release/luhdm_contours_v1.json

Cost. One oracle call is one attenuation ODE (``n_ode`` solves) plus one
``n_mc``-trial optimum-interval table, and at the v8 production fidelity
(n_ode 400, n_shm 3e5, n_mc 1e4) that is seconds to tens of seconds per call
-- strongly (m, alpha, lambda) dependent, with the finite-lambda surfaces the
expensive ones and the no-atmosphere plane (no ODE at all) ~20x cheaper than
any of them. A whole atmosphere surface is thousands of calls, ~85% of them in
the mass-insertion phase, so budget HOURS per finite-lambda surface on a
40-80 core box, not minutes. The sidecar is rewritten after every surface, so
a long run can be watched, and interrupted, without losing what is done.

  # From v7 on the release ships one file per hypothesis plane, so each surface
  # must be run against the file that carries its (f_DM, atmosphere) plane --
  # a mismatch is a clear file-level error, not a KeyError from the reader.
  # ``--surfaces all`` is the FIGURE_SURFACES list restricted to the planes the
  # given cube actually has (v8 File A carries the four f_DM = 1 atmosphere
  # surfaces; the f_DM = 0.1 atmosphere surface is in no v8 file):
  python scripts/refine_contours.py \\
      --surfaces massless_f1,2mm_f1,200um_f1,20um_f1 \\
      --release release/luhdm_datarelease_v9_A_f1_atm.h5 --out /tmp/a.json
  python scripts/refine_contours.py --surfaces 20um_f0p1_noatm \\
      --release release/luhdm_datarelease_v9_B_f0p1_noatm.h5 --out /tmp/b.json
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")   # match build_release: 1 proc/core
os.environ.setdefault("TQDM_DISABLE", "1")

import functools
import multiprocessing
import platform
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import optimum_interval
import scipy

# Progress must survive `> log` redirection (long phases would otherwise sit
# silent in the block buffer); same intent as build_release's flush=True.
print = functools.partial(print, flush=True)  # noqa: A001

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import luhdm  # noqa: E402
from luhdm import (  # noqa: E402
    atmosphere, config, cross_section, efficiency, halo, limits, rate, release,
)

# The determinism-critical pieces are imported from the builder, not copied:
# PerMuTable (per-rounded-mu seeded MC tables), the pinned seed and q-grid
# reference, the f_DM scale, and the efficiency dof hypothesis.
from build_release import (  # noqa: E402
    DF, F_SCALE_F1, PerMuTable, Q_HI_REF, SEED, sha256_file,
)

#: Two-tier MC: default extremeness above which a cell is re-evaluated on the
#: hi-tier table. Named ``P_HI_LO`` in the shelved fine-grid variant of
#: ``build_release.py`` (which is where the two-tier rule lives); the SHIPPED
#: builder has no two-tier MC, so the constant is defined here rather than
#: imported, and is used only as the fallback when a cube's fidelity dict
#: carries an ``n_mc_hi`` without naming its own ``p_hi_lo``. Cubes built by the
#: shipped pipeline carry no ``n_mc_hi`` at all, so the whole path is inert for
#: them and the oracle is a single base-tier evaluation -- exactly the cell body
#: of ``build_release._process_chunk``.
P_HI_LO = 0.90

#: The projected-dsigma/dq kernel of cubes that predate the flag: they carry no
#: ``projection_kernel`` attribute and were in fact built with exactly this
#: convention. A cube that names a kernel has it dispatched into every
#: ``rate.make_xsec`` the refiner builds; a cube naming a kernel this build of
#: ``luhdm.cross_section`` does not implement is a hard stop rather than a
#: silent mismatch.
KERNEL_DEFAULT = cross_section.KERNEL_DEFAULT

#: Provenance ships with the release and must carry no absolute home paths or
#: usernames (and no host identifiers at all); every path string is stored
#: home-relative ('~/...'), which stays copy-pasteable through shell expansion.
#: Same rule and same spelling as ``assemble_release.scrub_home``.
HOME = str(Path.home())


def scrub_home(x):
    """Recursively home-relativise every string in a provenance tree."""
    if isinstance(x, str):
        return x.replace(HOME, "~")
    if isinstance(x, dict):
        return {k: scrub_home(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [scrub_home(v) for v in x]
    return x


# --------------------------------------------------------------------------- #
# Tolerances (defaults; all overridable on the command line, all recorded in
# the sidecar provenance)
# --------------------------------------------------------------------------- #
TOL_ALPHA_DEX = 0.005    # edge resolution target in log10(alpha)
TOL_WALL_DEX = 0.05      # adjacent-column edge jump that triggers mass insertion
MASS_RES_DEX = 0.01      # mass-recursion floor (walls render vertical below this)
BRACKET_WIDEN_SE = 3.0   # widen a coarse bracket if its cube p is this close
                         # (in binomial se units) to the level
MAX_WIDEN_CELLS = 3      # max extra coarse cells when validating an off-grid
                         # bracket endpoint
TIP_SCAN_POINTS = 5      # alpha pre-scan points for the tip predicate
TIP_GOLDEN_CALLS = 4     # golden-section refinement calls after the pre-scan
MAX_INSERT_PER_SURFACE = 400   # hard cap on inserted columns (runaway guard)

#: Storage precision of the cube's extremeness (``results/extremeness``). Two
#: comparisons in this script must be made in it rather than in float64:
#:
#: * ``--spot``: the oracle returns the float64 the builder computed BEFORE
#:   h5py narrowed it, so bit-exactness means "equal after the same narrowing".
#:   A float64 comparison can never pass -- it measures the quantization
#:   (~1e-8 at p ~ 0.5), not a disagreement about the statistic.
#: * the cube's own excluded mask: ``float32(0.95) = 0.949999988...``, so a
#:   cell whose builder p was EXACTLY the level narrows to a stored value just
#:   under it, and a float64 ``>= 0.95`` would drop a genuinely excluded cell
#:   (one such cell exists in the v8 200 um f_DM = 1 plane, at the island's
#:   right end). p lives on the 1/n_mc grid, so nothing else can alias onto
#:   ``float32(level)``: comparing at the storage precision is exactly the
#:   builder's own ``p >= level``, and is what the figure's reader does too.
CUBE_P_DTYPE = np.float32

#: Surface table: name -> (lambda tag, f_DM, atmosphere). Names are stable
#: sidecar keys. The first five are the figure surfaces (paper_fig_limits.py):
#: mode 1, atmosphere on, massless + the three finite figure lambdas at
#: f_DM = 1 (left panel) and the 20 um / f_DM = 0.1 benchmark (right panel);
#: they are what ``--surfaces all`` selects. ``20um_f0p1_noatm`` is the
#: no-overburden companion plane and must be named explicitly -- from v7 on it
#: lives in its own release file (File B), and its excluded region is open at
#: the top of the alpha axis.
SURFACES = {
    "massless_f1": ("massless", 1.0, True),
    "2mm_f1": ("2mm", 1.0, True),
    "200um_f1": ("200um", 1.0, True),
    "20um_f1": ("20um", 1.0, True),
    "20um_f0p1": ("20um", 0.1, True),
    "20um_f0p1_noatm": ("20um", 0.1, False),
}

#: What ``--surfaces all`` means: the five atmosphere figure surfaces,
#: RESTRICTED to the planes the cube handed to ``--release`` actually carries.
#: Since v7 the release ships one file per hypothesis plane, so no single file
#: holds all five (v8 File A carries the four f_DM = 1 ones; the f_DM = 0.1
#: atmosphere surface is in no v8 file). Naming a missing surface explicitly is
#: still a hard error -- only 'all' filters, and it prints what it dropped.
FIGURE_SURFACES = ("massless_f1", "2mm_f1", "200um_f1", "20um_f1",
                   "20um_f0p1")

# --------------------------------------------------------------------------- #
# Shared read-only state; set in the parent BEFORE each surface's pool forks.
# --------------------------------------------------------------------------- #
QS = None          # momentum grid (fixed for every cell)
EFF = None         # efficiency eps_mode(QS) for the selected mode
EVENTS = None      # observed impulses [GeV] for the selected mode
V_I = None         # SHM initial-speed samples (seeded)
XS = None          # rate.make_xsec handle (per surface; carries the b cap and
                   # the cube's projection-kernel convention)
LAMB_ODE = None    # atmospheric-ODE regulator range [m] (per surface)
ATMOSPHERE = True  # per surface: False skips the ODE entirely (bare SHM)
FID = None         # fidelity dict incl. mu_cap, optional two-tier n_mc_hi
Q_MIN = None       # analysis threshold [GeV] (from the cube)
T_TOTAL = None     # exposure [s] (from the cube; asserted == config.T_EXPOSURE)
F_SCALE = 1.0      # 1.0 at the baseline f_DM, F_SCALE_F1 (=10) at f_DM = 1
LEVEL = 0.95       # confidence level (from the cube)
SE_P = None        # binomial se of p at LEVEL for the cube's n_mc

_worker_state: dict = {}


def _worker_tables():
    """Per-process MC tables: the base tier and, when the cube's fidelity asks
    for it, the two-tier hi table on its own seed (build_release's contract --
    ``generate`` regenerates from the advanced RNG state, so a shared table
    would make p evaluation-order dependent)."""
    if "table" not in _worker_state:
        _worker_state["table"] = PerMuTable(seed=SEED)
        if FID.get("n_mc_hi"):
            _worker_state["table_hi"] = PerMuTable(seed=SEED + 1)
    return _worker_state


# A stalled atmosphere ODE (step-size collapse at an extreme off-grid probe)
# runs forever at 100% CPU and blocks its whole insertion wave; 600 s is 60x
# the median oracle call, far beyond any healthy cell. On timeout the cell is
# treated exactly like a build_release status-1 cell: p = NaN reads as "not
# excluded" everywhere downstream, which can only shrink the island.
ORACLE_TIMEOUT_S = 600


class OracleTimeout(Exception):
    pass


def _oracle_alarm(signum, frame):
    raise OracleTimeout


def oracle(alpha, m, history=None, kind=""):
    """p, mu at one off- or on-grid (alpha, m) -- the build_release cell body."""
    t0 = time.time()
    signal.signal(signal.SIGALRM, _oracle_alarm)
    signal.alarm(ORACLE_TIMEOUT_S)
    try:
        state = _worker_tables()
        if ATMOSPHERE:
            v_min = Q_MIN / m / 10  # ODE floor follows the analysis threshold
            v_f_samples = atmosphere.compute_v_f_distribution(
                alpha, LAMB_ODE, m, V_I, v_min=v_min, n_grid=FID["n_ode"])
            f_v_f = atmosphere.compute_f_vf(v_f_samples, v_min)[0]
        else:
            # no-overburden pass: the arrival distribution IS the halo one, the
            # same for every (m, alpha) -- build_release._arrival_f_v_f, NO_ATM.
            f_v_f = halo.standard_halo_model
        raw = rate.differential_rate_trapz(QS, alpha, m, f_v_f, XS, eff=None)
        detected = raw * EFF * F_SCALE
        p, mu = limits.extremeness_and_mu(
            state["table"], EVENTS, QS, detected, T_TOTAL,
            n_mc=FID["n_mc"], mu_cap=FID["mu_cap"])
        # Two-tier MC (v7 fidelity contract), verbatim build_release.eval_extremeness:
        # a base p at or above p_hi_lo (but not shortcut-saturated) is re-evaluated
        # on the hi-tier table. Cubes without n_mc_hi take the base result as-is.
        n_hi = FID.get("n_mc_hi")
        if n_hi and FID.get("p_hi_lo", P_HI_LO) <= p < 1.0:
            p, mu = limits.extremeness_and_mu(
                state["table_hi"], EVENTS, QS, detected, T_TOTAL,
                n_mc=n_hi, mu_cap=FID["mu_cap"])
    except OracleTimeout:
        print(f"    ORACLE TIMEOUT ({ORACLE_TIMEOUT_S} s) at "
              f"alpha={alpha:.6e} m={m:.6e} kind={kind}: p=NaN (not excluded)",
              flush=True)
        p, mu = float("nan"), float("nan")
    finally:
        signal.alarm(0)
    if history is not None:
        history.append((float(m), float(alpha), float(p), float(mu),
                        time.time() - t0, kind))
    return p, mu


# --------------------------------------------------------------------------- #
# Edge bisection
# --------------------------------------------------------------------------- #
def bisect_edge(a_lo, a_hi, m, rising, tol_dex, history, kind):
    """Log-bisect the p(alpha) = LEVEL crossing inside a validated bracket.

    ``rising`` selects the invariant: floor (p < LEVEL at ``a_lo``,
    p >= LEVEL at ``a_hi``) or ceiling (the reverse). The final bracket
    endpoints sit within ``tol_dex`` of the returned edge on opposite sides
    of the level -- that *is* the plan's 3-point post-check, at zero extra
    cost. Returns (edge, (a_lo, a_hi)).
    """
    while np.log10(a_hi / a_lo) > tol_dex:
        mid = float(np.sqrt(a_lo * a_hi))
        p, _mu = oracle(mid, m, history, kind)
        excluded = p >= LEVEL
        if rising == excluded:
            a_hi = mid
        else:
            a_lo = mid
    return float(np.sqrt(a_lo * a_hi)), (float(a_lo), float(a_hi))


def _monotonicity_audit(history, edge_kind, rising):
    """Flag > 5 se violations of p-monotonicity among this edge's evals.

    The per-mu MC roughness is ~1 se across adjacent rounded-mu bins;
    violations well beyond it would mean genuine non-monotone structure
    (risk 1 of the plan) and are reported, never silently absorbed.
    """
    pts = sorted((h for h in history if h[5] == edge_kind), key=lambda h: h[1])
    worst = 0.0
    for (_, _, p0, _, _, _), (_, _, p1, _, _, _) in zip(pts, pts[1:]):
        viol = (p0 - p1) if rising else (p1 - p0)
        worst = max(worst, viol)
    return float(worst) if worst > 5.0 * SE_P else 0.0


def refine_column(task):
    """Refine both edges of one mass column. Runs inside a worker process.

    ``task`` carries the mass, the coarse bracket for each edge and, for
    off-grid (inserted/tip) columns, ``validate=True`` -- their brackets come
    from neighbours rather than from cube cells, so the endpoints are probed
    (and widened outward, up to MAX_WIDEN_CELLS coarse cells) before
    bisection. Grid columns skip that: their endpoint p values ARE the cube's
    (same statistic, same seeding), already on opposite sides of the level.

    An OPEN-TOPPED column (``open_top=True``: the excluded set runs off the
    top of the alpha axis) has no ceiling; only the floor is bisected and
    ``ceiling`` comes back None. A validated open-topped column still probes
    the axis top once -- if the exclusion does close there after all, the
    column reverts to a normal two-edge refinement.
    """
    t0 = time.time()
    history: list = []
    open_top = bool(task.get("open_top", False))
    out = dict(m=task["m"], im=task.get("im", -1), origin=task["origin"],
               flags=[], excluded=True, open_top=open_top)
    cell = task["cell_dex"]
    try:
        f_lo, f_hi = task["floor_bracket"]
        c_lo, c_hi = (None, None) if open_top else task["ceil_bracket"]
        if task.get("validate", False):
            # interior of the expected band (its top is the axis top when the
            # band is open there)
            a_mid = float(np.sqrt(f_hi * (task["alpha_top"] if open_top
                                          else c_lo)))
            p_mid, _ = oracle(a_mid, task["m"], history, "probe")
            if p_mid < LEVEL:
                got = _rescue_scan(task, history)
                if got is None:
                    out["excluded"] = False
                    out["wall_s"] = time.time() - t0
                    out["history"] = history
                    return out
                f_lo, f_hi, c_lo, c_hi, open_top = got
                out["open_top"] = open_top
                out["flags"].append("rescue_scan")
            else:
                # floor: bottom endpoint must not be excluded; widen down.
                f_lo, ok = _ensure_side(f_lo, task["m"], False, -cell, history,
                                        "floor")
                if not ok:
                    raise AssertionError("floor bracket bottom stays excluded "
                                         f"after {MAX_WIDEN_CELLS} cells")
                # floor top: use it if excluded, else fall back to a_mid.
                p_top, _ = oracle(f_hi, task["m"], history, "floor")
                if p_top < LEVEL:
                    f_hi = a_mid
                    out["flags"].append("floor_top_fallback")
                if open_top:
                    # inherited "open" from a neighbour: verify at the axis top
                    p_ax, _ = oracle(task["alpha_top"], task["m"], history,
                                     "ceiling")
                    if p_ax < LEVEL:
                        open_top = False
                        c_lo, c_hi = a_mid, task["alpha_top"]
                        out["open_top"] = False
                        out["flags"].append("open_top_closed_at_axis_top")
                if not open_top:
                    # ceiling: bottom must be excluded, else fall back to a_mid.
                    p_bot, _ = oracle(c_lo, task["m"], history, "ceiling")
                    if p_bot < LEVEL:
                        c_lo = a_mid
                        out["flags"].append("ceiling_bottom_fallback")
                    c_hi, ok = _ensure_side(c_hi, task["m"], False, +cell,
                                            history, "ceiling")
                    if not ok:
                        raise AssertionError("ceiling bracket top stays "
                                             f"excluded after "
                                             f"{MAX_WIDEN_CELLS} cells")
        floor, fbr = bisect_edge(f_lo, f_hi, task["m"], True,
                                 task["tol_dex"], history, "floor")
        assert f_lo <= floor <= f_hi, \
            "refined edge escaped its bracket"      # structural; cannot fire
        if open_top:
            ceiling, cbr = None, None
        else:
            ceiling, cbr = bisect_edge(c_lo, c_hi, task["m"], False,
                                       task["tol_dex"], history, "ceiling")
            assert c_lo <= ceiling <= c_hi, \
                "refined edge escaped its bracket"  # structural; cannot fire
            assert floor < ceiling, \
                f"floor {floor:.3e} >= ceiling {ceiling:.3e} " \
                f"at m={task['m']:.3e}"
        for kind, rising in (("floor", True), ("ceiling", False)):
            worst = _monotonicity_audit(history, kind, rising)
            if worst:
                out["flags"].append(f"nonmonotone_{kind}_{worst:.4f}")
        out.update(floor=floor, ceiling=ceiling,
                   floor_bracket=[f_lo, f_hi],
                   ceil_bracket=(None if open_top else [c_lo, c_hi]),
                   floor_final=list(fbr),
                   ceil_final=(None if open_top else list(cbr)))
    except Exception as err:  # noqa: BLE001 -- conservative, like the figure
        out["excluded"] = False
        out["flags"].append(f"oracle_error:{err}")
    out["wall_s"] = time.time() - t0
    out["history"] = history
    return out


def _ensure_side(a, m, want_excluded, step_dex, history, kind):
    """Probe ``a``; widen by ``step_dex`` until its side matches, or give up."""
    for _ in range(MAX_WIDEN_CELLS + 1):
        p, _ = oracle(a, m, history, kind)
        if (p >= LEVEL) == want_excluded:
            return a, True
        a = float(a * 10.0 ** step_dex)
    return a, False


def _rescue_scan(task, history, n=9):
    """9-point column scan when the band-interior probe was not excluded.

    Returns validated (f_lo, f_hi, c_lo, c_hi, open_top), or None if the whole
    scan finds nothing excluded (a genuine gap/tip column). For an open-topped
    task the scan runs up to the alpha-axis top and exclusion reaching it is
    the expected outcome, not a bail-out -- the ceiling brackets come back
    None; an exclusion that does close below the top demotes the column to a
    normal two-edge one.
    """
    open_top = bool(task.get("open_top", False))
    f_lo, _ = task["floor_bracket"]
    c_hi = task["alpha_top"] if open_top else task["ceil_bracket"][1]
    grid = np.geomspace(f_lo, c_hi, n)
    ps = np.array([oracle(a, task["m"], history, "probe")[0] for a in grid])
    idx = np.where(ps >= LEVEL)[0]
    if not idx.size:
        return None
    i0, i1 = idx[0], idx[-1]
    if i0 == 0:
        return None                 # exclusion reaches the scan floor: bail
    if i1 == n - 1:
        if not open_top:
            return None             # exclusion reaches the scan top: bail
        return float(grid[i0 - 1]), float(grid[i0]), None, None, True
    return (float(grid[i0 - 1]), float(grid[i0]),
            float(grid[i1]), float(grid[i1 + 1]), False)


# --------------------------------------------------------------------------- #
# Coarse brackets from the cube (grid columns)
# --------------------------------------------------------------------------- #
def grid_column_tasks(plane, alphas, ms, tol_dex, columns=None):
    """One task per excluded cube column, brackets from its bracketing cells.

    Parity assertion: each excluded column must have a contiguous p >= LEVEL
    set that either sits strictly interior to the alpha axis (exactly TWO
    level crossings) or runs off the TOP of it (an OPEN-TOPPED column: one
    crossing, the floor). (0 of 370 columns violate this in the v5 atmosphere
    cube; every excluded column of the no-atmosphere plane is open-topped --
    without an overburden the exclusion never closes again at strong
    coupling.) Anything else -- a gap, a second island, or a floor sitting on
    the bottom of the alpha axis -- would send a naive whole-column root-find
    to an arbitrary branch, so it is fatal here, not warned. Cube cells whose
    p lies within BRACKET_WIDEN_SE binomial se of the level widen the bracket
    one cell outward.

    Open-topped columns get ``open_top=True`` and ``ceil_bracket=None``: no
    ceiling exists to bisect, and the sidecar records ``null`` for it.

    The cube's mask is taken at the cube's OWN storage precision
    (``CUBE_P_DTYPE``), not in float64: a builder p of exactly the level
    narrows to a stored float32 just below it, and a float64 test would drop
    that cell -- and with it a whole mass column the figure draws.
    """
    cell = float(np.diff(np.log10(alphas)).mean())
    plane = np.nan_to_num(np.asarray(plane, float), nan=0.0)
    level = float(CUBE_P_DTYPE(LEVEL))
    n_a, n_m = plane.shape
    alpha_top = float(alphas[-1])
    tasks, widened = [], []
    sel = set(columns) if columns is not None else None
    for im in range(n_m):
        col = plane[:, im]
        mask = col >= level
        idx = np.where(mask)[0]
        if not idx.size or (sel is not None and im not in sel):
            continue
        i0, i1 = int(idx[0]), int(idx[-1])
        open_top = bool(mask[-1])
        interior = int(np.abs(np.diff(mask.astype(int))).sum())
        if open_top:
            assert 0 < i0, (
                f"column im={im} (m={ms[im]:.3e}) reaches the top of the "
                f"alpha axis and is excluded at the bottom of it as well; "
                f"there is no bracketing cell pair for the floor")
            assert interior == 1, (
                f"column im={im} (m={ms[im]:.3e}) reaches the top of the "
                f"alpha axis but has {interior} interior level crossings, "
                f"not 1; the excluded set is not one top-open interval and "
                f"bracketed bisection is unsafe here")
        else:
            crossings = int(interior + (col[0] >= level) + (col[-1] >= level))
            assert crossings == 2, (
                f"column im={im} (m={ms[im]:.3e}) has {crossings} level "
                f"crossings, not 2; the excluded set is not one interior "
                f"interval and bracketed bisection is unsafe here")
            assert 0 < i0 and i1 < n_a - 1, (
                f"column im={im} (m={ms[im]:.3e}) is excluded up to the "
                f"alpha-axis edge; there is no bracketing cell pair "
                f"(island_is_closed would reject this cube for the figure too)")
        lo0, hi0 = i0 - 1, i0            # floor bracket cells
        lo1, hi1 = i1, i1 + 1            # ceiling bracket cells (closed only)
        w = []
        if col[i0] - LEVEL < BRACKET_WIDEN_SE * SE_P and hi0 + 1 <= i1:
            hi0 += 1; w.append("floor_top")
        if LEVEL - col[i0 - 1] < BRACKET_WIDEN_SE * SE_P and lo0 - 1 >= 0:
            lo0 -= 1; w.append("floor_bottom")
        if not open_top:
            if col[i1] - LEVEL < BRACKET_WIDEN_SE * SE_P and lo1 - 1 >= i0:
                lo1 -= 1; w.append("ceiling_bottom")
            if LEVEL - col[i1 + 1] < BRACKET_WIDEN_SE * SE_P \
                    and hi1 + 1 <= n_a - 1:
                hi1 += 1; w.append("ceiling_top")
        if w:
            widened.append((im, w))
        tasks.append(dict(
            m=float(ms[im]), im=im, origin="grid", tol_dex=tol_dex,
            cell_dex=cell, validate=False,
            open_top=open_top, alpha_top=alpha_top,
            floor_bracket=(float(alphas[lo0]), float(alphas[hi0])),
            ceil_bracket=(None if open_top
                          else (float(alphas[lo1]), float(alphas[hi1]))),
            coarse_cells=dict(floor=[lo0, hi0],
                              ceiling=(None if open_top else [lo1, hi1])),
            cube_p=dict(floor=[float(col[lo0]), float(col[hi0])],
                        ceiling=(None if open_top
                                 else [float(col[lo1]), float(col[hi1])]))))
    return tasks, widened, cell


def inserted_task(m_new, va, vb, cell, tol_dex, alpha_top):
    """Task for a mass inserted between refined columns ``va`` and ``vb``.

    An insertion between neighbours of which either is open-topped starts out
    open-topped; ``refine_column`` probes the axis top and demotes it to a
    two-edge column if the exclusion does close there.
    """
    widen = 10.0 ** cell
    f = sorted([va["floor"], vb["floor"]])
    open_top = va["ceiling"] is None or vb["ceiling"] is None
    c = None if open_top else sorted([va["ceiling"], vb["ceiling"]])
    return dict(
        m=float(m_new), im=-1, origin="inserted", tol_dex=tol_dex,
        cell_dex=cell, validate=True,
        open_top=open_top, alpha_top=float(alpha_top),
        floor_bracket=(f[0] / widen, f[1] * widen),
        ceil_bracket=(None if open_top else (c[0] / widen, c[1] * widen)))


# --------------------------------------------------------------------------- #
# Mass cut (the release's right edge)
# --------------------------------------------------------------------------- #
def resolve_m_cut(attrs, f_dm, enabled=True):
    """The flux-argument mass cut this cube records for the f_DM plane.

    From v8 the release carries one ``m_cut_<cap>_f<f_DM:g>_gev`` root
    attribute per hypothesis plane (``m_cut_10cm_f1_gev``,
    ``m_cut_10cm_f0.1_gev``) beside ``m_cut_n_transits_required`` and
    ``m_cut_b_cap_m``, and leaves the stored surfaces uncapped
    (``m_cut_applied_to_stored_surfaces`` False). Exactly one key can match a
    given plane; a cube carrying none -- or ``enabled=False`` -- gives a null
    cut and no truncation anywhere. The companion ``..._gev_derivation`` string
    does not match the suffix.
    """
    suffix = f"_f{float(f_dm):g}_gev"
    keys = sorted(k for k in attrs
                  if k.startswith("m_cut_") and k.endswith(suffix))
    if not enabled or not keys:
        return dict(m_cut=None, n_req=None, b_cap=None, attr=None)
    assert len(keys) == 1, (
        f"cube records {len(keys)} mass cuts for f_dm={float(f_dm):g}: {keys}; "
        f"the plane's cut is ambiguous")

    def _opt(k):
        return None if attrs.get(k) is None else float(attrs[k])
    return dict(m_cut=float(attrs[keys[0]]),
                n_req=_opt("m_cut_n_transits_required"),
                b_cap=_opt("m_cut_b_cap_m"), attr=keys[0])


def drop_above_m_cut(tasks, m_cut):
    """Drop the column tasks the release truncates away. (kept, n_dropped).

    Above m_cut the exclusion is not a limit, so those columns carry no vertex
    and are not worth an oracle call in any phase.
    """
    if m_cut is None:
        return tasks, 0
    keep = [t for t in tasks if t["m"] <= m_cut]
    return keep, len(tasks) - len(keep)


# --------------------------------------------------------------------------- #
# Tip localization (worker task: one whole tip trace)
# --------------------------------------------------------------------------- #
def trace_tip(task):
    """Bisect the island end in mass; refine any newly-found excluded column.

    Between the outermost refined excluded column and the first coarse column
    with no exclusion, bisect log10(m) on the predicate "some alpha excluded"
    down to ``mass_res`` dex. The predicate is a TIP_SCAN_POINTS log-spaced
    alpha scan across the nearest excluded neighbour's band (widened half a
    coarse cell each way) -- the unimodality guard -- then, if nothing
    reached the level, a golden-section refinement of the scan's peak.
    Excluded probe masses get their edges refined (origin "tip") from the
    scan's own bracketing points and join the polyline.

    Open-topped bands have no ceiling to trace: the alpha scan then runs up to
    the alpha-axis top, and a probe mass whose exclusion reaches that top
    yields another open-topped column (floor refined, ceiling None). The mass
    bisection -- the tip itself -- is unchanged.
    """
    t0 = time.time()
    history: list = []
    cell, tol_dex = task["cell_dex"], task["tol_dex"]
    alpha_top = task["alpha_top"]
    m_in, m_out = task["m_excluded"], task["m_not_excluded"]
    band = dict(task["band"])          # floor/ceiling of the current inner col
    verts, n_scans = [], 0
    while abs(np.log10(m_out / m_in)) > task["mass_res"]:
        m_try = float(np.sqrt(m_in * m_out))
        open_top = band["ceiling"] is None
        lo = band["floor"] * 10.0 ** (-0.5 * cell)
        hi = alpha_top if open_top else band["ceiling"] * 10.0 ** (0.5 * cell)
        grid = np.geomspace(lo, hi, TIP_SCAN_POINTS)
        ps = np.array([oracle(a, m_try, history, "tip")[0] for a in grid])
        n_scans += 1
        excluded = bool((ps >= LEVEL).any())
        if not excluded:
            # unimodality guard + golden-section: refine around the peak.
            k = int(np.argmax(ps))
            interior = 0 < k < TIP_SCAN_POINTS - 1
            g_lo = grid[max(k - 1, 0)]
            g_hi = grid[min(k + 1, TIP_SCAN_POINTS - 1)]
            best = ps[k]
            if interior:
                phi = (np.sqrt(5.0) - 1.0) / 2.0
                x_lo, x_hi = np.log10(g_lo), np.log10(g_hi)
                for _ in range(TIP_GOLDEN_CALLS):
                    x1 = x_hi - phi * (x_hi - x_lo)
                    x2 = x_lo + phi * (x_hi - x_lo)
                    p1, _ = oracle(10 ** x1, m_try, history, "tip")
                    p2, _ = oracle(10 ** x2, m_try, history, "tip")
                    best = max(best, p1, p2)
                    if best >= LEVEL:
                        break
                    if p1 >= p2:
                        x_hi = x2
                    else:
                        x_lo = x1
            excluded = bool(best >= LEVEL)
        if excluded:
            pts = sorted((h for h in history
                          if h[5] == "tip" and h[0] == m_try),
                         key=lambda h: h[1])
            a_arr = np.array([h[1] for h in pts])
            p_arr = np.array([h[2] for h in pts])
            exc = np.where(p_arr >= LEVEL)[0]
            i0, i1 = exc[0], exc[-1]
            f_lo = a_arr[i0 - 1] if i0 > 0 else a_arr[0] * 10 ** (-cell)
            # the sub-column stays open-topped only if this band is open AND
            # its exclusion runs all the way to the scan's top (the axis top)
            sub_open = bool(open_top and i1 == len(a_arr) - 1)
            c_hi = a_arr[i1 + 1] if i1 < len(a_arr) - 1 \
                else a_arr[-1] * 10 ** cell
            sub = dict(m=m_try, im=-1, origin="tip", tol_dex=tol_dex,
                       cell_dex=cell, validate=True,
                       open_top=sub_open, alpha_top=alpha_top,
                       floor_bracket=(float(f_lo), float(a_arr[i0])),
                       ceil_bracket=(None if sub_open
                                     else (float(a_arr[i1]), float(c_hi))))
            res = refine_column(sub)
            history.extend(res.pop("history"))
            if res["excluded"]:
                verts.append(res)
                band = dict(floor=res["floor"], ceiling=res["ceiling"])
                m_in = m_try
            else:
                m_out = m_try
        else:
            m_out = m_try
    return dict(side=task["side"], vertices=verts,
                m_excluded=float(m_in), m_not_excluded=float(m_out),
                n_scans=n_scans, history=history, wall_s=time.time() - t0)


# --------------------------------------------------------------------------- #
# Per-surface driver
# --------------------------------------------------------------------------- #
def _run_pool(tasks, fn, workers, ctx):
    if not tasks:
        return []
    if workers == 1:
        return [fn(t) for t in tasks]
    out = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
        futs = [ex.submit(fn, t) for t in tasks]
        for f in as_completed(futs):
            out.append(f.result())
    return out


def refine_surface(name, plane, alphas, ms, args, ctx, mcut=None):
    """All three phases for one surface; returns (vertices, meta, evals).

    ``mcut`` is :func:`resolve_m_cut`'s dict for this surface's plane. The
    truncation is active only when the exclusion actually runs past the cut
    (excluded cube columns above it): the surface then loses those columns,
    gains an exact column at m = m_cut, and traces no right tip. A surface
    whose island closes below the cut -- or a cube that records none -- is
    refined exactly as before.
    """
    t0 = time.time()
    columns = ([int(c) for c in args.columns.split(",")]
               if args.columns else None)
    alpha_top = float(alphas[-1])
    tasks, widened, cell = grid_column_tasks(plane, alphas, ms,
                                             args.tol_alpha, columns)
    mcut = mcut or dict(m_cut=None, n_req=None, b_cap=None)
    m_cut = mcut["m_cut"]
    tasks, n_dropped = drop_above_m_cut(tasks, m_cut)
    truncated = bool(n_dropped)
    n_open = sum(bool(t["open_top"]) for t in tasks)
    print(f"  [{name}] {len(tasks)} excluded cube columns"
          + (f" ({n_open} open-topped: no ceiling)" if n_open else "")
          + (f" (restricted to {columns})" if columns else "")
          + (f"; {len(widened)} bracket(s) widened at marginal cells"
             if widened else ""))
    if truncated:
        print(f"  [{name}] m_cut = {m_cut:.4e} GeV (N_req={mcut['n_req']}, "
              f"b_cap={mcut['b_cap']} m): {n_dropped} excluded cube column(s) "
              f"above it dropped; an exact m_cut column will close the polyline")
    results = _run_pool(tasks, refine_column, args.workers, ctx)
    verts = [r for r in results if r["excluded"]]
    evals = [h for r in results for h in r["history"]]
    for r in results:
        r.pop("history", None)
        if not r["excluded"]:
            print(f"  [{name}] WARNING: grid column im={r['im']} "
                  f"m={r['m']:.3e} failed: {r['flags']}")

    # -- phase 2a: the exact m_cut column ----------------------------------- #
    # Refined by the wall-insertion path (validated brackets, one coarse alpha
    # cell either side of the last refined column's edges, open-topped iff that
    # column is), so the release's right edge is an ordinary refined vertex --
    # same oracle, same seeding, same tolerance. It is inserted BEFORE the
    # insertion waves so the wall between it and the last cube column is
    # resolved like any other.
    n_inserted = 0
    if truncated and verts:
        verts.sort(key=lambda v: v["m"])
        v_last = verts[-1]
        res = _run_pool([inserted_task(m_cut, v_last, v_last, cell,
                                       args.tol_alpha, alpha_top)],
                        refine_column, 1, ctx)[0]
        evals.extend(res.pop("history"))
        if res["excluded"]:
            res["flags"].append("m_cut_truncation")
            verts.append(res)
            n_inserted += 1
            print(f"  [{name}] m_cut column refined: floor={res['floor']:.4e}"
                  + (", no ceiling (open-topped)" if res["ceiling"] is None
                     else f", ceiling={res['ceiling']:.4e}"))
        else:
            print(f"  [{name}] WARNING: no exclusion at m_cut "
                  f"({m_cut:.4e} GeV): {res['flags']}; the polyline stops at "
                  f"the last refined column below the cut")

    # -- phase 2: adaptive mass insertion (waves; pairs are independent) ---- #
    if not args.no_insert:
        while True:
            verts.sort(key=lambda v: v["m"])
            wave = []
            for va, vb in zip(verts, verts[1:]):
                gap = np.log10(vb["m"] / va["m"])
                if gap <= args.mass_res:
                    continue
                jump = abs(np.log10(vb["floor"] / va["floor"]))
                if va["ceiling"] is not None and vb["ceiling"] is not None:
                    jump = max(jump,
                               abs(np.log10(vb["ceiling"] / va["ceiling"])))
                m_new = float(np.sqrt(va["m"] * vb["m"]))
                if m_cut is not None and m_new > m_cut:
                    continue                 # never refine past the right edge
                if jump > args.tol_wall:
                    wave.append(inserted_task(m_new, va, vb, cell,
                                              args.tol_alpha, alpha_top))
            if not wave or n_inserted >= args.max_insert:
                if wave:
                    print(f"  [{name}] WARNING: insertion cap "
                          f"({args.max_insert}) reached with "
                          f"{len(wave)} jumps unresolved")
                break
            room = args.max_insert - n_inserted
            wave = wave[:room]
            res = _run_pool(wave, refine_column, args.workers, ctx)
            n_inserted += len(wave)
            for r in res:
                evals.extend(r.pop("history"))
                if r["excluded"]:
                    verts.append(r)
            print(f"  [{name}] insertion wave: {len(wave)} columns "
                  f"({n_inserted} total)")

    # -- phase 3: tips ------------------------------------------------------ #
    tips = {}
    if not args.no_tips and columns is None and verts:
        verts.sort(key=lambda v: v["m"])
        grid_ims = sorted(v["im"] for v in verts if v["im"] >= 0)
        tip_tasks = []
        for side, v_edge, im_edge, step in (
                ("left", verts[0], grid_ims[0], -1),
                ("right", verts[-1], grid_ims[-1], +1)):
            if side == "right" and truncated:
                # No tip on this side to find: the island does not end inside
                # the cube, the release's mass cut ends it, and the polyline's
                # last vertex is the exact m_cut column.
                tips[side] = dict(side=side, cut_at_m_cut=True,
                                  m_cut_gev=float(m_cut),
                                  n_req=mcut["n_req"], b_cap_m=mcut["b_cap"],
                                  n_scans=0, wall_s=0.0)
                print(f"  [{name}] right island end is the mass cut "
                      f"(m_cut={m_cut:.4e} GeV); no tip to trace")
                continue
            im_out = im_edge + step
            if not (0 <= im_out < len(ms)):
                # The exclusion runs off the end of the cube's mass axis on
                # this side (the massless f_DM = 1 surface is still excluded at
                # m_Pl), so there is no coarse column beyond it to bisect
                # against and no tip inside the cube to localize. The polyline
                # simply ends at the axis; say so instead of failing, and never
                # invent a tip past the last computed column.
                tips[side] = dict(side=side, open_at_mass_axis_edge=True,
                                  m_excluded=float(v_edge["m"]),
                                  m_not_excluded=None, n_scans=0, wall_s=0.0)
                print(f"  [{name}] {side} island end sits on the cube's "
                      f"mass-axis edge (m={v_edge['m']:.4e}); no tip to trace")
                continue
            tip_tasks.append(dict(
                side=side, m_excluded=v_edge["m"],
                m_not_excluded=float(ms[im_out]),
                band=dict(floor=v_edge["floor"], ceiling=v_edge["ceiling"]),
                cell_dex=cell, tol_dex=args.tol_alpha, alpha_top=alpha_top,
                mass_res=args.mass_res))
        res = _run_pool(tip_tasks, trace_tip, min(args.workers, 2), ctx)
        for r in res:
            evals.extend(r.pop("history"))
            verts.extend(r.pop("vertices"))
            side = r.pop("side")
            tips[side] = r
            print(f"  [{name}] {side} tip: excluded down/up to "
                  f"m={r['m_excluded']:.4e} ({r['n_scans']} mass probes)")

    verts.sort(key=lambda v: v["m"])
    wall = time.time() - t0
    calls = len(evals)
    meta = dict(cell_dex=cell, n_grid_columns=len(tasks),
                # open-topped: no ceiling. Over the whole polyline, and over
                # the cube's own columns (the subset of n_grid_columns).
                n_open_top_columns=int(sum(v["ceiling"] is None
                                           for v in verts)),
                n_open_top_grid_columns=int(n_open),
                n_inserted=n_inserted, widened=widened, tips=tips,
                # the release's right edge: the cut this cube records for the
                # surface's f_DM plane (null when it records none, or
                # --no-m-cut), and whether it actually truncated this island.
                m_cut_gev=(None if m_cut is None else float(m_cut)),
                m_cut_truncated=bool(truncated),
                n_oracle_calls=calls, wall_s=wall)
    med = float(np.median([h[4] for h in evals])) if evals else float("nan")
    print(f"  [{name}] {len(verts)} vertices, {calls} oracle calls "
          f"(median {med:.2f} s/call), wall {wall:.0f} s")
    return verts, meta, evals


# --------------------------------------------------------------------------- #
# Spot check: the oracle must reproduce cube cells bit-for-bit
# --------------------------------------------------------------------------- #
def spot_check(tasks, plane, alphas, n_spot, rng):
    """Recompute ``n_spot`` coarse bracket cells; compare to the cube exactly.

    Returns the worst |dp| in the cube's storage precision -- 0.0 when every
    recomputed cell is bit-identical to the released one.
    """
    if not tasks:
        return 0.0
    plane = np.nan_to_num(np.asarray(plane, float), nan=0.0)
    picks = rng.choice(len(tasks), size=min(n_spot, len(tasks)), replace=False)
    worst = 0.0
    for k in picks:
        t = tasks[int(k)]
        for edge in ("floor", "ceiling"):
            cells = t["coarse_cells"][edge]
            if cells is None:
                continue            # open-topped column: no ceiling cell pair
            ia = cells[0 if edge == "floor" else 1]
            p, _mu = oracle(float(alphas[ia]), t["m"])
            p_cube = plane[ia, t["im"]]
            d = abs(float(CUBE_P_DTYPE(p)) - float(CUBE_P_DTYPE(p_cube)))
            worst = max(worst, d)
            status = ("OK (bit-exact in the cube's float32)" if d == 0.0
                      else f"DIFF {d:.3e}")
            print(f"    spot im={t['im']:3d} ia={ia:2d} "
                  f"p_cube={p_cube:.6f} p_oracle={p:.6f}  {status}")
    if worst != 0.0:
        print(f"    WARNING: spot check max |dp| = {worst:.3e} (expected 0); "
              f"the environment does not reproduce the cube bit-for-bit and "
              f"the refined boundary is only approximately tied to it")
    return float(worst)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _missing_plane(rel, f_dm, atm):
    """Why this cube cannot serve the (f_dm, atmosphere) plane; [] if it can."""
    have_f = [float(v) for v in rel.f_dm_values]
    have_a = [bool(v) for v in rel.atmosphere_values]
    miss = []
    if not any(np.isclose(v, float(f_dm), rtol=1e-12, atol=0.0)
               for v in have_f):
        miss.append(f"f_dm={f_dm} (this file carries {have_f})")
    if bool(atm) not in have_a:
        miss.append(f"atmosphere={bool(atm)} (this file carries {have_a})")
    return miss


def _require_plane(rel, path, name, f_dm, atm):
    """Fail with a file-level message when this cube has no such plane.

    From v7 the release ships one file per hypothesis plane (File A: f_DM = 1
    with atmosphere; File B: f_DM = 0.1 without), and the reader's raw
    KeyError/ValueError does not say which file to reach for.
    """
    miss = _missing_plane(rel, f_dm, atm)
    if not miss:
        return
    want = ("the f_DM = 1 / atmosphere file (v8: File A, "
            "release/luhdm_datarelease_v9_A_f1_atm.h5)"
            if (float(f_dm) == 1.0 and atm) else
            "the f_DM = 0.1 / no-atmosphere file (v8: File B, "
            "release/luhdm_datarelease_v9_B_f0p1_noatm.h5)"
            if (float(f_dm) == 0.1 and not atm) else
            "the release file that carries this plane")
    raise SystemExit(
        f"surface {name!r} needs a plane this cube does not have: "
        + "; ".join(miss) + f".\n  cube: {path}\n"
        f"  point --release at {want}, or drop {name!r} from --surfaces "
        f"(from v7 the release splits the hypotheses across files; "
        f"'--surfaces all' is the figure surfaces THIS cube carries).")


def main():
    global QS, EFF, EVENTS, V_I, XS, LAMB_ODE, ATMOSPHERE, FID, Q_MIN, T_TOTAL
    global F_SCALE, LEVEL, SE_P

    ap = argparse.ArgumentParser(
        description="Refine 95% CL island boundaries by bracketed bisection "
                    "of the release cube's own statistic.")
    ap.add_argument("--release", type=Path, default=release.DEFAULT_PATH)
    ap.add_argument("--mode", type=int, choices=(1, 2, 3), default=1)
    ap.add_argument("--surfaces", default="all",
                    help=f"comma list of {sorted(SURFACES)} or 'all' "
                         f"(= the figure surfaces {list(FIGURE_SURFACES)} "
                         f"that the --release cube actually carries)")
    ap.add_argument("--out", type=Path, required=True,
                    help="sidecar JSON path (refined polylines + provenance)")
    ap.add_argument("--evals-dir", type=Path, default=None,
                    help="write one NPZ of every oracle evaluation per "
                         "surface here (audit/debug)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--tol-alpha", type=float, default=TOL_ALPHA_DEX)
    ap.add_argument("--tol-wall", type=float, default=TOL_WALL_DEX)
    ap.add_argument("--mass-res", type=float, default=MASS_RES_DEX)
    ap.add_argument("--columns", default=None,
                    help="comma list of coarse mass-column indices "
                         "(validation runs; disables tips)")
    ap.add_argument("--max-insert", type=int, default=MAX_INSERT_PER_SURFACE,
                    help="runaway guard: stop inserting mass columns after "
                         "this many per surface (default: %(default)s). A run "
                         "that hits it says so and leaves those wall jumps at "
                         "coarse resolution")
    ap.add_argument("--no-insert", action="store_true")
    ap.add_argument("--no-tips", action="store_true")
    ap.add_argument("--no-m-cut", action="store_true",
                    help="refine the stored (uncapped) surfaces past the "
                         "cube's flux mass cut: no dropped columns, no exact "
                         "m_cut column, right tips traced as usual. Audit "
                         "escape hatch; the released contour truncates at "
                         "m_cut")
    ap.add_argument("--spot", type=int, default=0,
                    help="recompute this many coarse bracket cells per surface "
                         "and compare to the cube bit-for-bit")
    ap.add_argument("--massless-lamb", type=float, default=2.0,
                    help="ODE regulator range for the massless slice [m] "
                         "(must equal the cube build's value)")
    args = ap.parse_args()
    args.workers = args.workers or os.cpu_count()

    want_all = args.surfaces == "all"
    names = sorted(FIGURE_SURFACES) if want_all else args.surfaces.split(",")
    for n in names:
        if n not in SURFACES:
            ap.error(f"unknown surface {n!r}; choose from {sorted(SURFACES)}")

    t_start = time.time()
    with release.open_release(args.release) as rel:
        attrs = dict(rel.attrs)
        FID = json.loads(attrs["fid_json"])
        Q_MIN = float(attrs["q_thresh_gev"])
        T_TOTAL = float(attrs["t_exposure_s"])
        LEVEL = float(attrs.get("confidence_recommended", 0.95))
        b_cap = rel.b_constrained_max
        seed = int(attrs["seed"])
        assert seed == SEED, f"cube seed {seed} != pinned {SEED}"
        assert T_TOTAL == float(config.T_EXPOSURE), \
            "config.T_EXPOSURE differs from the cube's exposure; set " \
            "LUHDM_T_EXPOSURE to match or the oracle will not reproduce it"
        eff_path = Path(efficiency.table_path())
        eff_sha = sha256_file(eff_path)
        assert eff_sha == attrs["efficiency_npz_sha256"], \
            f"efficiency table {eff_path} sha mismatch vs the cube"
        SE_P = float(np.sqrt(LEVEL * (1 - LEVEL) / FID["n_mc"]))

        # Two conventions travel with the cube as optional attributes: the
        # massless slice's kinematic endpoint factor (that slice only) and the
        # projection kernel (every slice). luhdm.cross_section implements both
        # kernels, so the cube's kernel is DISPATCHED into every make_xsec
        # below; luhdm.rate has no endpoint-factor switch, so a cube asking for
        # anything but the historical 1.0 cannot be reproduced by this build
        # and must not be silently refined against the wrong physics.
        q_epf = float(attrs.get("massless_q_endpoint_factor", 1.0))
        kernel = str(attrs.get("projection_kernel", KERNEL_DEFAULT))
        assert q_epf == 1.0, (
            f"cube asks for massless_q_endpoint_factor={q_epf}; this build of "
            f"luhdm.rate implements only the historical 1.0 convention")
        assert kernel in cross_section._KERNELS, (
            f"cube asks for projection_kernel={kernel!r}; this build of "
            f"luhdm.cross_section implements {cross_section._KERNELS}")

        EVENTS = rel.events(args.mode).astype(float)
        if want_all:
            # 'all' is the figure surfaces THIS cube carries; a plane in
            # another file of the split release is skipped out loud, never
            # silently. An explicitly named surface still hard-errors below.
            keep = [n for n in names
                    if not _missing_plane(rel, SURFACES[n][1], SURFACES[n][2])]
            dropped = [n for n in names if n not in keep]
            if dropped:
                print(f"--surfaces all: {sorted(dropped)} not carried by this "
                      f"cube (f_dm {rel.f_dm_values}, atmosphere "
                      f"{rel.atmosphere_values}); refining {sorted(keep)}")
            if not keep:
                raise SystemExit(
                    f"this cube carries none of the figure surfaces "
                    f"{sorted(FIGURE_SURFACES)}\n  cube: {args.release}")
            names = keep
        planes, lam_m, axes_of = {}, {}, {}
        for n in names:
            tag, f_dm, atm = SURFACES[n]
            _require_plane(rel, args.release, n, f_dm, atm)
            planes[n] = rel.mass_plane("extremeness", mode=args.mode, lam=tag,
                                       atmosphere=atm, f_dm=f_dm)
            grp = "atm" if atm else "noatm"
            ax = rel.axes if atm else rel.axes_noatm
            axes_of[n] = (ax.alpha_n.copy(), ax.mass_gev.copy())
            il = rel.at_lambda(tag, grp)
            lam_m[n] = float(ax.lambda_m[il])
    # h5 handle closed before any fork.

    QS = np.geomspace(Q_MIN, FID["q_span"] * Q_HI_REF, FID["n_q"])
    EFF = efficiency.make_efficiency(args.mode, DF)(QS)
    V_I = atmosphere.sample_shm(FID["n_shm"], rng=np.random.default_rng(SEED))
    cube_sha = sha256_file(args.release)
    try:
        git_sha = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[1]),
             "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        git_sha = None

    print(f"release: {args.release} ({attrs.get('version_tag')})  "
          f"b_cap={b_cap}  kernel={kernel}  FID={FID}")
    print(f"mode {args.mode}, level {LEVEL}, tol_alpha {args.tol_alpha} dex, "
          f"tol_wall {args.tol_wall} dex, mass_res {args.mass_res} dex, "
          f"workers {args.workers}")
    if args.no_m_cut:
        print("--no-m-cut: contours run past the cube's flux mass cut "
              "(the released convention truncates at m_cut)")

    def write_sidecar(done):
        """Assemble and atomically write the sidecar for the surfaces DONE.

        Called after every surface, not only at the end: a finite-lambda
        surface is hours of ODE, and a run that is interrupted (or watched)
        must leave a complete, self-describing sidecar for everything already
        refined rather than nothing at all. ``surfaces_done`` names them in the
        order they were refined, so a partial file is never mistaken for a
        full one.
        """
        out = dict(
            format="luhdm-refined-contours", schema_version=1,
            confidence=LEVEL,
            provenance=dict(
                cube_path=str(args.release), cube_sha256=cube_sha,
                # what this file covers: a run interrupted between surfaces
                # leaves a complete sidecar for a SUBSET, and these two say so.
                surfaces_requested=list(names), surfaces_done=list(done),
                cube_version_tag=attrs.get("version_tag"),
                cube_git_commit=attrs.get("git_commit"),
                cube_fid=FID, seed=SEED, mu_round_dex=0.02,
                seed_policy=("fresh optimum_interval table per "
                             "0.02-dex-rounded mu, seed identical to the "
                             "cube build "
                             "(build_release.PerMuTable); p is a pure "
                             "function of (alpha, m, surface)"),
                b_constrained_max_m=b_cap, t_exposure_s=T_TOTAL,
                q_thresh_gev=Q_MIN, massless_lamb_ode_m=args.massless_lamb,
                massless_q_endpoint_factor=q_epf, projection_kernel=kernel,
                efficiency_npz_sha256=attrs.get("efficiency_npz_sha256"),
                events_sha256=attrs.get(f"events_mode{args.mode}_sha256"),
                # --spot evidence: max |p_oracle - p_cube| over the recomputed
                # bracket cells, in the cube's float32 storage precision.
                # 0.0 == the oracle reproduced the released cells bit for bit.
                spot_n_cells=args.spot, spot_max_dp=spot_max,
                tolerances=dict(tol_alpha_dex=args.tol_alpha,
                                tol_wall_dex=args.tol_wall,
                                mass_res_dex=args.mass_res,
                                bracket_widen_se=BRACKET_WIDEN_SE,
                                max_widen_cells=MAX_WIDEN_CELLS,
                                max_insert_per_surface=args.max_insert),
                # No hostname: the sidecar ships with the release and
                # carries no host identifiers (package versions cover
                # reproducibility) -- the same rule as
                # assemble_release.write_provenance.
                refiner_git_sha=git_sha, argv=sys.argv,
                created=datetime.now(timezone.utc).isoformat(),
                wall_s=time.time() - t_start,
                # the MC calibration is a pure function of (rounded mu,
                # seed, n_mc) only within one build of these three.
                packages=dict(numpy=np.__version__,
                              scipy=scipy.__version__,
                              optimum_interval=getattr(optimum_interval,
                                                       "__version__", "?"),
                              python=platform.python_version(),
                              luhdm=getattr(luhdm, "__version__", "?"))),
            surfaces=surfaces_out)
        # Home-relativise every path string before the sidecar is written:
        # it ships with the release and carries no home paths or usernames.
        out = scrub_home(out)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.out.with_suffix(args.out.suffix + ".tmp")
        tmp.write_text(json.dumps(out, indent=1))
        os.replace(tmp, args.out)
        print(f"wrote {args.out}  ({time.time() - t_start:.0f} s total)")

    ctx = multiprocessing.get_context("fork")
    rng = np.random.default_rng(0)
    surfaces_out, spot_max = {}, {}
    for n in names:
        tag, f_dm, atm = SURFACES[n]
        alphas, ms = axes_of[n]
        lamb = None if not np.isfinite(lam_m[n]) else lam_m[n]
        LAMB_ODE = args.massless_lamb if lamb is None else lamb
        ATMOSPHERE = atm
        # Tabulation density is resolved from xi inside rate.make_xsec, so the
        # handle matches the builder's byte for byte (build_release, same call).
        XS = rate.make_xsec(lamb, b_constrained_max=b_cap,
                            projection_kernel=kernel)
        F_SCALE = f_dm / float(config.F_X)
        print(f"[surface {n}] lambda={tag} ({lam_m[n]} m), f_dm={f_dm}, "
              f"atmosphere={atm}, F_SCALE={F_SCALE:g}"
              + (f", q_endpoint_factor={q_epf:g}" if lamb is None else ""))
        mcut = resolve_m_cut(attrs, f_dm, enabled=not args.no_m_cut)
        if args.spot:
            tasks, _w, _c = grid_column_tasks(
                planes[n], alphas, ms, args.tol_alpha,
                [int(c) for c in args.columns.split(",")] if args.columns
                else None)
            # spot-check the cells the refinement will actually use
            tasks, _n_cut = drop_above_m_cut(tasks, mcut["m_cut"])
            spot_max[n] = spot_check(tasks, planes[n], alphas, args.spot, rng)
        verts, meta, evals = refine_surface(n, planes[n], alphas, ms,
                                            args, ctx, mcut)
        surfaces_out[n] = dict(
            mode=args.mode, lambda_tag=tag,
            lambda_m=(None if lamb is None else lamb), f_dm=f_dm,
            atmosphere=bool(atm),
            mass_gev=[v["m"] for v in verts],
            floor_alpha_n=[v["floor"] for v in verts],
            # null ceiling == the band is open at the top of the alpha axis
            ceiling_alpha_n=[v["ceiling"] for v in verts],
            open_top=[bool(v["ceiling"] is None) for v in verts],
            origin=[v["origin"] for v in verts],
            coarse_im=[v["im"] for v in verts],
            floor_bracket_alpha=[v["floor_bracket"] for v in verts],
            ceiling_bracket_alpha=[v["ceil_bracket"] for v in verts],
            flags={str(v["m"]): v["flags"] for v in verts if v["flags"]},
            **{k: v for k, v in meta.items() if k != "widened"},
            widened_columns=meta["widened"])
        if args.evals_dir and evals:
            args.evals_dir.mkdir(parents=True, exist_ok=True)
            h = np.array([e[:5] for e in evals], float)
            np.savez(args.evals_dir / f"evals_{n}.npz",
                     m=h[:, 0], alpha=h[:, 1], p=h[:, 2], mu=h[:, 3],
                     call_s=h[:, 4], kind=np.array([e[5] for e in evals]))
        # bank this surface before starting the next one
        write_sidecar([k for k in names if k in surfaces_out])


if __name__ == "__main__":
    main()
