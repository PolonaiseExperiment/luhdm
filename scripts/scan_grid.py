#!/usr/bin/env python3
"""Parallel (m_DM, alpha_n) extremeness scan for the limit contours.

Runs the same physics as notebooks/01_the_limit.ipynb, through the shared
luhdm.rate module (so script and notebook cannot drift), parallelized over
grid points (each is independent: attenuation ODE -> dR/dq ->
optimum-interval extremeness). Designed for a many-core node; writes an npz
cache that the notebook loads if present.

    python scripts/scan_grid.py --lamb 2e-4 --out scan7_200um.npz
    python scripts/scan_grid.py --massless --lamb 2.0 --out scan7_massless.npz
    python scripts/scan_grid.py --quick         # small smoke test
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one process per core, no BLAS threads
os.environ.setdefault("TQDM_DISABLE", "1")

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from luhdm import atmosphere, config, efficiency, halo, limits, rate

# --- fiducials (as in the notebook) ---
Q_THRESH = config.Q_THRESH
# Fixed upper-momentum reference for the qs grid (upper edge = q_span x this).
# Decoupled from Q_THRESH / --q-min so lowering the analysis threshold does not
# shrink the integration window (it must stay well above the highest event).
Q_HI_REF = 8.4e3
T_TOTAL = config.T_EXPOSURE  # dataset live-time (single source of truth in luhdm.config)
SEED = 20260702
# Optimum-interval p==1 shortcut used when the caller's fidelity dict carries no
# explicit "mu_cap". Kept at the historical 40.0 so shards/caches built before
# the cap was recorded still recompute bit-for-bit; build_release.py pins its own
# (currently 85.0) into the fidelity string it writes.
MU_CAP_DEFAULT = 40.0
# Two-tier MC: lower edge of the upgrade band used when the caller's fidelity
# dict names an "n_mc_hi" but no explicit "p_hi_lo". Same value as
# build_release.P_HI_LO and refine_contours.P_HI_LO.
P_HI_LO_DEFAULT = 0.90
# Optimum-interval MC calibration granularity [dex of mu] used when the caller's
# fidelity dict names no explicit "mu_dex". Kept at limits.extremeness_and_mu's
# own default so every shard/cache built before mu_dex was recorded recomputes
# bit-for-bit; a build_release --mu-dex build pins its value in the fidelity
# string, and this path (verify_release V2) must honour it or every near-boundary
# cell of a fine-mu cube lands in the wrong mu bin. Same value as
# build_release.MU_DEX and refine_contours.MU_DEX.
MU_DEX_DEFAULT = 0.02
# default observed-event list; --data overrides it (e.g. per-mode data_mode{n}.txt)
DEFAULT_DATA = Path(__file__).resolve().parent.parent / "notebooks" / "data_mode1.txt"

# Shared, read-only state; set in main() BEFORE the fork so children inherit it.
LAMB = None      # atmospheric-regulator range [m] (also the sensor range
                 # unless --massless)
XS = None        # cross-section handle from rate.make_xsec
V_I_SAMPLES = None
FID = None       # fidelity dict; optional "mu_cap" key selects the
                 # optimum-interval p==1 shortcut (absent -> MU_CAP_DEFAULT, the
                 # historical value every pre-mu_cap shard was built with).
                 # Optional "n_mc_hi"/"p_hi_lo" keys select the two-tier MC
                 # upgrade (absent -> single tier, the historical behaviour).
                 # Optional "mu_dex" key sets the MC calibration granularity
                 # (absent -> MU_DEX_DEFAULT, the historical 0.02 dex)
EVENTS = None
Q_MIN = None     # lower edge of the momentum grid / analysis threshold [GeV]
EFF = None       # detection-efficiency callable eps(q_GeV), or None for raw rate
NO_ATM = False   # skip overburden attenuation; use the bare halo velocity dist

_worker_state: dict = {}


def _worker_init_lazy():
    """Per-process state, built on first use in each child.

    When the fidelity dict carries "n_mc_hi" (the two-tier MC contract written
    by build_release under --n-mc-hi), a SECOND seeded table serves the hi tier:
    upgraded cells must draw from a table that only ever generated n_mc_hi
    trials (seed SEED+1), exactly as the builder's PerMuTable pair does, or a
    single-cell recompute would not be bit-identical.
    """
    if "table" not in _worker_state:
        _worker_state["table"] = limits.new_table(seed=SEED)
        if FID and FID.get("n_mc_hi"):
            _worker_state["table_hi"] = limits.new_table(seed=SEED + 1)
    return _worker_state


def scan_point(task):
    """One (alpha_n, m) grid point -> (ia, im, extremeness, mu, n_transit)."""
    ia, im, alpha_n, m = task
    state = _worker_init_lazy()
    try:
        v_min = Q_MIN / m / 10   # ODE floor follows the analysis threshold
        if NO_ATM:
            # bare standard halo distribution; no overburden attenuation, so the
            # velocity distribution is the same for every (m, alpha) grid point
            f_v_f = halo.standard_halo_model
        else:
            v_f_samples = atmosphere.compute_v_f_distribution(
                alpha_n, LAMB, m, V_I_SAMPLES, v_min=v_min, n_grid=FID["n_ode"])
            f_v_f = atmosphere.compute_f_vf(v_f_samples, v_min)[0]
        qs = np.geomspace(Q_MIN, FID["q_span"] * Q_HI_REF, FID["n_q"])
        diff_rate = rate.differential_rate_trapz(qs, alpha_n, m, f_v_f, XS, eff=EFF)
        # the MC calibration granularity is part of the cell contract (it picks
        # the mu bin the seeded table is keyed on), so it comes from the caller's
        # fidelity dict — a fine-mu cube's cells only recompute bit-exact here
        mu_dex = FID.get("mu_dex", MU_DEX_DEFAULT)
        p, mu = limits.extremeness_and_mu(
            state["table"], EVENTS, qs, diff_rate, T_TOTAL, n_mc=FID["n_mc"],
            mu_cap=FID.get("mu_cap", MU_CAP_DEFAULT), mu_dex=mu_dex)
        # two-tier MC: near-boundary cells are re-evaluated on the hi-tier
        # table — same rule as build_release.eval_extremeness. Inert for
        # fidelity dicts without "n_mc_hi" (every single-tier cache/shard).
        # Both tiers share mu_dex: it is a property of the mu axis, not of n_mc.
        n_hi = FID.get("n_mc_hi")
        if n_hi and FID.get("p_hi_lo", P_HI_LO_DEFAULT) <= p < 1.0:
            p, mu = limits.extremeness_and_mu(
                state["table_hi"], EVENTS, qs, diff_rate, T_TOTAL, n_mc=n_hi,
                mu_cap=FID.get("mu_cap", MU_CAP_DEFAULT), mu_dex=mu_dex)
        n_t = rate.expected_transits(alpha_n, m, f_v_f, XS, T_TOTAL)
    except Exception as err:  # absurd-coupling corners: report, exclude nothing
        print(f"point (a={alpha_n:.1e}, m={m:.1e}) failed: {err}", flush=True)
        p, mu, n_t = 0.0, 0.0, 0.0
    return ia, im, p, mu, n_t


def main():
    global XS, V_I_SAMPLES, FID, EVENTS, LAMB, Q_MIN, EFF, NO_ATM

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--lamb", type=float, default=2e-4,
                    help="mediator range in meters")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="scan_results.npz")
    ap.add_argument("--q-min", type=float, default=Q_THRESH,
                    help="lower edge of the momentum grid / analysis threshold "
                         "in GeV (default: config.Q_THRESH = %(default)g)")
    ap.add_argument("--m-min", type=float, default=10 ** 6.4,
                    help="lowest dark-matter mass in the grid, GeV "
                         "(default: %(default).3g)")
    ap.add_argument("--a-min", type=float, default=10 ** -8.7,
                    help="lowest coupling alpha_n in the grid "
                         "(default: %(default).3g)")
    ap.add_argument("--mode", type=int, choices=(1, 2, 3), default=None,
                    help="sensor mode: fold in its measured detection "
                         "efficiency (default: none, raw rate)")
    ap.add_argument("--df", type=int, choices=(2, 3), default=3,
                    help="efficiency dof hypothesis (default: 3)")
    ap.add_argument("--flat-efficiency", action="store_true",
                    help="debug: force efficiency == 1 everywhere (must "
                         "reproduce the no-efficiency result)")
    ap.add_argument("--n-m", type=int, default=None,
                    help="override number of mass grid points (coarser = faster)")
    ap.add_argument("--n-a", type=int, default=None,
                    help="override number of coupling grid points")
    ap.add_argument("--n-mc", type=int, default=None,
                    help="override optimum-interval Monte-Carlo pseudo-experiments")
    ap.add_argument("--n-mc-hi", type=int, default=None,
                    help="two-tier MC: re-evaluate points whose extremeness "
                         "lands in [--p-hi-lo, 1.0) on a separate table with "
                         "this many trials (seed SEED+1); default off")
    ap.add_argument("--p-hi-lo", type=float, default=P_HI_LO_DEFAULT,
                    help="lower edge of the two-tier upgrade band "
                         "(default %(default)g; ignored without --n-mc-hi)")
    ap.add_argument("--mu-dex", type=float, default=MU_DEX_DEFAULT,
                    help="optimum-interval MC calibration granularity [dex of "
                         "mu] (default %(default)g); recorded in the cache's "
                         "fidelity string only when it differs from the default")
    ap.add_argument("--n-ode", type=int, default=None,
                    help="override attenuation-ODE grid points")
    ap.add_argument("--no-atmosphere", action="store_true",
                    help="ignore overburden attenuation; use the bare standard "
                         "halo velocity distribution for all (m, alpha)")
    ap.add_argument("--data", default=str(DEFAULT_DATA),
                    help="event list file, one impulse per line in eV "
                         "(default: notebooks/data.txt)")
    ap.add_argument("--massless", action="store_true",
                    help="analytic Rutherford dsigma/dq at the sensor; --lamb "
                         "then only regulates the atmospheric Coulomb log")
    ap.add_argument("--projection-kernel",
                    choices=("planar-signed", "isotropic-folded"),
                    default="planar-signed",
                    help="projected-dsigma/dq kernel convention (default: the "
                         "shipped planar-signed kernel, byte-identical)")
    args = ap.parse_args()
    LAMB = args.lamb
    Q_MIN = args.q_min
    if args.flat_efficiency:
        EFF = lambda q: np.ones_like(np.asarray(q, dtype=float))
        eff_desc = "flat 1.0 (regression)"
    elif args.mode is not None:
        EFF = efficiency.make_efficiency(args.mode, args.df)
        eff_desc = f"mode {args.mode} df{args.df}"
    else:
        EFF = None
        eff_desc = "none (raw rate)"
    print(f"mediator range lambda = {LAMB} m"
          + (" (massless analytic at sensor)" if args.massless else ""))
    print(f"analysis threshold q_min = {Q_MIN} GeV")
    print(f"detection efficiency = {eff_desc}")
    NO_ATM = args.no_atmosphere
    print(f"overburden attenuation = {'OFF (bare halo)' if NO_ATM else 'on'}")

    if args.quick:
        FID = dict(n_ode=60, n_shm=int(2e4), n_q=120, q_span=1e4, n_mc=1500,
                   n_m=8, n_a=8)
    else:
        FID = dict(n_ode=400, n_shm=int(3e5), n_q=240, q_span=3e4, n_mc=10000,
                   n_m=60, n_a=44)
    for key, val in (("n_m", args.n_m), ("n_a", args.n_a),
                     ("n_mc", args.n_mc), ("n_ode", args.n_ode)):
        if val is not None:
            FID[key] = val
    if args.n_mc_hi:
        # in the fidelity dict, so the cache file records the whole contract
        FID["n_mc_hi"] = args.n_mc_hi
        FID["p_hi_lo"] = args.p_hi_lo
    if args.mu_dex != MU_DEX_DEFAULT:
        # only when overridden, so a default cache's fidelity string is unchanged
        FID["mu_dex"] = float(args.mu_dex)
    print(f"grid {FID['n_m']}x{FID['n_a']}  n_mc={FID['n_mc']}  n_ode={FID['n_ode']}"
          + (f"  n_mc_hi={FID['n_mc_hi']} (p >= {FID['p_hi_lo']:g})"
             if args.n_mc_hi else "")
          + (f"  mu_dex={FID['mu_dex']:g}" if "mu_dex" in FID else ""))

    # masses to the Planck scale, couplings capped at alpha_n = 1
    ms = np.logspace(np.log10(args.m_min), np.log10(1.22e19), FID["n_m"])
    alphas_n = np.logspace(np.log10(args.a_min), 0.0, FID["n_a"])

    # observed events from the analysis input (the same file the notebook reads,
    # so script and notebook cannot drift)
    EVENTS = np.atleast_1d(np.loadtxt(args.data)) / 1e9   # eV -> GeV
    print(f"{EVENTS.size} events from {args.data} [GeV]: {np.round(EVENTS, 1)}")

    V_I_SAMPLES = atmosphere.sample_shm(FID["n_shm"],
                                        rng=np.random.default_rng(SEED))

    print("building cross-section handle ...")
    XS = rate.make_xsec(None if args.massless else LAMB,
                        projection_kernel=args.projection_kernel)

    tasks = [(ia, im, float(a), float(m))
             for im, m in enumerate(ms) for ia, a in enumerate(alphas_n)]
    # Longest-processing-time-first: high mass (full-span ODE) and high coupling
    # (stiff ODE) are the slow points; front-load them so the tail backfills
    # with cheap points instead of straggling.
    tasks.sort(key=lambda t: (t[3], t[2]), reverse=True)
    print(f"scanning {len(tasks)} points "
          f"({FID['n_m']} masses x {FID['n_a']} couplings), slowest first ...")

    P = np.zeros((alphas_n.size, ms.size))
    MU = np.zeros_like(P)
    NT = np.zeros_like(P)
    t0 = time.time()
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futures = [ex.submit(scan_point, t) for t in tasks]  # one task per grab
        for k, fut in enumerate(as_completed(futures)):
            ia, im, p, mu, n_t = fut.result()
            P[ia, im], MU[ia, im], NT[ia, im] = p, mu, n_t
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(tasks)} done  ({time.time() - t0:.0f}s)",
                      flush=True)

    np.savez(args.out, ms=ms, alphas_n=alphas_n, extremeness=P, counts=MU,
             n_transit=NT, events=EVENTS, lamb=LAMB, massless=args.massless,
             q_min=Q_MIN, a_min=args.a_min, no_atmosphere=NO_ATM,
             mode=(args.mode if args.mode is not None else 0),
             df=args.df, flat_efficiency=args.flat_efficiency,
             t_total=T_TOTAL, seed=SEED, fidelity=str(FID),
             projection_kernel=args.projection_kernel)
    print(f"wrote {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
