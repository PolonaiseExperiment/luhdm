#!/usr/bin/env python3
"""Parallel (m_DM, alpha_n) extremeness scan for the limit contours.

Runs the same physics as notebooks/limit_contour.ipynb, through the shared
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

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one process per core, no BLAS threads
os.environ.setdefault("TQDM_DISABLE", "1")

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from luhdm import atmosphere, config, limits, rate

# --- fiducials (as in the notebook) ---
Q_THRESH = config.Q_THRESH
T_TOTAL = 3600 * 24 * 7 * 2
SEED = 20260702

# Shared, read-only state; set in main() BEFORE the fork so children inherit it.
LAMB = None      # atmospheric-regulator range [m] (also the sensor range
                 # unless --massless)
XS = None        # cross-section handle from rate.make_xsec
V_I_SAMPLES = None
FID = None       # fidelity dict
EVENTS = None

_worker_state: dict = {}


def _worker_init_lazy():
    """Per-process state, built on first use in each child."""
    if "table" not in _worker_state:
        _worker_state["table"] = limits.new_table(seed=SEED)
    return _worker_state


def scan_point(task):
    """One (alpha_n, m) grid point -> (ia, im, extremeness, mu, n_transit)."""
    ia, im, alpha_n, m = task
    state = _worker_init_lazy()
    try:
        v_min = Q_THRESH / m / 10
        v_f_samples = atmosphere.compute_v_f_distribution(
            alpha_n, LAMB, m, V_I_SAMPLES, v_min=v_min, n_grid=FID["n_ode"])
        f_v_f = atmosphere.compute_f_vf(v_f_samples, v_min)[0]
        qs = np.geomspace(Q_THRESH, FID["q_span"] * Q_THRESH, FID["n_q"])
        diff_rate = rate.differential_rate_trapz(qs, alpha_n, m, f_v_f, XS)
        p, mu = limits.extremeness_and_mu(
            state["table"], EVENTS, qs, diff_rate, T_TOTAL, n_mc=FID["n_mc"])
        n_t = rate.expected_transits(alpha_n, m, f_v_f, XS, T_TOTAL)
    except Exception as err:  # absurd-coupling corners: report, exclude nothing
        print(f"point (a={alpha_n:.1e}, m={m:.1e}) failed: {err}", flush=True)
        p, mu, n_t = 0.0, 0.0, 0.0
    return ia, im, p, mu, n_t


def main():
    global XS, V_I_SAMPLES, FID, EVENTS, LAMB

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--lamb", type=float, default=2e-4,
                    help="mediator range in meters")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="scan_results.npz")
    ap.add_argument("--massless", action="store_true",
                    help="analytic Rutherford dsigma/dq at the sensor; --lamb "
                         "then only regulates the atmospheric Coulomb log")
    args = ap.parse_args()
    LAMB = args.lamb
    print(f"mediator range lambda = {LAMB} m"
          + (" (massless analytic at sensor)" if args.massless else ""))

    if args.quick:
        FID = dict(n_ode=60, n_shm=int(2e4), n_q=120, q_span=1e4, n_mc=1500,
                   n_m=8, n_a=8)
    else:
        FID = dict(n_ode=400, n_shm=int(3e5), n_q=240, q_span=3e4, n_mc=10000,
                   n_m=60, n_a=44)

    # masses to the Planck scale, couplings capped at alpha_n = 1
    ms = np.logspace(6.4, np.log10(1.22e19), FID["n_m"])
    alphas_n = np.logspace(-8.7, 0.0, FID["n_a"])

    # identical event draw to the notebook
    rng = np.random.default_rng(SEED)
    EVENTS = 10 ** rng.uniform(np.log10(Q_THRESH), np.log10(3 * Q_THRESH), size=1)
    print(f"events [GeV]: {np.round(EVENTS, 1)}")

    np.random.seed(SEED)
    V_I_SAMPLES = atmosphere.sample_shm(FID["n_shm"])

    print("building cross-section handle ...")
    XS = rate.make_xsec(None if args.massless else LAMB)

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
             t_total=T_TOTAL, seed=SEED, fidelity=str(FID))
    print(f"wrote {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
