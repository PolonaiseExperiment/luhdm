#!/usr/bin/env python3
"""Parallel (m_DM, alpha_n) extremeness scan for the limit contour.

Runs the same physics as notebooks/limit_contour.ipynb, parallelized over grid
points (each is independent: attenuation ODE -> dR/dq -> optimum-interval
extremeness). Designed for a many-core node; writes scan_results.npz, which the
notebook loads if present.

    python scripts/scan_grid.py                 # full fidelity (~80 cores)
    python scripts/scan_grid.py --quick         # small smoke test
    python scripts/scan_grid.py --out results.npz --workers 40
"""

from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one process per core, no BLAS threads

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from scipy.interpolate import interp1d

from luhdm import atmosphere, config, cross_section, halo, limits, units

# --- fiducials (as in the notebook) ---
LAMB = 3e-3
R_EFF = config.R_EFF
Q_THRESH = config.Q_THRESH
T_TOTAL = 3600 * 24 * 7 * 2
SEED = 20260702

# Shared, read-only state; set in main() BEFORE the fork so children inherit it.
DS_QT = None
DS_VALS = None
V_I_SAMPLES = None
FID = None  # fidelity dict
EVENTS = None

_worker_state: dict = {}


def _worker_init_lazy():
    """Per-process state, built on first use in each child."""
    if "interp" not in _worker_state:
        _worker_state["interp"] = interp1d(DS_QT, DS_VALS, kind="linear")
        _worker_state["table"] = limits.new_table(seed=SEED)
    return _worker_state


def differential_rate_trapz(qs, alpha_n, lamb, mu, R_eff, f_v_f, dsigma_interpolant):
    """dR/dq via trapz (mu = m_DM in the original notation); as in the notebook."""
    alpha = alpha_n * config.N_NEUTRONS
    n_dm = halo.number_density_dm(mu)
    v_min_global = qs.min() / mu
    vs_global = np.geomspace(v_min_global, config.VESC, 500)
    f_vf_grid = f_v_f(vs_global)
    results = []
    for q in qs:
        mask = vs_global >= q / mu
        vs = vs_global[mask]
        integrand = n_dm * f_vf_grid[mask] * vs * cross_section.dsigma_dq(
            q, alpha, lamb, R_eff, vs, dsigma_interpolant)
        results.append(np.trapezoid(integrand, vs) * units.CONV2RATE)
    return np.maximum(np.array(results), 0)


def scan_point(task):
    """One (alpha_n, m) grid point -> (ia, im, extremeness, mu)."""
    ia, im, alpha_n, m = task
    state = _worker_init_lazy()
    try:
        v_min = Q_THRESH / m / 10
        v_f_samples = atmosphere.compute_v_f_distribution(
            alpha_n, LAMB, m, V_I_SAMPLES, v_min=v_min, n_grid=FID["n_ode"])
        f_v_f = atmosphere.compute_f_vf(v_f_samples, v_min)[0]
        qs = np.geomspace(Q_THRESH, FID["q_span"] * Q_THRESH, FID["n_q"])
        rate = differential_rate_trapz(qs, alpha_n, LAMB, m, R_EFF,
                                       f_v_f, state["interp"])
        p, mu = limits.extremeness_and_mu(
            state["table"], EVENTS, qs, rate, T_TOTAL, n_mc=FID["n_mc"])
    except Exception as err:  # absurd-coupling corners: report, exclude nothing
        print(f"point (a={alpha_n:.1e}, m={m:.1e}) failed: {err}", flush=True)
        p, mu = 0.0, 0.0
    return ia, im, p, mu


def main():
    global DS_QT, DS_VALS, V_I_SAMPLES, FID, EVENTS

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="scan_results.npz")
    args = ap.parse_args()

    if args.quick:
        FID = dict(n_ode=60, n_shm=int(2e4), n_q=120, q_span=1e4, n_mc=1500,
                   n_m=8, n_a=8)
    else:
        FID = dict(n_ode=400, n_shm=int(3e5), n_q=240, q_span=3e4, n_mc=10000,
                   n_m=44, n_a=44)

    ms = np.logspace(6.4, 15.6, FID["n_m"])
    alphas_n = np.logspace(-8.7, 2.0, FID["n_a"])

    # identical event draw to the notebook
    rng = np.random.default_rng(SEED)
    EVENTS = 10 ** rng.uniform(np.log10(Q_THRESH), np.log10(3 * Q_THRESH), size=1)
    print(f"events [GeV]: {np.round(EVENTS, 1)}")

    np.random.seed(SEED)
    V_I_SAMPLES = atmosphere.sample_shm(FID["n_shm"])

    print("tabulating dsigma/dq_tilde ...")
    xi = R_EFF / LAMB
    from scipy.special import kn
    DS_QT = np.geomspace(1e-25, kn(1, xi) * (1 - 1e-9), 1000)
    DS_VALS = np.array([cross_section.dsigma_dq_tilde(
        qt, xi, cross_section.interpolant_k1_inverse) for qt in DS_QT])

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
    t0 = time.time()
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futures = [ex.submit(scan_point, t) for t in tasks]  # one task per grab
        for k, fut in enumerate(as_completed(futures)):
            ia, im, p, mu = fut.result()
            P[ia, im], MU[ia, im] = p, mu
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(tasks)} done  ({time.time() - t0:.0f}s)",
                      flush=True)

    np.savez(args.out, ms=ms, alphas_n=alphas_n, extremeness=P, counts=MU,
             events=EVENTS, lamb=LAMB, t_total=T_TOTAL, seed=SEED,
             fidelity=str(FID))
    print(f"wrote {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
