#!/usr/bin/env python3
"""Parallel (lambda, alpha_n) extremeness scan at a fixed DM mass.

The companion to scan_grid.py for the mediator-range figure: fix m_DM at the
most sensitive mass and scan the mediator range against the coupling. Same
physics through luhdm.rate (log-space tabulation for every range, so all
ranges pay the same, small, tabulation cost); writes an npz cache that the
notebook loads if present.

    python scripts/scan_lambda.py --mass 1.82e7 --out scan_lambda.npz
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

from luhdm import atmosphere, config, efficiency, limits, rate

Q_THRESH = config.Q_THRESH
Q_HI_REF = 8.4e3   # fixed qs upper-momentum reference (see scan_grid.py)
T_TOTAL = config.T_EXPOSURE  # dataset live-time (single source of truth in luhdm.config)
SEED = 20260702
# default observed-event list; --data overrides it (e.g. per-mode data_mode{n}.txt)
DEFAULT_DATA = Path(__file__).resolve().parent.parent / "notebooks" / "data_mode1.txt"

# Shared, read-only state; set in main() BEFORE the fork so children inherit it.
M_DM = None
XS_BY_IL = None   # list of xs handles, one per lambda grid point
LAMBS = None
V_I_SAMPLES = None
FID = None
EVENTS = None
EFF = None        # detection-efficiency callable eps(q_GeV), or None for raw rate

_worker_state: dict = {}


def _worker_init_lazy():
    if "table" not in _worker_state:
        _worker_state["table"] = limits.new_table(seed=SEED)
    return _worker_state


def scan_point(task):
    """One (lambda, alpha_n) grid point -> (il, ia, extremeness, mu, n_transit)."""
    il, ia, alpha_n = task
    state = _worker_init_lazy()
    lamb = LAMBS[il]
    xs = XS_BY_IL[il]
    try:
        v_min = Q_THRESH / M_DM / 10
        v_f_samples = atmosphere.compute_v_f_distribution(
            alpha_n, lamb, M_DM, V_I_SAMPLES, v_min=v_min, n_grid=FID["n_ode"])
        f_v_f = atmosphere.compute_f_vf(v_f_samples, v_min)[0]
        qs = np.geomspace(Q_THRESH, FID["q_span"] * Q_HI_REF, FID["n_q"])
        diff_rate = rate.differential_rate_trapz(qs, alpha_n, M_DM, f_v_f, xs,
                                                 eff=EFF)
        p, mu = limits.extremeness_and_mu(
            state["table"], EVENTS, qs, diff_rate, T_TOTAL, n_mc=FID["n_mc"])
        n_t = rate.expected_transits(alpha_n, M_DM, f_v_f, xs, T_TOTAL)
    except Exception as err:  # over-stopped/stiff corners: report, exclude nothing
        print(f"point (lamb={lamb:.1e}, a={alpha_n:.1e}) failed: {err}",
              flush=True)
        p, mu, n_t = 0.0, 0.0, 0.0
    return il, ia, p, mu, n_t


def main():
    global M_DM, XS_BY_IL, LAMBS, V_I_SAMPLES, FID, EVENTS, EFF

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--mass", type=float, required=True,
                    help="fixed DM mass in GeV (the most sensitive mass)")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default="scan_lambda.npz")
    ap.add_argument("--lamb-min", type=float, default=2e-6,
                    help="lowest mediator range in the grid, m (default: %(default)g)")
    ap.add_argument("--lamb-max", type=float, default=2.0,
                    help="highest mediator range in the grid, m (default: %(default)g)")
    ap.add_argument("--a-min", type=float, default=10 ** -8.7,
                    help="lowest coupling alpha_n in the grid (default: %(default).3g)")
    ap.add_argument("--n-l", type=int, default=None,
                    help="override number of mediator-range grid points")
    ap.add_argument("--n-a", type=int, default=None,
                    help="override number of coupling grid points")
    ap.add_argument("--mode", type=int, choices=(1, 2, 3), default=None,
                    help="sensor mode: fold in its measured detection "
                         "efficiency (default: none, raw rate)")
    ap.add_argument("--df", type=int, choices=(2, 3), default=3,
                    help="efficiency dof hypothesis (default: 3)")
    ap.add_argument("--data", default=str(DEFAULT_DATA),
                    help="event list file, one impulse per line in eV "
                         "(default: notebooks/data.txt)")
    args = ap.parse_args()
    M_DM = args.mass
    print(f"fixed m_DM = {M_DM:.3e} GeV")

    if args.quick:
        FID = dict(n_ode=60, n_shm=int(2e4), n_q=120, q_span=1e4, n_mc=1500,
                   n_l=6, n_a=8)
    else:
        FID = dict(n_ode=400, n_shm=int(3e5), n_q=240, q_span=3e4, n_mc=10000,
                   n_l=49, n_a=44)
    for key, val in (("n_l", args.n_l), ("n_a", args.n_a)):
        if val is not None:
            FID[key] = val

    if args.mode is not None:
        EFF = efficiency.make_efficiency(args.mode, args.df)
        eff_desc = f"mode {args.mode} df{args.df}"
    else:
        EFF = None
        eff_desc = "none (raw rate)"
    print(f"detection efficiency = {eff_desc}")

    # mediator range and couplings (log-spaced); defaults span the planned
    # ranges (2 um to 2 m) and the mass-scan couplings
    LAMBS = np.geomspace(args.lamb_min, args.lamb_max, FID["n_l"])
    alphas_n = np.logspace(np.log10(args.a_min), 0.0, FID["n_a"])

    # observed events from the analysis input (the same file the notebook reads,
    # so script and notebook cannot drift)
    EVENTS = np.atleast_1d(np.loadtxt(args.data)) / 1e9   # eV -> GeV
    print(f"{EVENTS.size} events from {args.data} [GeV]: {np.round(EVENTS, 1)}")

    V_I_SAMPLES = atmosphere.sample_shm(FID["n_shm"],
                                        rng=np.random.default_rng(SEED))

    print(f"tabulating {LAMBS.size} cross sections (log-space) ...")
    t0 = time.time()
    XS_BY_IL = [rate.make_xsec(lamb, force_ln=True) for lamb in LAMBS]
    print(f"  done in {time.time() - t0:.0f}s")

    # high coupling first: the stiff attenuation ODE dominates the runtime
    tasks = [(il, ia, float(a))
             for il in range(LAMBS.size) for ia, a in enumerate(alphas_n)]
    tasks.sort(key=lambda t: t[2], reverse=True)
    print(f"scanning {len(tasks)} points "
          f"({FID['n_l']} ranges x {FID['n_a']} couplings), slowest first ...")

    P = np.zeros((alphas_n.size, LAMBS.size))
    MU = np.zeros_like(P)
    NT = np.zeros_like(P)
    t0 = time.time()
    ctx = multiprocessing.get_context("fork")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
        futures = [ex.submit(scan_point, t) for t in tasks]
        for k, fut in enumerate(as_completed(futures)):
            il, ia, p, mu, n_t = fut.result()
            P[ia, il], MU[ia, il], NT[ia, il] = p, mu, n_t
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(tasks)} done  ({time.time() - t0:.0f}s)",
                      flush=True)

    np.savez(args.out, lambs=LAMBS, alphas_n=alphas_n, extremeness=P,
             counts=MU, n_transit=NT, events=EVENTS, m_dm=M_DM,
             mode=(args.mode if args.mode is not None else 0), df=args.df,
             t_total=T_TOTAL, seed=SEED, fidelity=str(FID))
    print(f"wrote {args.out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
