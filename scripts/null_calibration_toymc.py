#!/usr/bin/env python
"""Noise-only null calibration of the transient statistic T (audit item C3).

Statistic (main.tex eq:transient-q; data/analyse_transients_data.ipynb cell 16).
A segment yields M complex differences d_1..d_M.  For each tested index i,

    R_i(w) = |w d_i + (1-w) d_{i+1}|^2 / (w^2 + (1-w)^2),   w in [0,1]
    Rmax(i) = max_w R_i(w),   S = sum_k |d_k|^2
    T_i = -2 M ln(1 - Rmax(i)/S)

Under H0 the d_k are iid CN(0, sigma^2).  Three results, in order:

(1) CLOSED FORM for Rmax.  With A the 2x2 real Gram matrix of the 2-D real
    vectors (Re d_i, Im d_i) and (Re d_{i+1}, Im d_{i+1}) -- i.e.
    A11 = |d_i|^2, A22 = |d_{i+1}|^2, A12 = Re(d_i conj(d_{i+1})) -- the
    quantity R_i(w) is the Rayleigh quotient v^T A v / v^T v at v = (w, 1-w),
    so maximising over w in [0,1] means maximising over first-quadrant
    directions:

        Rmax = lambda_max(A)          if A12 >= 0
        Rmax = max(A11, A22)          if A12 <  0   (boundary, w = 1 or w = 0)

    No grid scan is needed.  Verified against a 2e5-point w grid to 2e-11
    relative, and against the shipped transient_q_1 arrays to 3e-14 absolute.

(2) EXACT NULL.  With v_k = |d_k|^2/S the vector (v_1..v_M) is Dirichlet
    (1,...,1), independent of S and of the phases, so T is exactly scale free
    (sigma cancels).  sign(A12) is independent of (A11, A22, |A12|, S) and
    P(A12 >= 0) = 1/2, so the two branches decouple.  Writing
    s = v_i + v_{i+1} ~ Beta(2, M-2), u = v_i/s ~ U(0,1) and phi ~ U(0, pi/2)
    for the relative phase, lambda_max/S = s * g with
    g(u,phi) = (1 + sqrt(1 - 4u(1-u) sin^2 phi))/2 in [1/2, 1], giving

        P(T > t) = 1/2 <Q_Beta(x/g)>_{u,phi} + 1/2 [2(1-x)^{M-1} - (1-2x)_+^{M-1}]
        x        = 1 - exp(-t/(2M))
        Q_Beta(y)= (1-y)^{M-2} (1 + (M-2) y)   for y < 1, else 0

    Note (1-x)^{M-1} = exp(-t/2) exp(t/(2M)): the single-bin branch is exactly
    the chi^2_2 tail times a finite-M correction.  This is exact algebra, not
    an asymptotic, so it evaluates the T > 100 selection directly instead of
    extrapolating the Monte Carlo 12.6 decades.

(3) TOY MC.  Whole segments of M iid CN(0,1) differences are simulated and all
    M-1 tested indices pooled, exactly as the data are pooled: this preserves
    both correlations in the analysis (adjacent indices share a difference
    sample; every index in a segment shares the denominator S).  4.0e9 draws
    reproduce (2) to within 1.4 sigma everywhere the MC has power.

Usage:  python null_calibration_toymc.py [n_batches]      (default 400 -> 4e9 draws)
Writes:  null_calibration_toymc_reproduced.json  next to this file (the curated
         artifact null_calibration_toymc.json is not overwritten).
"""
import json
import os
import time

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from scipy import optimize, stats

M = 271                     # complex differences per segment (len(transient_q)+1)
NIDX = M - 1                # 270 tested indices per segment
MASTER_SEED = 20260806
SEG_PER_BATCH = 37037       # 37037 * 270 = 9,999,990 draws per batch
SUBCHUNK = 8000
INV_BW = 500.0              # histogram bin width 0.002 (thresholds land on edges)
NBINS = 60000               # covers T in [0, 120)
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------- closed form
def rmax(a, b):
    """max over w in [0,1] of |w a + (1-w) b|^2 / (w^2 + (1-w)^2)."""
    a11 = a.real**2 + a.imag**2
    a22 = b.real**2 + b.imag**2
    a12 = a.real * b.real + a.imag * b.imag
    half = 0.5 * (a11 - a22)
    lmax = 0.5 * (a11 + a22) + np.sqrt(half * half + a12 * a12)
    return np.where(a12 >= 0.0, lmax, np.maximum(a11, a22))


# ---------------------------------------------------------------- exact null
@lru_cache(maxsize=8)
def _grid(n):
    xu, wu = np.polynomial.legendre.leggauss(n)
    u = 0.25 * (xu + 1.0)                       # u in [0, 1/2]; g is symmetric in u
    xp, wp = np.polynomial.legendre.leggauss(n)
    phi = 0.25 * np.pi * (xp + 1.0)             # phi in [0, pi/2], density 2/pi
    g = 0.5 * (1.0 + np.sqrt(np.maximum(0.0, 1.0 - np.outer(4.0 * u * (1.0 - u),
                                                            np.sin(phi) ** 2))))
    return g.ravel(), np.outer(0.5 * wu, 0.5 * wp).ravel()


def sf_exact(t, m=M, n=400):
    """Exact per-index survival P(T > t).  Converged to 9 digits by n = 300."""
    t_arr = np.atleast_1d(np.asarray(t, float))
    x = -np.expm1(-t_arr / (2.0 * m))
    g, w = _grid(n)
    out = np.empty_like(t_arr)
    for j, xj in enumerate(x):
        y = xj / g
        ok = y < 1.0
        term1 = float(np.sum(w[ok] * np.exp((m - 2) * np.log1p(-y[ok])) * (1.0 + (m - 2) * y[ok])))
        lo = np.exp((m - 1) * np.log1p(-xj))
        hi = np.exp((m - 1) * np.log1p(-2.0 * xj)) if 2.0 * xj < 1.0 else 0.0
        out[j] = 0.5 * term1 + 0.5 * (2.0 * lo - hi)
    return out if np.ndim(t) else float(out[0])


def quantile_exact(p, m=M, n=400):
    lo, hi = 1e-9, 500.0
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        if sf_exact(mid, m, n) > 1.0 - p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------- toy MC
def _segment_block(rng, nseg):
    a = rng.standard_normal((nseg, M))
    b = rng.standard_normal((nseg, M))
    p = a * a + b * b                                   # proportional to |d_k|^2
    s = p.sum(axis=1, keepdims=True)
    a11, a22 = p[:, :-1], p[:, 1:]
    a12 = a[:, :-1] * a[:, 1:] + b[:, :-1] * b[:, 1:]
    half = 0.5 * (a11 - a22)
    lmax = 0.5 * (a11 + a22) + np.sqrt(half * half + a12 * a12)
    rm = np.where(a12 >= 0.0, lmax, np.maximum(a11, a22))
    return ((-2.0 * M) * np.log1p(-rm / s)).ravel()


def _batch(args):
    seedseq, nseg = args
    rng = np.random.default_rng(seedseq)
    hist = np.zeros(NBINS + 1, dtype=np.int64)
    ssum = ssq = 0.0
    n = 0
    mx = 0.0
    done = 0
    while done < nseg:
        k = min(SUBCHUNK, nseg - done)
        done += k
        t = _segment_block(rng, k)
        hist += np.bincount((t * INV_BW).astype(np.int64).clip(0, NBINS), minlength=NBINS + 1)
        ssum += float(t.sum())
        ssq += float((t ** 2).sum())
        n += t.size
        mx = max(mx, float(t.max()))
    return hist, ssum, ssq, n, mx


def run_mc(n_batch=400, workers=None):
    children = np.random.SeedSequence(MASTER_SEED).spawn(n_batch)
    hist = np.zeros(NBINS + 1, dtype=np.int64)
    per_batch = np.zeros((n_batch, NBINS + 1), dtype=np.int32)
    ssum = ssq = 0.0
    ntot = 0
    mx = 0.0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, (h, s, q, n, m) in enumerate(
                ex.map(_batch, [(c, SEG_PER_BATCH) for c in children], chunksize=1)):
            hist += h
            per_batch[i] = h.astype(np.int32)
            ssum += s
            ssq += q
            ntot += n
            mx = max(mx, m)
    print("toy MC: %d draws (%d segments) in %.0f s; mean %.5f, max %.2f"
          % (ntot, n_batch * SEG_PER_BATCH, time.time() - t0, ssum / ntot, mx))
    return dict(hist=hist, per_batch=per_batch, ssum=ssum, ssq=ssq, ntot=ntot,
                mx=mx, n_batch=n_batch)


# ---------------------------------------------------------------- summary
def summarise(mc):
    hist = mc["hist"][:-1].astype(float)
    pbh = mc["per_batch"][:, :-1].astype(float)
    ntot, nbatch = mc["ntot"], mc["n_batch"]
    bw = 1.0 / INV_BW

    def quant(h, p):
        c = np.cumsum(h)
        k = np.searchsorted(c, p * c[-1], side="left")
        below = c[k - 1] if k else 0.0
        return (k + (p * c[-1] - below) / h[k] if h[k] else k) * bw

    out = {"setup": dict(
        M_differences=M, indices_per_segment=NIDX, n_draws=ntot,
        n_segments=nbatch * SEG_PER_BATCH, n_batches=nbatch, master_seed=MASTER_SEED,
        rng="numpy PCG64, SeedSequence(%d).spawn(%d)" % (MASTER_SEED, nbatch),
        mean=mc["ssum"] / ntot, var=mc["ssq"] / ntot - (mc["ssum"] / ntot) ** 2,
        max_T=mc["mx"], overflow_above_120=int(mc["hist"][-1]))}

    out["quantiles"] = {}
    for p in (0.05, 0.50, 0.90, 0.99, 0.999):
        qmc = quant(hist, p)
        se = np.std([quant(pbh[i], p) for i in range(nbatch)], ddof=1) / np.sqrt(nbatch)
        out["quantiles"]["%g" % p] = dict(
            toy_mc=qmc, toy_mc_se=float(se), exact=quantile_exact(p),
            chi2_2=float(stats.chi2.ppf(p, 2)), chi2_3=float(stats.chi2.ppf(p, 3)),
            chi2_3_dev_pct=100 * (stats.chi2.ppf(p, 3) - qmc) / qmc)

    out["tail"] = {}
    for t in (10., 15., 20., 25., 30., 35., 40., 45., 50.):
        k = int(round(t * INV_BW))
        p_mc = hist[k:].sum() / ntot
        pbv = pbh[:, k:].sum(axis=1) / pbh.sum(axis=1)
        se = float(pbv.std(ddof=1) / np.sqrt(nbatch))
        p_ex = sf_exact(t)
        out["tail"]["%g" % t] = dict(
            n_above=int(hist[k:].sum()), p_mc=p_mc, p_mc_se=se, p_exact=p_ex,
            pull=(p_mc - p_ex) / se if se else None,
            chi2_2=float(stats.chi2.sf(t, 2)), chi2_3=float(stats.chi2.sf(t, 3)))

    r_thr = 2 * M * np.expm1(100.0 / (2 * M))
    p_thr = sf_exact(100.0)
    n_idx = 842388                                    # measured, per mode, 3122 segments
    out["threshold"] = dict(
        T_thr=100.0, matched_filter_power_thr=float(r_thr), p_per_index_exact=p_thr,
        p_per_index_chi2_2=float(stats.chi2.sf(100., 2)),
        p_per_index_chi2_3=float(stats.chi2.sf(100., 3)),
        p_per_index_chi2_3_at_110=float(stats.chi2.sf(r_thr, 3)),
        n_indices_per_mode=n_idx, n_indices_total=3 * n_idx,
        expected_accidentals_three_modes=p_thr * 3 * n_idx)

    keff = lambda t: float(optimize.brentq(
        lambda k: np.log(stats.chi2.sf(t, k)) - np.log(sf_exact(t)), 0.5, 12))
    out["effective_dof"] = dict(
        from_mean=out["setup"]["mean"], from_var=out["setup"]["var"] / 2.0,
        from_median=float(optimize.brentq(
            lambda k: stats.chi2.ppf(0.5, k) - quant(hist, 0.5), 0.5, 12)),
        local_keff_vs_t={"%g" % t: keff(t) for t in (2, 5, 10, 20, 50, 100)})
    return out


if __name__ == "__main__":
    import sys
    nb = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    res = summarise(run_mc(nb))
    with open(os.path.join(HERE, "null_calibration_toymc_reproduced.json"), "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("wrote", os.path.join(HERE, "null_calibration_toymc_reproduced.json"))
