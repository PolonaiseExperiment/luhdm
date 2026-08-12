#!/usr/bin/env python3
"""Sanity gate: sidecar lambda scan vs the released File A plane at each best mass."""
from pathlib import Path

import numpy as np

from luhdm.release import open_release

import argparse
_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--scan-dir", type=Path, default=Path("."),
                 help="directory holding scan_lambda_mode{1,2,3}.npz")
SCRATCH = _ap.parse_args().scan_dir
CUBE = "/home/tunnell/code/luhdm/release/luhdm_datarelease_v8_A_f1_atm.h5"
BEST_IM = {1: 27, 2: 24, 3: 28}


def band(alpha, p, level=0.95):
    idx = np.where(p >= level)[0]
    if not idx.size:
        return (np.nan, np.nan, 0)
    return (alpha[idx[0]], alpha[idx[-1]], idx.size)


def main():
    with open_release(CUBE) as r:
        ax = r._axis_for("atm")
        ds, pre = r._cube("extremeness", "atm", 1.0)
        dm, _ = r._cube("mu", "atm", 1.0)
        lam_cube = ax.lambda_finite
        for mode in (1, 2, 3):
            d = np.load(SCRATCH / f"scan_lambda_mode{mode}.npz", allow_pickle=True)
            lam, al, P, MU = d["lambs"], d["alphas_n"], d["extremeness"], d["counts"]
            im = BEST_IM[mode]
            k = mode - 1
            assert np.allclose(al, ax.alpha_n, rtol=0, atol=0), "alpha axis mismatch"
            print(f"\n{'='*78}\nmode {mode}   best mass = {float(d['m_dm']):.6e} GeV "
                  f"(cube mass axis [{im}] = {ax.mass_gev[im]:.6e})")

            # -- long-range end vs the released massless slice ---------------- #
            il_s = int(np.argmax(lam))                       # 2.0 m
            p_s = P[:, il_s]
            lo_s, hi_s, n_s = band(al, p_s)
            print(f"  scan  lambda = {lam[il_s]:.4g} m      : 95% excluded alpha_n "
                  f"[{lo_s:.4e}, {hi_s:.4e}]  ({n_s} pts)")
            for il_c, tag in ((4, "massless (lambda = inf)"), (3, "lambda = 200 m")):
                p_c = ds[pre + (k, slice(None), im, il_c)]
                lo_c, hi_c, n_c = band(al, p_c)
                dl = (np.log10(lo_s / lo_c) if np.isfinite(lo_s * lo_c) else np.nan)
                dh = (np.log10(hi_s / hi_c) if np.isfinite(hi_s * hi_c) else np.nan)
                print(f"  cube  {tag:<22}: 95% excluded alpha_n "
                      f"[{lo_c:.4e}, {hi_c:.4e}]  ({n_c} pts)   "
                      f"dlog10(lo) = {dl:+.3f}  dlog10(hi) = {dh:+.3f}  "
                      f"[grid step = {np.log10(al[1]/al[0]):.3f} dex]")

            # -- exact shared tags -------------------------------------------- #
            print("  exact-tag cross-checks (lambda values shared with the cube):")
            for il_c in range(3):
                L = lam_cube[il_c]
                il_s2 = int(np.argmin(np.abs(np.log(lam) - np.log(L))))
                assert abs(lam[il_s2] / L - 1) < 1e-9, f"no exact scan member at {L}"
                p_c = ds[pre + (k, slice(None), im, il_c)]
                m_c = dm[pre + (k, slice(None), im, il_c)]
                lo_c, hi_c, n_c = band(al, p_c)
                lo_2, hi_2, n_2 = band(al, P[:, il_s2])
                ok = m_c > 0
                rat = MU[ok, il_s2] / np.maximum(m_c[ok], 1e-300)
                same = (lo_2 == lo_c) and (hi_2 == hi_c)
                print(f"    lambda = {L:.0e} m: scan [{lo_2:.4e}, {hi_2:.4e}] ({n_2}) "
                      f"vs cube [{lo_c:.4e}, {hi_c:.4e}] ({n_c})  "
                      f"{'IDENTICAL' if same else 'DIFFER'};  "
                      f"mu ratio med {np.median(rat):.4f} "
                      f"[{rat.min():.4f}, {rat.max():.4f}]")

            # -- shortest excluded range in the sidecar ----------------------- #
            any_ex = (P >= 0.95).any(axis=0)
            if any_ex.any():
                print(f"  sidecar shortest excluded mediator range: "
                      f"{lam[np.argmax(any_ex)]:.4g} m "
                      f"(of {lam.size} ranges, {int(any_ex.sum())} with exclusion)")


if __name__ == "__main__":
    main()
