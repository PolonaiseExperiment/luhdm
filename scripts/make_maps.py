#!/usr/bin/env python3
"""Precompute the transit-count / reach maps cache for the limit notebook.

Writes notebooks/flux_maps.npz through the same luhdm.rate functions the
notebook uses inline when the cache is absent, so the two cannot drift.
Cheap enough for a laptop; provided as a script so it can run on remote-node
alongside scan_grid.py.

    python scripts/make_maps.py --lamb 2e-4 --out notebooks/flux_maps.npz
"""

from __future__ import annotations

import argparse

import numpy as np

from luhdm import config, rate

T_TOTAL = config.T_EXPOSURE  # dataset live-time (single source of truth in luhdm.config)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lamb", type=float, default=2e-4,
                    help="finite reference range in meters (massless panel is "
                         "always included)")
    ap.add_argument("--out", default="notebooks/computation_cache/flux_maps.npz")
    args = ap.parse_args()

    ms_map = np.geomspace(1e5, 1.22e19, 64)
    al_map = np.geomspace(2e-11, 1.0, 64)

    print("massless (analytic Coulomb reach) ...")
    nt_ml, b_ml = rate.transit_maps(ms_map, al_map, rate.make_xsec(None),
                                    T_TOTAL)
    print(f"finite range lambda = {args.lamb} m ...")
    nt_fr, b_fr = rate.transit_maps(ms_map, al_map, rate.make_xsec(args.lamb),
                                    T_TOTAL)

    np.savez(args.out, ms_map=ms_map, al_map=al_map,
             nt_massless=nt_ml, b_massless=b_ml,
             nt_finite=nt_fr, b_finite=b_fr,
             lamb=args.lamb, t_total=T_TOTAL)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
