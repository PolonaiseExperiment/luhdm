#!/usr/bin/env python3
"""Bundle the measured detection-efficiency curves into the luhdm package.

Reads the (git-ignored, local-only) analysis product
``data/selected_data_efficiency_curves.npz`` -- the per-segment-mean detection
efficiency vs imparted momentum for each sensor mode -- converts the momentum
axis from SI (kg m/s) to GeV, and writes a small committed table that the rate
pipeline loads on any host (remote-node has no data/ dir).

    python scripts/export_efficiency.py

Run locally whenever the efficiency measurement changes, then commit
``luhdm/reference_data/efficiency_curves.npz``.
"""
from pathlib import Path

import numpy as np

# exact natural-unit conversion, same constants as notebook 00
C_LIGHT = 2.99792458e8       # m/s
E_CHARGE = 1.602176634e-19   # J/eV
GEV_PER_SI = C_LIGHT / E_CHARGE / 1e9   # p[GeV] = p[kg m/s] * this

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "selected_data_efficiency_curves.npz"
OUT = REPO / "luhdm" / "reference_data" / "efficiency_curves.npz"
MODES = (1, 2, 3)


def main():
    src = np.load(SRC)
    out = {}
    for n in MODES:
        q_gev = src[f"dv_bins_{n}"] * GEV_PER_SI
        out[f"q_gev_{n}"] = q_gev
        out[f"eff_{n}_df2"] = src[f"efficiency_{n}_2"]
        out[f"eff_{n}_df3"] = src[f"efficiency_{n}_3"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, gev_per_si=GEV_PER_SI, **out)
    print(f"wrote {OUT}")
    for n in MODES:
        q, e = out[f"q_gev_{n}"], out[f"eff_{n}_df3"]
        print(f"  mode {n}: q_GeV {q.min():.3g}..{q.max():.3g}, "
              f"eff(df3) max={e.max():.4f}, eff@1TeV={np.interp(1e3, q, e):.3f}")


if __name__ == "__main__":
    main()
