#!/usr/bin/env python3
"""Merge per-surface refined-contour sidecars into one release sidecar.

``refine_contours.py`` only parallelises inside an oracle wave, so refining a
plane's surfaces as separate concurrent processes is much faster than one
sequential run -- but it yields one sidecar per surface. This stitches them
back into the single file the release ships and the notebooks read.

The merge is bookkeeping, not physics: every input must come from the SAME
cube (same sha256), the same code revision and the same fidelity, and that is
asserted rather than assumed. Provenance fields that describe the run rather
than the cube are combined conservatively -- the spot-check cell counts add,
and the worst per-surface ``spot_max_dp`` wins.

    python scripts/merge_contour_sidecars.py \
        --out release/luhdm_contours_v10_B_f0p1_noatm.json \
        ignore/scratchpad/luhdm_contours_v10hi_B_*.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# provenance keys that must agree across every input, or the merge is unsound
MUST_MATCH = (
    "cube_sha256", "cube_version_tag", "cube_git_commit", "cube_fid", "seed",
    "mu_round_dex", "b_constrained_max_m", "t_exposure_s", "q_thresh_gev",
    "projection_kernel", "v_earth_km_s", "efficiency_npz_sha256",
    "events_sha256", "massless_lamb_ode_m",
)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("inputs", nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cube-path", default=None,
                    help="rewrite provenance.cube_path (e.g. when the cube is "
                         "promoted to its released name after refinement)")
    a = ap.parse_args()

    docs = [json.loads(p.read_text()) for p in sorted(a.inputs)]
    if not docs:
        raise SystemExit("no inputs")
    base = docs[0]

    for key in MUST_MATCH:
        vals = {json.dumps(d["provenance"].get(key), sort_keys=True) for d in docs}
        if len(vals) != 1:
            raise SystemExit(
                f"FATAL: inputs disagree on provenance['{key}'] -- refusing to "
                f"merge sidecars from different builds:\n  " + "\n  ".join(vals))
    for key in ("confidence", "format", "schema_version"):
        if len({json.dumps(d.get(key)) for d in docs}) != 1:
            raise SystemExit(f"FATAL: inputs disagree on '{key}'")

    surfaces, done, requested = {}, [], []
    for d, p in zip(docs, sorted(a.inputs)):
        for name, surf in d["surfaces"].items():
            if name in surfaces:
                raise SystemExit(f"FATAL: surface '{name}' appears in more than "
                                 f"one input (second: {p})")
            surfaces[name] = surf
        done += d["provenance"].get("surfaces_done", [])
        requested += d["provenance"].get("surfaces_requested", [])

    prov = dict(base["provenance"])
    prov["surfaces_requested"] = sorted(set(requested))
    prov["surfaces_done"] = sorted(set(done))
    prov["spot_n_cells"] = sum(d["provenance"].get("spot_n_cells", 0) or 0
                               for d in docs)
    # spot_max_dp is keyed BY SURFACE, so the merge is a dict union rather than
    # a max; a collision would mean the same surface was refined twice.
    dps = {}
    for d in docs:
        for name, val in (d["provenance"].get("spot_max_dp") or {}).items():
            if name in dps and dps[name] != val:
                raise SystemExit(f"FATAL: conflicting spot_max_dp for '{name}'")
            dps[name] = val
    prov["spot_max_dp"] = {k: dps[k] for k in sorted(dps)}
    prov["merged_from"] = [p.name for p in sorted(a.inputs)]
    prov["merge_note"] = ("surfaces refined as separate concurrent processes "
                          "against one cube and merged; identical cube sha256, "
                          "revision and fidelity asserted at merge time")
    if a.cube_path:
        prov["cube_path"] = a.cube_path

    out = dict(base)
    out["provenance"] = prov
    out["surfaces"] = {k: surfaces[k] for k in sorted(surfaces)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}  ({a.out.stat().st_size/1024:.0f} KiB)")
    print(f"  surfaces: {', '.join(out['surfaces'])}")
    print(f"  cube    : {prov['cube_path']}  sha {prov['cube_sha256'][:16]}...")
    print(f"  fidelity: n_mc_hi={prov['cube_fid'].get('n_mc_hi')}, "
          f"mu_dex={prov['cube_fid'].get('mu_dex')}, "
          f"v_earth={prov.get('v_earth_km_s')} km/s")
    worst = max(prov['spot_max_dp'].values()) if prov['spot_max_dp'] else None
    print(f"  spot    : {prov['spot_n_cells']} cells over {len(prov['spot_max_dp'])} "
          f"surfaces, worst |dp| = {worst}")


if __name__ == "__main__":
    main()
