#!/usr/bin/env python3
"""Assemble the per-mode (lambda, alpha_n) scans into one release sidecar npz.

Reads the three scan_lambda.py outputs (one per sensor mode, each at that
mode's best DM mass from the v8 File A cube) and writes
release/luhdm_lambda_scan_v8.npz with a provenance JSON string.
"""
from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
import argparse
_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_ap.add_argument("--scan-dir", type=Path, default=Path("."),
                 help="directory holding scan_lambda_mode{1,2,3}.npz")
SCRATCH = _ap.parse_args().scan_dir
CUBE = "luhdm_datarelease_v8_A_f1_atm.h5"
OUT = REPO / "release" / "luhdm_lambda_scan_v8.npz"
MODES = (1, 2, 3)


def tilde(s):
    """Replace any /home/<user> prefix with '~' (no absolute user paths shipped)."""
    import re
    return re.sub(r"/home/[^/\s\"']+", "~", str(s))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha_from_sums(name):
    """sha256 of `name` as recorded in release/SHA256SUMS (the released value)."""
    for line in (REPO / "release" / "SHA256SUMS").read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == name:
            return parts[0]
    raise KeyError(f"{name} not in release/SHA256SUMS")


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True).stdout.strip()


def main():
    scans = {m: np.load(SCRATCH / f"scan_lambda_mode{m}.npz", allow_pickle=True)
             for m in MODES}
    argv = json.loads((SCRATCH / "scan_argv.json").read_text())

    # --- consistency checks across the three per-mode scans ------------------ #
    ref = scans[1]
    for m in MODES:
        d = scans[m]
        assert int(d["mode"]) == m, f"mode mismatch in scan {m}"
        assert float(d["f_dm"]) == 1.0, "sidecar is the f_DM = 1 hypothesis"
        assert float(d["mu_cap"]) == 85.0, "mu cap drifted from the release build"
        assert int(d["seed"]) == 20260702, "seed drifted from the release build"
        assert float(d["q_thresh"]) == 1e3, "q_thresh drifted"
        assert float(d["t_total"]) == 790778.0, "exposure drifted"
        assert np.isnan(float(d["b_constrained_max"])), "cross section must be uncapped"
        assert np.array_equal(d["lambs"], ref["lambs"]), "lambda axes differ"
        assert np.array_equal(d["alphas_n"], ref["alphas_n"]), "alpha axes differ"

    fid = str(ref["fidelity"])
    prov = {
        "product": "luhdm lambda-scan sidecar (mediator range x coupling "
                   "extremeness plane at each mode's best DM mass)",
        "cube": {
            "file": CUBE,
            "sha256": sha_from_sums(CUBE),
            "sha256_source": "release/SHA256SUMS",
            "f_dm": 1.0,
            "atmosphere": True,
            "group": "atm",
        },
        "best_mass_gev": {str(m): float(scans[m]["m_dm"]) for m in MODES},
        "best_mass_criterion":
            "luhdm.release.best_mass_index at 95% CL on the File A extremeness "
            "cube: widest excluded log-alpha area integrated over log-lambda, "
            "restricted to the masses reaching the globally shortest excluded "
            "mediator range. File A carries only the 4 finite lambda slices "
            "[2e-5, 2e-4, 2e-3, 2e2] m, so the criterion is evaluated over "
            "those - that is the available evidence in the released cube.",
        "lambda_axis":
            "build_release.build_lambda_axis(): the release build's 54-point "
            "finite mediator-range axis, 1e-7 m (0.1 um) to 2 m. The 20 um, "
            "200 um and 2 mm physics tags are exact members, so sidecar slices "
            "at those ranges are directly comparable to the cube.",
        "alpha_axis":
            "np.logspace(-10, 0, 44): bitwise the cube's axes/alpha_n.",
        "physics": {
            "q_thresh_gev": float(ref["q_thresh"]),
            "q_hi_ref_gev": 8400.0,
            "t_exposure_s": float(ref["t_total"]),
            "b_constrained_max_m": None,
            "b_constrained_max_note":
                "uncapped cross section (rate.make_xsec b_constrained_max=None), "
                "matching the v8 release whose b_constrained_max_m attribute is NaN",
            "f_dm": 1.0,
            "f_x_baseline": 0.1,
            "f_dm_note":
                "luhdm.rate carries the config.F_X = 0.1 baseline internally; "
                "the rate and n_transit are rescaled by f_DM/F_X = 10, the same "
                "linear scaling build_release uses for its f_DM = 1 surfaces",
            "atmosphere": True,
            "efficiency_df": 3,
            "cross_section_tabulation":
                "rate.make_xsec(force_ln=True) for every range (uniform "
                "log-space tabulation). Agrees with the release build's "
                "xi-dispatched default to ~0.4% in mu; the excluded alpha bands "
                "at the 20 um / 200 um / 2 mm tags are identical to the cube.",
        },
        "statistics": {
            "seed": int(ref["seed"]),
            "mu_cap": float(ref["mu_cap"]),
            "mu_floor": 0.2,
            "n_mc": 10000,
            "mc_table":
                "build_release.PerMuTable(seed): MC calibration sharded by "
                "rounded mu, so the result is independent of evaluation order "
                "and worker count",
            "fidelity": fid,
        },
        "inputs": {
            "efficiency_npz": "~/code/luhdm/luhdm/reference_data/efficiency_curves.npz",
            "efficiency_npz_sha256":
                sha256(REPO / "luhdm" / "reference_data" / "efficiency_curves.npz"),
            "events": {
                f"data_mode{m}.txt": {
                    "path": f"~/code/luhdm/notebooks/data_mode{m}.txt",
                    "sha256": sha256(REPO / "notebooks" / f"data_mode{m}.txt"),
                    "n_events": int(scans[m]["events"].size),
                } for m in MODES
            },
        },
        "code": {
            "scan_script": "scripts/scan_lambda.py",
            "scan_script_sha256": sha256(REPO / "scripts" / "scan_lambda.py"),
            "build_release_sha256": sha256(REPO / "scripts" / "build_release.py"),
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": bool(git("status", "--porcelain")),
        },
        "argv": {str(m): [tilde(a) for a in argv[str(m)]] for m in MODES},
        "arrays":
            "per mode n in {1,2,3}: lambda_m_mode{n} [m] (54), "
            "alpha_n_mode{n} (44), extremeness_mode{n} and mu_mode{n} both "
            "(alpha, lambda) = (44, 54), n_transit_mode{n} (44, 54), "
            "best_mass_gev_mode{n} scalar. Exclusion at 95%: extremeness >= 0.95.",
        "created": datetime.now(timezone.utc).isoformat(),
    }

    payload = {"provenance": json.dumps(prov, indent=1)}
    for m in MODES:
        d = scans[m]
        payload[f"lambda_m_mode{m}"] = np.asarray(d["lambs"], dtype=np.float64)
        payload[f"alpha_n_mode{m}"] = np.asarray(d["alphas_n"], dtype=np.float64)
        payload[f"extremeness_mode{m}"] = np.asarray(d["extremeness"], dtype=np.float64)
        payload[f"mu_mode{m}"] = np.asarray(d["counts"], dtype=np.float64)
        payload[f"n_transit_mode{m}"] = np.asarray(d["n_transit"], dtype=np.float64)
        payload[f"best_mass_gev_mode{m}"] = np.float64(d["m_dm"])
    payload["modes"] = np.array(MODES, dtype=np.int64)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **payload)
    print(f"wrote {OUT}  ({OUT.stat().st_size/1024:.0f} KiB)")

    # --- leakage gate: inspect the DECOMPRESSED contents of what was written -- #
    import socket
    hosts = {socket.gethostname().lower(), "fried"}
    back = np.load(OUT, allow_pickle=False)
    text = " ".join(back.files)
    for k in back.files:
        v = back[k]
        if v.dtype.kind in "US":
            text += " " + " ".join(np.atleast_1d(v).astype(str).tolist())
    low = text.lower()
    assert "/home/" not in text, "'/home/' leaked into the sidecar"
    for h in hosts:
        assert h and h not in low, f"hostname {h!r} leaked into the sidecar"
    json.loads(str(back["provenance"]))          # provenance must round-trip
    print(f"leakage gate: no '/home/' and no hostname ({sorted(hosts)}) in the sidecar")
    print(shlex.join(["sha256sum", str(OUT)]), "->", sha256(OUT))


if __name__ == "__main__":
    main()
