#!/usr/bin/env python3
"""Assemble the per-lambda npz shards into the release HDF5 (LOCAL machine only).

The compute node writes float64 npz shards (no h5py there); this script — run on
a host with h5py — stacks them into the cube named by ``--out``, the source every
notebook loads from. It also computes ``/reference_curves`` locally at production
fidelity (notebook 02's showcase point), copies the detector products, embeds
full provenance, and writes the matching provenance JSON + ``SHA256SUMS`` beside
it. ``--select`` splits the release by hypothesis: run it once per
``(f_dm, atmosphere)`` plane to get the v7 two-file layout, each file with its
own ``provenance_<stem>.json``; the default ``both`` keeps every plane in one
cube and the historical ``provenance.json`` name.

    python scripts/assemble_release.py \
        --atm-dir   ~/release_shards/atm \
        --noatm-dir ~/release_shards/noatm \
        --halo-dir  ~/release_shards/halo

Float64 shard dirs are archived as the verification source of record; the H5
carries float32 cubes (p granularity is MC-limited at ~1e-4, far coarser than
f4 spacing ~6e-8 near 0.95).

Layouts (``--layout``):

``axes`` (default, the release layout, file ``version`` 2)
    One ``/results/<quantity>`` array per quantity, shaped
    ``(f_dm, atmosphere, mode, alpha_n, mass, lambda)`` — so every element
    explicitly carries its (f_DM) x (atmosphere) hypothesis, named by
    ``axes/f_dm`` = [0.1, 1.0] and ``axes/atmosphere`` = [1, 0] (1 = attenuation
    on). The noatm shard dir fills the atmosphere=0 plane and atm the
    atmosphere=1 plane; unsuffixed shard arrays fill f_dm=0.1 and ``_f1`` fills
    f_dm=1.0. ``n_transit`` is materialised at full shape (it is
    atmosphere-dependent and exactly linear in f_DM). This needs BOTH f_DM
    surfaces and ONE shared mass axis across the two passes.

``groups`` (the v3 layout, file ``version`` 1)
    ``/atm`` and ``/noatm`` groups, each with ``extremeness``/``mu``/``status``
    at f_DM = 0.1 plus ``*_f1`` at f_DM = 1.0, and a mass axis per pass. Kept so
    v3-schema cubes can still be produced and compared against.

``luhdm.release`` reads both, keyed on schema detection.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# Recorded provenance ships with the data release (Zenodo) and must carry no
# absolute home paths or usernames; every path string is stored home-relative
# ('~/...'), which stays copy-pasteable through shell expansion.
HOME = str(Path.home())


def _scrub_str(s):
    return s.replace(HOME, "~")


def scrub_home(x):
    """Recursively home-relativise every string in a provenance tree.

    Also blanks any nested ``hostname`` field: shards built before the writers
    stopped recording hostnames replay them through run_config.json.
    """
    if isinstance(x, str):
        return _scrub_str(x)
    if isinstance(x, dict):
        return {k: ("" if k == "hostname" else scrub_home(v))
                for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [scrub_home(v) for v in x]
    return x

import h5py

from luhdm import atmosphere, config, efficiency, halo, rate, units

# ── pinned constants (shared contract) ────────────────────────────────────────
SEED = 20260702
Q_HI_REF = 8.4e3
CONFIDENCE = 0.95
M_PLANCK = 1.22e19
SCHEMA_VERSION = 1
FILE_FORMAT = "luhdm-datarelease"
FORMAT_VERSION = 1        # v3 group layout: /atm, /noatm (+ _f1 datasets)
FORMAT_VERSION_AXES = 2   # axis layout: /results over (f_dm, atmosphere, ...)

# canonical tag order everywhere in the release
TAGS = ["2m", "20cm", "2cm", "2mm", "200um", "20um", "10um", "2um"]
TAG_LAMBDA = {
    "2m": 2.0, "20cm": 0.2, "2cm": 2e-2, "2mm": 2e-3, "200um": 2e-4,
    "20um": 2e-5, "10um": 1e-5, "2um": 2e-6,
}
MODES = (1, 2, 3)

# reference-curve showcase point (notebook 02)
REF_M = 1e8
REF_ALPHA_N = 1e-3
REF_TAG_FOR_SPECTRA = "200um"


def _iso_now():
    return datetime.now(timezone.utc).isoformat()


# ── git / package provenance ──────────────────────────────────────────────────
def git_provenance():
    """(commit, dirty, dirty_files) from the repo; read-only, never fatal.

    ``dirty_files`` names the repo-relative paths behind a True ``dirty`` flag,
    so provenance shows *what* was uncommitted (e.g. display scripts held under
    review) instead of an unexplained dirty build.
    """
    commit, dirty, dirty_files = "unknown", None, []
    try:
        commit = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=15,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=15,
        ).stdout.strip()
        dirty = bool(porcelain)
        dirty_files = sorted(ln[3:] for ln in porcelain.splitlines())
    except Exception as err:  # noqa: BLE001
        print(f"WARNING: git provenance unavailable ({err})")
    return commit, dirty, dirty_files


def package_versions():
    import importlib.metadata as im
    out = {}
    for pkg in ("numpy", "scipy", "h5py", "optimum_interval", "luhdm",
                "matplotlib", "pandas"):
        try:
            out[pkg] = im.version(pkg)
        except Exception:  # noqa: BLE001
            out[pkg] = "unknown"
    out["python"] = platform.python_version()
    return out


# ── shard loading / validation ────────────────────────────────────────────────
def _scalar(v):
    """Unwrap a 0-d numpy array to a Python scalar."""
    a = np.asarray(v)
    return a.item() if a.ndim == 0 else a


def _parse_fidelity(fid_str):
    """str(FID) -> dict; robust to the exact scan_grid.py formatting."""
    try:
        d = ast.literal_eval(str(fid_str))
        return {k: d[k] for k in d}
    except Exception:  # noqa: BLE001
        return {}


def load_pass(pass_dir, pass_name):
    """Read + validate every shard in an atm/noatm dir, stack to a cube.

    Returns a dict with axes, cubes (mode, alpha, mass, lambda) [massless last],
    the finite-lambda axis, and the shared metadata pulled off the shards.
    Aborts (SystemExit) on missing shards or mixed fidelity/axes/events.
    """
    pass_dir = Path(pass_dir)
    if not pass_dir.is_dir():
        sys.exit(f"FATAL: {pass_name} shard dir does not exist: {pass_dir}")

    finite = {}   # il -> loaded dict
    massless = None
    for f in sorted(pass_dir.glob("shard_*.npz")):
        if f.name.startswith("shard_halo"):
            continue
        with np.load(f, allow_pickle=True) as d:
            rec = {k: d[k] for k in d.files}
        rec["_file"] = f.name
        if bool(_scalar(rec["massless"])):
            if massless is not None:
                sys.exit(f"FATAL: two massless shards in {pass_dir}")
            massless = rec
        else:
            il = int(_scalar(rec["il"]))
            if il in finite:
                sys.exit(f"FATAL: duplicate il={il} in {pass_dir}")
            finite[il] = rec

    if not finite and massless is None:
        sys.exit(f"FATAL: no shards found in {pass_dir}")

    # completeness: ils must be contiguous 0..n_finite-1, massless present
    n_finite = len(finite)
    missing = [il for il in range(n_finite) if il not in finite]
    if missing:
        have = sorted(finite)
        sys.exit(f"FATAL: {pass_name} missing finite il shards {missing} "
                 f"(present ils: {have}); expected contiguous 0..{n_finite - 1}")
    if massless is None:
        sys.exit(f"FATAL: {pass_name} missing shard_massless.npz")

    # order finite shards ascending in lambda (contract: finite ascending)
    fin_recs = [finite[il] for il in range(n_finite)]
    lam_by_il = np.array([float(_scalar(r["lamb"])) for r in fin_recs])
    order = np.argsort(lam_by_il, kind="stable")
    fin_recs = [fin_recs[i] for i in order]
    lambda_finite = lam_by_il[order].astype(np.float64)
    if not np.all(np.diff(lambda_finite) > 0):
        print(f"WARNING: {pass_name} finite lambda axis not strictly increasing "
              f"after sort: {lambda_finite}")

    ref = fin_recs[0]
    ms = np.asarray(ref["ms"], dtype=np.float64)
    alphas_n = np.asarray(ref["alphas_n"], dtype=np.float64)
    q_min = float(_scalar(ref["q_min"]))
    seed = int(_scalar(ref["seed"]))
    t_total = float(_scalar(ref["t_total"]))
    df = int(_scalar(ref["df"])) if "df" in ref else 3
    fid_str = str(_scalar(ref["fidelity"]))
    events = {n: np.asarray(ref[f"events_mode{n}"], dtype=np.float64)
              for n in MODES}

    n_a, n_m = alphas_n.size, ms.size

    # cross-shard identity (hard gate): axes, q_min, seed, t_total, fidelity, events
    all_recs = fin_recs + [massless]
    for rec in all_recs:
        tag = rec["_file"]
        if not np.array_equal(np.asarray(rec["ms"], dtype=np.float64), ms):
            sys.exit(f"FATAL: {pass_name}:{tag} mass axis differs from {ref['_file']}")
        if not np.array_equal(np.asarray(rec["alphas_n"], dtype=np.float64), alphas_n):
            sys.exit(f"FATAL: {pass_name}:{tag} alpha axis differs")
        if float(_scalar(rec["q_min"])) != q_min:
            sys.exit(f"FATAL: {pass_name}:{tag} q_min differs")
        if int(_scalar(rec["seed"])) != seed:
            sys.exit(f"FATAL: {pass_name}:{tag} seed differs")
        if float(_scalar(rec["t_total"])) != t_total:
            sys.exit(f"FATAL: {pass_name}:{tag} t_total differs")
        if str(_scalar(rec["fidelity"])) != fid_str:
            sys.exit(f"FATAL: {pass_name}:{tag} MIXED FIDELITY "
                     f"({str(_scalar(rec['fidelity']))!r} != {fid_str!r})")
        for n in MODES:
            if not np.array_equal(
                    np.asarray(rec[f"events_mode{n}"], dtype=np.float64), events[n]):
                sys.exit(f"FATAL: {pass_name}:{tag} events_mode{n} differ")

    # dual f_DM: shards from schema 2 on also carry the f_DM=1 surfaces. Older
    # shard dirs (f=0.1 only) stay assemblable — the _f1 datasets are then just
    # absent from the release.
    ordered = fin_recs + [massless]
    f1_missing = [r["_file"] for r in ordered if "p_f1" not in r]
    has_f1 = not f1_missing
    if f1_missing and len(f1_missing) != len(ordered):
        sys.exit(f"FATAL: {pass_name} MIXED f_DM schema: shards without the "
                 f"f_DM=1 arrays: {f1_missing}")
    f_dm_values = None
    if has_f1:
        vals = {tuple(np.asarray(r["f_dm_values"], dtype=np.float64).tolist())
                for r in ordered if "f_dm_values" in r}
        if len(vals) > 1:
            sys.exit(f"FATAL: {pass_name} MIXED f_dm_values across shards: {vals}")
        f_dm_values = list(vals.pop()) if vals else [float(config.F_X), 1.0]

    # stack cubes: lambda axis = finite (ascending) then massless last
    L = n_finite + 1
    p = np.full((3, n_a, n_m, L), np.nan, dtype=np.float64)
    mu = np.full((3, n_a, n_m, L), np.nan, dtype=np.float64)
    nt = np.full((n_a, n_m, L), np.nan, dtype=np.float64)
    st = np.zeros((3, n_a, n_m, L), dtype=np.uint8)
    p_f1 = np.full((3, n_a, n_m, L), np.nan, dtype=np.float64) if has_f1 else None
    mu_f1 = np.full((3, n_a, n_m, L), np.nan, dtype=np.float64) if has_f1 else None
    st_f1 = np.zeros((3, n_a, n_m, L), dtype=np.uint8) if has_f1 else None
    for li, rec in enumerate(ordered):
        p[:, :, :, li] = np.asarray(rec["p"], dtype=np.float64)
        mu[:, :, :, li] = np.asarray(rec["mu"], dtype=np.float64)
        nt[:, :, li] = np.asarray(rec["n_transit"], dtype=np.float64)
        st[:, :, :, li] = np.asarray(rec["status"], dtype=np.uint8)
        if has_f1:
            p_f1[:, :, :, li] = np.asarray(rec["p_f1"], dtype=np.float64)
            mu_f1[:, :, :, li] = np.asarray(rec["mu_f1"], dtype=np.float64)
            st_f1[:, :, :, li] = np.asarray(rec["status_f1"], dtype=np.uint8)

    # status<->NaN consistency (soft: warn with counts, never abort)
    _check_status_nan(pass_name, p, mu, nt, st)
    if has_f1:
        _check_status_nan(f"{pass_name}[f_dm=1]", p_f1, mu_f1, nt, st_f1)
        _check_f1_scaling(pass_name, mu, mu_f1, f_dm_values)

    lambda_ode = float(_scalar(massless.get("lamb_ode", np.nan)))
    b_cap, cap_unflagged = _cap_provenance(pass_name, all_recs)
    return dict(
        pass_name=pass_name, ms=ms, alphas_n=alphas_n,
        lambda_finite=lambda_finite, n_finite=n_finite,
        p=p, mu=mu, n_transit=nt, status=st,
        p_f1=p_f1, mu_f1=mu_f1, status_f1=st_f1,
        has_f1=has_f1, f_dm_values=f_dm_values,
        inputs=_shard_inputs(ordered),
        q_min=q_min, seed=seed, t_total=t_total, df=df,
        fidelity_str=fid_str, fidelity=_parse_fidelity(fid_str),
        events=events, lambda_ode=lambda_ode,
        b_constrained_max=b_cap, cap_unflagged_shards=cap_unflagged,
        shard_files=[r["_file"] for r in all_recs],
    )


def _cap_provenance(pass_name, recs):
    """Resolve the impact-parameter cap across a pass's shards.

    Shards written before ``--b-constrained-max`` existed carry no
    ``b_constrained_max`` key. Such shards are legitimately reusable in a capped
    cube only over the lambda range where the cap provably cannot bite (see
    release/README.md; the reuse is gated by a byte-identity check on the
    boundary shard), so they are recorded, not refused. Two *different* explicit
    cap values in one pass is genuinely mixed provenance and is fatal.

    Returns (cap_or_None, [files carrying no cap flag]).
    """
    caps, unflagged = {}, []
    for rec in recs:
        v = _scalar(rec["b_constrained_max"]) if "b_constrained_max" in rec else None
        try:
            v = None if v is None else float(v)
        except (TypeError, ValueError):
            v = None
        if v is None or not np.isfinite(v):
            unflagged.append(rec["_file"])
        else:
            caps[rec["_file"]] = v
    distinct = sorted(set(caps.values()))
    if len(distinct) > 1:
        sys.exit(f"FATAL: {pass_name} MIXED b_constrained_max across shards: "
                 f"{distinct}")
    return (distinct[0] if distinct else None), unflagged


def _shard_inputs(recs):
    """The per-run input provenance stamped on the shards (paths + sha256).

    ``inputs_json`` is written by build_release from the *runtime* resolution of
    the event dir, the efficiency table (LUHDM_EFFICIENCY_NPZ) and the live-time
    (LUHDM_T_EXPOSURE). Distinct values across a pass are reported as a list so
    a mixed-input pass is visible rather than silently collapsed.
    """
    seen = []
    for rec in recs:
        if "inputs_json" not in rec:
            continue
        try:
            v = json.loads(str(_scalar(rec["inputs_json"])))
        except Exception:  # noqa: BLE001
            continue
        if v not in seen:
            seen.append(v)
    if not seen:
        return None
    if len(seen) > 1:
        print(f"WARNING: shards disagree on their inputs provenance "
              f"({len(seen)} distinct records); all are kept in provenance.json")
        return seen
    return seen[0]


def _check_f1_scaling(pass_name, mu, mu_f1, f_dm_values):
    """mu(f=1) must be exactly the flux ratio times mu(f=0.1). Warn, never abort.

    f_DM is a pure normalisation, so this is an identity up to float round-off
    in the trapz of the scaled rate; a violation means the two surfaces were not
    computed from the same spectrum.
    """
    if f_dm_values is None or len(f_dm_values) != 2 or f_dm_values[0] == 0:
        return
    ratio = f_dm_values[1] / f_dm_values[0]
    ok = np.isfinite(mu) & np.isfinite(mu_f1) & (mu > 0)
    if not ok.any():
        return
    rel = np.abs(mu_f1[ok] / (ratio * mu[ok]) - 1.0)
    worst = float(rel.max())
    if worst > 1e-9:
        bad = int(np.count_nonzero(rel > 1e-9))
        print(f"WARNING: {pass_name} mu_f1 != {ratio:g}*mu in {bad} cells "
              f"(worst rel dev {worst:.2e})")
    else:
        print(f"  {pass_name}: mu_f1 == {ratio:g}*mu OK (worst rel dev "
              f"{worst:.1e})")


def _check_status_nan(pass_name, p, mu, nt, st):
    """Report (never abort) status<->NaN inconsistencies."""
    is1 = st == 1
    # status 1 cells must be NaN in p and mu
    bad_p = int(np.count_nonzero(is1 & ~np.isnan(p)))
    bad_mu = int(np.count_nonzero(is1 & ~np.isnan(mu)))
    # non-status-1 cells must be finite in p and mu
    bad_ok = int(np.count_nonzero(~is1 & np.isnan(p)))
    # n_transit NaN <=> all 3 modes status 1 at that (alpha, mass, lambda)
    all1 = np.all(is1, axis=0)
    bad_nt = int(np.count_nonzero(np.isnan(nt) != all1))
    if bad_p or bad_mu or bad_ok or bad_nt:
        print(f"WARNING: {pass_name} status<->NaN inconsistencies: "
              f"status1-but-finite p={bad_p} mu={bad_mu}; "
              f"ok-but-NaN p={bad_ok}; n_transit/all-mode-fail mismatch={bad_nt}")
    else:
        print(f"  {pass_name}: status<->NaN consistency OK")


def load_halo(halo_dir):
    """Read + validate halo shards; stack to (alpha_halo, mass_halo, lambda)."""
    halo_dir = Path(halo_dir)
    if not halo_dir.is_dir():
        sys.exit(f"FATAL: halo shard dir does not exist: {halo_dir}")

    finite, massless = {}, None
    for f in sorted(halo_dir.glob("shard_halo_*.npz")):
        with np.load(f, allow_pickle=True) as d:
            rec = {k: d[k] for k in d.files}
        rec["_file"] = f.name
        if bool(_scalar(rec["massless"])):
            massless = rec
        else:
            finite[int(_scalar(rec["il"]))] = rec

    if not finite and massless is None:
        sys.exit(f"FATAL: no halo shards found in {halo_dir}")
    n_finite = len(finite)
    missing = [il for il in range(n_finite) if il not in finite]
    if missing:
        sys.exit(f"FATAL: halo missing finite il shards {missing}")
    if massless is None:
        sys.exit("FATAL: halo missing shard_halo_massless.npz")

    fin_recs = [finite[il] for il in range(n_finite)]
    lam_by_il = np.array([float(_scalar(r["lamb"])) for r in fin_recs])
    order = np.argsort(lam_by_il, kind="stable")
    fin_recs = [fin_recs[i] for i in order]
    lambda_finite = lam_by_il[order].astype(np.float64)

    ref = fin_recs[0]
    ms = np.asarray(ref["ms"], dtype=np.float64)
    alphas_n = np.asarray(ref["alphas_n"], dtype=np.float64)
    for rec in fin_recs + [massless]:
        if not np.array_equal(np.asarray(rec["ms"], dtype=np.float64), ms):
            sys.exit(f"FATAL: halo:{rec['_file']} mass axis differs")
        if not np.array_equal(np.asarray(rec["alphas_n"], dtype=np.float64), alphas_n):
            sys.exit(f"FATAL: halo:{rec['_file']} alpha axis differs")

    n_a, n_m = alphas_n.size, ms.size
    L = n_finite + 1
    nt = np.full((n_a, n_m, L), np.nan, dtype=np.float64)
    bmax = np.full((n_a, n_m, L), np.nan, dtype=np.float64)
    for li, rec in enumerate(fin_recs + [massless]):
        nt[:, :, li] = np.asarray(rec["nt"], dtype=np.float64)
        bmax[:, :, li] = np.asarray(rec["bmax_m"], dtype=np.float64)
    b_cap, cap_unflagged = _cap_provenance("halo", fin_recs + [massless])
    return dict(ms=ms, alphas_n=alphas_n, lambda_finite=lambda_finite,
                n_finite=n_finite, n_transit=nt, bmax=bmax,
                b_constrained_max=b_cap, cap_unflagged_shards=cap_unflagged,
                inputs=_shard_inputs(fin_recs + [massless]),
                shard_files=[r["_file"] for r in (fin_recs + [massless])])


def cross_check_lambda(atm, noatm, halo_d):
    """The three passes must share the finite-lambda axis (hard gate)."""
    la, ln, lh = atm["lambda_finite"], noatm["lambda_finite"], halo_d["lambda_finite"]
    if la.shape != ln.shape or not np.array_equal(la, ln):
        sys.exit(f"FATAL: atm/noatm finite-lambda axes differ "
                 f"(atm n={la.size}, noatm n={ln.size})")
    if lh.shape != la.shape or not np.array_equal(lh, la):
        sys.exit(f"FATAL: halo finite-lambda axis differs from atm/noatm "
                 f"(halo n={lh.size})")
    return la


def cross_check_cap(atm, noatm, halo_d):
    """All three passes must share one impact-parameter cap (hard gate)."""
    caps = {p["pass_name"] if "pass_name" in p else "halo": p["b_constrained_max"]
            for p in (atm, noatm, halo_d)}
    distinct = sorted({v for v in caps.values() if v is not None})
    if len(distinct) > 1:
        sys.exit(f"FATAL: passes disagree on b_constrained_max: {caps}")
    cap = distinct[0] if distinct else None
    if cap is None:
        print("\nimpact-parameter cap: none (uncapped cube)")
        return None
    unflagged = sorted(f for p in (atm, noatm, halo_d)
                       for f in p["cap_unflagged_shards"])
    missing = [k for k, v in caps.items() if v is None]
    if missing:
        sys.exit(f"FATAL: b_constrained_max={cap} m in {sorted(caps)} but "
                 f"pass(es) {missing} carry no cap at all")
    print(f"\nimpact-parameter cap: b_constrained_max = {cap} m")
    if unflagged:
        print(f"  {len(unflagged)} shard(s) reused from an uncapped run "
              f"(provably unaffected; see release/README.md): "
              f"{', '.join(unflagged[:6])}{' ...' if len(unflagged) > 6 else ''}")
    return cap


# ── status census ─────────────────────────────────────────────────────────────
STATUS_NAMES = {0: "ok(MC)", 1: "exception", 2: "mu<0.2", 3: "mu>mu_cap",
                4: "mu==0"}


def print_status_report(pass_name, pd):
    """Per-status counts + every status-1 cell. Warn, never abort."""
    st = pd["status"]
    print(f"\n── status census: {pass_name} "
          f"(cube {st.shape} = mode,alpha,mass,lambda) ──")
    total = st.size
    for code in sorted(STATUS_NAMES):
        c = int(np.count_nonzero(st == code))
        print(f"  status {code} {STATUS_NAMES[code]:>10}: {c:>10d}  "
              f"({100.0 * c / total:5.2f}%)")
    lambdas = list(pd["lambda_finite"]) + [np.inf]
    ones = np.argwhere(st == 1)
    print(f"  status-1 (exception) cells: {len(ones)}")
    for (k, ia, im, li) in ones:
        lam = lambdas[li]
        print(f"    [{pass_name}] il={li} lambda={lam:.3e} mode={k + 1} "
              f"alpha_n={pd['alphas_n'][ia]:.3e} m={pd['ms'][im]:.3e} GeV")


# ── /reference_curves (production fidelity, notebook 02 showcase) ─────────────
def compute_reference_curves(quick=False):
    """Notebook-02 showcase arrival-speed distributions + raw spectra.

    m = 1e8 GeV, alpha_n = 1e-3. Speed grid v = linspace(Q_THRESH/1e8/10, VESC,
    500). fv_shm = SHM; fv_<tag> via the attenuation ODE per tag. Raw dR/dq on
    q = geomspace(1e2, 1e5, 160) with the arrival f(v) FIXED at the 200 um
    attenuated distribution for every curve (mirrors notebook 02 exactly).
    """
    n_grid = 80 if quick else 400
    n_shm = int(1e5) if quick else int(3e5)
    print(f"\ncomputing /reference_curves (n_grid={n_grid}, n_shm={n_shm}) ...")
    t0 = time.time()

    v_min = config.Q_THRESH / REF_M / 10.0
    v = np.linspace(v_min, config.VESC, 500)
    q_gev = np.geomspace(1e2, 1e5, 160)

    v_i_samples = atmosphere.sample_shm(n_shm, rng=np.random.default_rng(SEED))

    out = {"v": v.astype(np.float64), "q_gev": q_gev.astype(np.float64),
           "fv": {}, "survival": {}, "drdq": {}}

    # SHM baseline (unattenuated); survival fraction = 1.0 by construction
    out["fv"]["shm"] = halo.standard_halo_model(v).astype(np.float32)
    out["survival"]["shm"] = 1.0

    f_v_f_ref = None   # the 200 um attenuated distribution, reused for all dR/dq
    for tag in TAGS:
        lamb = TAG_LAMBDA[tag]
        v_f_samples = atmosphere.compute_v_f_distribution(
            REF_ALPHA_N, lamb, REF_M, v_i_samples, v_min=v_min, n_grid=n_grid)
        f_v_f, f_survive = atmosphere.compute_f_vf(v_f_samples, v_min)
        out["fv"][tag] = np.asarray(f_v_f(v), dtype=np.float32)
        out["survival"][tag] = float(f_survive)
        if tag == REF_TAG_FOR_SPECTRA:
            f_v_f_ref = f_v_f
        print(f"    {tag:>6}: survival = {f_survive:.4f}")

    assert f_v_f_ref is not None, "200um tag missing from TAGS"

    # raw dR/dq for each tag + massless, arrival f(v) fixed at 200 um
    out["drdq"]["massless"] = np.asarray(
        rate.differential_rate_trapz(q_gev, REF_ALPHA_N, REF_M, f_v_f_ref,
                                     rate.make_xsec(None), eff=None),
        dtype=np.float32)
    for tag in TAGS:
        xs = rate.make_xsec(TAG_LAMBDA[tag])   # auto dispatch (not force_ln)
        out["drdq"][tag] = np.asarray(
            rate.differential_rate_trapz(q_gev, REF_ALPHA_N, REF_M, f_v_f_ref,
                                         xs, eff=None),
            dtype=np.float32)
    print(f"  reference curves done ({time.time() - t0:.0f}s)")
    return out


# ── /detector: events, efficiency curves, blip momenta ────────────────────────
def sha256_file(path):
    """sha256 of a file, or None if it cannot be read (never fatal)."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()
    except Exception as err:  # noqa: BLE001
        print(f"WARNING: sha256 unavailable for {path}: {err}")
        return None


def events_dir(data_dir=None):
    """Directory holding data_mode{n}.txt for this assembly."""
    return Path(data_dir) if data_dir else REPO / "notebooks"


def load_events(data_dir=None):
    """events_mode{n} in GeV from <data-dir>/data_mode{n}.txt (source of truth).

    ``data_dir`` must match the ``--data-dir`` the shards were built with (e.g.
    the cveto event dir); the recorded sha256s in provenance.json are of the
    files actually read here.
    """
    out = {}
    for n in MODES:
        f = events_dir(data_dir) / f"data_mode{n}.txt"
        if f.exists():
            out[n] = np.atleast_1d(np.loadtxt(f)).astype(np.float64) / 1e9
        else:
            out[n] = None
    return out


def load_efficiency_table(table=None):
    """The efficiency curves in use, copied verbatim (key names preserved).

    Defaults to :func:`luhdm.efficiency.table_path` so LUHDM_EFFICIENCY_NPZ (the
    veto-variant table) is honoured instead of silently assuming the committed
    one.
    """
    tbl = Path(table) if table else Path(efficiency.table_path())
    out = {}
    with np.load(tbl) as d:
        for n in MODES:
            for key in (f"q_gev_{n}", f"eff_{n}_df2", f"eff_{n}_df3"):
                out[key] = np.asarray(d[key], dtype=np.float64)
    return out


def assembly_inputs(data_dir, eff_table):
    """Paths + sha256 of the detector inputs this assembly actually read.

    Paths come back home-relativised (:func:`scrub_home`) so the file attrs and
    provenance.json built from them are clean at birth.
    """
    tbl = Path(eff_table) if eff_table else Path(efficiency.table_path())
    return scrub_home({
        "t_exposure_s": float(config.T_EXPOSURE),
        "efficiency_npz": str(tbl),
        "efficiency_npz_sha256": sha256_file(tbl),
        "events_dir": str(events_dir(data_dir)),
        "events": {
            f"data_mode{n}.txt": {
                "path": str(events_dir(data_dir) / f"data_mode{n}.txt"),
                "sha256": sha256_file(events_dir(data_dir) / f"data_mode{n}.txt"),
            } for n in MODES},
        "env": {k: os.environ[k] for k in
                ("LUHDM_T_EXPOSURE", "LUHDM_EFFICIENCY_NPZ") if k in os.environ},
    })


def extract_blips():
    """All blip momenta [eV] per mode, reproduced from the notebook-00 source.

    No cache exists, so we replay ``blip_momenta_eV`` against the raw hdf5 the
    notebook reads. If the raw data/ files are absent (or anything else breaks)
    we return None for that mode — the caller writes a placeholder with
    ``missing=true`` and warns loudly; assembly never fails on this.
    """
    data_dir = REPO / "data"
    hdf = data_dir / "fit_data_temp_lockin_transients_selected.hdf5"
    npz = data_dir / "selected_data_efficiency_curves.npz"
    if not hdf.exists() or not npz.exists():
        print(f"WARNING: raw blip source absent ({hdf.name}/{npz.name}); "
              "writing empty all_blips placeholders (missing=true)")
        return {n: None for n in MODES}
    try:
        import pandas as pd
        C_LIGHT = 2.99792458e8
        E_CHARGE = 1.602176634e-19
        EV_PER_SI = C_LIGHT / E_CHARGE
        df = pd.read_hdf(hdf)
        with np.load(npz) as d:
            dv_bins = {n: d[f"dv_bins_{n}"] for n in MODES}
        _dv_volts = np.logspace(-6, -3, 400)
        K = {n: float(np.median(dv_bins[n] / _dv_volts)) for n in MODES}

        def blip_momenta_eV(mode):
            out = []
            q_col = df[f"transient_q_{mode}"]
            dv_col = df[f"transient_dv_hat_{mode}"]
            up_col = df[f"upcrossings_{mode}"]
            for q, dv, ups in zip(q_col, dv_col, up_col):
                if not len(ups):
                    continue
                q = np.asarray(q)
                dv = np.asarray(dv)
                for start_i, end_i in ups:
                    k = start_i + int(np.argmax(q[start_i:end_i + 1]))
                    p_si = abs(dv[k]) * K[mode]
                    out.append(p_si * EV_PER_SI)
            return np.array(sorted(out), dtype=np.float64)

        blips = {n: blip_momenta_eV(n) for n in MODES}
        for n in MODES:
            print(f"  mode {n}: extracted {blips[n].size} blips [eV]")
        return blips
    except Exception as err:  # noqa: BLE001
        print(f"WARNING: blip extraction failed ({err}); "
              "writing empty all_blips placeholders (missing=true)")
        return {n: None for n in MODES}


# ── HDF5 writer ───────────────────────────────────────────────────────────────
def _ds(grp, name, data, units_s, desc, *, cube=False, chunks=None):
    """Create a dataset with units/description attrs, optionally compressed."""
    data = np.asarray(data)
    kw = {}
    if cube:
        kw = dict(compression="gzip", compression_opts=4, shuffle=True,
                  chunks=chunks)
    d = grp.create_dataset(name, data=data, **kw)
    d.attrs["units"] = units_s
    d.attrs["description"] = desc
    return d


def _attach_scales(dset, scale_dsets, labels, flat_axes):
    """Attach HDF5 dimension scales + a flat axes attr (order = dataset order)."""
    for i, (sc, lab) in enumerate(zip(scale_dsets, labels)):
        try:
            dset.dims[i].attach_scale(sc)
            dset.dims[i].label = lab
        except Exception as err:  # noqa: BLE001
            print(f"WARNING: could not attach scale {lab} to {dset.name}: {err}")
    dset.attrs["axes"] = flat_axes


def _mode_plane_chunks(shape, has_mode):
    """One mode-plane chunk: (1, n_a, n_m, L) or the whole (n_a, n_m, L)."""
    if has_mode:
        return (1,) + tuple(shape[1:])
    return tuple(shape)


def _shared_mass_axis(atm, noatm):
    """The single mass axis of the axis-based layout, or abort explaining why.

    ``results/*`` is one array spanning (f_dm, atmosphere, mode, alpha, mass,
    lambda), so the atm and noatm planes must live on the SAME mass grid. The
    default tiers do not (atm 119, noatm 600: only the two endpoints coincide),
    and resampling either onto the other would fabricate values, so this is a
    hard gate rather than a silent interpolation.
    """
    ma, mn = atm["ms"], noatm["ms"]
    if ma.shape == mn.shape and np.array_equal(ma, mn):
        return ma
    sys.exit(
        f"FATAL: the axis-based layout needs one mass axis, but atm has "
        f"{ma.size} masses and noatm has {mn.size} "
        f"({int(np.intersect1d(ma, mn).size)} exactly in common).\n"
        f"       Build both passes on the same grid (build_release.py "
        f"--m-tier N for BOTH --pass atm and --pass noatm), or assemble the "
        f"group-based layout instead (--layout groups), which keeps a mass "
        f"axis per pass.")


M_CUT_B_CAP_M = 0.1          # aperture radius of the flux argument [m]
M_CUT_N_TRANSITS = 3.0       # transits required within that aperture


def halo_mean_speed_m_s():
    """Flux-weighted <v> of the halo, in the pipeline's own convention.

    rate.expected_transits / rate.transit_count_halo build the transit flux as
    n_dm * <f(v) v> * area, with f the truncated standard halo model in units
    of c and speeds converted by units.C_M_S. The same normalised first moment
    is used here so m_cut and the stored n_transit surface describe the same
    halo, not two different ones.
    """
    vs = np.linspace(1e-8, config.VESC, 200000)
    f = halo.standard_halo_model(vs)
    return float(np.trapezoid(f * vs, vs) / np.trapezoid(f, vs)) * units.C_M_S


def mass_cut_flux(f_dm, b_cap=M_CUT_B_CAP_M, n_req=M_CUT_N_TRANSITS,
                  t_total=None):
    """Heaviest DM mass the halo still delivers through a b_cap aperture.

    Flux-through-aperture: the number of DM particles passing within b_cap of
    the sensor during the exposure is

        N(m) = f_DM * (rho_0 / m) * <v> * T_obs * pi * b_cap^2,

    with rho_0 = 0.3 GeV/cm^3 the local density the transit diagnostics use.
    N falls as 1/m, so requiring N >= n_req sets a largest mass

        m_cut = f_DM * rho_0 * <v> * T_obs * pi * b_cap^2 / n_req.

    Above m_cut the halo simply does not supply enough close passages for the
    exposure to constrain anything, however large the cross section is. This
    is a statement about the flux, not about the cross section, which is why
    it is applied as a post-facto cut in mass rather than as a cap inside the
    impact-parameter integral: the stored mu/extremeness surfaces stay
    uncapped and auditable, and the cut is a line a reader can move.

    Returns (m_cut [GeV], derivation string).
    """
    t_total = float(config.T_EXPOSURE if t_total is None else t_total)
    rho0 = 0.3 * 1e6                     # 0.3 GeV/cm^3 -> GeV/m^3
    v_mean = halo_mean_speed_m_s()
    m_cut = (float(f_dm) * rho0 * v_mean * t_total * np.pi * b_cap ** 2
             / float(n_req))
    derivation = (
        "m_cut = f_DM * rho_0 * <v> * T_obs * pi * b_cap^2 / N_req, the "
        "largest DM mass for which the halo delivers at least N_req particles "
        "within b_cap of the sensor during the exposure "
        "(N(m) = f_DM (rho_0/m) <v> T_obs pi b_cap^2 falls as 1/m). "
        f"Inputs: f_DM = {float(f_dm):g}, rho_0 = 0.3 GeV/cm^3 = {rho0:g} "
        f"GeV/m^3, <v> = {v_mean:.6g} m/s (flux-weighted first moment of the "
        f"truncated standard halo model, the same convention as the "
        f"n_transit surface), T_obs = {t_total:g} s, b_cap = {b_cap:g} m, "
        f"N_req = {float(n_req):g}  =>  m_cut = {m_cut:.6g} GeV. "
        "NOT baked into mu/extremeness: the stored surfaces are uncapped, and "
        "this is the mass line beyond which they should not be read as a "
        "limit.")
    return m_cut, derivation


def resolve_selection(choice, f_dm_values):
    """(f_DM indices, plane indices) written by ``--select``.

    The cube axes are f_dm = f_dm_values (baseline then the f_DM=1 surface)
    and atmosphere = [1, 0] (atm plane then noatm). A selection subsets those
    two axes only: the schema, the scales and every other group are unchanged,
    so a one-hypothesis file is read exactly like the full cube -- the axes
    just have length 1. Cross-checks still run over everything that was
    loaded; only what is WRITTEN is subset.
    """
    n_f = len(f_dm_values)
    i_f1 = n_f - 1                      # f_DM = 1.0 is the last f_dm value
    table = {
        "both": (list(range(n_f)), [0, 1]),
        "f1-atm": ([i_f1], [0]),        # File A: f_DM = 1, WITH atmosphere
        "f-base-noatm": ([0], [1]),     # File B: f_DM = baseline, NO atmosphere
    }
    if choice not in table:
        sys.exit(f"FATAL: unknown --select {choice!r}; "
                 f"expected one of {sorted(table)}")
    f_sel, plane_sel = table[choice]
    if choice != "both" and n_f < 2:
        sys.exit(f"FATAL: --select {choice} needs both f_DM surfaces; these "
                 f"shards carry only f_DM={f_dm_values}")
    return f_sel, plane_sel


def _write_results_axes(h5, atm, noatm, f_dm_values, L, scales,
                        f_sel=None, plane_sel=None):
    """``/results``: one array per quantity over (f_dm, atmosphere, ...).

    Every element explicitly carries its (f_DM, atmosphere) hypothesis: the
    atmosphere axis holds 1 (attenuation on, the atm pass) then 0 (off, noatm),
    and the f_dm axis holds the baseline then the f_DM=1 surface. n_transit is
    materialised at full shape too — it is atmosphere-dependent and exactly
    linear in f_DM.

    ``f_sel`` / ``plane_sel`` (see :func:`resolve_selection`) subset those two
    axes; None means write everything.
    """
    d_fdm, d_atm, d_mode, d_alpha, d_mass, d_lam = scales
    if f_sel is None:
        f_sel = list(range(len(f_dm_values)))
    if plane_sel is None:
        plane_sel = [0, 1]
    n_f = len(f_sel)
    n_p = len(plane_sel)
    n_a = atm["alphas_n"].size
    n_m = atm["ms"].size
    g = h5.create_group("results")

    # plane order follows the axis values: atmosphere = [1, 0] -> [atm, noatm]
    passes = [[atm, noatm][i] for i in plane_sel]
    scale_f1 = (f_dm_values[1] / f_dm_values[0]
                if len(f_dm_values) > 1 else None)

    def stack(key_base, dtype, fill):
        """(n_f, n_p, 3, n_a, n_m, L) from the per-pass, per-f_DM planes."""
        out = np.full((n_f, n_p) + (3, n_a, n_m, L), fill, dtype=dtype)
        for i_at, pd_ in enumerate(passes):
            for i_f, f_idx in enumerate(f_sel):
                # f_idx 0 = the unsuffixed baseline arrays, else the _f1 ones
                out[i_f, i_at] = pd_[key_base if f_idx == 0
                                     else key_base + "_f1"]
        return out

    cube = (n_f, n_p, 3, n_a, n_m, L)
    ch = (1, 1, 1) + cube[3:]              # one (alpha, mass, lambda) plane
    ext = _ds(g, "extremeness", stack("p", np.float32, np.nan), "1",
              "optimum-interval extremeness / confidence; NaN where status==1",
              cube=True, chunks=ch)
    muu = _ds(g, "mu", stack("mu", np.float32, np.nan), "counts",
              "expected signal counts mu; NaN where status==1. Exactly linear "
              "in f_DM (a pure flux normalisation).", cube=True, chunks=ch)
    stt = _ds(g, "status", stack("status", np.uint8, 0), "enum",
              "0=ok(MC) 1=exception 2=mu<0.2 3=mu>mu_cap 4=mu==0",
              cube=True, chunks=ch)
    for dset in (ext, muu, stt):
        _attach_scales(dset, (d_fdm, d_atm, d_mode, d_alpha, d_mass, d_lam),
                       ("f_dm", "atmosphere", "mode", "alpha_n", "mass_gev",
                        "lambda_m"),
                       "f_dm,atmosphere,mode,alpha_n,mass_gev,lambda_m")

    # n_transit: mode-less, so (n_f, n_p, n_a, n_m, L). The f_DM=1 plane is
    # materialised as scale_f1 x the baseline (n_dm ∝ f_DM, geometry unchanged).
    nt = np.full((n_f, n_p, n_a, n_m, L), np.nan, dtype=np.float32)
    for i_at, pd_ in enumerate(passes):
        base = np.maximum(pd_["n_transit"], 0.0).astype(np.float32)
        for i_f, f_idx in enumerate(f_sel):
            # scale the STORED float32 baseline, so the materialised plane is
            # exactly scale_f1 x what a reader sees at f_dm[0] (and identical to
            # what the group layout produces by scaling on read).
            nt[i_f, i_at] = (base if f_idx == 0
                             else base * np.float32(scale_f1))
    ntd = _ds(g, "n_transit", nt, "counts",
              "expected within-reach transits; clipped >=0 (KDE tail can "
              "oscillate slightly negative). Exactly linear in f_DM.",
              cube=True, chunks=(1, 1) + nt.shape[2:])
    ntd.attrs["clipped_nonnegative"] = True
    _attach_scales(ntd, (d_fdm, d_atm, d_alpha, d_mass, d_lam),
                   ("f_dm", "atmosphere", "alpha_n", "mass_gev", "lambda_m"),
                   "f_dm,atmosphere,alpha_n,mass_gev,lambda_m")


def write_h5(out_path, atm, noatm, halo_d, lambda_finite, ref, detector,
             version_tag, quick_reference, inputs, layout="axes",
             select="both", m_cut=None):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    axis_layout = layout == "axes"

    # lambda axis in file: finite ascending + inf last; m_phi = 1/conv, 0 at inf
    lambda_m = np.concatenate([lambda_finite, [np.inf]]).astype(np.float64)
    with np.errstate(divide="ignore"):
        m_phi = 1.0 / units.conv_m2pGeV(lambda_m)
    m_phi[-1] = 0.0
    L = lambda_m.size
    n_finite = lambda_finite.size

    commit, dirty, dirty_files = git_provenance()
    pkgs = package_versions()
    fid = atm["fidelity"]
    f_dm_values = atm["f_dm_values"] or [float(config.F_X)]
    f_sel, plane_sel = resolve_selection(select, f_dm_values)
    if axis_layout:
        shared_ms = _shared_mass_axis(atm, noatm)
        if not atm["has_f1"] or len(f_dm_values) < 2:
            sys.exit("FATAL: the axis-based layout needs both f_DM surfaces; "
                     "these shards carry only f_DM=" f"{f_dm_values}. Rebuild "
                     "with the dual-f_DM builder or use --layout groups.")

    with h5py.File(tmp, "w") as h5:
        # ── /axes ──
        ax = h5.create_group("axes")
        d_mode = ax.create_dataset("mode", data=np.array(MODES, dtype=np.uint8))
        d_mode.attrs["units"] = "1"
        d_mode.attrs["description"] = "sensor mode index (1,2,3)"
        d_alpha = _ds(ax, "alpha_n", atm["alphas_n"], "1",
                      "per-neutron coupling alpha_n")
        d_fdm = d_atmos = None
        if axis_layout:
            d_mass = _ds(ax, "mass_gev", shared_ms, "GeV",
                         "dark-matter mass (shared by both atmosphere planes)")
            d_mass_no = d_mass
            d_fdm = _ds(ax, "f_dm",
                        np.asarray(f_dm_values, dtype=np.float64)[f_sel],
                        "1", "dark-matter fraction hypothesis of this species; "
                        "a pure flux normalisation (n_dm ∝ f_DM)")
            d_atmos = _ds(ax, "atmosphere",
                          np.array([1, 0], dtype=np.int8)[plane_sel],
                          "bool", "1 = attenuation through the atmosphere/earth "
                          "applied (atm pass); 0 = bare halo flux (noatm pass)")
        else:
            d_mass = _ds(ax, "mass_gev", atm["ms"], "GeV",
                         "dark-matter mass (atm 119-tier grid)")
            d_mass_no = _ds(ax, "mass_noatm_gev", noatm["ms"], "GeV",
                            "dark-matter mass (noatm 600-tier grid)")
        d_lam = _ds(ax, "lambda_m", lambda_m, "m",
                    "mediator range; finite ascending then inf (massless) last")
        d_lam.attrs["n_finite"] = int(n_finite)
        d_lam.attrs["tags_json"] = json.dumps(TAG_LAMBDA)
        d_mphi = _ds(ax, "m_phi_gev", m_phi, "GeV",
                     "mediator mass = 1/conv_m2pGeV(lambda); exactly 0 at inf")
        d_mhalo = _ds(ax, "mass_halo_gev", halo_d["ms"], "GeV",
                      "dark-matter mass (halo/flux-map 64 grid)")
        d_ahalo = _ds(ax, "alpha_halo_n", halo_d["alphas_n"], "1",
                      "coupling alpha_n (halo/flux-map 64 grid)")
        seen = set()
        for dd in (d_mode, d_alpha, d_mass, d_mass_no, d_lam, d_mphi,
                   d_mhalo, d_ahalo, d_fdm, d_atmos):
            if dd is None or dd.name in seen:   # mass_noatm aliases mass_gev
                continue                        # in the axis layout
            seen.add(dd.name)
            dd.make_scale(dd.name.split("/")[-1])

        # ── /results (axis-based) or /atm + /noatm (v3 group layout) ──
        if axis_layout:
            _write_results_axes(h5, atm, noatm, f_dm_values, L,
                                (d_fdm, d_atmos, d_mode, d_alpha, d_mass,
                                 d_lam),
                                f_sel=f_sel, plane_sel=plane_sel)
        for grp_name, pd_, mass_scale in ([] if axis_layout else
                                          (("atm", atm, d_mass),
                                           ("noatm", noatm, d_mass_no))):
            g = h5.create_group(grp_name)
            n_a = pd_["alphas_n"].size
            n_m = pd_["ms"].size
            cube4 = (3, n_a, n_m, L)
            ch4 = _mode_plane_chunks(cube4, has_mode=True)
            ext = _ds(g, "extremeness", pd_["p"].astype(np.float32), "1",
                      f"optimum-interval extremeness / confidence ({grp_name} "
                      "pass); NaN where status==1", cube=True, chunks=ch4)
            muu = _ds(g, "mu", pd_["mu"].astype(np.float32), "counts",
                      f"expected signal counts mu ({grp_name}); NaN where "
                      "status==1", cube=True, chunks=ch4)
            ntc = np.maximum(pd_["n_transit"], 0.0).astype(np.float32)
            ntd = _ds(g, "n_transit", ntc, "counts",
                      f"expected within-reach transits ({grp_name}); clipped "
                      ">=0 (KDE tail can oscillate slightly negative)",
                      cube=True, chunks=_mode_plane_chunks((n_a, n_m, L), False))
            ntd.attrs["clipped_nonnegative"] = True
            stt = _ds(g, "status", pd_["status"], "enum",
                      "0=ok(MC) 1=exception 2=mu<0.2 3=mu>mu_cap 4=mu==0",
                      cube=True, chunks=ch4)
            per_mode = [ext, muu, stt]
            if pd_["has_f1"]:
                f_hi = pd_["f_dm_values"][1]
                ext1 = _ds(g, "extremeness_f1",
                           pd_["p_f1"].astype(np.float32), "1",
                           f"optimum-interval extremeness / confidence "
                           f"({grp_name} pass) for f_DM = {f_hi:g}; NaN where "
                           "status_f1==1", cube=True, chunks=ch4)
                muu1 = _ds(g, "mu_f1", pd_["mu_f1"].astype(np.float32), "counts",
                           f"expected signal counts mu ({grp_name}) for f_DM = "
                           f"{f_hi:g}; exactly {f_hi / pd_['f_dm_values'][0]:g}x "
                           "mu (f_DM is a pure flux normalisation)",
                           cube=True, chunks=ch4)
                stt1 = _ds(g, "status_f1", pd_["status_f1"], "enum",
                           f"status of the f_DM = {f_hi:g} surface: 0=ok(MC) "
                           "1=exception 2=mu<0.2 3=mu>mu_cap 4=mu==0",
                           cube=True, chunks=ch4)
                per_mode += [ext1, muu1, stt1]
            for dset in per_mode:
                _attach_scales(dset, (d_mode, d_alpha, mass_scale, d_lam),
                               ("mode", "alpha_n", "mass_gev", "lambda_m"),
                               "mode,alpha_n,mass_gev,lambda_m")
            _attach_scales(ntd, (d_alpha, mass_scale, d_lam),
                           ("alpha_n", "mass_gev", "lambda_m"),
                           "alpha_n,mass_gev,lambda_m")

        # ── /halo ──
        gh = h5.create_group("halo")
        hshape = (halo_d["alphas_n"].size, halo_d["ms"].size, L)
        hch = _mode_plane_chunks(hshape, has_mode=False)
        hnt = _ds(gh, "n_transit", halo_d["n_transit"].astype(np.float32),
                  "counts", "unattenuated-halo expected transits",
                  cube=True, chunks=hch)
        hbm = _ds(gh, "bmax", halo_d["bmax"].astype(np.float32), "m",
                  "flux-averaged threshold reach sqrt(<pi b^2>/pi)",
                  cube=True, chunks=hch)
        for dd in (hnt, hbm):
            _attach_scales(dd, (d_ahalo, d_mhalo, d_lam),
                           ("alpha_halo_n", "mass_halo_gev", "lambda_m"),
                           "alpha_halo_n,mass_halo_gev,lambda_m")

        # ── /detector ──
        gd = h5.create_group("detector")
        exps = gd.create_dataset("exposure_s", data=float(atm["t_total"]))
        exps.attrs["units"] = "s"
        exps.attrs["description"] = "total live-time exposure (config.T_EXPOSURE)"
        for n in MODES:
            ev = detector["events"][n]
            if ev is None:
                ev = np.empty(0, dtype=np.float64)
            _ds(gd, f"events_mode{n}", ev, "GeV",
                f"observed impulse candidates, mode {n} (from data_mode{n}.txt)")
            blip = detector["blips"][n]
            if blip is None:
                dblip = _ds(gd, f"all_blips_mode{n}",
                            np.empty(0, dtype=np.float64), "eV",
                            f"all q>100 blip momenta, mode {n} (raw source absent)")
                dblip.attrs["missing"] = True
            else:
                _ds(gd, f"all_blips_mode{n}", blip, "eV",
                    f"all q>100 blip momenta, mode {n}")
        for key, arr in detector["efficiency"].items():
            unit = "GeV" if key.startswith("q_gev") else "1"
            _ds(gd, key, arr, unit,
                f"efficiency table array '{key}' (verbatim from "
                "efficiency_curves.npz)")

        # ── /reference_curves ──
        gr = h5.create_group("reference_curves")
        _ds(gr, "v", ref["v"], "c",
            "arrival-speed grid v/c for the showcase point (m=1e8, alpha_n=1e-3)")
        d_shm = _ds(gr, "fv_shm", ref["fv"]["shm"], "(v/c)^-1",
                    "unattenuated SHM arrival-speed distribution")
        d_shm.attrs["survival_fraction"] = float(ref["survival"]["shm"])
        for tag in TAGS:
            dfv = _ds(gr, f"fv_{tag}", ref["fv"][tag], "(v/c)^-1",
                      f"attenuated arrival-speed distribution, lambda={tag}")
            dfv.attrs["survival_fraction"] = float(ref["survival"][tag])
        _ds(gr, "q_gev", ref["q_gev"], "GeV",
            "momentum-kick grid for the raw dR/dq spectra")
        _ds(gr, "drdq_massless", ref["drdq"]["massless"], "s^-1 GeV^-1",
            "raw dR/dq (massless), arrival f(v) fixed at 200um")
        for tag in TAGS:
            _ds(gr, f"drdq_{tag}", ref["drdq"][tag], "s^-1 GeV^-1",
                f"raw dR/dq (lambda={tag}), arrival f(v) fixed at 200um")

        # ── root attrs ──
        a = h5.attrs
        a["file_format"] = FILE_FORMAT
        a["version"] = FORMAT_VERSION_AXES if axis_layout else FORMAT_VERSION
        a["layout"] = "axes" if axis_layout else "groups"
        a["version_tag"] = version_tag
        a["schema_version"] = SCHEMA_VERSION
        a["created"] = _iso_now()
        a["git_commit"] = commit
        a["git_dirty"] = bool(dirty) if dirty is not None else False
        a["git_dirty_files_json"] = json.dumps(dirty_files)
        a["seed"] = SEED
        a["q_thresh_gev"] = float(config.Q_THRESH)
        a["r_eff_m"] = float(config.R_EFF)
        a["f_x"] = float(config.F_X)
        # dual f_DM: the unsuffixed cubes are f_DM = f_x (0.1); the *_f1 cubes
        # (when present) are the same cells at f_dm_values[1]. f_DM is a pure
        # flux normalisation: same ODE, same dR/dq shape, mu scales exactly.
        fdm = f_dm_values
        a["f_dm_values"] = np.asarray(fdm, dtype=np.float64)
        a["f_dm_default"] = float(fdm[0])
        if not axis_layout:
            a["f_dm_suffix_json"] = json.dumps(
                {f"{v:g}": ("" if i == 0 else "_f1") for i, v in enumerate(fdm)})
        a["rho_dm_gev4"] = float(config.RHO_DM)
        a["n_neutrons"] = float(config.N_NEUTRONS)
        a["t_exposure_s"] = float(atm["t_total"])
        a["m_planck_gev"] = float(M_PLANCK)
        a["confidence_recommended"] = float(CONFIDENCE)
        a["q_hi_ref_gev"] = float(Q_HI_REF)
        # impact-parameter cap: outer limit of the b-integral is
        # min(b_constrained_max, b_max(q)) in both dsigma/dq and the transit
        # reach. NaN = uncapped (the b-integral runs to the full b_max(q)).
        cap = atm["b_constrained_max"]
        a["b_constrained_max_m"] = float("nan") if cap is None else float(cap)
        # Post-facto mass cut. With the cross section uncapped, the right edge
        # of the excluded region is set by the halo flux through a b_cap
        # aperture rather than by an in-integral cap; see mass_cut_flux. One
        # value per f_DM actually written to this file.
        if m_cut is not None:
            for f_val in np.asarray(f_dm_values, dtype=np.float64)[f_sel]:
                mc, deriv = mass_cut_flux(f_val, b_cap=m_cut["b_cap"],
                                          n_req=m_cut["n_req"],
                                          t_total=atm["t_total"])
                key = f"m_cut_{m_cut['label']}_f{f_val:g}_gev"
                a[key] = float(mc)
                a[key + "_derivation"] = deriv
            a["m_cut_b_cap_m"] = float(m_cut["b_cap"])
            a["m_cut_n_transits_required"] = float(m_cut["n_req"])
            a["m_cut_applied_to_stored_surfaces"] = False
        a["fid_n_ode"] = int(fid.get("n_ode", -1))
        a["fid_n_shm"] = int(fid.get("n_shm", -1))
        a["fid_n_q"] = int(fid.get("n_q", -1))
        a["fid_q_span"] = float(fid.get("q_span", float("nan")))
        a["fid_n_mc"] = int(fid.get("n_mc", -1))
        # optimum-interval expected-count cap: cells above it are asserted
        # excluded without Monte Carlo. Shards built before the cap was
        # recorded carry no "mu_cap" key -> NaN, meaning "not recorded".
        a["fid_mu_cap"] = float(fid.get("mu_cap", float("nan")))
        a["fid_json"] = json.dumps(fid)
        a["df"] = int(atm["df"])
        a["reference_curves_fidelity"] = ("quick" if quick_reference
                                          else "production")
        # input provenance on the file itself, so a cube is self-describing even
        # when separated from provenance.json (env-overridable inputs included).
        a["efficiency_npz"] = inputs["efficiency_npz"]
        a["efficiency_npz_sha256"] = inputs["efficiency_npz_sha256"] or ""
        a["events_dir"] = inputs["events_dir"]
        for n in MODES:
            ev = inputs["events"].get(f"data_mode{n}.txt", {})
            a[f"events_mode{n}_sha256"] = ev.get("sha256") or ""
        a["inputs_json"] = json.dumps(inputs, sort_keys=True)
        for k, vv in pkgs.items():
            a[f"pkg_{k}"] = vv
        a["packages_json"] = json.dumps(pkgs)

    os.replace(tmp, out_path)
    return out_path, commit, dirty, dirty_files, pkgs


# ── provenance.json + SHA256SUMS ─────────────────────────────────────────────
def _read_run_config(shard_dir):
    f = Path(shard_dir) / "run_config.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except Exception as err:  # noqa: BLE001
        print(f"WARNING: could not parse {f}: {err}")
        return None


def write_provenance(release_dir, atm_dir, noatm_dir, halo_dir, atm, noatm,
                     halo_d, commit, dirty, dirty_files, pkgs, out_path,
                     version_tag, inputs, name=None):
    # No hostname: provenance ships with the release and carries no host
    # identifiers (platform/package versions cover reproducibility).
    prov = {
        "assembly": {
            "created": _iso_now(),
            "platform": platform.platform(),
            "argv": " ".join(sys.argv),
            "git_commit": commit,
            "git_dirty": bool(dirty) if dirty is not None else None,
            "git_dirty_files": dirty_files,
            "packages": pkgs,
            "version_tag": version_tag,
            "output": str(out_path),
            "fidelity": atm["fidelity"],
            "n_finite_lambda": int(atm["n_finite"]),
            "seed": SEED,
            # t_exposure_s comes off the shards; config.T_EXPOSURE is what THIS
            # process resolved (LUHDM_T_EXPOSURE-aware) and must agree.
            "t_exposure_s": float(atm["t_total"]),
            "f_dm_values": atm["f_dm_values"],
            "inputs": inputs,
        },
        "shard_dirs": {
            "atm": str(atm_dir), "noatm": str(noatm_dir), "halo": str(halo_dir),
        },
        "shard_inputs": {
            "atm": atm["inputs"], "noatm": noatm["inputs"],
            "halo": halo_d.get("inputs"),
        },
        "impact_parameter_cap": {
            "b_constrained_max_m": atm["b_constrained_max"],
            # shards reused from an uncapped run over the lambda range where the
            # cap provably cannot bite (byte-identity gated on the boundary shard)
            "shards_without_cap_flag": {
                "atm": atm["cap_unflagged_shards"],
                "noatm": noatm["cap_unflagged_shards"],
                "halo": halo_d["cap_unflagged_shards"],
            },
        },
        "run_config": {
            "atm": _read_run_config(atm_dir),
            "noatm": _read_run_config(noatm_dir),
            "halo": _read_run_config(halo_dir),
        },
    }
    # Shard-level provenance (inputs_json, run_config.json) was written on the
    # compute node with that node's paths; the tree scrub cleans those too.
    prov = scrub_home(prov)
    f = Path(release_dir) / (name or "provenance.json")
    f.write_text(json.dumps(prov, indent=2, default=str))
    print(f"wrote {f}")
    return f


def write_sha256(release_dir, out_path):
    """Record this file's sha256, KEEPING any other files already listed.

    A release can be more than one HDF5 (the v7 split writes one file per
    hypothesis), so this merges by filename instead of truncating: the entry
    for this output is replaced, every other entry is preserved.
    """
    h = hashlib.sha256()
    with open(out_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    name = Path(out_path).name
    f = Path(release_dir) / "SHA256SUMS"
    entries = {}
    if f.exists():
        for ln in f.read_text().splitlines():
            parts = ln.split(None, 1)
            if len(parts) == 2:
                entries[parts[1].strip()] = parts[0].strip()
    entries[name] = h.hexdigest()
    f.write_text("".join(f"{entries[k]}  {k}\n" for k in sorted(entries)))
    print(f"wrote {f}: {h.hexdigest()}  {name}")
    return f


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--atm-dir", required=True)
    ap.add_argument("--noatm-dir", required=True)
    ap.add_argument("--halo-dir", required=True)
    ap.add_argument("--out", default=str(REPO / "release" /
                                         "luhdm_datarelease_v2.h5"))
    ap.add_argument("--version-tag", default="v1.0")
    ap.add_argument("--layout", choices=("axes", "groups"), default="axes",
                    help="'axes' (default, the release layout): one /results "
                         "cube per quantity over (f_dm, atmosphere, mode, "
                         "alpha, mass, lambda) — needs both f_DM surfaces and "
                         "one shared mass axis. 'groups': the v3 layout "
                         "(/atm + /noatm, _f1 datasets alongside).")
    ap.add_argument("--data-dir", default=None,
                    help="dir holding data_mode{1,2,3}.txt for /detector "
                         "(default <repo>/notebooks); must match the "
                         "--data-dir the shards were built with")
    ap.add_argument("--efficiency-npz", default=None,
                    help="efficiency table copied into /detector (default: the "
                         "one luhdm.efficiency resolves, i.e. "
                         "LUHDM_EFFICIENCY_NPZ or the committed table)")
    ap.add_argument("--quick-reference", action="store_true",
                    help="reference curves at quick fidelity (n_grid=80, "
                         "n_shm=1e5) for smoke runs; default is production "
                         "(n_grid=400, n_shm=3e5)")
    ap.add_argument("--select", default="both",
                    choices=("both", "f1-atm", "f-base-noatm"),
                    help="which (f_DM, atmosphere) hypothesis to WRITE: "
                         "'both' (default, the full cube), 'f1-atm' "
                         "(f_DM=1 with atmosphere), or 'f-base-noatm' "
                         "(f_DM=f_x with no atmosphere). Subsets only the "
                         "f_dm and atmosphere axes; cross-checks still run "
                         "over every loaded shard.")
    ap.add_argument("--m-cut-b-cap", type=float, default=M_CUT_B_CAP_M,
                    help="aperture radius [m] of the post-facto flux mass cut "
                         "(default %(default)g); stored as an attribute, "
                         "never applied to the stored surfaces")
    ap.add_argument("--m-cut-n-transits", type=float,
                    default=M_CUT_N_TRANSITS,
                    help="transits required within the aperture for the mass "
                         "cut (default %(default)g)")
    ap.add_argument("--no-m-cut", action="store_true",
                    help="omit the post-facto mass-cut attributes")
    args = ap.parse_args()

    print("=" * 72)
    print("luhdm data-release assembly")
    print("=" * 72)

    print("\nloading atm shards ...")
    atm = load_pass(args.atm_dir, "atm")
    print(f"  atm: {atm['n_finite']} finite lambda + massless, "
          f"mass {atm['ms'].size}, alpha {atm['alphas_n'].size}")
    print("loading noatm shards ...")
    noatm = load_pass(args.noatm_dir, "noatm")
    print(f"  noatm: {noatm['n_finite']} finite lambda + massless, "
          f"mass {noatm['ms'].size}, alpha {noatm['alphas_n'].size}")
    print("loading halo shards ...")
    halo_d = load_halo(args.halo_dir)
    print(f"  halo: {halo_d['n_finite']} finite lambda + massless, "
          f"grid {halo_d['alphas_n'].size}x{halo_d['ms'].size}")

    lambda_finite = cross_check_lambda(atm, noatm, halo_d)
    print(f"\nfinite-lambda axis agrees across passes: {lambda_finite.size} values"
          f"  [{lambda_finite.min():.3e} .. {lambda_finite.max():.3e}] m")
    cross_check_cap(atm, noatm, halo_d)

    print_status_report("atm", atm)
    print_status_report("noatm", noatm)

    ref = compute_reference_curves(quick=args.quick_reference)

    print("\nloading detector products ...")
    ev_dir = events_dir(args.data_dir)
    events = load_events(args.data_dir)
    for n in MODES:
        if events[n] is None:
            print(f"WARNING: {ev_dir}/data_mode{n}.txt missing; "
                  f"using atm-shard events for mode {n}")
            events[n] = atm["events"][n]
        else:
            if not np.allclose(events[n], atm["events"][n],
                               rtol=1e-9, atol=0):
                print(f"WARNING: {ev_dir}/data_mode{n}.txt events differ from "
                      f"atm-shard events (mode {n}); using data_mode{n}.txt")
    efficiency_table = load_efficiency_table(args.efficiency_npz)
    inputs = assembly_inputs(args.data_dir, args.efficiency_npz)
    print(f"  events dir: {inputs['events_dir']}")
    print(f"  efficiency: {inputs['efficiency_npz']}")
    print(f"  t_exposure_s: {inputs['t_exposure_s']:.0f} "
          f"(shards: {atm['t_total']:.0f})")
    if abs(inputs["t_exposure_s"] - float(atm["t_total"])) > 1e-6:
        print("WARNING: config.T_EXPOSURE differs from the shards' t_total; "
              "the release uses the shards' value (set LUHDM_T_EXPOSURE to "
              "match when assembling a veto-variant cube)")
    blips = extract_blips()
    detector = {"events": events, "efficiency": efficiency_table, "blips": blips}

    m_cut = None if args.no_m_cut else dict(
        b_cap=args.m_cut_b_cap, n_req=args.m_cut_n_transits,
        label=f"{args.m_cut_b_cap * 100:g}cm")
    if m_cut is not None:
        for f_val in (atm["f_dm_values"] or [float(config.F_X)]):
            mc, _ = mass_cut_flux(f_val, b_cap=m_cut["b_cap"],
                                  n_req=m_cut["n_req"], t_total=atm["t_total"])
            print(f"  post-facto mass cut  f_DM={f_val:<5g} "
                  f"b_cap={m_cut['b_cap']:g} m  N>={m_cut['n_req']:g}  "
                  f"=>  m_cut = {mc:.4e} GeV")

    print(f"\nwriting HDF5 (layout={args.layout}, select={args.select}) ...")
    out_path, commit, dirty, dirty_files, pkgs = write_h5(
        args.out, atm, noatm, halo_d, lambda_finite, ref, detector,
        args.version_tag, args.quick_reference, inputs, args.layout,
        select=args.select, m_cut=m_cut)
    print(f"wrote {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")

    release_dir = out_path.parent
    # One provenance file per HDF5 when the release is split by hypothesis;
    # the unsplit default keeps the historical provenance.json name.
    prov_name = ("provenance.json" if args.select == "both"
                 else f"provenance_{out_path.stem}.json")
    write_provenance(release_dir, args.atm_dir, args.noatm_dir, args.halo_dir,
                     atm, noatm, halo_d, commit, dirty, dirty_files, pkgs,
                     out_path, args.version_tag, inputs, name=prov_name)
    write_sha256(release_dir, out_path)

    print("\n" + "=" * 72)
    print(f"ASSEMBLY COMPLETE: {out_path}")
    print(f"  git_commit={commit}{' (DIRTY)' if dirty else ''}")
    print("=" * 72)


if __name__ == "__main__":
    main()
