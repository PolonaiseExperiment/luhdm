#!/usr/bin/env python3
"""Verify the luhdm data-release shards (and optionally the assembled HDF5).

The float64 shard dirs are the source of truth. This runs four checks:

  V1  tag parity vs the committed scan7 caches (the +4 h launch-abort gate;
      works on PARTIAL shard dirs — verifies whatever tag shards exist).
  V2  single-cell bit-exact recompute via the OLD scan_grid.scan_point path
      with a fresh table (the PerMuTable determinism guarantee).
  V3  physics-consistency warnings (mu_atm <= mu_noatm, n_transit <= halo nt,
      p monotone-ish in alpha). Never fails.
  V4  status census + list of every status-1 cell.

    python scripts/verify_release.py \
        --shard-dir-atm   ~/release_shards/atm \
        --shard-dir-noatm ~/release_shards/noatm

Exit code is 0 only if every HARD gate (V1 real-mismatch / MC-band, V2) passes.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from luhdm import config, cross_section, efficiency, rate  # noqa: E402

# the shipped projected-dsigma/dq kernel; pre-kernel shards carry no key and
# were built with exactly this convention
KERNEL_DEFAULT = cross_section.KERNEL_DEFAULT

SEED = 20260702
Q_HI_REF = 8.4e3
MODES = (1, 2, 3)
TAG_LAMBDA = {
    "2m": 2.0, "20cm": 0.2, "2cm": 2e-2, "2mm": 2e-3, "200um": 2e-4,
    "20um": 2e-5, "10um": 1e-5, "2um": 2e-6,
}
DP_GATE = 0.05           # MC-band |Delta p| tolerance vs committed cache
DRIFT_RTOL = 1e-12       # environment-drift band for "bit-equal" quantities
STATUS_NAMES = {0: "ok(MC)", 1: "exception", 2: "mu<0.2", 3: "mu>mu_cap",
                4: "mu==0"}


# ── shard reading (no completeness requirement; partial dirs OK) ──────────────
def _scalar(v):
    a = np.asarray(v)
    return a.item() if a.ndim == 0 else a


def _opt_float(rec, key):
    """Read an optional float shard field that may be absent / stored as None.

    ``np.savez(b_constrained_max=None)`` round-trips as a 0-d object array, so
    ``_scalar`` yields None; a real cap comes back as a float. Missing key (a
    pre-cap shard) is also None = uncapped.
    """
    if key not in rec:
        return None
    val = _scalar(rec[key])
    if val is None:
        return None
    val = float(val)
    return None if np.isnan(val) else val


def _opt_str(rec, key):
    if key not in rec:
        return None
    val = _scalar(rec[key])
    return None if val is None else str(val)


def read_shards(shard_dir):
    """Read every atm/noatm shard in a dir into a list of dicts (partial OK)."""
    shard_dir = Path(shard_dir)
    if not shard_dir.is_dir():
        return []
    out = []
    for f in sorted(shard_dir.glob("shard_*.npz")):
        if f.name.startswith("shard_halo"):
            continue
        with np.load(f, allow_pickle=True) as d:
            rec = {k: d[k] for k in d.files}
        rec = {
            "file": f.name,
            "p": np.asarray(rec["p"], np.float64),
            "mu": np.asarray(rec["mu"], np.float64),
            "n_transit": np.asarray(rec["n_transit"], np.float64),
            "status": np.asarray(rec["status"], np.uint8),
            "ms": np.asarray(rec["ms"], np.float64),
            "alphas_n": np.asarray(rec["alphas_n"], np.float64),
            "lamb": float(_scalar(rec["lamb"])),
            "massless": bool(_scalar(rec["massless"])),
            "il": int(_scalar(rec["il"])),
            "q_min": float(_scalar(rec["q_min"])),
            "df": int(_scalar(rec["df"])) if "df" in rec else 3,
            "lamb_ode": float(_scalar(rec.get("lamb_ode", np.nan))),
            # the impact-parameter cap this shard was built with; V2 must
            # recompute with the SAME cap or nothing on a capped cube can match
            "b_constrained_max": _opt_float(rec, "b_constrained_max"),
            # projected-dsigma/dq kernel convention, same on every shard of a
            # campaign; absent on pre-kernel shards -> the shipped default
            "projection_kernel": _opt_str(rec, "projection_kernel"),
            "fidelity": str(_scalar(rec["fidelity"])),
            "events": {n: np.asarray(rec[f"events_mode{n}"], np.float64)
                       for n in MODES},
        }
        out.append(rec)
    return out


def read_halo_shards(shard_dir):
    if not shard_dir:
        return []
    shard_dir = Path(shard_dir)
    if not shard_dir.is_dir():
        return []
    out = []
    for f in sorted(shard_dir.glob("shard_halo_*.npz")):
        with np.load(f, allow_pickle=True) as d:
            rec = {k: d[k] for k in d.files}
        out.append({
            "file": f.name,
            "nt": np.asarray(rec["nt"], np.float64),
            "bmax": np.asarray(rec["bmax_m"], np.float64),
            "ms": np.asarray(rec["ms"], np.float64),
            "alphas_n": np.asarray(rec["alphas_n"], np.float64),
            "lamb": float(_scalar(rec["lamb"])),
            "massless": bool(_scalar(rec["massless"])),
        })
    return out


def match_tag(shard):
    """Return the tag name this shard corresponds to, or None (off-tag lambda)."""
    if shard["massless"]:
        return "massless"
    for tag, lam in TAG_LAMBDA.items():
        if np.isclose(shard["lamb"], lam, rtol=1e-9, atol=0):
            return tag
    return None


def _parse_fid(fid_str):
    import ast
    try:
        return dict(ast.literal_eval(fid_str))
    except Exception:  # noqa: BLE001
        return {}


# ── V1: tag parity vs committed scan7 caches ─────────────────────────────────
def _mass_slice(shard_nm, cache_nm):
    """How to index the shard mass axis onto the cache grid."""
    if shard_nm == cache_nm:
        return slice(None)
    if shard_nm == 2 * cache_nm - 1:      # atm 119-tier: ms119[::2] == ms60
        return slice(None, None, 2)
    return None


def _compare_bitequal(cube, cache, mask):
    """(exact, drift, max_rel) for the masked elements."""
    a, b = cube[mask], cache[mask]
    if a.size == 0:
        return True, False, 0.0
    exact = np.array_equal(a, b)
    if exact:
        return True, False, 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = np.where(np.abs(b) > 0, np.abs(b), 1.0)
        rel = np.abs(a - b) / denom
    max_rel = float(np.nanmax(rel)) if rel.size else 0.0
    drift = bool(np.allclose(a, b, rtol=DRIFT_RTOL, atol=0, equal_nan=True))
    return False, drift, max_rel


def v1_tag_parity(atm_shards, noatm_shards):
    print("\n" + "=" * 72)
    print("V1  tag parity vs committed scan7 caches")
    print("=" * 72)
    hard_fail = False
    n_compared = 0
    n_missing_cache = 0

    for pass_name, shards, cache_root in (
            ("atm", atm_shards, REPO / "notebooks" / "computation_cache"),
            ("noatm", noatm_shards, REPO / "notebooks" / "computation_cache_noatm")):
        for shard in shards:
            tag = match_tag(shard)
            if tag is None:
                continue     # off-tag lambda: nothing to compare
            for mode in MODES:
                cache_f = cache_root / f"mode{mode}" / f"scan7_{tag}.npz"
                if not cache_f.exists():
                    n_missing_cache += 1
                    continue
                with np.load(cache_f) as c:
                    cache = {k: c[k] for k in c.files}
                cache_ms = np.asarray(cache["ms"], np.float64)
                sl = _mass_slice(shard["ms"].size, cache_ms.size)
                if sl is None:
                    print(f"  [{pass_name}:{tag}:mode{mode}] SHAPE MISMATCH "
                          f"shard n_m={shard['ms'].size} cache n_m={cache_ms.size}"
                          " -> skip")
                    continue
                n_compared += 1
                mi = mode - 1

                # axes bit-equal
                if not np.array_equal(shard["ms"][sl], cache_ms):
                    print(f"  [{pass_name}:{tag}:mode{mode}] FAIL mass axis "
                          "not bit-equal")
                    hard_fail = True
                if not np.array_equal(shard["alphas_n"],
                                      np.asarray(cache["alphas_n"], np.float64)):
                    print(f"  [{pass_name}:{tag}:mode{mode}] FAIL alpha axis "
                          "not bit-equal")
                    hard_fail = True

                p_cube = shard["p"][mi][:, sl]
                mu_cube = shard["mu"][mi][:, sl]
                nt_cube = shard["n_transit"][:, sl]
                st_cube = shard["status"][mi][:, sl]
                p_cache = np.asarray(cache["extremeness"], np.float64)
                mu_cache = np.asarray(cache["counts"], np.float64)
                nt_cache = np.asarray(cache["n_transit"], np.float64)

                # mu / n_transit bit-equal where the cube is finite
                finite_mu = np.isfinite(mu_cube)
                ex, dr, mr = _compare_bitequal(mu_cube, mu_cache, finite_mu)
                if not ex:
                    kind = "environment drift" if dr else "REAL MISMATCH"
                    print(f"  [{pass_name}:{tag}:mode{mode}] mu not bit-equal "
                          f"({kind}, max_rel={mr:.2e})")
                    if not dr:
                        hard_fail = True

                finite_nt = np.isfinite(nt_cube)
                ex, dr, mr = _compare_bitequal(nt_cube, nt_cache, finite_nt)
                if not ex:
                    kind = "environment drift" if dr else "REAL MISMATCH"
                    print(f"  [{pass_name}:{tag}:mode{mode}] n_transit not "
                          f"bit-equal ({kind}, max_rel={mr:.2e})")
                    if not dr:
                        hard_fail = True

                # p bit-equal where status in {2,3,4} (shortcut / no-support)
                short = np.isin(st_cube, (2, 3, 4))
                ex, _, mr = _compare_bitequal(p_cube, p_cache, short)
                if not ex:
                    print(f"  [{pass_name}:{tag}:mode{mode}] p (shortcut cells) "
                          f"not bit-equal (max_rel={mr:.2e}; cache failures?)")

                # status-1 (cube NaN) vs cache (0,0,0) — report, not a failure
                st1 = st_cube == 1
                n_st1 = int(np.count_nonzero(st1))
                if n_st1:
                    cache0 = int(np.count_nonzero(p_cache[st1] == 0.0))
                    print(f"  [{pass_name}:{tag}:mode{mode}] {n_st1} status-1 "
                          f"cube cells (NaN); {cache0} are 0 in cache")

                # MC band over status-0 cells. The committed caches' p values
                # in the MC band are scheduling-dependent (the old scripts
                # shared one RNG stream across mu values per worker), so
                # single-cell noise up to ~0.06 is expected; gate on the
                # distribution, not the worst cell: mean <= 0.01, outlier
                # fraction (|dp| > DP_GATE) <= 1%, and no cell beyond 2*gate.
                band = st_cube == 0
                if np.any(band):
                    dp = np.abs(p_cube[band] - p_cache[band])
                    mean_dp, max_dp = float(np.mean(dp)), float(np.max(dp))
                    frac_out = float(np.mean(dp > DP_GATE))
                    # A large single-cell |dp| only matters if it moves the
                    # exclusion boundary: flag cells whose p>=0.95 status flips
                    # with a non-trivial |dp|.
                    flips = ((p_cube[band] >= 0.95) != (p_cache[band] >= 0.95)) \
                        & (dp > DP_GATE)
                    n_flip = int(np.count_nonzero(flips))
                    ok = mean_dp <= 0.01 and frac_out <= 0.01 and n_flip == 0
                    tag_s = "ok" if ok else "FAIL"
                    print(f"  [{pass_name}:{tag}:mode{mode}] MC-band |dp| "
                          f"mean={mean_dp:.4f} max={max_dp:.4f} "
                          f"frac>{DP_GATE}={frac_out:.4%} "
                          f"contour-flips={n_flip} "
                          f"(gates: mean<=0.01, frac<=1%, flips==0) "
                          f"-> {tag_s}")
                    if not ok:
                        hard_fail = True

    if n_compared == 0:
        print("  no matching tags in these shard dirs / nothing to compare")
    else:
        print(f"\n  compared {n_compared} (pass,tag,mode) slices; "
              f"{n_missing_cache} cache files absent")
    print(f"V1 verdict: {'FAIL' if hard_fail else 'PASS'}")
    return not hard_fail


# ── V2: single-cell bit-exact recompute via scan_grid.scan_point ─────────────
def _stratified_cells(shards, n_spot):
    """Pick cells stratified over {MC-band, shortcut, off-tag, massless}."""
    rng = np.random.default_rng(SEED)
    strata = {"mc": [], "shortcut": [], "offtag": [], "massless": []}
    for si, shard in enumerate(shards):
        tag = match_tag(shard)
        n_a, n_m = shard["alphas_n"].size, shard["ms"].size
        for mi in range(3):
            st = shard["status"][mi]
            for ia in range(n_a):
                for im in range(n_m):
                    s = int(st[ia, im])
                    if s == 1:
                        continue          # exception cells excluded from V2
                    cell = (si, mi, ia, im)
                    if shard["massless"]:
                        strata["massless"].append(cell)
                    elif tag is None:
                        strata["offtag"].append(cell)
                    if s == 0:
                        strata["mc"].append(cell)
                    elif s in (2, 3):
                        strata["shortcut"].append(cell)
    picks = []
    per = max(1, n_spot // len(strata))
    for key, cells in strata.items():
        if not cells:
            continue
        idx = rng.choice(len(cells), size=min(per, len(cells)), replace=False)
        picks.extend(cells[i] for i in idx)
    # de-dup and cap
    seen, uniq = set(), []
    for c in picks:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq[:n_spot]


def _recompute_cell(scan_grid, shard, pass_name, mode, ia, im, caches):
    """Fresh-table single-cell recompute; returns (p, mu, n_t)."""
    m = float(shard["ms"][im])
    alpha = float(shard["alphas_n"][ia])
    df = shard["df"]
    massless = shard["massless"]
    lamb = shard["lamb_ode"] if massless else shard["lamb"]

    fid = _parse_fid(shard["fidelity"])
    # The optimum-interval mu>cap shortcut is part of the cell's identity: a
    # recompute at a different cap flips p between 1.0 and its MC value. Shards
    # built before the cap was recorded carry no "mu_cap" key -> the historical
    # default, which is exactly what those shards were built with.
    fid.setdefault("mu_cap", scan_grid.MU_CAP_DEFAULT)
    b_cap = shard.get("b_constrained_max")
    # the impact-parameter cap and the projection-kernel convention are both
    # part of the cross-section identity (the kernel is baked into the
    # finite-lambda interpolant), so both belong in the cache key
    kernel = shard.get("projection_kernel") or KERNEL_DEFAULT
    key_xs = ("massless" if massless else round(shard["lamb"], 18), b_cap,
              kernel)
    if key_xs not in caches["xs"]:
        caches["xs"][key_xs] = rate.make_xsec(
            None if massless else shard["lamb"], b_constrained_max=b_cap,
            projection_kernel=kernel)
    # V_I_SAMPLES built once per n_shm (same seed as the campaign)
    vkey = fid["n_shm"]
    if vkey not in caches["visamp"]:
        from luhdm import atmosphere
        caches["visamp"][vkey] = atmosphere.sample_shm(
            int(fid["n_shm"]), rng=np.random.default_rng(SEED))

    scan_grid.LAMB = lamb
    scan_grid.XS = caches["xs"][key_xs]
    scan_grid.V_I_SAMPLES = caches["visamp"][vkey]
    scan_grid.FID = fid
    scan_grid.EVENTS = shard["events"][mode]
    scan_grid.Q_MIN = shard["q_min"]
    scan_grid.EFF = efficiency.make_efficiency(mode, df)
    scan_grid.NO_ATM = (pass_name == "noatm")
    scan_grid._worker_state.clear()   # fresh PerMuTable-equivalent table

    _, _, p, mu, n_t = scan_grid.scan_point((ia, im, alpha, m))
    return p, mu, n_t


def _feq(a, b):
    return (a == b) or (np.isnan(a) and np.isnan(b))


def v2_spot_recompute(atm_shards, noatm_shards, n_spot):
    print("\n" + "=" * 72)
    print(f"V2  single-cell bit-exact recompute (n_spot={n_spot})")
    print("=" * 72)
    try:
        spec = importlib.util.spec_from_file_location(
            "scan_grid_verify", REPO / "scripts" / "scan_grid.py")
        scan_grid = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scan_grid)
    except Exception as err:  # noqa: BLE001
        print(f"  could not import scan_grid.py: {err}")
        print("V2 verdict: FAIL")
        return False

    caps = sorted({s.get("b_constrained_max") for s in atm_shards + noatm_shards},
                  key=lambda c: (c is not None, c))
    print(f"  recomputing with the shards' own impact-parameter caps: {caps}")
    kernels = sorted({s.get("projection_kernel") or KERNEL_DEFAULT
                      for s in atm_shards + noatm_shards})
    print(f"  recomputing with the shards' own projection kernels: {kernels}")
    mu_caps = sorted({_parse_fid(s["fidelity"]).get("mu_cap", scan_grid.MU_CAP_DEFAULT)
                      for s in atm_shards + noatm_shards})
    print(f"  recomputing with the shards' own optimum-interval mu caps: {mu_caps}")

    caches = {"xs": {}, "visamp": {}}
    hard_fail = False
    n_checked = 0
    for pass_name, shards in (("atm", atm_shards), ("noatm", noatm_shards)):
        cells = _stratified_cells(shards, n_spot)
        for (si, mi, ia, im) in cells:
            shard = shards[si]
            mode = mi + 1
            p_cube = shard["p"][mi][ia, im]
            mu_cube = shard["mu"][mi][ia, im]
            nt_cube = shard["n_transit"][ia, im]
            try:
                p, mu, n_t = _recompute_cell(
                    scan_grid, shard, pass_name, mode, ia, im, caches)
            except Exception as err:  # noqa: BLE001
                print(f"  [{pass_name}:{shard['file']}:mode{mode}:"
                      f"a{ia},m{im}] recompute raised: {err}")
                hard_fail = True
                continue
            n_checked += 1
            ok = _feq(p, p_cube) and _feq(mu, mu_cube) and _feq(n_t, nt_cube)
            if not ok:
                print(f"  [{pass_name}:{shard['file']}:mode{mode}:"
                      f"a{ia},m{im}] MISMATCH "
                      f"p {p!r} vs {p_cube!r}; mu {mu!r} vs {mu_cube!r}; "
                      f"nt {n_t!r} vs {nt_cube!r}")
                hard_fail = True
    if n_checked == 0:
        print("  no cells available to recompute")
    else:
        print(f"\n  recomputed {n_checked} cells bit-exact")
    print(f"V2 verdict: {'FAIL' if hard_fail else 'PASS'}")
    return not hard_fail


# ── V3: physics-consistency warnings (never fails) ───────────────────────────
def _index_by_lambda(shards):
    out = {}
    for s in shards:
        key = "massless" if s["massless"] else round(s["lamb"], 15)
        out[key] = s
    return out


def v3_physics(atm_shards, noatm_shards, halo_shards):
    print("\n" + "=" * 72)
    print("V3  physics-consistency warnings (never fails)")
    print("=" * 72)
    atm_by = _index_by_lambda(atm_shards)
    noatm_by = _index_by_lambda(noatm_shards)

    # mu_atm <= mu_noatm at shared cells (interpolate noatm mu onto atm masses)
    tot_viol = 0
    for key in sorted(atm_by.keys(), key=str):
        if key not in noatm_by:
            continue
        a, no = atm_by[key], noatm_by[key]
        la, lno = np.log10(a["ms"]), np.log10(no["ms"])
        for mi in range(3):
            for ia in range(a["alphas_n"].size):
                mu_a = a["mu"][mi, ia]
                mu_no = no["mu"][mi, ia]
                fin = np.isfinite(mu_a)
                if not np.any(fin):
                    continue
                mu_no_i = np.interp(la, lno, mu_no)
                viol = np.count_nonzero(
                    mu_a[fin] > mu_no_i[fin] * (1 + 1e-6) + 1e-12)
                tot_viol += int(viol)
    print(f"  mu_atm <= mu_noatm: {tot_viol} cell violations "
          f"(attenuation should only reduce mu)")

    # n_transit_atm <= halo nt (nearest halo cell)
    if halo_shards:
        halo_by = _index_by_lambda(halo_shards)
        nt_viol = 0
        for key in sorted(atm_by.keys(), key=str):
            if key not in halo_by:
                continue
            a, h = atm_by[key], halo_by[key]
            la, laa = np.log10(a["ms"]), np.log10(a["alphas_n"])
            lhm, lha = np.log10(h["ms"]), np.log10(h["alphas_n"])
            im_near = np.clip(np.searchsorted(lhm, la), 0, h["ms"].size - 1)
            ia_near = np.clip(np.searchsorted(lha, laa), 0, h["alphas_n"].size - 1)
            nt_a = a["n_transit"]
            for ia in range(a["alphas_n"].size):
                hrow = h["nt"][ia_near[ia], im_near]
                fin = np.isfinite(nt_a[ia])
                nt_viol += int(np.count_nonzero(
                    nt_a[ia][fin] > hrow[fin] * (1 + 1e-3)))
        print(f"  n_transit_atm <= halo nt: {nt_viol} cell violations "
              f"(nearest halo cell)")
    else:
        print("  n_transit vs halo: skipped (no --shard-dir-halo)")

    # p monotone-ish in alpha away from the attenuation ceiling
    non_mono = 0
    for s in atm_shards:
        for mi in range(3):
            p = s["p"][mi]
            for im in range(s["ms"].size):
                col = p[:, im]
                fin = np.isfinite(col)
                c = col[fin]
                if c.size < 3:
                    continue
                # count strict decreases below the ceiling (p<0.999)
                below = c < 0.999
                dec = np.diff(c)
                non_mono += int(np.count_nonzero((dec < -1e-3) & below[1:]))
    print(f"  p(alpha) monotonicity: {non_mono} downward steps below the "
          f"ceiling (informational)")
    print("V3 verdict: warnings only (does not gate)")
    return True


# ── V4: status census ────────────────────────────────────────────────────────
def v4_status(atm_shards, noatm_shards):
    print("\n" + "=" * 72)
    print("V4  status census + status-1 cells")
    print("=" * 72)
    for pass_name, shards in (("atm", atm_shards), ("noatm", noatm_shards)):
        if not shards:
            continue
        counts = {c: 0 for c in STATUS_NAMES}
        total = 0
        ones = []
        for s in shards:
            st = s["status"]
            total += st.size
            for c in STATUS_NAMES:
                counts[c] += int(np.count_nonzero(st == c))
            lam = np.inf if s["massless"] else s["lamb"]
            for (k, ia, im) in np.argwhere(st == 1):
                ones.append((s["file"], lam, k + 1,
                             float(s["alphas_n"][ia]), float(s["ms"][im])))
        print(f"\n  {pass_name}: {total} cells")
        for c in STATUS_NAMES:
            print(f"    status {c} {STATUS_NAMES[c]:>10}: {counts[c]:>10d}")
        print(f"    status-1 cells: {len(ones)}")
        for (fn, lam, k, al, m) in ones[:200]:
            print(f"      {fn} lambda={lam:.3e} mode={k} "
                  f"alpha_n={al:.3e} m={m:.3e}")
        if len(ones) > 200:
            print(f"      ... ({len(ones) - 200} more)")
    return True


# ── optional: f4 spot check vs the assembled H5 ──────────────────────────────
def release_spot_check(release_path, atm_shards, noatm_shards, n_spot):
    print("\n" + "=" * 72)
    print("Vh  f4 spot check vs assembled HDF5")
    print("=" * 72)
    try:
        import h5py
    except Exception as err:  # noqa: BLE001
        print(f"  h5py unavailable ({err}); skipping")
        return True
    rng = np.random.default_rng(SEED + 1)
    ok = True
    with h5py.File(release_path, "r") as h5:
        # The f4 value check below cannot see a wrong kernel ATTRIBUTE: the
        # numbers still match the shards they came from. Gate the attribute
        # itself, or a v9 cube could ship claiming the historical kernel.
        h5_kernel = str(h5.attrs.get("projection_kernel", KERNEL_DEFAULT))
        shard_kernels = sorted({s.get("projection_kernel") or KERNEL_DEFAULT
                                for s in atm_shards + noatm_shards})
        if shard_kernels and shard_kernels != [h5_kernel]:
            print(f"  projection_kernel mismatch: h5={h5_kernel!r} "
                  f"shards={shard_kernels}")
            ok = False
        lam_axis = h5["axes/lambda_m"][:]
        # Spot-check every (f_dm, atmosphere) plane the FILE carries (from v7
        # a release file is a --select'ed single plane, not the full cube).
        # The shards' "p" array is the f_DM = config.F_X baseline plane only;
        # other-f_dm planes are derived at assembly, so they are checked
        # through mu, which rescales linearly by f_dm/F_X (extremeness does
        # not: it re-enters the MC table). v3 layout: one group per pass.
        axis_layout = "results" in h5
        if axis_layout:
            f_axis = h5["axes/f_dm"][:]
            atmos = h5["axes/atmosphere"][:]
        for pass_name, shards in (("atm", atm_shards), ("noatm", noatm_shards)):
            if not shards:
                continue
            planes = []          # (i_f, i_at, f_dm) the file has for this pass
            if axis_layout:
                want = 1 if pass_name == "atm" else 0
                for i_f, f_val in enumerate(f_axis):
                    for i_at in np.where(atmos == want)[0]:
                        planes.append((i_f, int(i_at), float(f_val)))
                if not planes:
                    print(f"  [{pass_name}] no such plane in this file; skip")
                    continue
            else:
                planes = [(None, None, float(config.F_X))]
            for i_f, i_at, f_val in planes:
                baseline = np.isclose(f_val, config.F_X, rtol=1e-9)
                if axis_layout:
                    surf = h5["results/extremeness" if baseline
                              else "results/mu"][i_f, i_at]
                else:
                    surf = h5[pass_name]["extremeness"]
                scale = 1.0 if baseline else f_val / config.F_X
                for s in shards[: max(1, n_spot // 4)]:
                    if s["massless"]:
                        hits = np.where(np.isinf(lam_axis))[0]
                    else:
                        hits = np.where(np.isclose(lam_axis, s["lamb"],
                                                   rtol=1e-9, atol=0))[0]
                    if hits.size == 0:
                        continue
                    li = int(hits[0])
                    for _ in range(3):
                        mi = int(rng.integers(0, 3))
                        ia = int(rng.integers(0, s["alphas_n"].size))
                        im = int(rng.integers(0, s["ms"].size))
                        fv = float(surf[mi, ia, im, li])
                        cv = float((s["p"] if baseline else s["mu"])
                                   [mi, ia, im]) * scale
                        if np.isnan(cv):
                            continue
                        # p: absolute f4 tolerance. mu: relative (spans
                        # decades; f4 storage rounds at ~6e-8 relative).
                        bad = (abs(fv - cv) > 1e-4 if baseline
                               else not np.isclose(fv, cv, rtol=1e-5,
                                                   atol=1e-30))
                        if bad:
                            print(f"  [{pass_name} f_dm={f_val:g} "
                                  f"{'p' if baseline else 'mu'}] f4 mismatch "
                                  f"mode{mi+1} a{ia} m{im}: h5={fv} "
                                  f"shard={cv}")
                            ok = False
    print(f"Vh verdict: {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-dir-atm", required=True)
    ap.add_argument("--shard-dir-noatm", required=True)
    ap.add_argument("--shard-dir-halo", default=None,
                    help="optional; enables the V3 n_transit<=halo check")
    ap.add_argument("--release", default=None,
                    help="optional assembled H5 for f4 spot checks")
    ap.add_argument("--n-spot", type=int, default=20)
    ap.add_argument("--skip-v2", action="store_true")
    ap.add_argument("--skip-v1", action="store_true",
                    help="skip the scan7-cache parity gate: V1 only means "
                         "anything when the committed caches share the "
                         "shards' conventions (kernel, efficiency, axes); a "
                         "convention-changing build verifies on V2 until the "
                         "caches are regenerated")
    args = ap.parse_args()

    atm_shards = read_shards(args.shard_dir_atm)
    noatm_shards = read_shards(args.shard_dir_noatm)
    halo_shards = read_halo_shards(args.shard_dir_halo)
    print(f"loaded {len(atm_shards)} atm, {len(noatm_shards)} noatm, "
          f"{len(halo_shards)} halo shards")

    hard = {}
    if args.skip_v1:
        print("\nV1 skipped (--skip-v1: committed scan7 caches do not share "
              "these shards' conventions)")
    else:
        hard["V1"] = v1_tag_parity(atm_shards, noatm_shards)
    if args.skip_v2:
        print("\nV2 skipped (--skip-v2)")
    else:
        hard["V2"] = v2_spot_recompute(atm_shards, noatm_shards, args.n_spot)
    v3_physics(atm_shards, noatm_shards, halo_shards)
    v4_status(atm_shards, noatm_shards)
    if args.release:
        hard["Vh"] = release_spot_check(args.release, atm_shards,
                                        noatm_shards, args.n_spot)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    all_pass = True
    for k, v in hard.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
        all_pass = all_pass and v
    print(f"\nOVERALL: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
