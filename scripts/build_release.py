#!/usr/bin/env python3
"""Build the UHDM data-release cube, one lambda-shard at a time (all passes).

Fans the (mode x alpha x mass) extremeness/mu/n_transit computation across a
many-core node, one mediator-range (lambda) shard per invocation-slice, writing
npz shards that scripts/assemble_release.py later stitches into the HDF5 release.
Same physics as scan_grid.py / scan_lambda.py through luhdm.rate, so the cube and
the historical scans cannot drift; the efficiency post-multiply lets one dR/dq +
one attenuation-ODE per (m, alpha) cell feed all three sensor modes.

Determinism: MC calibration is routed through :class:`PerMuTable`, which gives
every rounded mu its own freshly-seeded optimum_interval table, so cube values are
independent of worker count / task order / resume boundaries and a single-cell
recompute matches bit-for-bit.

Shard schema (atm/ and noatm/ dirs, one file per lambda index; massless is the
virtual index n_finite, written to shard_massless.npz with stored il = -1):

Dual f_DM: the DM fraction is a pure flux normalisation — the attenuated
velocity distribution and the dR/dq *shape* do not depend on it, only the
normalisation does. So each cell computes the rate ONCE (at the baseline
config.F_X = 0.1) and evaluates the optimum-interval statistic TWICE: once on
that rate and once on ``F_SCALE_F1 = 1.0/F_X`` times it. The f=1 surfaces are
written as parallel ``*_f1`` arrays; the unsuffixed arrays keep their exact
f=0.1 meaning, so every existing consumer is untouched.

  file : shard_il{il:02d}.npz  |  shard_massless.npz
  key            dtype   shape          meaning
  p              f8      (3,n_a,n_m)    optimum-interval extremeness, modes [1,2,3]
  mu             f8      (3,n_a,n_m)    expected detected counts, per mode
  n_transit      f8      (n_a,n_m)      expected flybys within threshold reach
  status         u1      (3,n_a,n_m)    0 ok / 1 exc / 2 mu<0.2 / 3 mu>mu_cap / 4 mu==0
  p_f1           f8      (3,n_a,n_m)    as p, for f_DM = 1.0 (rate x 10)
  mu_f1          f8      (3,n_a,n_m)    as mu, for f_DM = 1.0 (== 10 x mu)
  status_f1      u1      (3,n_a,n_m)    as status, for f_DM = 1.0
  f_dm_values    f8      (2,)           [0.1, 1.0]: unsuffixed / _f1 surfaces
  ms             f8      (n_m,)         DM mass axis [GeV]
  alphas_n       f8      (n_a,)         per-neutron coupling axis
  lamb           f8      ()             mediator range [m]; NaN for massless
  massless       bool    ()            True on the massless shard
  lamb_ode       f8      ()             ODE Coulomb-log regulator range [m]
  il             i8      ()             stable axis index; -1 for massless
  pass_name      str     ()            'atm' or 'noatm'
  q_min          f8      ()            analysis / grid-floor momentum [GeV]
  t_total        f8      ()            exposure live-time [s]
  seed           i8      ()            MC seed
  df             i8      ()            efficiency dof hypothesis (3)
  fidelity       str     ()            str(FID); includes mu_cap, so a
                                        single-cell recompute (verify_release V2
                                        -> scan_grid.scan_point) reuses the same
                                        optimum-interval shortcut as the build
  events_mode1/2/3 f8    (n_ev,)       observed impulses [GeV]
  schema_version i8      ()            2
  created,argv,hostname str; wall_s f8  provenance
  inputs_json    str     ()            event/efficiency file paths + sha256

Halo shard schema (halo/ dir, shard_halo_il{il:02d}.npz | shard_halo_massless.npz):
  nt f8 (n_a,n_m); bmax_m f8 (n_a,n_m); status u1 (n_a,n_m) [0 ok / 1 exc];
  ms, alphas_n (64-pt halo axes); lamb; massless; il; pass_name='halo'; t_total;
  seed; schema_version; created; argv; hostname; wall_s; inputs_json.
  (The halo pass is pure geometry: n_transit scales linearly with f_DM, so it
  carries no _f1 arrays.)

Usage
-----
  # local smoke (all three passes, tiny axes + quick fidelity):
  python scripts/build_release.py --pass atm   --shard-dir /tmp/rel/atm   --quick
  python scripts/build_release.py --pass noatm --shard-dir /tmp/rel/noatm --quick
  python scripts/build_release.py --pass halo  --shard-dir /tmp/rel/halo  --quick

  # compute-node production (one pass; tags + massless first as a launch-abort gate):
  python scripts/build_release.py --pass atm --shard-dir ~/release_shards/atm \
      --order tags-first --workers 80

  # resume: re-run the identical command; completed shard files are skipped
  # (atomic .tmp + os.replace means an existing shard is a complete shard).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")  # one process per core, no BLAS threads
os.environ.setdefault("TQDM_DISABLE", "1")

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

import luhdm
from luhdm import atmosphere, config, efficiency, halo, limits, rate

# --- pinned constants (shared contract) ---
SEED = 20260702
T_TOTAL = config.T_EXPOSURE                 # 790_778.0 s (single source of truth)
Q_HI_REF = 8.4e3                            # fixed qs upper-momentum reference
M_PLANCK = 1.22e19                          # top of the mass axis [GeV]
DF = 3                                      # efficiency dof hypothesis
CONFIDENCE = 0.95
MU_FLOOR = 0.2                              # matches limits.extremeness_and_mu
# Optimum-interval "no MC needed, p == 1" shortcut. This is passed EXPLICITLY to
# limits.extremeness_and_mu (whose own default is the historical 40.0) and is
# recorded in the shard fidelity string, so build and single-cell recompute
# always agree. Raised 40 -> 85 because the measured extremeness at mu ~ 40 is
# far below 1 (as low as 0.0000), i.e. the 40 cap over-excluded. The 85 cap has
# real margin only in modes 1 and 2 (max mu among MC cells with p < 0.95 is
# 16.2 and 38.9). Mode 3 has MC cells at mu = 84.9 with p as low as 0.0003, so
# there its status-3 cells are an assertion of exclusion, not a computed result
# -- see release/README.md section 9 "Known limitations". Do not read this cap
# as validated for mode 3, and do not raise it on the strength of that reading.
MU_CAP = 85.0
SCHEMA_VERSION = 2                          # 2: added the f_DM=1 (*_f1) surfaces

# Dual f_DM. config.F_X (0.1) is the baseline the unsuffixed arrays carry; the
# *_f1 arrays are the same cells at f_DM = 1.0. f_DM is a pure flux
# normalisation (n_dm ∝ f_DM), so the ODE and the dR/dq shape are identical and
# only the normalisation moves: one rate, two optimum-interval evaluations.
F_DM_BASE = float(config.F_X)               # 0.1
F_DM_HIGH = 1.0
F_SCALE_F1 = F_DM_HIGH / F_DM_BASE          # 10.0
F_DM_VALUES = (F_DM_BASE, F_DM_HIGH)

# exact members of the finite axis; 0.2 (20 cm) fills the 2 cm -> 2 m gap
TAGS = [2e-6, 1e-5, 2e-5, 2e-4, 2e-3, 2e-2, 0.2, 2.0]

# --lambda-set v7quick: the reduced finite axis for the v7-quick campaign.
# The three physics ranges (20 um, 200 um, 2 mm) are the figure set, and are
# exact members of the full axis too, so a v7quick slice and a full-axis slice
# at the same lambda are directly comparable. 200 m is a VALIDATION-ONLY slice
# (m_phi ~ 1e-9 eV): it is far longer than every impact parameter in play, so
# the finite-lambda code path must reproduce the analytic massless slice there.
# It is not a physics point and must not be plotted as one.
LAMBDA_SET_V7QUICK = [2e-5, 2e-4, 2e-3, 200.0]
LAMBDA_VALIDATION = 200.0
# The validation slice needs a denser dsigma/dq_tilde tabulation than the
# physics slices; that rule lives in rate.tabulation_n_points, keyed on xi, so
# every consumer (builder, verifier, contour refiner) resolves it identically.

FID_PROD = dict(n_ode=400, n_shm=300000, n_q=240, q_span=3e4, n_mc=10000,
                mu_cap=MU_CAP)
FID_QUICK = dict(n_ode=60, n_shm=20000, n_q=120, q_span=1e4, n_mc=1500,
                 mu_cap=MU_CAP)

# Shared, read-only state; set in main() BEFORE the fork so children inherit it.
# Pass-constant globals:
MS = None            # mass axis (analysis axis, or halo axis for the halo pass)
ALPHAS = None        # coupling axis
QS = None            # momentum grid (fixed for every cell)
EFF_QS = None        # (3, n_q) efficiency eps_mode(QS), modes [1,2,3]
EVENTS_BY_MODE = None # [events_mode1, events_mode2, events_mode3] in GeV
V_I_SAMPLES = None   # SHM initial-speed samples (seeded)
FID = None
NO_ATM = False       # True on the noatm pass (bare halo, no attenuation ODE)
Q_MIN = None
# Per-shard globals (reset before each shard's pool is spawned):
LAMB = None          # sensor mediator range [m]; None for massless
XS = None            # rate.make_xsec handle for this shard
MASSLESS = False
LAMB_ODE = None      # atmospheric-ODE regulator range [m]

_worker_state: dict = {}


# --------------------------------------------------------------------------- #
# Per-mu MC calibration (order-independent, bit-reproducible)
# --------------------------------------------------------------------------- #
class PerMuTable:
    """Duck-type of OptimumIntervalTable that shards by rounded mu.

    scan_extremeness (optimum_interval.scanning) calls ``generate(mu, n)``,
    ``optimum_interval_statistic(events, mu, spectrum_cdf=...)`` and
    ``extremeness_of_opt_itv_stat(stat, mu)`` all with the SAME rounded mu for a
    given scan point. Routing each rounded mu to its own limits.new_table(seed)
    makes the MC calibration a pure function of (mu, seed, n) -- independent of
    evaluation order, worker count and resume boundaries -- so a fresh
    single-cell recompute reproduces the cube bit-for-bit.
    """

    def __init__(self, seed=SEED):
        self._seed = seed
        self._tables: dict = {}

    def _table_for(self, mu):
        t = self._tables.get(mu)
        if t is None:
            t = limits.new_table(seed=self._seed)
            self._tables[mu] = t
        return t

    def generate(self, mu, n):
        self._table_for(mu).generate(mu, n)

    def optimum_interval_statistic(self, events, mu, spectrum_cdf=None):
        return self._table_for(mu).optimum_interval_statistic(
            events, mu, spectrum_cdf=spectrum_cdf)

    def extremeness_of_opt_itv_stat(self, stat, mu):
        return self._table_for(mu).extremeness_of_opt_itv_stat(stat, mu)


def _worker_init_lazy():
    """Per-process state, built on first use in each child (one PerMuTable)."""
    if "table" not in _worker_state:
        _worker_state["table"] = PerMuTable(seed=SEED)
    return _worker_state


def _derive_status(p, mu):
    """Status code from (p, mu) per the contract (see module docstring)."""
    if np.isnan(p) or np.isnan(mu):
        return 1
    if mu == 0.0:
        return 4          # no spectrum support (extremeness_and_mu returned 0,0)
    if mu < MU_FLOOR:
        return 2          # mu<0.2 shortcut, p exactly 0
    if mu > MU_CAP:
        return 3          # mu>mu_cap shortcut, p exactly 1
    return 0              # MC ran


# --------------------------------------------------------------------------- #
# Worker chunk functions
# --------------------------------------------------------------------------- #
def _process_chunk(chunk):
    """atm/noatm: a contiguous block of (im, ia) cells -> per-cell results."""
    state = _worker_init_lazy()
    table = state["table"]
    out = []
    for im, ia in chunk:
        m = float(MS[im])
        alpha_n = float(ALPHAS[ia])
        try:
            v_min = Q_MIN / m / 10   # ODE floor follows the analysis threshold
            if NO_ATM:
                f_v_f = halo.standard_halo_model
            else:
                v_f_samples = atmosphere.compute_v_f_distribution(
                    alpha_n, LAMB_ODE, m, V_I_SAMPLES,
                    v_min=v_min, n_grid=FID["n_ode"])
                f_v_f = atmosphere.compute_f_vf(v_f_samples, v_min)[0]
            raw = rate.differential_rate_trapz(QS, alpha_n, m, f_v_f, XS, eff=None)
            n_t = rate.expected_transits(alpha_n, m, f_v_f, XS, T_TOTAL)
            p3 = np.empty(3)
            mu3 = np.empty(3)
            st3 = np.empty(3, np.uint8)
            p3_f1 = np.empty(3)
            mu3_f1 = np.empty(3)
            st3_f1 = np.empty(3, np.uint8)
            for k in range(3):
                detected = raw * EFF_QS[k]       # dR/dq at f_DM = F_DM_BASE
                p, mu = limits.extremeness_and_mu(
                    table, EVENTS_BY_MODE[k], QS, detected, T_TOTAL,
                    n_mc=FID["n_mc"], mu_cap=FID["mu_cap"])
                p3[k], mu3[k], st3[k] = p, mu, _derive_status(p, mu)
                # f_DM = 1: same spectrum shape, same events, same MC table —
                # only the normalisation scales (mu -> F_SCALE_F1 * mu).
                p_f1, mu_f1 = limits.extremeness_and_mu(
                    table, EVENTS_BY_MODE[k], QS, detected * F_SCALE_F1, T_TOTAL,
                    n_mc=FID["n_mc"], mu_cap=FID["mu_cap"])
                p3_f1[k], mu3_f1[k] = p_f1, mu_f1
                st3_f1[k] = _derive_status(p_f1, mu_f1)
        except Exception as err:  # a raised ODE/rate corner: NaNs + status 1
            print(f"cell FAIL im={im} ia={ia} m={m:.3e} a={alpha_n:.3e}: {err}",
                  flush=True)
            p3 = np.full(3, np.nan)
            mu3 = np.full(3, np.nan)
            st3 = np.ones(3, np.uint8)
            p3_f1 = np.full(3, np.nan)
            mu3_f1 = np.full(3, np.nan)
            st3_f1 = np.ones(3, np.uint8)
            n_t = np.nan
        out.append((im, ia, p3, mu3, float(n_t), st3, p3_f1, mu3_f1, st3_f1))
    return out


def _process_chunk_halo(chunk):
    """halo: a contiguous block of (im, ia) cells -> (nt, bmax, status)."""
    out = []
    for im, ia in chunk:
        m = float(MS[im])
        alpha_n = float(ALPHAS[ia])
        try:
            n_t, a_eff = rate.transit_count_halo(m, alpha_n, XS, T_TOTAL)
            bmax = float(np.sqrt(a_eff / np.pi))
            st = np.uint8(0)
        except Exception as err:
            print(f"halo cell FAIL im={im} ia={ia} m={m:.3e} a={alpha_n:.3e}: "
                  f"{err}", flush=True)
            n_t, bmax, st = np.nan, np.nan, np.uint8(1)
        out.append((im, ia, float(n_t), bmax, st))
    return out


# --------------------------------------------------------------------------- #
# Axes (deterministic; single implementation lives here)
# --------------------------------------------------------------------------- #
def build_lambda_axis():
    """Finite mediator-range axis per the contract. Returns the sorted array.

    All 7 TAG values and all 14 zoom values are exact float members; n_finite
    (== axis size) is the virtual index of the appended massless slice.
    """
    base = np.geomspace(1e-7, 2.0, 45)
    zoom = np.geomspace(1e-6, 2e-5, 14)
    exact = np.unique(np.concatenate([TAGS, zoom]))
    le = np.log10(exact)
    kept = [b for b in base if np.min(np.abs(np.log10(b) - le)) > 0.04]
    return np.sort(np.unique(np.concatenate([exact, kept])))


def build_mass_axis(m_tier, m_min=1e5, m_max=M_PLANCK):
    """Mass axis for the requested tier (60 / 119 / 600)."""
    if m_tier == 119:
        ms60 = np.logspace(np.log10(m_min), np.log10(m_max), 60)
        out = np.empty(119)
        out[::2] = ms60                          # ms119[::2] bitwise == ms60
        out[1::2] = np.sqrt(ms60[:-1] * ms60[1:])
        return out
    return np.logspace(np.log10(m_min), np.log10(m_max), m_tier)


def build_alpha_axis(a_min, n_a):
    return np.logspace(np.log10(a_min), 0.0, n_a)


def find_tag_ils(lam_finite, tags):
    """Axis indices of the tag ranges, ordered largest -> smallest lambda."""
    tag_ils = []
    for t in sorted(tags, reverse=True):
        idx = np.where(lam_finite == t)[0]
        if idx.size:
            tag_ils.append(int(idx[0]))
    return tag_ils


def build_processing_order(selected, order, tag_ils, n_finite):
    """Sequence in which the selected ils are processed (names use axis index)."""
    sel = set(selected)
    if order == "axis":
        return sorted(selected)
    if order == "small-first":
        fin = sorted(i for i in selected if i < n_finite)
        return fin + ([n_finite] if n_finite in sel else [])
    # tags-first: tag ils (large->small lambda), massless, then remaining
    # finite ils smallest-lambda first.
    seq = [i for i in tag_ils if i in sel]
    if n_finite in sel:
        seq.append(n_finite)
    seq += sorted(j for j in selected if j < n_finite and j not in tag_ils)
    return seq


# --------------------------------------------------------------------------- #
# Shard IO / provenance
# --------------------------------------------------------------------------- #
def shard_path(shard_dir, pass_name, il, n_finite):
    massless = il == n_finite
    if pass_name == "halo":
        name = "shard_halo_massless.npz" if massless else f"shard_halo_il{il:02d}.npz"
    else:
        name = "shard_massless.npz" if massless else f"shard_il{il:02d}.npz"
    return shard_dir / name


def sha256_file(path):
    """sha256 of a file, or None if it cannot be read (never fatal)."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for blk in iter(lambda: fh.read(1 << 20), b""):
                h.update(blk)
        return h.hexdigest()
    except Exception as err:  # noqa: BLE001
        print(f"WARNING: sha256 unavailable for {path}: {err}", flush=True)
        return None


def input_provenance(data_dir):
    """Paths + sha256 of the inputs this run actually consumed.

    Both the event files (``--data-dir``) and the efficiency table
    (``LUHDM_EFFICIENCY_NPZ``) can be redirected per run, and the live-time can
    be redirected by ``LUHDM_T_EXPOSURE``; recording them here is what makes an
    env-overridden campaign (e.g. the cveto variant) reconstructable from the
    shards alone.
    """
    eff_table = Path(efficiency.table_path())
    prov = {
        "t_exposure_s": float(config.T_EXPOSURE),
        "f_dm_values": list(F_DM_VALUES),
        "f_x_base": F_DM_BASE,
        "efficiency_npz": str(eff_table),
        "efficiency_npz_sha256": sha256_file(eff_table),
        "events": {},
        "env": {k: os.environ[k] for k in
                ("LUHDM_T_EXPOSURE", "LUHDM_EFFICIENCY_NPZ") if k in os.environ},
    }
    if data_dir is not None:
        for n in (1, 2, 3):
            f = Path(data_dir) / f"data_mode{n}.txt"
            prov["events"][f"data_mode{n}.txt"] = {
                "path": str(f), "sha256": sha256_file(f)}
    return prov


def _atomic_savez(path, **arrays):
    # np.savez appends ".npz" unless the name already ends in it, so keep the
    # temp name ".npz"-terminated or os.replace would miss the real output.
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def append_run_config(shard_dir, record):
    path = shard_dir / "run_config.json"
    records = []
    if path.exists():
        try:
            records = json.loads(path.read_text())
            if not isinstance(records, list):
                records = [records]
        except Exception:
            records = []
    records.append(record)
    tmp = path.with_name("run_config.json.tmp")
    tmp.write_text(json.dumps(records, indent=2))
    os.replace(tmp, path)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    global MS, ALPHAS, QS, EFF_QS, EVENTS_BY_MODE, V_I_SAMPLES, FID, NO_ATM, Q_MIN
    global LAMB, XS, MASSLESS, LAMB_ODE

    ap = argparse.ArgumentParser(
        description="Build one pass of the UHDM data-release cube (npz shards).")
    ap.add_argument("--pass", dest="pass_name", required=True,
                    choices=("atm", "noatm", "halo"))
    ap.add_argument("--shard-dir", required=True,
                    help="output directory for this pass's shards")
    ap.add_argument("--il-start", type=int, default=None,
                    help="first lambda index (inclusive); default 0")
    ap.add_argument("--il-end", type=int, default=None,
                    help="one past the last lambda index (half-open); default "
                         "n_finite+1 (includes the massless index n_finite)")
    ap.add_argument("--workers", type=int, default=None,
                    help="worker processes (default: all cores)")
    ap.add_argument("--chunk", type=int, default=None,
                    help="cells per pool task (default 2 atm / 200 noatm / 500 halo)")
    ap.add_argument("--order", choices=("tags-first", "small-first", "axis"),
                    default="tags-first",
                    help="processing order of the selected ils (names use the "
                         "stable axis index regardless)")
    ap.add_argument("--m-tier", type=int, choices=(60, 119, 600), default=None,
                    help="mass tier (default 119 atm / 600 noatm; halo ignores)")
    ap.add_argument("--n-a", type=int, default=44, help="coupling grid points")
    ap.add_argument("--m-min", type=float, default=1e5, help="lowest DM mass [GeV]")
    ap.add_argument("--a-min", type=float, default=1e-10, help="lowest coupling")
    ap.add_argument("--n-mc", type=int, default=None)
    ap.add_argument("--n-ode", type=int, default=None)
    ap.add_argument("--n-shm", type=int, default=None)
    ap.add_argument("--n-q", type=int, default=None)
    ap.add_argument("--q-span", type=float, default=None)
    ap.add_argument("--q-min", type=float, default=config.Q_THRESH,
                    help="analysis / grid-floor momentum [GeV] (default: "
                         "config.Q_THRESH = %(default)g)")
    ap.add_argument("--massless-lamb", type=float, default=2.0,
                    help="atmospheric-ODE regulator range for the massless slice [m]")
    ap.add_argument("--b-constrained-max", type=float, default=None,
                    help="impact-parameter cap [m] applied to the cross section "
                         "dsigma/dq only (finite and massless); None (default) "
                         "= uncapped, reproduces the current build exactly")
    ap.add_argument("--data-dir", default=None,
                    help="dir holding data_mode{1,2,3}.txt (default <repo>/notebooks)")
    ap.add_argument("--print-order", action="store_true",
                    help="print the il processing sequence and exit")
    ap.add_argument("--print-lambdas", action="store_true",
                    help="print the il -> lambda table and exit")
    ap.add_argument("--lambda-set", choices=("full", "v7quick"), default="full",
                    help="finite mediator-range axis: 'full' (the contract "
                         "axis, default) or 'v7quick' (20 um / 200 um / 2 mm "
                         "plus the 200 m massless-equivalence validation "
                         "slice); --quick overrides both")
    ap.add_argument("--quick", action="store_true",
                    help="tiny smoke: lambda = the 7 tags + massless, quick FID, "
                         "n_a=6, 8-point mass axis")
    args = ap.parse_args()

    pass_name = args.pass_name
    quick = args.quick

    # --- lambda axis (independent of pass) ---
    if quick:
        # all 7 tags so tag-slicing consumers (notebooks) can smoke end-to-end
        lam_finite = np.sort(np.array(TAGS))
        tags = TAGS
    elif args.lambda_set == "v7quick":
        lam_finite = np.sort(np.array(LAMBDA_SET_V7QUICK))
        tags = list(LAMBDA_SET_V7QUICK)
    else:
        lam_finite = build_lambda_axis()
        tags = TAGS
    n_finite = lam_finite.size
    tag_ils = find_tag_ils(lam_finite, tags)

    if args.print_lambdas:
        zoom = np.geomspace(1e-6, 2e-5, 14)
        print("il   lamb_m         flags")
        for il, lam in enumerate(lam_finite):
            flags = []
            if any(lam == t for t in TAGS):
                flags.append("TAG")
            if any(lam == z for z in zoom):
                flags.append("zoom")
            print(f"{il:3d}  {lam:.6e}  {' '.join(flags)}")
        print(f"massless  il={n_finite}  lamb=inf")
        print(f"n_finite = {n_finite}")
        return

    # --- selected ils + processing order ---
    il_start = 0 if args.il_start is None else args.il_start
    il_end = (n_finite + 1) if args.il_end is None else args.il_end
    selected = [i for i in range(il_start, il_end) if 0 <= i <= n_finite]
    seq = build_processing_order(selected, args.order, tag_ils, n_finite)

    if args.print_order:
        print(" ".join(str(i) for i in seq))
        return

    # --- fidelity ---
    FID = dict(FID_QUICK if quick else FID_PROD)
    for key, val in (("n_mc", args.n_mc), ("n_ode", args.n_ode),
                     ("n_shm", args.n_shm), ("n_q", args.n_q),
                     ("q_span", args.q_span)):
        if val is not None:
            FID[key] = val

    # --- axes for this pass ---
    n_a = 6 if quick else args.n_a
    if pass_name == "halo":
        if quick:
            MS = np.geomspace(args.m_min, M_PLANCK, 8)
            ALPHAS = np.geomspace(2e-11, 1.0, n_a)
        else:
            MS = np.geomspace(1e5, M_PLANCK, 64)   # bitwise == make_maps.py grids
            ALPHAS = np.geomspace(2e-11, 1.0, 64)
    else:
        m_tier = args.m_tier if args.m_tier is not None else (
            600 if pass_name == "noatm" else 119)
        if quick:
            MS = np.logspace(np.log10(args.m_min), np.log10(M_PLANCK), 8)
        else:
            MS = build_mass_axis(m_tier, args.m_min)
        ALPHAS = build_alpha_axis(args.a_min, n_a)
    n_m = MS.size
    n_a = ALPHAS.size

    NO_ATM = pass_name == "noatm"
    Q_MIN = args.q_min
    # The analysis window enters by two independent routes: Q_MIN sets the rate
    # integral's lower endpoint (and, via Q_MIN/m/10, the atmospheric ODE floor),
    # while rate.expected_transits / rate.transit_count_halo read config.Q_THRESH
    # directly for the threshold reach b_max(q_thresh). If the two disagree the
    # n_transit surface describes a different window than mu does, which is
    # exactly the kind of drift that is invisible in the output.
    if float(Q_MIN) != float(config.Q_THRESH):
        print("*" * 72, flush=True)
        print(f"WARNING: --q-min ({Q_MIN:g} GeV) != config.Q_THRESH "
              f"({config.Q_THRESH:g} GeV).", flush=True)
        print("         mu/p use --q-min; n_transit uses config.Q_THRESH. "
              "The two surfaces", flush=True)
        print("         in this cube will describe DIFFERENT analysis windows.",
              flush=True)
        print("*" * 72, flush=True)

    # --- pass-constant shared state (halo needs only MS/ALPHAS/XS) ---
    if pass_name != "halo":
        QS = np.geomspace(Q_MIN, FID["q_span"] * Q_HI_REF, FID["n_q"])
        EFF_QS = np.array([efficiency.make_efficiency(mode, DF)(QS)
                           for mode in (1, 2, 3)])
        repo = Path(__file__).resolve().parents[1]
        data_dir = Path(args.data_dir) if args.data_dir else repo / "notebooks"
        EVENTS_BY_MODE = [np.atleast_1d(np.loadtxt(data_dir / f"data_mode{n}.txt"))
                          / 1e9 for n in (1, 2, 3)]
        V_I_SAMPLES = atmosphere.sample_shm(
            FID["n_shm"], rng=np.random.default_rng(SEED))
    else:
        data_dir = None
    inputs = input_provenance(data_dir)
    inputs_json = json.dumps(inputs, sort_keys=True)

    chunk = args.chunk if args.chunk is not None else (
        500 if pass_name == "halo" else (200 if NO_ATM else 2))

    shard_dir = Path(args.shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)
    hostname = socket.gethostname()
    argv_str = " ".join(sys.argv)
    worker_fn = _process_chunk_halo if pass_name == "halo" else _process_chunk

    print(f"pass={pass_name}  shard-dir={shard_dir}  order={args.order}")
    print(f"axes: n_m={n_m} n_a={n_a} n_finite={n_finite} (lambda "
          f"{lam_finite.min():.2e}..{lam_finite.max():.2e})")
    print(f"FID={FID}  chunk={chunk}  workers={args.workers}")
    print(f"f_dm_values={list(F_DM_VALUES)}  t_exposure_s={T_TOTAL:.0f}  "
          f"efficiency_npz={efficiency.table_path()}")
    print(f"processing {len(seq)} ils: {seq}", flush=True)

    # --- expensive-first cell list (same for every shard of this pass) ---
    cells = [(im, ia) for im in range(n_m) for ia in range(n_a)]
    cells.sort(key=lambda c: (MS[c[0]], ALPHAS[c[1]]), reverse=True)
    chunks = [cells[i:i + chunk] for i in range(0, len(cells), chunk)]
    total_cells = len(cells)
    n_chunks = len(chunks)

    start_iso = datetime.now(timezone.utc).isoformat()
    ils_done = []
    ctx = multiprocessing.get_context("fork")

    for il in seq:
        massless = il == n_finite
        label = "massless" if massless else str(il)
        path = shard_path(shard_dir, pass_name, il, n_finite)
        if path.exists():
            print(f"SKIP il={label} (exists)", flush=True)
            continue

        lamb = None if massless else float(lam_finite[il])
        LAMB = lamb
        MASSLESS = massless
        LAMB_ODE = args.massless_lamb if massless else lamb   # unused on noatm/halo
        # Tabulation density is resolved from xi inside rate.make_xsec, so the
        # builder and any later single-cell recompute cannot disagree about it.
        XS = rate.make_xsec(None if massless else lamb,       # auto dispatch
                            b_constrained_max=args.b_constrained_max)

        print(f"[il={label}] lamb={lamb}  building shard "
              f"({total_cells} cells, {n_chunks} chunks) ...", flush=True)
        t0 = time.time()

        if pass_name == "halo":
            NT = np.full((n_a, n_m), np.nan)
            BMAX = np.full((n_a, n_m), np.nan)
            ST = np.ones((n_a, n_m), np.uint8)
        else:
            P = np.full((3, n_a, n_m), np.nan)
            MU = np.full((3, n_a, n_m), np.nan)
            NT = np.full((n_a, n_m), np.nan)
            ST = np.ones((3, n_a, n_m), np.uint8)
            P_F1 = np.full((3, n_a, n_m), np.nan)
            MU_F1 = np.full((3, n_a, n_m), np.nan)
            ST_F1 = np.ones((3, n_a, n_m), np.uint8)

        cells_done = 0
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as ex:
            futures = [ex.submit(worker_fn, ch) for ch in chunks]
            for k, fut in enumerate(as_completed(futures)):
                res = fut.result()
                if pass_name == "halo":
                    for im, ia, n_t, bmax, st in res:
                        NT[ia, im] = n_t
                        BMAX[ia, im] = bmax
                        ST[ia, im] = st
                else:
                    for im, ia, p3, mu3, n_t, st3, p3f, mu3f, st3f in res:
                        P[:, ia, im] = p3
                        MU[:, ia, im] = mu3
                        NT[ia, im] = n_t
                        ST[:, ia, im] = st3
                        P_F1[:, ia, im] = p3f
                        MU_F1[:, ia, im] = mu3f
                        ST_F1[:, ia, im] = st3f
                cells_done += len(res)
                if (k + 1) % 50 == 0:
                    el = time.time() - t0
                    cps = cells_done / el if el > 0 else 0.0
                    eta = (total_cells - cells_done) / cps if cps > 0 else float("inf")
                    print(f"  [il={label}] {k + 1}/{n_chunks} chunks  "
                          f"{cells_done}/{total_cells} cells  {el:.0f}s  "
                          f"{cps:.1f} cells/s  ETA {eta:.0f}s", flush=True)

        wall_s = time.time() - t0
        created = datetime.now(timezone.utc).isoformat()
        il_stored = -1 if massless else il

        if pass_name == "halo":
            _atomic_savez(
                path, nt=NT, bmax_m=BMAX, status=ST, ms=MS, alphas_n=ALPHAS,
                lamb=(np.nan if massless else lamb), massless=massless,
                il=il_stored, pass_name=pass_name, t_total=T_TOTAL, seed=SEED,
                b_constrained_max=args.b_constrained_max,
                schema_version=SCHEMA_VERSION, created=created, argv=argv_str,
                hostname=hostname, wall_s=wall_s, inputs_json=inputs_json)
        else:
            _atomic_savez(
                path, p=P, mu=MU, n_transit=NT, status=ST,
                p_f1=P_F1, mu_f1=MU_F1, status_f1=ST_F1,
                f_dm_values=np.array(F_DM_VALUES, dtype=np.float64), ms=MS,
                alphas_n=ALPHAS, lamb=(np.nan if massless else lamb),
                massless=massless, lamb_ode=LAMB_ODE, il=il_stored,
                pass_name=pass_name, q_min=Q_MIN, t_total=T_TOTAL, seed=SEED,
                b_constrained_max=args.b_constrained_max,
                df=DF, fidelity=str(FID),
                events_mode1=EVENTS_BY_MODE[0], events_mode2=EVENTS_BY_MODE[1],
                events_mode3=EVENTS_BY_MODE[2], schema_version=SCHEMA_VERSION,
                created=created, argv=argv_str, hostname=hostname, wall_s=wall_s,
                inputs_json=inputs_json)

        ils_done.append(label)
        print(f"[il={label}] wrote {path.name} in {wall_s:.0f}s", flush=True)

    append_run_config(shard_dir, dict(
        argv=sys.argv, pass_name=pass_name,
        b_constrained_max=args.b_constrained_max,
        axes=dict(n_m=n_m, n_a=n_a, n_l=n_finite + 1, m_tier=(
            None if pass_name == "halo" else (
                args.m_tier if args.m_tier is not None else
                (600 if pass_name == "noatm" else 119))),
            lambda_min=float(lam_finite.min()),
            lambda_max=float(lam_finite.max())),
        fid=FID, hostname=hostname, workers=args.workers,
        start=start_iso, end=datetime.now(timezone.utc).isoformat(),
        ils_completed=ils_done,
        schema_version=SCHEMA_VERSION,
        # env-overridable inputs, resolved at runtime (closes the gap where a
        # LUHDM_T_EXPOSURE / LUHDM_EFFICIENCY_NPZ / --data-dir override left no
        # trace in run_config.json)
        inputs=inputs,
        luhdm_version=getattr(luhdm, "__version__", "?"),
        numpy_version=np.__version__))

    print(f"PASS_DONE {pass_name}  ils this run: {ils_done}", flush=True)


if __name__ == "__main__":
    main()
