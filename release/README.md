# POLONAISE UHDM data release

Single-file HDF5 release of the levitated-sensor ultra-heavy dark-matter search:
the full optimum-interval analysis as a matrix over
(sensor mode, coupling α_n, DM mass, mediator range λ), plus the detector
inputs and reference curves needed to reproduce every figure in `notebooks/`.

- File: `luhdm_datarelease_v2.h5` (not tracked in git; distributed via Zenodo
  at publication — regenerate locally per **Regeneration** below)
- Integrity: `sha256sum -c SHA256SUMS`
- Provenance: `provenance.json` (builder + assembly records) and the root
  attributes embedded in the file itself
- Tutorial: `notebooks/06_datarelease.ipynb`
- Loader: `from luhdm import release; rel = release.open_release()`

Everything is plain HDF5 with dimension scales and per-dataset `units` /
`description` attributes — `h5py`, `h5ls`, MATLAB, or xarray (via h5netcdf)
all work with no luhdm code.

## Quickstart

```python
from luhdm import release
rel = release.open_release()                       # release/luhdm_datarelease_v2.h5
P = rel.mass_plane("extremeness", mode=1, lam="200um")   # (n_alpha, n_mass)
lo, hi = rel.excluded_alpha_band(mass=1e8, lam="200um", mode=1)
rel.cell(mass=1e8, alpha=1e-3, lam="200um", mode=1)      # one-cell dump
rel.close()
```

Raw h5py, no luhdm:

```python
import h5py
f = h5py.File("release/luhdm_datarelease_v2.h5", "r")
p = f["atm/extremeness"]          # (mode, alpha, mass, lambda), float32
alpha = f["axes/alpha_n"][:]
```

## Layout

Cube axis order is `(mode, alpha, mass, lambda)`; `L` is the lambda-axis length
(54 finite values + the massless sentinel, see below).

| path | shape | dtype | units | description |
|---|---|---|---|---|
| `/axes/mode` | (3,) | u1 | 1 | sensor modes [1,2,3] |
| `/axes/alpha_n` | (44,) | f8 | 1 | coupling per neutron, logspace(1e-10, 1) |
| `/axes/mass_gev` | (119,) | f8 | GeV | DM mass, 1e5..1.22e19 (`/atm` axis) |
| `/axes/mass_noatm_gev` | (600,) | f8 | GeV | DM mass (`/noatm` axis) |
| `/axes/lambda_m` | (L,) | f8 | m | mediator range, ascending; **last = inf** |
| `/axes/m_phi_gev` | (L,) | f8 | GeV | mediator mass = 1/conv_m2pGeV(λ); 0.0 at inf |
| `/axes/mass_halo_gev`, `/axes/alpha_halo_n` | (64,) | f8 | | `/halo` axes (α down to 2e-11) |
| `/atm/extremeness` | (3,44,119,L) | f4 | 1 | optimum-interval extremeness p; exclusion at CL C = the p ≥ C level set |
| `/atm/mu` | (3,44,119,L) | f4 | events | expected detected events (efficiency folded in) |
| `/atm/n_transit` | (44,119,L) | f4 | transits | expected flybys within reach (mode-independent) |
| `/atm/status` | (3,44,119,L) | u1 | enum | see **Status codes** |
| `/noatm/…` | (·,44,600,L) | | | same four datasets, bare-halo velocity distribution |
| `/halo/n_transit` | (64,64,L) | f4 | transits | unattenuated halo transit count |
| `/halo/bmax` | (64,64,L) | f4 | m | flux-averaged threshold reach √⟨b²⟩ |
| `/detector/exposure_s` | () | f8 | s | live-time (1,691,020 s = 469.7 h) |
| `/detector/events_mode{1,2,3}` | (66/99/443,) | f8 | GeV | candidate impulses (= `data_mode{n}.txt`/1e9) |
| `/detector/all_blips_mode{1,2,3}` | | f8 | eV | all reconstructed blip momenta (pre-selection) |
| `/detector/q_gev_{n}`, `/detector/eff_{n}_df{2,3}` | (400,) | f8 | | measured efficiency curves per mode |
| `/reference_curves/v`, `fv_shm`, `fv_<tag>` | (500,) | | 1/c | arrival-speed distributions at m=1e8 GeV, α_n=1e-3 (attr `survival_fraction`) |
| `/reference_curves/q_gev`, `drdq_<tag>` | (160,) | | s⁻¹ GeV⁻¹ | raw dR/dq at the showcase point (no efficiency; arrival f(v) fixed at λ=200 µm) |

## The lambda axis

The finite axis is the union of a ~6-points-per-decade grid over 1e-7..2.0 m, a
14-point zoom grid over 1–20 µm, and the seven named tag values — deduplicated
at 0.04 dex, always keeping the exact tag value. The named tags are **exact
float members** of the axis, so tag-based figures are pure integer slices:

| tag | λ (m) | | tag | λ (m) |
|---|---|---|---|---|
| `2m` | 2.0 | | `20um` | 2e-5 |
| `2cm` | 2e-2 | | `10um` | 1e-5 |
| `2mm` | 2e-3 | | `2um` | 2e-6 |
| `200um` | 2e-4 | | `massless` | inf |

**Massless sentinel:** the last axis element is `inf` (`m_phi_gev` exactly 0.0)
— the analytic-Rutherford (massless mediator at the sensor) slice. Its
atmosphere ODE uses a λ = 2 m Coulomb-log regulator, matching the historical
production convention. The loader resolves the tag `"massless"` to this index;
`rel.axes.lambda_finite` drops it.

## The impact-parameter cap (v2)

v2 applies an **impact-parameter cap** `b_constrained_max = 0.1 m (10 cm)`: the
outer limit of the impact-parameter integral is

```
dσ/dq = ∫_{R_eff}^{min(b_constrained_max, b_max(q))} 2b / (q₀ √(1 − (q/q₀)²)) db
```

i.e. flybys further than 10 cm from the sensor are excluded. The cap applies to
**both** the cross section `dσ/dq` **and** the geometric transit reach, so
`n_transit` and the `halo` `bmax` / `n_transit` diagnostics stay consistent with
the rate (one clip at the single reach chokepoint,
`rate.impact_parameter_max_any`). Its value is in the root attribute
`b_constrained_max_m` (NaN ⇒ uncapped); `rel.b_constrained_max` exposes it.

Which λ the cap touches (α_n = 1, the strongest coupling on the grid):

| channel | v floor | first affected λ index |
|---|---|---|
| transit reach (`n_transit`, halo) | 1e-8 (`expected_transits`) | il37, λ = 4.43 mm (b_max = 0.130 m) |
| cross section `dσ/dq` | q_min/m_max = 8.2e-18 | il35, λ = 2.0 mm (b_max = 0.1015 m) |

il34 (λ = 1.41 mm, b_max = 0.0719 m) is the highest **provably unaffected**
shard, so v2 reuses the uncapped v1 shards il00–il34 verbatim; il35 upward and
the massless slice were recomputed with the cap. The reuse was gated by
recomputing il34 *with* the cap and requiring byte-identity against v1 (it
passed for both the atm and noatm passes). `provenance.json` →
`impact_parameter_cap.shards_without_cap_flag` lists the reused shards, and
`assemble_release.py` refuses to mix two different explicit cap values.

Confirmed against v1: every quantity in every group is bit-identical for
λ index < 35, and changes only above it — `n_transit` first at il37 and the
extremeness first at il38, exactly as the table predicts.

The massless (Coulomb) slice uses the exact capped closed form
`dσ/dq|capped = dσ/dq|uncapped × retained(r)`, `r = b_max(q)/b_constrained_max`,
`retained(r) = 1 − (2/π)(√(r²−1)/r² + arctan√(r²−1))` for `r > 1`, else 1.

## Status codes (`status` datasets)

| code | meaning |
|---|---|
| 0 | Monte-Carlo extremeness computed (0.2 ≤ μ ≤ 40) |
| 1 | cell raised an exception; stored values are NaN |
| 2 | μ < 0.2 shortcut — p is exactly 0 |
| 3 | μ > 40 shortcut — p is exactly 1 |
| 4 | spectrum has no support (μ = 0) |

Codes 2/3/4 are exact deterministic outcomes, not failures. Status is per-mode
(μ depends on each mode's efficiency); a cell-level failure marks all three
modes with code 1.

## Known issues

- **61,116 status-1 cells** (2,607 atm + 58,509 noatm; ~1.3% of the noatm cube)
  in the λ ≈ 8–45 µm shards at the heaviest masses / strongest couplings: the
  direct-path cross-section interpolant underflows its tabulation floor and the
  cell is recorded as NaN + status 1. Every such cell lies ≥ 55 grid steps from
  any 95% exclusion contour (deep in the non-excluded p≈0 corner), so the
  physics results are unaffected. Historical scans hit the same failure but
  recorded silent zeros; v1 labels them honestly. A lower-edge guard
  (dσ/dq̃ → 0) is a candidate v2 fix.
- In the Monte-Carlo band (0.2 < μ < 40) the extremeness p carries toy noise
  (n_mc = 10⁴) and the *historical* caches' values were additionally
  scheduling-dependent; this release's values are order-independent and
  bit-reproducible (see PerMuTable in `scripts/build_release.py`).

## Conventions

- Natural units, c = ħ = 1; momenta in GeV/c (threshold q_th = 100 GeV = 0.1 TeV/c).
- Lengths in metres; `luhdm.units.conv_m2pGeV` converts m → GeV⁻¹.
- Root attributes carry every fiducial (seed 20260702, exposure, Q_THRESH,
  R_EFF, F_X, RHO_DM, N_NEUTRONS, MC fidelity `fid_*`, git commit).
- Cubes are float32 (p granularity is MC-limited at 1/n_mc = 1e-4, three orders
  above float32 spacing near the 0.95 contour); float64 shards are archived as
  the verification source.

## Regeneration

```bash
# 1. cube shards (per pass; ~2 days for atm on an 80-core node, ~1 h noatm)
python scripts/build_release.py --pass atm   --shard-dir release_shards/atm   --workers N
python scripts/build_release.py --pass noatm --shard-dir release_shards/noatm --workers N
python scripts/build_release.py --pass halo  --shard-dir release_shards/halo
# (campaign driver with resume + sentinels: scripts/release_campaign_driver.sh)

# 2. assemble the HDF5 (writes the .h5, provenance.json, SHA256SUMS)
python scripts/assemble_release.py --atm-dir release_shards/atm \
    --noatm-dir release_shards/noatm --halo-dir release_shards/halo

# 3. verify against the historical caches / spot recomputes
python scripts/verify_release.py --shard-dir-atm release_shards/atm \
    --shard-dir-noatm release_shards/noatm
```

## Citation

Placeholder — citation and DOI will be added when the release is published on
Zenodo alongside the paper.

## Changelog

- **v2** (`v2.0-bcap10cm`) — adds the 10 cm impact-parameter cap
  `b_constrained_max` to the cross section and the transit reach (see **The
  impact-parameter cap**). Identical to v1 for λ index < 35. New root attribute
  `b_constrained_max_m`; new provenance block `impact_parameter_cap`.
- **v1** — initial release.
