# Data release: First Search for Ultraheavy Dark Matter Using a Magnetically Levitated Particle

The complete limit-setting calculation behind the paper, as two self-describing
HDF5 files. For every point of a grid over sensor mode, coupling, dark-matter
mass and mediator range each file stores the expected signal, the expected
number of in-reach transits, and the optimum-interval extremeness from which the
exclusion at any confidence level is a level set. You can re-derive the published
limit, quote it at a different confidence level, compare against it, or
reinterpret it, without rerunning any of the physics and without installing any
of our software.

|  |  |
|---|---|
| **Paper** | D. G. Uitenbroek, D. W. P. Amaral, J. Qin, J. Langendorff, A. Gingerich, T. H. Oosterkamp and C. D. Tunnell, *First Search for Ultraheavy Dark Matter Using a Magnetically Levitated Particle*. arXiv identifier and journal reference to be assigned. |
| **Release version** | `v10`, `version_tag` = `v10.0-night-m0p356mg-q1TeV-nocap-wmargnight-a18iso-mudex0p002-vE245` |
| **Dataset DOI** | to be assigned |
| **Date** | 2026-08-18 |
| **License** | Data: [CC BY 4.0](LICENSE). Code, including `luhdm_release.py`: GPL-3.0-or-later. See [§13](#13-license-and-contact). |
| **Code repository** | <https://github.com/PolonaiseExperiment/luhdm> |
| **Contact** | Dorian W. P. Amaral, <damaral@ifae.es>, or open an issue on the code repository |
| **Requirements** | Python with `numpy` and `h5py`. Nothing else. |

> **Trust the file over this text.** Every version tag, axis length, event count
> and physics constant quoted below is also stored inside the HDF5 files and is
> read back from them. `python luhdm_release.py luhdm_datarelease_v10_A_f1_atm.h5`
> prints all of them; [§1](#1-quickstart) shows how to read them with five lines
> of `h5py`. Every example in this document was executed against the released
> files and its output pasted verbatim. All of them assume you are working in the
> directory that holds the release files; otherwise give a full path.

**Contents.** [1 Quickstart](#1-quickstart) &middot;
[2 Files](#2-files-in-this-release) &middot;
[3 What the data is](#3-what-the-data-is) &middot;
[4 Data dictionary](#4-file-layout-and-data-dictionary) &middot;
[5 Conventions](#5-conventions-you-have-to-get-right) &middot;
[6 Worked example](#6-worked-example-the-published-limit) &middot;
[7 Standalone reader](#7-the-standalone-reader) &middot;
[8 Intended use](#8-intended-use) &middot;
[9 Known limitations](#9-known-limitations) &middot;
[10 Integrity and provenance](#10-integrity-provenance-and-environment) &middot;
[11 Versions](#11-versions) &middot;
[12 How to cite](#12-how-to-cite) &middot;
[13 License and contact](#13-license-and-contact) &middot;
[14 Glossary](#14-glossary)

---

## 1. Quickstart

Download `luhdm_datarelease_v10_A_f1_atm.h5`, then:

```python
import h5py, numpy as np

with h5py.File("luhdm_datarelease_v10_A_f1_atm.h5", "r") as f:
    ext = f["results/extremeness"]
    print(f.attrs["version_tag"], ext.shape, list(ext.attrs["DIMENSION_LABELS"]))
    p = ext[0, 0, 0]                     # the file's one hypothesis; mode 1
    print("excluded at 95% CL:", int((p >= 0.95).sum()), "of", p.size, "grid points")
```

```
v10.0-night-m0p356mg-q1TeV-nocap-wmargnight-a18iso-mudex0p002-vE245 (1, 1, 3, 44, 119, 5) ['f_dm', 'atmosphere', 'mode', 'alpha_n', 'mass_gev', 'lambda_m']
excluded at 95% CL: 8117 of 26180 grid points
```

That is the whole interface: a dense array, its axis names attached, and one
comparison. Exclusion at confidence `C` is the level set `extremeness >= C`.

Two things that snippet does not do, and that real code must:

* the leading `0, 0` are the `f_dm` and `atmosphere` axes, which are length 1 in
  each file because **each file is one hypothesis** ([§5.1](#51-two-files-one-hypothesis-each));
* the excluded region has a **right-hand edge in mass** that is not in the
  stored surfaces. It is the flux cut `m_cut`, an attribute, and you have to
  apply it yourself ([§5.4](#54-the-mass-cut-m_cut-and-how-to-apply-it)).

From the shell, if you have the HDF5 command-line tools (they ship with the
HDF5 C library, not with `pip install h5py`):

```console
$ h5ls -r luhdm_datarelease_v10_A_f1_atm.h5 | head
```

```
/                        Group
/axes                    Group
/axes/alpha_halo_n       Dataset {64}
/axes/alpha_n            Dataset {44}
/axes/atmosphere         Dataset {1}
/axes/f_dm               Dataset {1}
/axes/lambda_m           Dataset {5}
/axes/m_phi_gev          Dataset {5}
/axes/mass_gev           Dataset {119}
/axes/mass_halo_gev      Dataset {64}
```

The files are plain HDF5 with dimension scales and per-dataset `units` and
`description` attributes, so `h5py`, `h5dump`, MATLAB, Julia, IDL and xarray
(through h5netcdf) all open them. There is no collaboration software to install.

---

## 2. Files in this release

| file | size | what it is |
|---|---|---|
| `luhdm_datarelease_v10_A_f1_atm.h5` | 0.56 MB | **Dataset A.** `f_DM = 1`, atmospheric propagation **on**. The plane the paper's `alpha_n` limits are quoted on. |
| `luhdm_datarelease_v10_B_f0p1_noatm.h5` | 0.59 MB | **Dataset B.** `f_DM = 0.1`, atmospheric propagation **off**. The plane the composite cross-section benchmark is quoted on. |
| `luhdm_contours_v10_A_f1_atm.json` | 199 kB | **Refined contours for dataset A.** The 95% boundary of each of A's four mode-1 exclusion surfaces, root-found rather than read off the grid. This is the boundary the paper draws. See [§5.5](#55-the-sidecar-files). |
| `luhdm_contours_v10_B_f0p1_noatm.json` | 132 kB | The same, for dataset B's four mode-1 surfaces. |
| `luhdm_lambda_scan_v10.npz` | 34 kB | **Mediator-range sidecar to dataset A.** For sensor mode 1, the (coupling × range) plane on a 54-point range axis from 0.1 µm to 2 m, at that mode's best dark-matter mass — the range resolution the cube's four finite slices cannot give. See [§5.5](#55-the-sidecar-files). |
| `luhdm_release.py` | 53 kB | **Optional** single-file reader. `numpy` and `h5py` only, `pandas` optional. Copy it next to the HDF5 files and import it. Described in [§7](#7-the-standalone-reader). |
| `README.md` | 125 kB | This document. |
| `SHA256SUMS` | 1 kB | SHA-256 digest of every file in the release. See [§10](#10-integrity-provenance-and-environment). |
| `provenance_luhdm_datarelease_v10_A_f1_atm.json` | 37 kB | Build-side record for dataset A ([§10](#10-integrity-provenance-and-environment)): assembly command line, per-shard run records, per-input digests, impact-parameter-cap block. Not needed to use the data; the same information is in the file's own attributes. |
| `provenance_luhdm_datarelease_v10_B_f0p1_noatm.json` | 37 kB | The same, for dataset B. |
| `CITATION.cff` | 3.5 kB | Machine-readable citation metadata. See [§12](#12-how-to-cite). |
| `exclusion_massless_mode1.png` | 63 kB | The figure produced by [§6](#6-worked-example-the-published-limit), for reference. Regenerated by section 7 of `notebooks/05_using_the_data_release.ipynb`. |
| `LICENSE` | 19 kB | CC BY 4.0, the licence of the **data**. `luhdm_release.py` is code and is GPL-3.0-or-later instead. See [§13](#13-license-and-contact). |

**The files you need are the one HDF5 that carries your hypothesis and this
README.** Add `luhdm_release.py` if you want value-based selection instead of
integer indices. Everything else is provenance and convenience — the three
sidecars included, since everything in them is derived from the cubes. One
caveat worth having early: the exclusion boundary the paper draws is the
root-found one in `luhdm_contours_v10_A_f1_atm.json`, not the grid crossing you
get by contouring the cube. The two agree to within one coupling grid cell, and
[§5.5](#55-the-sidecar-files) says exactly where they part company.

The two HDF5 files have the **same schema, the same axes and the same detector
inputs**; they differ only in which single `(f_dm, atmosphere)` hypothesis they
carry, and in the numbers that follow from it. Code written against one runs
against the other unchanged.

Notebooks that reproduce the figures in the paper and its Supplemental Material
from these files live in
[`notebooks/`](https://github.com/PolonaiseExperiment/luhdm/tree/main/notebooks)
in the code repository.

---

## 3. What the data is

A search for **ultraheavy dark matter** with a **levitated micro-sensor**. A
magnetically levitated microsphere (0.356 mg, effective radius 0.26 mm,
N = 1.07 × 10²⁰ neutrons) held in a superconducting trap is monitored for sudden
momentum impulses. A dark-matter particle of mass `m_DM` flying past the sphere
with impact parameter `b` transfers momentum through a new Yukawa-type
interaction with the sphere's neutrons, of strength `alpha_n` per neutron and
range `lambda` (mediator mass `m_phi = 1/lambda` in natural units; `lambda → ∞`
is the massless, Coulomb-like limit). Because the dark-matter number density
falls as `1/m_DM` while the momentum kick grows with it, a single sphere is
sensitive to masses far above the WIMP range, here 10⁵ to 1.22 × 10¹⁹ GeV.
Strongly coupled candidates also lose energy crossing the atmosphere and
overburden before reaching the detector, which both attenuates and reshapes the
arrival-velocity distribution. This release carries the analysis **with** that
propagation (dataset A) and **without** it (dataset B), at four mediator ranges
— 2 mm, 200 µm, 20 µm and a 200 m validation slice — together with the massless
limit.

### The selection it encodes

The release describes the **night selection**. The dark-matter dataset was
recorded between 22 December 2025 and 21 January 2026. Only night-time data are
analysed, 19:00 to 07:00 local time, when nobody is expected to be in the
laboratory; 19 and 20 January are excluded, when work on the still suspension
and a deliberate vibration test disturbed the apparatus, as are the periods when
the calibration drive was on. What remains is a live time of
`T_obs = 790 778 s` (219.66 h, 9.15 days, 31% of the 30.0-day run), recorded in
`attrs['t_exposure_s']` and in `detector/exposure_s`.

Three translational eigenmodes of the trapped sphere, at 51.2365, 59.4663 and
94.86 Hz, are demodulated from the same readout stream and analysed
independently. They are indexed 1, 2, 3 by ascending frequency. **The paper
reports the search on mode 1**; modes 2 and 3 are carried through the identical
chain as cross-checks. All three are in both files.

### The halo is evaluated in the laboratory frame

**This is the change that defines `v10`, and it moves every number in the
release.** The standard halo model is an isotropic truncated Maxwellian in the
*Galactic* rest frame, with `v_0` = 220 km/s and an escape speed
`v_esc` = 544 km/s. The detector does not sit in that frame: the Earth moves
through the halo, so the arrival-speed distribution seen in the laboratory is
the Maxwellian boosted by the observer's velocity and integrated over arrival
direction (Lewin & Smith 1996). This release evaluates the halo in that
laboratory frame, at

```
attrs['v_earth_km_s'] = 245.0 km/s
```

the value used by Monteiro (2020) and Tseng (2025) — the two levitated-sensor
results this analysis overlays — so the halo here is entirely in their
convention. Every release up to and including `v9.1` carried the Galactic
rest-frame distribution and no such attribute; **absence of `v_earth_km_s`
means 0, the rest frame**, and that is how a reader should treat any earlier
cube.

Two consequences run through everything below:

* **the halo's support now ends at `v_esc + v_E` = 789 km/s**, not at
  `v_esc` = 544 km/s. A particle sitting at the Galactic escape speed and met
  head-on by the Earth arrives at the sum, so that — not `v_esc` — is the
  ceiling of every speed integral, every speed grid and the rate integral's
  upper limit. It is what sets the kinematic wall of the next subsection;
* **the flux is faster on average.** The flux-weighted mean arrival speed
  `<v>`, the first moment that normalises the transit rate, is 338 173 m/s
  against 245 972 m/s in the rest frame, a factor 1.3748. `<v>` enters the
  flux cut `m_cut` linearly ([§5.4](#54-the-mass-cut-m_cut-and-how-to-apply-it)),
  which moves right by the same factor.

The boost is a **stated convention, not a fit**, and it is fixed for a whole
cube: nothing you read out of a released file needs to know about it. It
matters the moment you **recompute** something and compare it against the file,
exactly as the projection kernel does ([§5.5](#55-the-sidecar-files)) — a
spectrum recomputed in the wrong frame disagrees with the cube for a reason
that has nothing to do with the physics being checked. Read
`attrs['v_earth_km_s']` and thread it through — with the analysis code that
means setting the `LUHDM_V_EARTH` environment variable to this value before
importing `luhdm`, which the refiner and the release verifier both check and
hard-stop on. The `/halo` diagnostic maps
([§4.4](#44-halo)) and the `/reference_curves` showcase spectra
([§4.5](#45-reference_curves)) are in the same frame as `/results`.

### The analysis window starts at 1 TeV

The momentum threshold of the analysis is `attrs['q_thresh_gev']` = **1000 GeV**
(1 TeV/c). It is the lower endpoint of the signal spectrum's support, so an
impulse below it contributes nothing to the limit. It is a stated momentum
threshold and nothing more: it is not placed at any particular point of the
measured efficiency, and it is not a point where the efficiency has saturated.
At 1 TeV the measured efficiency is 0.16 for mode 1, 0.58 for mode 2 and 0.00
for mode 3, whose turn-on lies far above the edge
([§7.1](#71-detector-inputs)). The efficiency is folded into `mu` point by point
across the window, so where the window opens and where each mode becomes
efficient are two separate statements, and both travel with the files.

The stored candidate lists are the upstream **100 GeV** reconstruction
selection, unchanged from earlier builds and identified by the input digests
in [§10](#10-integrity-provenance-and-environment); the 1 TeV window is applied
by the limit calculation, not by pre-cutting the lists. In practice this only
matters for mode 2:

```python
import h5py, numpy as np

with h5py.File("luhdm_datarelease_v10_A_f1_atm.h5", "r") as f:
    q_thresh = float(f.attrs["q_thresh_gev"])
    print("analysis window starts at q_thresh_gev =", q_thresh, "GeV")
    for m in (1, 2, 3):
        ev = f[f"detector/events_mode{m}"][:]
        below = np.sort(ev[ev < q_thresh])
        print(f"  mode {m}: {ev.size:3d} stored, {int((ev >= q_thresh).sum()):3d} inside the window"
              f"   below: {np.round(below, 1).tolist()}")
```

```
analysis window starts at q_thresh_gev = 1000.0 GeV
  mode 1:   8 stored,   8 inside the window   below: []
  mode 2:  26 stored,  24 inside the window   below: [554.2, 945.8]
  mode 3: 126 stored, 126 inside the window   below: []
```

The threshold also fixes the **left edge of the excluded region**. A halo
particle cannot arrive faster than the top of the distribution's support, which
in the laboratory frame is `v_max = v_esc + v_E` = 789 km/s
([above](#the-halo-is-evaluated-in-the-laboratory-frame)), so a mass below
`m_wall = q_thresh / v_max` = 3.80 × 10⁵ GeV cannot deliver a 1 TeV impulse at
any coupling. That is a hard kinematic wall, not a sensitivity statement: no
exclusion exists to its left, and the first excluded mass on the grid is
5.20 × 10⁵ GeV. The wall moves with `1/v_max`, so it sits lower here than in
the Galactic-rest-frame releases, where the same threshold gave
5.51 × 10⁵ GeV.

The larger lists (66, 99 and 443 entries) that ship alongside as
`detector/all_blips_mode{1,2,3}` are **not** the night pre-selection of the
candidates: they are every up-crossing above the 100 GeV reconstruction
threshold over the whole unvetoed run, about 469.7 h, so `detector/exposure_s`
does **not** normalise them. They carry no per-blip time, segment or drive-state
metadata, so the night selection cannot be re-derived or re-cut from them; they
ship for context, to show the scale of the raw transient population the
selection acts on. **They are also in different units** — eV, against GeV for
the candidates: see [§4.3](#43-detector).

### The interaction model, in one paragraph

The differential cross section for a flyby is built from the impulse delivered
along a straight-line trajectory. The impact-parameter integral has **one**
cutoff, the inner one: the sphere's own effective radius. No trajectory
approaches closer than `R_eff` (`attrs['r_eff_m']` = 2.6 × 10⁻⁴ m), so in the
massless limit the impulse saturates at

```
q_max = 2 alpha_n / (v R_eff)
```

and `dsigma/dq` vanishes identically above it. This is the massless limit of the
finite-range cutoff the Yukawa branch already applies, and it removes the
contribution of arbitrarily close approaches.

There is **no outer cutoff**: `attrs['b_constrained_max_m']` is `NaN`, meaning
uncapped, and the large-`b` wedge is integrated in full. Earlier versions of
this analysis truncated the integral at the 10 cm scale of the cryogenic
hardware around the trap; that truncation has been removed, and the hardware
scale now enters only *after* the calculation, as the aperture of the flux cut
described in [§5.4](#54-the-mass-cut-m_cut-and-how-to-apply-it). Anything read
off these surfaces at a mass beyond that cut is not a limit.

---

## 4. File layout and data dictionary

Every dataset under `/results` and `/halo` carries HDF5 **dimension scales**, so
its axes are self-identifying: `ds.attrs['DIMENSION_LABELS']` names them in
order and each is attached to the matching `/axes/<name>` dataset. Read the axis
order from there rather than assuming it. Every dataset also carries `units` and
`description` attributes.

Symbolic lengths, with this release's values in the last column. They are the
same in both files:

| symbol | meaning | value |
|---|---|---|
| `n_f` | dark-matter fraction hypotheses **in one file** | 1 |
| `n_atm` | atmosphere hypotheses **in one file** | 1 |
| `n_mode` | sensor modes | 3 |
| `n_alpha` | coupling grid points | 44 |
| `n_mass` | dark-matter mass grid points | 119 |
| `n_lam` | mediator ranges, **finite plus 1 massless sentinel** | 5 (4 + 1) |
| `n_halo` | coupling and mass points of the halo diagnostic maps | 64 |

`n_f` and `n_atm` are 1 rather than absent so that the array layout, the
dimension scales and every index expression are identical to the multi-plane
layout of earlier releases. Code that resolves indices from the axes
([§5.1](#51-two-files-one-hypothesis-each)) ports over unchanged.

### 4.1 `/axes`

The coordinate arrays. No missing-value codes: every entry is a valid grid
point, and the only non-finite value anywhere is the deliberate `inf` sentinel
in `lambda_m`.

| dataset | shape | dtype | units | meaning |
|---|---|---|---|---|
| `f_dm` | (`n_f`,) | f8 | 1 | fraction of the local dark-matter density carried by this species. A **pure flux normalisation**: `mu` and `n_transit` are exactly linear in it. `[1.0]` in file A, `[0.1]` in file B. |
| `atmosphere` | (`n_atm`,) | i1 | bool | `1` = propagation through atmosphere and overburden applied; `0` = bare halo flux. `[1]` in file A, `[0]` in file B. |
| `mode` | (`n_mode`,) | u1 | 1 | sensor mode label (1, 2, 3), by ascending eigenfrequency (51.2365, 59.4663, 94.86 Hz). Modes differ in threshold and efficiency, hence in their event lists. |
| `alpha_n` | (`n_alpha`,) | f8 | 1 | per-neutron coupling, log-spaced 10⁻¹⁰ to 1, 0.2326 dex per step. This is the parameter the limit is set on. |
| `mass_gev` | (`n_mass`,) | f8 | GeV | dark-matter mass, log-spaced 10⁵ to 1.22 × 10¹⁹ (the Planck mass), 0.1194 dex per step. |
| `lambda_m` | (`n_lam`,) | f8 | m | mediator range, 2 × 10⁻⁵ m to 2 × 10² m: finite values **ascending**, then `inf` last. See below. |
| `m_phi_gev` | (`n_lam`,) | f8 | GeV | mediator mass `1/lambda` in natural units, parallel to `lambda_m`; exactly `0.0` at the `inf` entry. |
| `alpha_halo_n` | (`n_halo`,) | f8 | 1 | coupling axis of the `/halo` maps, extending down to 2 × 10⁻¹¹. |
| `mass_halo_gev` | (`n_halo`,) | f8 | GeV | mass axis of the `/halo` maps. |

**The massless slice.** The last element of `lambda_m` is `inf`, with
`m_phi_gev = 0.0`: the analytic, Coulomb-like limit of a massless mediator. The
number of finite entries is the `n_finite` attribute of `axes/lambda_m` (4
here), so the finite part is `lambda_m[:n_finite]` and the massless index is the
one where `~np.isfinite(lambda_m)`. Never assume it is `-1` by arithmetic on a
hard-coded length. Read `n_finite`.

**Four finite ranges, three of them physics.** The finite axis is 20 µm, 200 µm,
2 mm and 200 m. The first three are the physics slices: a decade apart, they
bracket the sphere's 0.26 mm effective radius. **200 m is a validation slice**,
not a physics point. It is there to check that the finite-range branch of the
cross section converges onto the analytic massless branch when the range is far
larger than every other scale in the problem, which it does
([§9](#9-known-limitations)). Quote it as a numerical cross-check; do not quote
it as a physics result at `lambda` = 200 m.

**Named ranges, and a trap in them.** `axes/lambda_m` carries a `tags_json`
attribute mapping short names to exact axis values. It is inherited from the
parent scan and lists **every tag the scan knows**, including ranges this file
does not carry, so a tag is not a promise that the slice is here:

```python
import h5py, json, numpy as np

with h5py.File("luhdm_datarelease_v10_A_f1_atm.h5", "r") as f:
    lam  = f["axes/lambda_m"][:]
    tags = json.loads(f["axes/lambda_m"].attrs["tags_json"])
print("axis:", lam)
print("tags on the axis:    ", {k: v for k, v in tags.items() if np.isin(v, lam)})
print("tags NOT on the axis:", {k: v for k, v in tags.items() if not np.isin(v, lam)})
```

```
axis: [2.e-05 2.e-04 2.e-03 2.e+02    inf]
tags on the axis:     {'2mm': 0.002, '200um': 0.0002, '20um': 2e-05}
tags NOT on the axis: {'2m': 2.0, '20cm': 0.2, '2cm': 0.02, '10um': 1e-05, '2um': 2e-06}
```

So three of the eight tags resolve, `'massless'` is the reader-side name for the
`inf` sentinel rather than a stored tag, and the 200 m validation slice has no
tag at all — select it by value. Asking the standalone reader for an absent tag
raises and lists what the axis actually holds; it never silently snaps to a
neighbour.

### 4.2 `/results`

The analysis cube.

| dataset | shape | dtype | units | meaning |
|---|---|---|---|---|
| `extremeness` | (`n_f`, `n_atm`, `n_mode`, `n_alpha`, `n_mass`, `n_lam`) | f4 | 1 | optimum-interval extremeness: the probability that a background-free pseudo-experiment under this hypothesis looks *less* extreme than the data. **Exclusion at confidence `C` is the level set `extremeness >= C`**, subject to the mass cut of [§5.4](#54-the-mass-cut-m_cut-and-how-to-apply-it). `NaN` where `status == 1`. |
| `mu` | same | f4 | counts | expected detected signal events, efficiency folded in. `NaN` where `status == 1`. Exactly linear in `f_DM`. |
| `status` | same | u1 | enum | how the cell was obtained. See [§4.6](#46-status-codes-and-the-nan-policy). |
| `n_transit` | (`n_f`, `n_atm`, `n_alpha`, `n_mass`, `n_lam`) | f4 | counts | expected number of dark-matter transits within threshold reach. **No `mode` axis**: the flyby rate does not depend on which sensor mode you read out. Clipped at `>= 0` (`clipped_nonnegative` attribute); exactly linear in `f_DM`. `NaN` at cells where the calculation raised. |

`NaN` is the missing-value code throughout, and it is exactly coincident with
`status == 1`.

`n_transit` counts transits **within threshold reach**, that is inside
`b_max(q_thresh)`, which the uncapped cross section lets grow without bound at
strong coupling. It is not the transit count inside the 10 cm aperture that
defines `m_cut`; that one is the closed-form `N(m)` of
[§5.4](#54-the-mass-cut-m_cut-and-how-to-apply-it), which you compute from
attributes rather than read from a dataset.

### 4.3 `/detector`

The analysis inputs. Byte-identical in the two files.

| dataset | shape | dtype | units | meaning |
|---|---|---|---|---|
| `exposure_s` | scalar | f8 | s | total live time, 790 778 s. |
| `events_mode{1,2,3}` | (8,) (26,) (126,) | f8 | **GeV** | **the analysis event lists**: momentum kicks surviving the full selection above the 100 GeV reconstruction threshold. The limit is set on these, inside the 1 TeV analysis window ([§3](#the-analysis-window-starts-at-1-tev)). |
| `all_blips_mode{1,2,3}` | (66,) (99,) (443,) | f8 | **eV** | every reconstructed up-crossing above the 100 GeV threshold over the **whole unvetoed run** (~469.7 h), not the night pre-selection. `exposure_s` does not apply to them and they carry no time or drive-state metadata. Context only. |
| `q_gev_{1,2,3}` | (400,) | f8 | GeV | momentum grid of the measured efficiency curves. |
| `eff_{1,2,3}_df{2,3}` | (400,) | f8 | 1 | measured detection efficiency ε(q) per mode, averaged over the phase of the mode oscillation at which the impulse arrives, for the two degrees-of-freedom hypotheses of the efficiency fit. The analysis used `df` = `attrs['df']` (3). See [§7.1](#71-detector-inputs). |

> **The two impulse lists are in different units.** `events_mode{n}` is in GeV
> and `all_blips_mode{n}` is in eV, a factor of 1e9 between two datasets in the
> same group. Plotted on one axis without conversion they look plausible and are
> wrong by nine orders of magnitude. Divide the blip momenta by 1e9 before
> comparing. Each dataset carries its own `units` attribute; read it.

```python
import h5py, numpy as np

with h5py.File("luhdm_datarelease_v10_A_f1_atm.h5", "r") as f:
    ev = f["detector/events_mode1"][:]          # GeV
    bl = f["detector/all_blips_mode1"][:]       # eV
    print("events_mode1   units =", f["detector/events_mode1"].attrs["units"],
          f"  range {ev.min():.1f} .. {ev.max():.1f}")
    print("all_blips_mode1 units =", f["detector/all_blips_mode1"].attrs["units"],
          f"  range {bl.min():.3e} .. {bl.max():.3e}")

blips_gev = bl / 1e9                            # eV -> GeV
print("after conversion the events are a subset of the blips:",
      all(np.isclose(blips_gev, e).any() for e in ev))
```

```
events_mode1   units = GeV   range 1520.7 .. 12790.7
all_blips_mode1 units = eV   range 8.610e+11 .. 1.265e+14
after conversion the events are a subset of the blips: True
```

Event-list lengths are data, not schema: they change with the selection.
[§7.1](#71-detector-inputs) works through reading these.

### 4.4 `/halo`

Flux diagnostics, optional.

| dataset | shape | dtype | units | meaning |
|---|---|---|---|---|
| `n_transit` | (`n_halo`, `n_halo`, `n_lam`) | f4 | counts | unattenuated-halo expected transits on the coarser diagnostic grid. |
| `bmax` | (`n_halo`, `n_halo`, `n_lam`) | f4 | m | flux-averaged threshold reach √⟨b²⟩, how far out a transit can still push the sensor over threshold. |

These are an **independently sampled, coarser** map, with their own
`alpha_halo_n` and `mass_halo_gev` axes, used for intuition and figures. They
are stored at the **build-side baseline `f_DM`** (`attrs['f_dm_default']` = 0.1)
with no `f_dm` axis, in both files; multiply by `f_DM / 0.1` for another
fraction. **For anything quantitative use `/results`.**

They are computed in the **same laboratory-frame halo** as `/results`
(`attrs['v_earth_km_s']` = 245 km/s,
[§3](#the-halo-is-evaluated-in-the-laboratory-frame)), so the two are directly
comparable; the assembly step refuses to write a cube whose halo pass and
results pass disagree on the frame. Both maps therefore differ from their
Galactic-rest-frame counterparts in `v9.1` and earlier.

With the cap removed, `bmax` at strong coupling and long range runs well past
the 10 cm hardware scale — that is the physical reach of a threshold-crossing
flyby, and it is exactly why the mass cut of
[§5.4](#54-the-mass-cut-m_cut-and-how-to-apply-it) exists as a separate,
explicit statement rather than as a truncation buried in the integral.

### 4.5 `/reference_curves`

Showcase spectra for figures, all at the single point `m_DM = 10⁸ GeV`,
`alpha_n = 10⁻³`. Not needed to use the limits.

These are **standalone curves, not slices of the cube**, and their `<tag>` set is
its own: eight showcase ranges from 2 m down to 2 µm, fixed when the curves were
computed. It does not track the `lambda_m` axis of this file
([§4.1](#41-axes)) and is not indexed by it; read the dataset names to see what
is here. They are the widest view of the range dependence that travels with
these files.

| dataset | shape | units | meaning |
|---|---|---|---|
| `v` | (500,) | c | arrival-speed grid, shared by every `fv_*`. |
| `fv_<tag>` | (500,) | (v/c)⁻¹ | attenuated arrival-speed distribution for each of the eight showcase ranges (`2m`, `20cm`, `2cm`, `2mm`, `200um`, `20um`, `10um`, `2um`). Each carries a `survival_fraction` attribute, 0.80 to 0.86 across the eight. |
| `fv_shm` | (500,) | (v/c)⁻¹ | the unattenuated standard-halo-model distribution, `survival_fraction` 1.0. |
| `q_gev` | (160,) | GeV | momentum-kick grid, shared by every `drdq_*`. |
| `drdq_<tag>`, `drdq_massless` | (160,) | s⁻¹ GeV⁻¹ | raw differential rate dR/dq with no efficiency applied, one per showcase range plus the massless limit. |

There are nine `drdq_*` curves (eight showcase ranges plus `massless`) but only
eight attenuated `fv_*` curves. That asymmetry is deliberate and is recorded in
the `description` attributes: attenuation is computed per finite range, so there
is no massless arrival distribution, and **every `drdq_*` curve, including
`drdq_massless`, is drawn with the 200 µm arrival distribution** so that the
curves differ only through the cross section. `fv_shm` completes the set on the
`fv_*` side as the unattenuated reference.

These curves are computed at `attrs['reference_curves_fidelity']` =
`production` with the uncapped cross section, so they are not the curves the
capped scheme produced. No efficiency is applied to them, so nothing on the
detector side moves them; what does move them is the halo. They are drawn in
the **laboratory frame** of
[§3](#the-halo-is-evaluated-in-the-laboratory-frame), so the `v` grid runs to
`v_esc + v_E` = 789 km/s (2.632 × 10⁻³ c) rather than to 544 km/s, the whole
`fv_*` family sits at higher speed than its `v9.1` counterpart, and the
survival fractions rise with it, from 0.53–0.76 there to 0.80–0.86 here.

### 4.6 Status codes and the NaN policy

`/results/status` records how each cell was obtained. Its `description`
attribute is the authoritative short form; the long form:

| code | meaning | `extremeness` |
|---|---|---|
| `0` | the optimum-interval Monte Carlo ran | MC value in (0, 1) |
| `1` | **the cell raised an exception** | `NaN`, and `mu` and `n_transit` are `NaN` too |
| `2` | `mu` below the MC floor (0.2), nothing to expect | exactly `0.0` |
| `3` | `mu` above the MC cap (85, `attrs['fid_mu_cap']`), asserted excluded | exactly `1.0` |
| `4` | the spectrum has no support, `mu == 0` | exactly `0.0` |

**Codes 2, 3 and 4 are deterministic shortcuts, not failures.** They are how the
scan avoids Monte Carlo where the answer is taken to be known, and they are the
majority of each cube. Code 3 in particular is an *assertion* rather than a
computation; see [§9](#9-known-limitations).

```python
import h5py, numpy as np

for path in ("luhdm_datarelease_v10_A_f1_atm.h5", "luhdm_datarelease_v10_B_f0p1_noatm.h5"):
    with h5py.File(path, "r") as f:
        st  = f["results/status"][:]
        ext = f["results/extremeness"][:]
    codes, counts = np.unique(st, return_counts=True)
    print(path)
    print("  ", {int(c): int(n) for c, n in zip(codes, counts)},
          "of", st.size, "cells")
    print("   NaN extremeness exactly where status==1:",
          np.array_equal(np.isnan(ext), st == 1),
          f"| status-1 cells: {int((st == 1).sum())} "
          f"({100 * (st == 1).mean():.2f}%)")
```

```
luhdm_datarelease_v10_A_f1_atm.h5
   {0: 9926, 1: 75, 2: 36509, 3: 19991, 4: 12039} of 78540 cells
   NaN extremeness exactly where status==1: True | status-1 cells: 75 (0.10%)
luhdm_datarelease_v10_B_f0p1_noatm.h5
   {0: 10805, 1: 75, 2: 40198, 3: 21543, 4: 5919} of 78540 cells
   NaN extremeness exactly where status==1: True | status-1 cells: 75 (0.10%)
```

For file A that is 12.6% code 0, 0.1% code 1, 46.5% code 2, 25.5% code 3 and
15.3% code 4; for file B, 13.8%, 0.1%, 51.2%, 27.4% and 7.5%.

**Code 1 is a failure and needs care.** `extremeness` is `NaN` there, and
`NaN >= 0.95` is `False`, so **a failed cell silently reads as "not excluded"**
in any naive level set. That is the published convention and this release does
not change it, but you should know where those cells are; see
[§9](#9-known-limitations) for how far they sit from any contour.

The failures are a cross-section interpolant underflowing its tabulation floor
at the strongest couplings and heaviest masses in the shortest mediator ranges.
All 75 in each file are in the 20 µm slice, the shortest range carried here; the
2 mm, 200 µm, 200 m and massless slices are free of them in both files. A cell-level failure marks all three modes, so `n_transit`, which
has no mode axis, is `NaN` at exactly the cells where any mode has
`status == 1`.

The standalone reader's `excluded_band()` counts them for you
(`band.n_undefined` per mass) and warns unless you pass `nan_policy='ignore'`.

---

## 5. Conventions you have to get right

Four of them. The fourth has no counterpart in earlier releases and is the one
that changes a plot.

### 5.1 Two files, one hypothesis each

Earlier versions of this release carried one cube with a `f_dm` axis of length 2
and an `atmosphere` axis of length 2, four parallel analyses in one file. This
version ships the **two hypotheses the paper uses**, one per file:

| file | `f_dm` | `atmosphere` | what it is for |
|---|---|---|---|
| `luhdm_datarelease_v10_A_f1_atm.h5` | 1.0 | 1 (attenuated) | this species is all of the dark matter, propagated through the overburden. **The plane the paper's `alpha_n` limits are quoted on.** |
| `luhdm_datarelease_v10_B_f0p1_noatm.h5` | 0.1 | 0 (bare halo) | this species is a tenth of the dark matter, with no overburden. **The plane the composite cross-section benchmark is quoted on.** |

* **`f_dm`** is the fraction of the local dark-matter density carried by this
  species. It enters only as a flux normalisation, so `mu` and `n_transit` scale
  exactly linearly with it. The `extremeness` does not: it is a non-linear
  function of `mu`, so you cannot rescale one file into the other. The coupling
  limits are quoted at `f_DM = 1`, the presentation convention of the optically
  levitated searches; the composite benchmark at `f_DM = 0.1`, the conventional
  subdominant choice that evades self-interaction constraints.
* **`atmosphere`** selects whether the arrival flux has been propagated through
  the atmosphere and overburden. With attenuation ON, strongly coupled
  candidates are slowed or stopped before reaching the sensor, so the exclusion
  can close from above in `alpha_n`. With it OFF there is no such mechanism at
  all; see [§9](#9-known-limitations).

Both axes are still present, length 1, so the layout is unchanged and index
resolution still works. Do it by value, never by position:

```python
import h5py

for path in ("luhdm_datarelease_v10_A_f1_atm.h5", "luhdm_datarelease_v10_B_f0p1_noatm.h5"):
    with h5py.File(path, "r") as f:
        print(f"{path}:  f_dm = {f['axes/f_dm'][:]}, "
              f"atmosphere = {f['axes/atmosphere'][:]}, "
              f"results/mu {f['results/mu'].shape}")
        print(f"    f_dm_default attr = {float(f.attrs['f_dm_default'])} "
              f"(build-side baseline, NOT this file's plane)")
```

```
luhdm_datarelease_v10_A_f1_atm.h5:  f_dm = [1.], atmosphere = [1], results/mu (1, 1, 3, 44, 119, 5)
    f_dm_default attr = 0.1 (build-side baseline, NOT this file's plane)
luhdm_datarelease_v10_B_f0p1_noatm.h5:  f_dm = [0.1], atmosphere = [0], results/mu (1, 1, 3, 44, 119, 5)
    f_dm_default attr = 0.1 (build-side baseline, NOT this file's plane)
```

**`attrs['f_dm_default']` is 0.1 in both files, and in file A that is not a
value on the axis.** It records the build-side baseline the flux was normalised
from, not the plane the file carries. Any reader that falls back to it will ask
file A for a hypothesis it does not have: pass `f_dm=1.0` explicitly when
reading file A ([§7](#7-the-standalone-reader)). The `f_x` attribute, also 0.1
in both files, is the same baseline and carries the same warning. The axis is
the authority.

### 5.2 Selecting a mediator range

`lambda_m` values are exact floats, so `==` works on them; three of the four
finite entries also carry a name ([§4.1](#41-axes)). For anything else, snap to
the nearest point in log space and *check* what you got.

```python
import h5py, numpy as np

with h5py.File("luhdm_datarelease_v10_A_f1_atm.h5", "r") as f:
    lam = f["axes/lambda_m"][:]
    n_finite = int(f["axes/lambda_m"].attrs["n_finite"])

i_massless = int(np.flatnonzero(~np.isfinite(lam))[0])
i_200um    = int(np.flatnonzero(lam == 2e-4)[0])
i_2mm      = int(np.flatnonzero(lam == 2e-3)[0])
i_200m     = int(np.flatnonzero(lam == 200.0)[0])
i_near     = int(np.argmin(np.abs(np.log10(lam[:n_finite]) - np.log10(3.7e-5))))

print(f"n_finite = {n_finite}, axis = {lam}")
print(f"massless -> index {i_massless} (lambda={lam[i_massless]})")
print(f"200 um   -> index {i_200um}")
print(f"2 mm     -> index {i_2mm}")
print(f"200 m    -> index {i_200m}   (validation slice, no tag)")
print(f"3.7e-5 m -> nearest index {i_near}, lambda = {lam[i_near]:.6g} m")
```

```
n_finite = 4, axis = [2.e-05 2.e-04 2.e-03 2.e+02    inf]
massless -> index 4 (lambda=inf)
200 um   -> index 1
2 mm     -> index 2
200 m    -> index 3   (validation slice, no tag)
3.7e-5 m -> nearest index 0, lambda = 2e-05 m
```

The last line is the warning in miniature: the physics part of the axis is three
points wide, so an off-grid request snaps a long way. 3.7 × 10⁻⁵ m lands on
2 × 10⁻⁵ m, nearly a factor of two away. Always print what you got. And note
that a naive nearest-neighbour search over the whole finite axis can land on the
200 m validation slice, which is not a physics point.

### 5.3 The exclusion convention

`extremeness` is the probability that a background-free pseudo-experiment under
a given hypothesis looks *less* extreme than the observed data, computed with
Yellin's optimum-interval method. A grid point is **excluded at confidence `C`**
when

```
extremeness >= C          # C = 0.95 for the published 95% CL limits
```

`attrs['confidence_recommended']` records the level the release is quoted at
(0.95). A two-dimensional exclusion region is that level set, contoured
directly — **inside the mass window of [§5.4](#54-the-mass-cut-m_cut-and-how-to-apply-it)**.

For a **boundary in the coupling** the project quotes the *interpolated*
crossing rather than the last excluded grid point. At each mass, take the run of
`alpha_n` indices with `extremeness >= C` and find where the level is crossed by
linear interpolation in `log10(alpha_n)` between the two bracketing points. If
the excluded run already starts at the first, or ends at the last, scanned
coupling, there is nothing to bracket and the edge **saturates** at that end of
the grid. Such an edge is a property of the scan range, not a measurement.

[§6](#6-worked-example-the-published-limit) is the complete implementation.

That interpolation is what the released files support, and it is what the cubes
are meant to be read with. The release additionally ships the same crossing
*root-found* instead of interpolated, as a sidecar per hypothesis file, and it is
the root-found boundary the paper draws:
[§5.5](#55-the-sidecar-files) says what is in those files and how far the two
boundaries sit apart.

### 5.4 The mass cut `m_cut`, and how to apply it

**This is the one thing in this release that the stored surfaces do not do for
you.** The cross section is uncapped ([§3](#the-interaction-model-in-one-paragraph)),
so `extremeness` keeps reporting exclusion at masses where the halo delivers
essentially no particles anywhere near the apparatus: the number density falls
as `1/m_DM`, and at some point a "signal" is a flyby that never happens within
any distance you would call the laboratory. The stored surfaces are honest about
the statistics of a spectrum and say nothing about whether a transit occurred.
The mass cut is the statement that closes the region from the right.

Require that the halo deliver at least `N_req` particles within a `b_cap` = 10 cm
aperture — the scale of the cryogenic hardware around the trap — during the
exposure:

```
N(m) = f_DM (rho_0 / m) <v> T_obs pi b_cap^2  >=  N_req
```

`N(m)` falls as `1/m`, so the requirement is a **ceiling in mass**:

```
m_cut = f_DM rho_0 <v> T_obs pi b_cap^2 / N_req
```

with `rho_0` = 0.3 GeV/cm³ = 3 × 10⁵ GeV/m³, `<v>` = 338 173 m/s (the
flux-weighted first moment of the truncated standard halo model, the same
convention as the `n_transit` surface), `T_obs` = 790 778 s and `b_cap` = 0.1 m.

`<v>` is **the one input to this cut that the laboratory frame changes**
([§3](#the-halo-is-evaluated-in-the-laboratory-frame)): the same moment of the
Galactic-rest-frame distribution is 245 972 m/s, so the boost multiplies `<v>`,
and with it `m_cut`, by 1.3748. The exact value used, and this derivation, are
in the `m_cut_10cm_*_gev_derivation` attribute beside the cut itself.

> **`N_req = 3` is an assumption of this release, not a measurement.** It is the
> Poisson-zero standard, the same one the limit itself uses: three expected
> transits and none seen is the point at which absence starts to mean something.
> It is stated here, stored in `attrs['m_cut_n_transits_required']`, and it is a
> single number you can change — `m_cut` is exactly inversely proportional to it.

Each file carries its own cut, at its own `f_DM`, in an attribute named for the
aperture and the fraction, with a one-paragraph derivation beside it:

| file | attribute | value |
|---|---|---|
| A (`f_DM = 1`) | `m_cut_10cm_f1_gev` | 8.40 × 10¹⁴ GeV |
| B (`f_DM = 0.1`) | `m_cut_10cm_f0.1_gev` | 8.40 × 10¹³ GeV |

`attrs['m_cut_applied_to_stored_surfaces']` is `False` in both files. That is
deliberate: the surfaces stay a pure statement about the statistics, the cut
stays a stated assumption, and you can move it.

```python
import h5py, numpy as np

with h5py.File("luhdm_datarelease_v10_A_f1_atm.h5", "r") as f:
    a = dict(f.attrs)
m_cut  = float(a["m_cut_10cm_f1_gev"])
b_cap  = float(a["m_cut_b_cap_m"])
n_req  = float(a["m_cut_n_transits_required"])
print(f"m_cut = {m_cut:.4g} GeV   (b_cap = {b_cap} m, N_req = {n_req:g}, "
      f"applied to stored surfaces: {bool(a['m_cut_applied_to_stored_surfaces'])})")

rho_0, v_mean, t_obs = 3.0e5, 338173.0, float(a["t_exposure_s"])   # GeV/m^3, m/s, s
f_dm = 1.0
check = f_dm * rho_0 * v_mean * t_obs * np.pi * b_cap**2 / n_req
print(f"recomputed from the attributes: {check:.4g} GeV")
for n in (1.0, 3.0, 6.8):
    print(f"  N_req = {n:>3}  ->  m_cut = {m_cut * n_req / n:.3g} GeV")
```

```
m_cut = 8.401e+14 GeV   (b_cap = 0.1 m, N_req = 3, applied to stored surfaces: False)
recomputed from the attributes: 8.401e+14 GeV
  N_req = 1.0  ->  m_cut = 2.52e+15 GeV
  N_req = 3.0  ->  m_cut = 8.4e+14 GeV
  N_req = 6.8  ->  m_cut = 3.71e+14 GeV
```

**How to apply it.** One line, wherever you build the mask you plot or quote
from:

```
ok = (extremeness >= C) & (mass_gev <= m_cut)     # mass_gev broadcast along the mass axis
```

That is all. Do not clip the arrays in the file, do not renormalise anything,
and do not carry the cut into `mu` or `n_transit`, which are physical
expectations that remain correct above it. If you contour without the cut, the
massless region runs to the Planck mass and you are reading a statistical
statement about a flux that is not there.
[§6](#6-worked-example-the-published-limit) shows it in place.

The last line of the block above is the historical bridge: `N_req = 6.8` puts
the cut at 3.71 × 10¹⁴ GeV, against 2.70 × 10¹⁴ GeV, the right-hand edge that
earlier, cap-truncated versions of this analysis closed at. The two schemes
agree on where the region ends to within a factor of about three in mass; the
difference between them is a choice of standard, and this release makes that
choice visible instead of burying it in an integration limit.

### 5.5 The sidecar files

Three files ship beside the two cubes: two refined-contour files, one per cube,
and one mediator-range scan. None of them is a new measurement: everything in
them is the cubes' own calculation, run at finer resolution or on a finer axis.
All three are listed in
`SHA256SUMS`, and each carries in its own provenance record the SHA-256 of the
cube it was computed against, so a sidecar can always be matched to the file it
belongs to:

```console
$ grep -E 'contours|lambda_scan' SHA256SUMS | sha256sum -c -
```

```
luhdm_contours_v10_A_f1_atm.json: OK
luhdm_contours_v10_B_f0p1_noatm.json: OK
luhdm_lambda_scan_v10.npz: OK
```

Each also records the halo frame it was refined in, as `provenance.v_earth_km_s`
= 245.0, beside the projection kernel; a sidecar cannot be read as belonging to
a cube built under a different convention
([§3](#the-halo-is-evaluated-in-the-laboratory-frame)).

(`SHA256SUMS` lists the files of the current release only; superseded cubes and
their sidecars are not carried forward.)

**The refined contours.** `luhdm_contours_v10_A_f1_atm.json` and
`luhdm_contours_v10_B_f0p1_noatm.json` carry the 95% boundary itself. Where [§5.3](#53-the-exclusion-convention) interpolates
the level crossing between the two grid cells that bracket it, these files
root-find it, on the same statistic and against the same level:

* at each mass column, the coarse cells bracketing an edge — the floor, and the
  ceiling where the column has one — start a bisection in `log10(alpha_n)` that
  stops at 0.005 dex, about a fiftieth of a coupling cell;
* wherever neighbouring columns' edges jump by more than 0.05 dex, a new mass
  column is inserted at the geometric midpoint and refined the same way,
  recursively, down to 0.01 dex in mass;
* each island end is localized by bisection along the mass axis, on whether
  *anything* is excluded at that mass, since at an end there is no coupling
  crossing left to bracket.

There is one refined surface per exclusion surface the paper draws. **`v10`
refines sensor mode 1 only**, four surfaces per cube, eight in all: massless,
2 mm, 200 µm and 20 µm, at `f_DM` = 1 with atmosphere in file A and at
`f_DM` = 0.1 bare-halo in file B. That is a deliberate narrowing from `v9.1`,
which refined eleven surfaces — mode 1 on file A and all three sensor modes on
file B, in one file per mode. Mode 1 is the mode the paper reports
([§3](#3-what-the-data-is)), so it is the mode whose boundary is drawn, and the
refinement is expensive; modes 2 and 3 are still in both cubes at grid
resolution and are still contoured per
[§5.3](#53-the-exclusion-convention), which is what the refined mode-2 and
mode-3 files of `v9.1` were a convenience on top of. If you want a refined
boundary at another mode, `scripts/refine_contours.py --mode 2` is the same
command with a different argument; the recipe is below.

The massless surface has no right-hand end to bisect — it is still excluded at
the last mass the cube carries ([§5.4](#54-the-mass-cut-m_cut-and-how-to-apply-it))
— so its polyline is **truncated at `m_cut`** instead, and says so:
`m_cut_truncated` is `true`, `m_cut_gev` records the cut it was truncated at, and
the right entry of `tips` is `cut_at_m_cut` with the `N_req` and `b_cap` behind
it. The polyline therefore ends exactly where the claim does. If you disagree
with `N_req = 3` you cannot extend the polyline past its last vertex — the
refinement was never run there — but the cube's own surfaces are uncapped and
[§5.4](#54-the-mass-cut-m_cut-and-how-to-apply-it) shows how to draw a different
cut on them.

The format is `luhdm-refined-contours`, `schema_version` 1. Top level:
`confidence` (0.95), `provenance`, and `surfaces`. Each surface is a set of
parallel arrays, one entry per vertex, ordered in mass:

| key | what it is |
|---|---|
| `mass_gev` | the vertex masses, ascending |
| `floor_alpha_n`, `ceiling_alpha_n` | the two coupling edges. `ceiling_alpha_n` is JSON `null` where the excluded band leaves the top of the coupling axis and there is no ceiling to find (`open_top` flags the same columns) |
| `origin` | `grid` for one of the cube's own mass columns, `inserted` for a column the refinement added, `tip` for an island end |
| `coarse_im` | index into the cube's `axes/mass_gev` for a `grid` vertex, −1 otherwise |
| `floor_bracket_alpha`, `ceiling_bracket_alpha` | the two coarse couplings each edge was found between, so every refined vertex traces back to the grid cells it came from |
| `tips` | the mass bracket at each end: the last mass still excluded and the first that is not. An island still excluded at the end of the cube's mass axis says `open_at_mass_axis_edge` instead; one whose polyline was stopped at the flux cut says `cut_at_m_cut` (with `m_cut_gev`, `n_req` and `b_cap_m` beside it) rather than reporting a bracket it never found |
| `m_cut_gev`, `m_cut_truncated` | the flux cut of [§5.4](#54-the-mass-cut-m_cut-and-how-to-apply-it) read off the cube, and whether this surface's polyline was truncated at it. `true` only where the surface is still excluded at `m_cut` — the massless one — so the polyline stops where the claim stops instead of running to the Planck mass |
| `cell_dex`, `n_grid_columns`, `n_inserted`, `n_open_top_columns`, `n_oracle_calls`, `wall_s` | the cube's coupling cell (0.2326 dex) and the run's counts |
| `widened_columns`, `flags` | the few columns where the starting bracket had to be widened, or where an edge search fell back or needed a rescue scan, keyed by mass. Read them before quoting a single vertex: in file A the massless surface carries one (the `m_cut` truncation), the 2 mm surface five (columns where the ceiling is non-monotone in mass), and the 200 µm and 20 µm surfaces none |

`provenance` holds the cube's path, SHA-256, version tag and git commit; the
fidelity, seed and seed policy read back out of that cube; the tolerances above;
the refiner's git SHA, command line and timings; and the spot-check result. It
also repeats the cube's `projection_kernel` and `v_earth_km_s`, so a sidecar
cannot be read as belonging to a cube built under a different convention.

**The projection kernel.** Both cubes record the convention their projected
`dsigma/dq` was built with, as the root attribute `projection_kernel`, and in
this release it is `isotropic-folded`: the absolute one-axis projection of the
impulse under the isotropic arrival model, with coefficient `8 pi / 3` and shell
fraction `x^3` for the massless slice, and `pi * int beta dbeta / K1(beta)` for a
finite range. Earlier cubes carry no such attribute and were built
`planar-signed` — the signed projection, coefficient `2 pi`, arcsine shell
fraction. This is a stated convention, not a fit, and it is fixed for a whole
cube: nothing you read out of a released file needs to know about it. It matters
the moment you **recompute** something and compare it against the file, because
the two kernels differ by a constant factor in the Coulomb limit, so a spectrum
recomputed under the wrong one disagrees with the cube for a reason that has
nothing to do with the physics being checked. Read the attribute and thread it
through:

```python
import h5py
from luhdm import rate, release             # or your own cross-section code

with h5py.File("luhdm_datarelease_v10_A_f1_atm.h5", "r") as f:
    kernel = f.attrs.get("projection_kernel", "planar-signed")   # pre-flag default

xs = rate.make_xsec(None, projection_kernel=kernel)              # explicit, or
rel = release.open_release("luhdm_datarelease_v10_A_f1_atm.h5")
xs = rel.make_xsec(None)          # the reader threads the file's own kernel and cap
```

`luhdm.release.Release.make_xsec` is the supported route: it fills in both this
attribute and `b_constrained_max_m` from the file that is open, so a recomputed
`dsigma/dq`, `dR/dq` or `mu` is comparable with the stored surface by
construction. `scripts/refine_contours.py` dispatches the same attribute into
every cross section it builds, which is why the spot check above reproduces the
cube bit for bit.

```python
import json

doc = json.load(open("luhdm_contours_v10_A_f1_atm.json"))
prov = doc["provenance"]
print(doc["format"], "schema", doc["schema_version"], "at C =", doc["confidence"])
print("refined from", prov["cube_path"], prov["cube_version_tag"])
for name, s in doc["surfaces"].items():
    n_open = sum(c is None for c in s["ceiling_alpha_n"])
    print(f"  {name:>11}: {len(s['mass_gev']):3d} vertices, "
          f"{s['n_grid_columns']} on the cube's mass columns, "
          f"{s['n_inserted']} inserted, {n_open} with no ceiling")

s = doc["surfaces"]["massless_f1"]
i = min(range(len(s["mass_gev"])), key=lambda j: s["floor_alpha_n"][j])
print(f"massless left end: {s['mass_gev'][0]:.4g} GeV (origin {s['origin'][0]}), "
      f"floor {s['floor_alpha_n'][0]:.4g}")
print(f"massless deepest floor: {s['floor_alpha_n'][i]:.4g} "
      f"at {s['mass_gev'][i]:.4g} GeV (origin {s['origin'][i]})")
print("spot check against the cube:", prov["spot_n_cells"], "cells per surface, "
      "max |dp| =", max(prov["spot_max_dp"].values()))
```

```
luhdm-refined-contours schema 1 at C = 0.95
refined from release/luhdm_datarelease_v10_A_f1_atm.h5 v10.0-night-m0p356mg-q1TeV-nocap-wmargnight-a18iso-mudex0p002-vE245
  massless_f1: 165 vertices, 78 on the cube's mass columns, 83 inserted, 47 with no ceiling
       2mm_f1: 275 vertices, 65 on the cube's mass columns, 205 inserted, 0 with no ceiling
     200um_f1: 224 vertices, 48 on the cube's mass columns, 171 inserted, 0 with no ceiling
      20um_f1: 136 vertices, 32 on the cube's mass columns, 99 inserted, 0 with no ceiling
massless left end: 4.021e+05 GeV (origin tip), floor 6.784e-08
massless deepest floor: 3.207e-09 at 5.382e+06 GeV (origin inserted)
spot check against the cube: 6 cells per surface, max |dp| = 0.0
```

**Which boundary is which.** The paper quotes the **refined** boundary; the
**grid** is what the released cubes store, and contouring it per
[§5.3](#53-the-exclusion-convention) remains the right way to read a cube. The
two are the same claim at different resolution, and the gap between them is
bounded by the cell size: for the massless mode-1 surface the interpolated floor
sits at most 0.176 dex above the refined one and the interpolated ceiling at most
0.219 dex below it, against a coupling cell of 0.233 dex. Where they part
company visibly is at the ends, because a grid contour cannot begin before the
first mass column that is excluded at all: that column is 5.20 × 10⁵ GeV for the
massless surface, while the island really ends at 4.02 × 10⁵ GeV. The ends are
bisected in mass, so the sidecar puts them where they are rather than on the
nearest column — the 200 µm island's right end is 2.35 × 10¹¹ GeV, bracketed
against 2.39 × 10¹¹ GeV where nothing is excluded any more. Compare curve to
curve with the published figure using the sidecar; re-derive from the cube and
you have the grid boundary, which is what you should say you have.

**Rebuilding them.** `scripts/refine_contours.py` in the code repository, once
per hypothesis file — each surface must be run against the file that carries its
`(f_DM, atmosphere)` plane. The shipped files record their own command lines in
`provenance.argv`:

```
scripts/refine_contours.py --release release/luhdm_datarelease_v10_A_f1_atm.h5 \
    --surfaces massless_f1,2mm_f1,200um_f1,20um_f1 --mode 1 --spot 6 --workers 50 \
    --max-insert 800 --out release/luhdm_contours_v10_A_f1_atm.json
scripts/refine_contours.py --release release/luhdm_datarelease_v10_B_f0p1_noatm.h5 \
    --surfaces massless_f0p1_noatm,2mm_f0p1_noatm,200um_f0p1_noatm,20um_f0p1_noatm \
    --mode 1 --spot 6 --workers 25 \
    --max-insert 800 --out release/luhdm_contours_v10_B_f0p1_noatm.json
```

The refiner reads the cube's `v_earth_km_s` and applies it to its own halo
before recomputing anything, and hard-stops if the two disagree; there is no way
to refine a lab-frame cube in the rest frame by accident.

**Re-checking them.** The same script does it: `--spot N` re-evaluates N of the
cube's own grid cells with the refining calculation and compares against the
stored values. Both shipped files were made with `--spot 6` and record the
outcome in `provenance.spot_max_dp`, which is `0.0` for every refined surface — at the
cube's own grid points the refinement reproduces the released numbers bit for bit
in `float32` storage, which is what ties the refined boundary to the release.
`--columns i,j,k --no-insert --no-tips --workers 1` is the cheap partial run for
checking a handful of columns without paying for a whole surface.

**What it costs.** Hours per hypothesis at the release fidelity, not minutes.
One call of the refining calculation is one attenuation ODE plus one freshly
seeded optimum-interval table, and a surface takes thousands of them, most of
them in the column-insertion phase. File A's four surfaces took 13 619 such
calls and 7.2 hours of wall time on 50 workers. File B's four surfaces solve no
ODE at all, so they are cheap by comparison: 7 341 calls and 52 minutes on 25
workers. The sidecar is rewritten after each surface,
so a long run can be watched, and interrupted, without losing what is already
done.

**The mediator-range scan.**
`luhdm_lambda_scan_v10.npz` answers the question the cube's `lambda_m` axis cannot:
four finite ranges are too few to draw a band in the (coupling, range) plane, or
to say where it closes. The sidecar is that plane scanned properly, for dataset A
and **sensor mode 1** only — the same narrowing of scope as the refined contours
above. It holds `extremeness_mode1`, `mu_mode1` and
`n_transit_mode1` on a 44 × 54 (coupling × range) grid running from 0.1 µm to
2 m, at that mode's best dark-matter mass `best_mass_gev_mode1` — the mass whose
exclusion is widest in the (coupling, range) plane, ties broken towards reach to
the shortest range, by `luhdm.release.best_mass_index` evaluated on the cube's
four finite slices. `modes` names the modes actually present, so read it rather
than assuming three. The
axes travel with the planes: `alpha_n_mode1` is bitwise the cube's
`axes/alpha_n`, and `lambda_m_mode1` contains the 20 µm, 200 µm and 2 mm tags
exactly, so those columns are directly comparable to the cube. `provenance` is a
JSON string carrying the cube's name and digest, the best-mass criterion, the
range and coupling axis definitions, the physics and statistics settings
(including `v_earth_km_s` and `mu_dex` in the fidelity string), the
input digests and the scan command line.

```python
import json, numpy as np

d = np.load("luhdm_lambda_scan_v10.npz", allow_pickle=True)
prov = json.loads(str(d["provenance"]))
level = float(np.float32(0.95))              # the storage-precision convention
print("scanned against", prov["cube"]["file"], "f_dm =", prov["cube"]["f_dm"],
      "atmosphere =", prov["cube"]["atmosphere"])
for m in d["modes"]:
    lam, p = d[f"lambda_m_mode{m}"], d[f"extremeness_mode{m}"]
    band = (p >= level).any(axis=0)
    print(f"  mode {m}: best mass {float(d[f'best_mass_gev_mode{m}']):.4g} GeV, "
          f"plane {p.shape} (alpha_n x lambda), 95% band "
          f"{lam[band][0] * 1e6:.3g} um .. {lam[band][-1]:g} m")
```

```
scanned against luhdm_datarelease_v10_A_f1_atm.h5 f_dm = 1.0 atmosphere = True
  mode 1: best mass 5.019e+08 GeV, plane (44, 54) (alpha_n x lambda), 95% band 7.96 um .. 2 m
```

The short end of that band is where the exclusion pinches off in mediator range:
7.96 µm, a mediator mass of 24.8 meV. In the Galactic rest frame of `v9.1` the
same scan pinched off at 6.32 µm (31.2 meV) at a best mass of 2.20 × 10⁸ GeV,
so the lab-frame flux moves both the optimizing mass and the closing range.

**Rebuilding and re-checking it.** `scripts/scan_lambda.py` runs the scan at the
mode's best mass, and `scripts/assemble_lambda_sidecar.py`
gathers the output into the npz with the provenance record.
`scripts/lambda_sidecar_gate.py` is the gate: it compares the scan against the
released cube everywhere the two overlap — the 20 µm, 200 µm and 2 mm columns
must reproduce the cube's excluded coupling bands, and the long end of the range
axis, 2 m, must reproduce the cube's massless slice. The same
comparison is written into the file's own provenance.

---

## 6. Worked example: the published limit

This reproduces the paper's headline number from the released file, in plain
numpy. It is the same arithmetic the analysis code runs, and the standalone
reader's `excluded_band()` is a wrapper around it — with the mass cut applied on
top, since the reader does not apply it either.

```python
import h5py, numpy as np

C = 0.95                                    # confidence level, attrs['confidence_recommended']

with h5py.File("luhdm_datarelease_v10_A_f1_atm.h5", "r") as f:
    alpha = f["axes/alpha_n"][:]
    mass  = f["axes/mass_gev"][:]
    lam   = f["axes/lambda_m"][:]
    i_mod = int(np.flatnonzero(f["axes/mode"][:] == 1)[0])         # mode 1
    i_lam = int(np.flatnonzero(~np.isfinite(lam))[0])              # massless mediator
    m_cut = float(f.attrs["m_cut_10cm_f1_gev"])                    # flux cut, N_req = 3
    p = f["results/extremeness"][0, 0, i_mod, :, :, i_lam]         # (alpha_n, mass_gev)

lo = np.full(mass.size, np.nan)             # lower edge of the excluded band, per mass
hi = np.full(mass.size, np.nan)             # upper edge
for j in range(mass.size):
    above = np.flatnonzero(p[:, j] >= C)                  # NaN compares False: not excluded
    if above.size == 0:
        continue
    a, b = above[0], above[-1]
    lo[j] = alpha[0] if a == 0 else 10 ** np.interp(
        C, p[a - 1:a + 1, j], np.log10(alpha[a - 1:a + 1]))
    hi[j] = alpha[-1] if b == alpha.size - 1 else 10 ** np.interp(
        -C, -p[b:b + 2, j], np.log10(alpha[b:b + 2]))

ok = np.isfinite(lo) & (mass <= m_cut)      # the flux cut of section 5.4, N_req = 3
j = int(np.nanargmin(np.where(ok, lo, np.nan)))           # strongest coupling reached
print(f"best limit  alpha_n = {lo[j]:.2g}  at m_DM = {mass[j]:.2g} GeV  ({C:.0%} CL)")
print(f"excluded at {ok.sum()} of {mass.size} masses, "
      f"m_DM = {mass[ok].min():.3g} .. {mass[ok].max():.3g} GeV")
print(f"({int((np.isfinite(lo) & (mass > m_cut)).sum())} further masses are excluded by the "
      f"surfaces but lie above m_cut = {m_cut:.3g} GeV)")
print(f"upper edge at the best mass: {hi[j]:.3g}; "
      f"{int(np.nansum(hi[ok] >= alpha[-1]))} of the {int(ok.sum())} in-window masses "
      f"run to the top of the coupling grid")
```

```
best limit  alpha_n = 4e-09  at m_DM = 8.1e+06 GeV  (95% CL)
excluded at 78 of 119 masses, m_DM = 5.2e+05 .. 8.09e+14 GeV
(35 further masses are excluded by the surfaces but lie above m_cut = 8.4e+14 GeV)
upper edge at the best mass: 0.00166; 23 of the 78 in-window masses run to the top of the coupling grid
```

Those are the paper's numbers for a massless mediator: couplings excluded down
to `alpha_n = 4.0 × 10⁻⁹` at 95% CL at a mass of 8.1 × 10⁶ GeV — 3.2 × 10⁻⁹ at
5.4 × 10⁶ GeV once the boundary is root-found rather than read off the grid
([§5.5](#55-the-sidecar-files)) — over a mass window running from the kinematic
wall at 3.80 × 10⁵ GeV to the flux cut at 8.40 × 10¹⁴ GeV. Three features of the
printout are worth reading twice:

* the **left** end, 5.20 × 10⁵ GeV, is the first mass grid point that is
  excluded at all, above the kinematic wall
  ([§3](#the-analysis-window-starts-at-1-tev)). Nothing is
  excluded below the wall because nothing *can* be. It is a grid point, not the
  edge of the claim: the refined contour sidecar of
  [§5.5](#55-the-sidecar-files) traces the island's tip to 4.02 × 10⁵ GeV;
* the **right** end is set by `m_cut` and by nothing in the surfaces: 35 further
  masses, all the way to the Planck mass, satisfy `extremeness >= 0.95`. Without
  the cut the region would not close on the right at all;
* the band is **open at the top over part of the window**. With the cap removed,
  atmospheric attenuation no longer closes the massless region from above
  everywhere: 23 of the 78 in-window masses run to `alpha_n = 1`, the top of the
  scanned grid, and there the upper edge is a property of the scan and not a
  ceiling ([§9](#9-known-limitations)).

The negations in the `hi` branch exist only to make the descending side
increasing for `np.interp`, which requires increasing `x`.

Now draw it. This block is **standalone** — it reopens the file and redoes the
band, so it can be pasted into a cold interpreter on its own; that is also why
it is the generator of record for the shipped figure. `matplotlib` is not part
of the `numpy` and `h5py` floor, so it needs one more package:

```python
import h5py, numpy as np
import matplotlib.pyplot as plt

C = 0.95
with h5py.File("luhdm_datarelease_v10_A_f1_atm.h5", "r") as f:
    alpha = f["axes/alpha_n"][:]
    mass  = f["axes/mass_gev"][:]
    lam   = f["axes/lambda_m"][:]
    i_mod = int(np.flatnonzero(f["axes/mode"][:] == 1)[0])
    i_lam = int(np.flatnonzero(~np.isfinite(lam))[0])
    m_cut = float(f.attrs["m_cut_10cm_f1_gev"])
    p = f["results/extremeness"][0, 0, i_mod, :, :, i_lam]

lo = np.full(mass.size, np.nan)
hi = np.full(mass.size, np.nan)
for k in range(mass.size):
    above = np.flatnonzero(p[:, k] >= C)
    if above.size == 0:
        continue
    a, b = above[0], above[-1]
    lo[k] = alpha[0] if a == 0 else 10 ** np.interp(
        C, p[a - 1:a + 1, k], np.log10(alpha[a - 1:a + 1]))
    hi[k] = alpha[-1] if b == alpha.size - 1 else 10 ** np.interp(
        -C, -p[b:b + 2, k], np.log10(alpha[b:b + 2]))
ok = np.isfinite(lo) & (mass <= m_cut)
j = int(np.nanargmin(np.where(ok, lo, np.nan)))

fig, ax = plt.subplots(figsize=(6.0, 4.2), constrained_layout=True)
ax.fill_between(mass[ok], lo[ok], hi[ok], alpha=0.30, label="excluded, 95% CL")
ax.plot(mass[ok], lo[ok], lw=1.6)
ax.plot(mass[j], lo[j], "o", ms=5, label=f"best: {lo[j]:.2g} at {mass[j]:.2g} GeV")
ax.axvline(m_cut, color="0.35", ls="--", lw=1.2)
ax.text(m_cut, 0.97, r" $m_{\rm cut}$ ($N_{\rm req}=3$)", transform=ax.get_xaxis_transform(),
        ha="left", va="top", fontsize=8, color="0.35")
ax.set(xscale="log", yscale="log", xlabel=r"$m_{\rm DM}$ [GeV]", ylabel=r"$\alpha_n$",
       title="Massless mediator, mode 1, $f_{\\rm DM}=1$, atmosphere on")
ax.legend(loc="lower right", frameon=False)
fig.savefig("exclusion_massless_mode1.png", dpi=160)
print("wrote exclusion_massless_mode1.png")
```

```
wrote exclusion_massless_mode1.png
```

![Excluded region in the coupling versus dark-matter mass plane for a massless mediator, mode 1, f_DM = 1, with atmospheric attenuation applied. The region is bounded on the left by the kinematic wall, below by the sensitivity floor, and on the right by the dashed flux cut at 8.40e14 GeV; over the upper decades of mass it runs to the top of the coupling grid.](exclusion_massless_mode1.png)

The file that block writes is shipped as
[`exclusion_massless_mode1.png`](exclusion_massless_mode1.png), so you can check
your copy against ours. It is regenerated, from exactly this block, by section 7
of `notebooks/05_using_the_data_release.ipynb` in the analysis repository —
which is what keeps the shipped png at the same release version as the cube.

For a different mediator range, replace the `i_lam` line with
`int(np.flatnonzero(lam == 2e-4)[0])`. For a different confidence level, change
`C`. For a different mode, change the value matched on `axes/mode`. For the
other hypothesis, open file B and read its `m_cut_10cm_f0.1_gev` instead —
nothing else in the block changes.

---

## 7. The standalone reader

[`luhdm_release.py`](luhdm_release.py) is a **single self-contained file**. Copy
it next to the HDF5 files and import it: there is no package to install, no
relative imports, no data files, and it never imports the analysis code. Its only
requirements are `numpy` and `h5py`; `pandas` is imported lazily inside
`to_dataframe()` and is optional. It is also readable as reference, since
everything it does is a few lines of numpy.

What it adds over raw `h5py`:

* **selection by physical value**: `mode=2`, `lam='200um'` or `lam=20e-6` or
  `lam='massless'`, `f_dm=1.0`, `atmosphere=True`, `mass=1e8`, `alpha=1e-3`,
  with errors that list the available values when a request misses;
* slices returned **with their axes attached**, and the resolved grid values
  echoed back so you can see what `mass=1e8` snapped to;
* `excluded_band()` implementing [§5.3](#53-the-exclusion-convention), including
  saturation flags and an explicit count of undefined (status-1) cells;
* `efficiency_curve()`, `events()`, `all_blips()` and `exposure_s` for the
  detector inputs, see [§7.1](#71-detector-inputs);
* `summary()`, which prints everything in this README read out of the file you
  actually have;
* `to_dataframe()`, a tidy long-format table for a chosen hypothesis;
* it is **schema-driven**: axis names, lengths, units, tags and the massless
  sentinel all come from the file, so it keeps working across cube versions.

Two things it deliberately does **not** do. It does not apply the mass cut of
[§5.4](#54-the-mass-cut-m_cut-and-how-to-apply-it) — `excluded_band()` returns
every mass the surfaces exclude, and masking at `m_cut` is yours. And it falls
back to `attrs['f_dm_default']` = 0.1 when a caller names no fraction, which
file A does not carry: **pass `f_dm=1.0` explicitly when you read file A**, or
you will get a `KeyError` naming the axis it does have.

Run it on a file to see what you have:

```console
$ python luhdm_release.py luhdm_datarelease_v10_A_f1_atm.h5
```

```
==============================================================================
POLONAISE UHDM data release   v10.0-night-m0p356mg-q1TeV-nocap-wmargnight-a18iso-mudex0p002-vE245
==============================================================================
file            : luhdm_datarelease_v10_A_f1_atm.h5
format          : luhdm-datarelease version 2 (schema 1)
created         : 2026-08-18T14:01:23.802961+00:00
exposure        : 790,778 s  (219.66 h)
impact-param cap: none (uncapped)
recommended CL  : 0.95

hypothesis axes
  f_dm          : [1.0]  (default 0.1) -- DM fraction, pure flux scale
  atmosphere    : [1] -> [True]  (1/True = attenuation applied)
  mode          : [1, 2, 3]

axes
  alpha_halo_n    n=64    2e-11 .. 1                   [1]
                  coupling alpha_n (halo/flux-map 64 grid)
  alpha_n         n=44    1e-10 .. 1                   [1]
                  per-neutron coupling alpha_n
  atmosphere      n=1     [1]                          [bool]
                  1 = attenuation through the atmosphere/earth applied (atm pass); 0 = bare halo flux (noatm pass)
  f_dm            n=1     [1]                          [1]
                  dark-matter fraction hypothesis of this species; a pure flux normalisation (n_dm ∝ f_DM)
  lambda_m        n=5     [2e-05, 0.0002, 0.002, 200, inf] [m]  [4 finite + 1 massless sentinel(inf); tags: 10um, 200um, 20cm, 20um, 2cm, 2m, 2mm, 2um, massless]
                  mediator range; finite ascending then inf (massless) last
  m_phi_gev       n=5     [9.866e-12, 9.866e-13, 9.866e-14, 9.866e-19, 0] [GeV]
                  mediator mass = 1/conv_m2pGeV(lambda); exactly 0 at inf
  mass_gev        n=119   1e+05 .. 1.22e+19            [GeV]
                  dark-matter mass (shared by both atmosphere planes)
  mass_halo_gev   n=64    1e+05 .. 1.22e+19            [GeV]
                  dark-matter mass (halo/flux-map 64 grid)
  mode            n=3     [1, 2, 3]                    [1]
                  sensor mode index (1,2,3)

results  (axis order in parentheses)
  extremeness     (1, 1, 3, 44, 119, 5)      float32  [1]
                  (f_dm, atmosphere, mode, alpha_n, mass_gev, lambda_m)
                  optimum-interval extremeness / confidence; NaN where status==1
  mu              (1, 1, 3, 44, 119, 5)      float32  [counts]
                  (f_dm, atmosphere, mode, alpha_n, mass_gev, lambda_m)
                  expected signal counts mu; NaN where status==1. Exactly linear in f_DM (a pure flux normalisation).
  n_transit       (1, 1, 44, 119, 5)         float32  [counts]
                  (f_dm, atmosphere, alpha_n, mass_gev, lambda_m)
                  expected within-reach transits; clipped >=0 (KDE tail can oscillate slightly negative). Exactly linear in f_DM.
  status          (1, 1, 3, 44, 119, 5)      uint8    [enum]
                  (f_dm, atmosphere, mode, alpha_n, mass_gev, lambda_m)
                  0=ok(MC) 1=exception 2=mu<0.2 3=mu>mu_cap 4=mu==0

status codes  (counts over the whole cube)
  0         9,926  (12.64%)  ok(MC): the optimum-interval Monte Carlo ran
  1            75  ( 0.10%)  exception: the cell raised; extremeness/mu/n_transit are NaN, and NaN reads as NOT excluded
  2        36,509  (46.48%)  mu<0.2: expected counts below the MC floor; extremeness is exactly 0
  3        19,991  (25.45%)  mu>mu_cap: expected counts above the MC cap; extremeness is exactly 1 (excluded)
  4        12,039  (15.33%)  mu==0: the spectrum has no support; extremeness is exactly 0

detector
  exposure_s     790,778 s
  mode 1:    8 analysis events (1521 .. 1.279e+04 GeV), 66 raw blips
  mode 2:   26 analysis events (554.2 .. 8473 GeV), 99 raw blips
  mode 3:  126 analysis events (1569 .. 1.723e+04 GeV), 443 raw blips
  efficiency     q_gev_<mode>, eff_<mode>_df<2|3>; analysis used df=3

halo diagnostics (own coarser alpha/mass grids)
  bmax            (64, 64, 5)                [m]  flux-averaged threshold reach sqrt(<pi b^2>/pi)
  n_transit       (64, 64, 5)                [counts]  unattenuated-halo expected transits

reference_curves: 20 datasets (showcase spectra / arrival-speed distributions)

provenance
  git_commit     4193b74c0439b13d5fbea7f31df940dea0b3ccdf (dirty=True)
  seed           20260702
  MC fidelity    n_mc=10000 n_ode=400 n_shm=300000 n_q=240
  packages       {"numpy": "2.5.1", "scipy": "1.18.0", "h5py": "3.16.0", "optimum_interval": "0.3.0", "luhdm": "0.1.0", "matplotlib": "3.11.1", "pandas": "3.0.3", "python": "3.14.7"}
  events_mode1_sha256 9bdc69c90b6f9e80db114821e1af363157a1a55c260907e2d4ebfc0641c1f5b6
  events_mode2_sha256 9b78181c959266873dafabe2db4ae8227ed61e10e1b363282be521123ea0ea50
  events_mode3_sha256 31e611787b087d6f6494422d8485a5a51d97168eb6b3b34dd6920840c396f105
  efficiency_npz_sha256  6505e5e39621f094fc5b902e24f9ad4cd802c53c2a1c2930d6e5b7921202d497
==============================================================================
```

Three things about that summary deserve a note. `impact-param cap: none
(uncapped)` is the headline change of the `v7` line and still holds
([§3](#the-interaction-model-in-one-paragraph)).
The `tags:` list on `lambda_m` names ranges the axis does not carry; see the
trap in [§4.1](#41-axes). And the summary does **not** print the halo frame:
the reader shipped here predates the attribute, so read
`attrs['v_earth_km_s']` yourself if you are recomputing anything
([§3](#the-halo-is-evaluated-in-the-laboratory-frame)). It is one line of
`h5py`, and absence of the attribute means the Galactic rest frame.

`python luhdm_release.py --help` prints the module documentation and the usage
line. The API is in the docstrings: `help(luhdm_release)` and
`help(luhdm_release.Release)` work offline.

```python
import numpy as np
import luhdm_release

with luhdm_release.open_release("luhdm_datarelease_v10_A_f1_atm.h5") as rel:
    print(rel)
    print("exposure", rel.exposure_s, "s, cap", rel.b_constrained_max_m)

    sl = rel.get("extremeness", mode=1, lam="200um", f_dm=1.0, atmosphere=True)
    print(sl)                                   # axes travel with the array
    print("excluded cells:", int((sl.values >= 0.95).sum()))

    band = rel.excluded_band(mode=1, lam="200um", f_dm=1.0)   # 95% CL, per mass
    print(band)
    j = int(np.flatnonzero(np.isfinite(band.alpha_lo))[0])
    print("at m =", band.mass_gev[j], "alpha_n in",
          (band.alpha_lo[j], band.alpha_hi[j]))

    cell = rel.cell(mode=2, alpha=1e-3, mass=1e8, lam="200um", f_dm=1.0)
    print({k: cell[k] for k in ("mass_gev", "alpha_n", "mu", "extremeness", "status")})
```

```
<Release 'luhdm_datarelease_v10_A_f1_atm.h5' (v10.0-night-m0p356mg-q1TeV-nocap-wmargnight-a18iso-mudex0p002-vE245)>
exposure 790778.0 s, cap None
<Slice extremeness [1] (alpha_n=44, mass_gev=119) at f_dm=1, atmosphere=True, mode=1, lambda_m=0.0002>
excluded cells: 981
<ExcludedBand 95% CL at f_dm=1, atmosphere=True, mode=1, lambda_m=0.0002: 48/119 masses excluded, mass range 5.203e+05..2.123e+11 GeV>
at m = 520302.1070058871 alpha_n in (np.float64(3.5197484401264716e-08), np.float64(0.0003336228720393256))
{'mass_gev': 96471330.72266711, 'alpha_n': 0.0009478599776522384, 'mu': 5508.9423828125, 'extremeness': 1.0, 'status': 3}
```

`cap None` is the uncapped cross section. The 200 µm band above ends at
2.1 × 10¹¹ GeV, well below `m_cut`, so for that range the mass cut changes
nothing; it is the massless and 2 mm slices where it bites.

`to_dataframe()` is the one method that needs `pandas`, so it is in its own
block:

```python
import luhdm_release                                  # this block also needs pandas

with luhdm_release.open_release("luhdm_datarelease_v10_A_f1_atm.h5") as rel:
    df = rel.to_dataframe(mode=1, lam="massless", mass=1e12, f_dm=1.0)
    print(df.shape, list(df.columns))
```

```
(44, 13) ['f_dm', 'atmosphere', 'mode', 'alpha_n', 'mass_gev', 'lambda_m', 'm_phi_gev', 'extremeness', 'mu', 'status', 'status_meaning', 'n_transit', 'excluded']
```

### 7.1 Detector inputs

The measured detection efficiency travels **inside the files**, so a
reinterpretation never has to reconstruct it. Each mode has its own momentum
grid `detector/q_gev_{mode}` and its efficiency `detector/eff_{mode}_df{2,3}`,
where `df` is the degrees-of-freedom hypothesis of the efficiency fit; the
analysis used `attrs['df']` = 3. The live time is `detector/exposure_s` and the
candidate lists are `detector/events_mode{n}`.

**What the curves are.** ε(q) is the per-mode detection efficiency delivered by
the detector group, and in these files it is **averaged over the phase of the
mode oscillation at which the impulse arrives**. The curves the earlier builds
of this analysis used held that phase fixed at its most favourable value;
averaging over it is the only input that changed between the last of those
builds and this one. It lowers ε through the turn-on — the mode-1 curve now
reaches 50% at 1.22 TeV, and the saturated value is 0.998 rather than exactly
1 — and every `mu` in `/results` was computed with the curves that ship here,
whose digest is `attrs['efficiency_npz_sha256']`. A second averaging of the same
measurement, over the night selection alone rather than over the whole run, is
committed alongside these curves in the analysis repository; the two differ at
the few-permille level, and that variant is to be taken up at the next rebuild
rather than partway through this one.

With the standalone reader, where `df` defaults to the file's own `attrs['df']`:

```python
import numpy as np
import luhdm_release

with luhdm_release.open_release("luhdm_datarelease_v10_A_f1_atm.h5") as rel:
    print("exposure_s :", rel.exposure_s, "s")
    for m in rel.modes:
        q, eff = rel.efficiency_curve(m)          # df defaults to attrs['df']
        ev = rel.events(m)
        i50 = int(np.argmin(np.abs(eff - 0.5)))
        print(f"mode {m}: {len(ev):3d} candidates, "
              f"q = {ev.min():7.1f} .. {ev.max():9.1f} GeV | "
              f"eff grid {q.size} pts, 50% at q = {q[i50]:7.1f} GeV")

    q2, e2 = rel.efficiency_curve(1, df=2)
    q3, e3 = rel.efficiency_curve(1, df=3)
    print("mode 1 efficiency at q = 2000 GeV:",
          f"df=2 {np.interp(2000.0, q2, e2):.4f},",
          f"df=3 {np.interp(2000.0, q3, e3):.4f}")
```

```
exposure_s : 790778.0 s
mode 1:   8 candidates, q =  1520.7 ..   12790.7 GeV | eff grid 400 pts, 50% at q =  1199.9 GeV
mode 2:  26 candidates, q =   554.2 ..    8473.1 GeV | eff grid 400 pts, 50% at q =   954.9 GeV
mode 3: 126 candidates, q =  1569.0 ..   17234.7 GeV | eff grid 400 pts, 50% at q =  5493.6 GeV
mode 1 efficiency at q = 2000 GeV: df=2 0.9720, df=3 0.9725
```

Read those 50% points against the 1 TeV window edge: mode 2 crosses 50% just
below it, mode 1 just above it at 1.22 TeV, and mode 3 not until 5.49 TeV. The
edge is a stated momentum threshold ([§3](#the-analysis-window-starts-at-1-tev))
and falls where it falls on each curve: ε(1 TeV) is 0.16, 0.58 and 0.00 for
modes 1, 2 and 3. Mode 3 contributes nothing at the bottom of the window, and
its sensitivity comes entirely from impulses several TeV above it.

Or straight from `h5py`, with no reader at all:

```python
import h5py

with h5py.File("luhdm_datarelease_v10_A_f1_atm.h5", "r") as f:
    df = int(f.attrs["df"])
    print("analysis df =", df, "| exposure_s =", float(f["detector/exposure_s"][()]))
    for m in (1, 2, 3):
        q   = f[f"detector/q_gev_{m}"][:]
        eff = f[f"detector/eff_{m}_df{df}"][:]
        ev  = f[f"detector/events_mode{m}"][:]
        print(f"  mode {m}: q_gev_{m} {q.shape}, eff_{m}_df{df} {eff.shape}, "
              f"events_mode{m} {ev.shape}, eff_max {eff.max():.4f}")
```

```
analysis df = 3 | exposure_s = 790778.0
  mode 1: q_gev_1 (400,), eff_1_df3 (400,), events_mode1 (8,), eff_max 0.9993
  mode 2: q_gev_2 (400,), eff_2_df3 (400,), events_mode2 (26,), eff_max 0.9993
  mode 3: q_gev_3 (400,), eff_3_df3 (400,), events_mode3 (126,), eff_max 1.0000
```

The efficiency is already folded into `results/mu`. You need these curves only
if you are recomputing rates yourself or folding a different spectrum.

**If you have cloned the analysis repository** instead of taking the two-file
path, `luhdm.release` exposes the same three quantities under the same method
names, so code written against either reader ports over unchanged. Install it
with `pip install -e ".[dev,notebooks]"` from a clone of
<https://github.com/PolonaiseExperiment/luhdm>; the repository README has the
full instructions.

```python
import numpy as np
from luhdm import release

rel = release.open_release("luhdm_datarelease_v10_A_f1_atm.h5")
print("exposure_s :", rel.attrs["t_exposure_s"], "s")
for m in (1, 2, 3):
    q, eff = rel.efficiency_curve(m, df=int(rel.attrs["df"]))
    ev = rel.events(m)
    i50 = int(np.argmin(np.abs(eff - 0.5)))
    print(f"mode {m}: {len(ev):3d} candidates, "
          f"q = {ev.min():7.1f} .. {ev.max():9.1f} GeV | "
          f"eff grid {q.size} pts, 50% at q = {q[i50]:7.1f} GeV")
rel.close()
```

```
exposure_s : 790778.0 s
mode 1:   8 candidates, q =  1520.7 ..   12790.7 GeV | eff grid 400 pts, 50% at q =  1199.9 GeV
mode 2:  26 candidates, q =   554.2 ..    8473.1 GeV | eff grid 400 pts, 50% at q =   954.9 GeV
mode 3: 126 candidates, q =  1569.0 ..   17234.7 GeV | eff grid 400 pts, 50% at q =  5493.6 GeV
```

---

## 8. Intended use

**This release is for** re-deriving the published exclusion, quoting it at a
different confidence level, reading it at any of the mediator ranges on the
grid, comparing a new result against it, and recomputing rates with the shipped
efficiency curves, candidate lists and live time.

**It is not for** the following, each for a stated reason:

* **Quoting an exclusion above `m_cut`.** The surfaces are defined everywhere on
  the grid, but above the flux cut of
  [§5.4](#54-the-mass-cut-m_cut-and-how-to-apply-it) the halo does not deliver
  the transits the statement would need. Apply the cut, or state clearly that
  you have chosen a different `N_req` and what it is.
* **Reading the 200 m slice as a physics point.** It is a convergence check on
  the finite-range branch of the cross section. See [§4.1](#41-axes).
* **Combining the three sensor modes.** They are separate measurements with
  different thresholds, efficiencies and candidate lists, demodulated from the
  same readout stream. Whether and how to combine them is an analysis choice the
  release does not make for you; the paper reports mode 1 and carries modes 2
  and 3 as cross-checks. See [§9](#9-known-limitations).
* **Treating `all_blips_mode{n}` as a signal candidate list, or as the night
  pre-selection.** It is the raw up-crossing population of the whole unvetoed
  run, published for context; it has no time metadata, so the night selection
  cannot be re-derived from it, and `exposure_s` does not normalise it. The
  limit is set on `events_mode{n}`.
* **Quoting a saturated upper edge as a ceiling.** Wherever the excluded band
  runs to `alpha_n = 1` the upper edge is where the coupling grid stopped, not
  where the physics closed. This is everything in file B and part of file A.
  See [§9](#9-known-limitations).
* **Quantitative use of `/halo`.** It is a coarser, independently sampled
  diagnostic map for figures and intuition. Use `/results/n_transit`.
* **Reading a range dependence off three points.** The physics ranges are a
  decade apart. Take each slice on its own terms; do not interpolate between
  them, and do not interpolate across the 200 m validation slice, which is not
  on the same footing.
* **Rescaling `extremeness` between the two files.** `mu` and `n_transit` are
  exactly linear in `f_DM`; `extremeness` is not. Open the file whose hypothesis
  you want.
* **Extrapolating outside the scanned grid.** The coupling axis stops at
  `alpha_n = 1` and the mass axis at the Planck mass. Edges that saturate there
  are properties of the scan.

---

## 9. Known limitations

**The halo frame is a convention, and it is the largest single difference from
every earlier release.** The lab-frame boost `v_earth_km_s` = 245 km/s
([§3](#the-halo-is-evaluated-in-the-laboratory-frame)) is a stated choice, taken
to match Monteiro (2020) and Tseng (2025) so that the overlaid limits are read
in one frame. It is not a fit and it carries no uncertainty band here: the boost
is applied as a single fixed speed over the whole exposure, so any annual
variation in the Earth's motion through the halo is neither modelled nor
propagated. Nor is the choice reducible to a rescaling of the `v9.1` surfaces —
the boost changes the shape of the arrival-speed distribution, not only its
mean, and `extremeness` is a non-linear function of what follows. If you want
the Galactic rest frame, you want a different cube, not a factor.

**The mass cut is an assumption, and it is yours to own.** `N_req = 3` transits
within a 10 cm aperture is a stated standard, not a measurement, and the
right-hand edge of every excluded region moves inversely with it
([§5.4](#54-the-mass-cut-m_cut-and-how-to-apply-it)). It is the largest
discretionary choice in this release. The cut is also a *sharp* line drawn
across a continuous fall-off in transit rate: nothing changes physically at
8.40 × 10¹⁴ GeV, the expected number of flybys simply passes 3 there.

**Cells above the expected-count cap are asserted excluded, not computed.**
Where `mu` exceeds `attrs['fid_mu_cap']` = 85 the optimum-interval Monte Carlo is
skipped and `extremeness` is set to exactly `1.0` (`status == 3`, 25.5% of file
A and 27.4% of file B). The assertion is that such a hypothesis is
overwhelmingly excluded. It is validated for **modes 1 and 2**: no Monte-Carlo
cell with `p < 0.95` has `mu` above 36, leaving a wide margin below the cap. For
**mode 3** it is not: Monte-Carlo cells with `mu` up to about 84 still show
`p < 0.95`, so the assertion is applied right where the computed answer can still
fall below threshold. **Mode-3 exclusion boundaries near high expected counts
therefore carry additional uncertainty** beyond the quoted Monte-Carlo noise.
Mask on `status == 3` if you need a purely computed boundary — but know how much
of the region that removes: the shortcut accounts for 86% of the excluded
mode-1 cells in file A and 85% in file B. The boundary itself is computed; the
interior, overwhelmingly, is asserted.

**The mode-3 massless floor is resolved to about one coupling grid step.** The
`alpha_n` axis is 0.2326 dex per step, and for the massless slice in mode 3 the
lower edge of the excluded band is determined to roughly that spacing. Treat the
mode-3 massless floor as grid-resolution-limited rather than as a precisely
located number.

**Undefined (status-1) cells.** 75 cells in each file (0.10% of
each) raised during evaluation and carry `NaN` in `extremeness`, `mu` and
`n_transit`; `NaN` is exactly coincident with `status == 1`. They sit deep in
the non-excluded corner, at strong coupling and heavy mass, all of them in
the 20 µm slice, the shortest range carried here. The published limits are
unaffected. If you contour a different confidence level, or work in that corner,
mask on `status == 1` explicitly rather than trusting the `NaN` comparison.

**Upper edges that saturate are grid artefacts, not ceilings.** With attenuation
ON, strong couplings can be stopped by the overburden and the exclusion closes
from above. With the impact-parameter cap removed, that no longer happens
everywhere: part of the file-A region, and all of file B, runs past the top of
the scanned coupling grid.

```python
import h5py, numpy as np

for path in ("luhdm_datarelease_v10_A_f1_atm.h5", "luhdm_datarelease_v10_B_f0p1_noatm.h5"):
    with h5py.File(path, "r") as f:
        ext = f["results/extremeness"][0, 0]      # (mode, alpha, mass, lambda)
        atm = int(f["axes/atmosphere"][0])
    above = ext >= 0.95
    any_exc, top_exc = above.any(axis=1), above[:, -1]
    n = int(any_exc.sum())
    print(f"atmosphere={atm}: {n} excluded (mode, mass, lambda) columns, "
          f"{100 * (any_exc & top_exc).sum() / n:.1f}% saturate at alpha_n = 1")
```

```
atmosphere=1: 1105 excluded (mode, mass, lambda) columns, 31.5% saturate at alpha_n = 1
atmosphere=0: 1032 excluded (mode, mass, lambda) columns, 100.0% saturate at alpha_n = 1
```

Quote a saturated column as a one-sided lower bound (`alpha_n > lo`), not as a
band. The reader's `band.saturated_hi` flags this per mass.

**The 200 m slice is a convergence check, and it passes.** Taking the
interpolated lower edge per mass, mode 1, file A, the 200 m slice and the
massless slice exclude the same 113 masses, agree on the edge to 0.06% in the
median, and reach the same floor to the precision printed — well inside the
0.2326 dex coupling grid step:

```python
import h5py, numpy as np

def lower_edge(path, i_lam, mode=1, C=0.95):
    with h5py.File(path, "r") as f:
        alpha = f["axes/alpha_n"][:]
        mass  = f["axes/mass_gev"][:]
        i_mod = int(np.flatnonzero(f["axes/mode"][:] == mode)[0])
        p = f["results/extremeness"][0, 0, i_mod, :, :, i_lam]
    lo = np.full(mass.size, np.nan)
    for j in range(mass.size):
        above = np.flatnonzero(p[:, j] >= C)
        if above.size == 0:
            continue
        a = above[0]
        lo[j] = alpha[0] if a == 0 else 10 ** np.interp(
            C, p[a - 1:a + 1, j], np.log10(alpha[a - 1:a + 1]))
    return mass, lo

P = "luhdm_datarelease_v10_A_f1_atm.h5"
with h5py.File(P, "r") as f:
    lam = f["axes/lambda_m"][:]
mass, lo_200m = lower_edge(P, int(np.flatnonzero(lam == 200.0)[0]))
_,    lo_ml   = lower_edge(P, int(np.flatnonzero(~np.isfinite(lam))[0]))
ok = np.isfinite(lo_200m) & np.isfinite(lo_ml)
d = np.abs(lo_200m[ok] - lo_ml[ok]) / lo_ml[ok]
print(f"masses excluded: 200 m {int(np.isfinite(lo_200m).sum())}, "
      f"massless {int(np.isfinite(lo_ml).sum())}")
print(f"floor: 200 m {np.nanmin(lo_200m):.4g}, massless {np.nanmin(lo_ml):.4g}")
print(f"lower edge, 200 m vs massless: median |diff| {np.median(d):.2%}, "
      f"90th pct {np.percentile(d, 90):.2%}, max {d.max():.1%}")
```

```
masses excluded: 200 m 113, massless 113
floor: 200 m 3.993e-09, massless 3.993e-09
lower edge, 200 m vs massless: median |diff| 0.06%, 90th pct 2.42%, max 26.2%
```

The tail of that distribution is Monte-Carlo noise at individual masses, not a
systematic offset: the two slices are computed through different code branches
(finite-range Yukawa against the analytic Coulomb limit) with independent toy
draws, and where a contour crosses shallowly a `1/n_mc` wobble in
`extremeness` moves the interpolated edge by a visible fraction of a grid step.
The agreement of the excluded mass set, and of the floor to well inside one grid
step, is the content of the check.

**Monte-Carlo granularity.** Cells with `status == 0` carry toy noise at the
`1/n_mc` level, and this release uses two tiers of it
([§11](#11-versions)): 10⁻⁴ in the bulk, and 10⁻⁵ for any cell whose base
extremeness landed in `[0.9, 1)`, which is the band the 95% contour lives in. So
a contour near 0.95 wobbles at the finer of the two scales, and a contour drawn
at some other level — 0.5, say — wobbles at the coarser one. Cubes are stored as
float32, whose spacing near 0.95 is orders of magnitude finer than either, so the
precision loss is irrelevant.

**Monte-Carlo *calibration* granularity is a second, separate floor, and it is
the one `v10` moves.** The optimum-interval calibration is a pure function of
the expected count, so the build rounds `mu` onto a log grid of step
`attrs['fid_mu_dex']` before drawing the toy table, and shards the seeded Monte
Carlo by the rounded value. That is what makes the cube bit-reproducible
regardless of worker count or evaluation order — and it also makes
`extremeness` a *step* function of `mu`, so a boundary can only be located to
within one `mu` bin. In coupling, one bin is
`dlog(alpha) = mu_dex / (dlog mu / dlog alpha)` wide, so wherever the
sensitivity `dlog mu / dlog alpha` is small the plateau is wide: on the
bare-halo plane at high dark-matter mass it falls to about 0.05, where the
`mu_dex` = 0.02 of every earlier release gave plateaus up to 0.4 dex in
`alpha_n` — nearly two coupling cells, and **no number of Monte-Carlo trials
removes them**. `v10` uses `mu_dex` = 0.002, ten times finer, which narrows
those plateaus by the same factor. It is still a floor, an order of magnitude
lower: do not read a refined boundary as located to better than one `mu` bin.

**Modes are separate measurements.** The three sensor modes have different
thresholds, efficiencies and event lists. Combining them is an analysis choice
the release does not make for you. Figures in the paper that show a single curve
take the per-mode maximum of the extremeness, the most constraining mode at each
point, which is a valid but conservative choice.

**`f_DM` is not a free knob on `extremeness`.** `mu` and `n_transit` scale
exactly linearly with `f_DM`, but `extremeness` does not: it is a non-linear
function of `mu`. The two files are the two planes the paper uses; there is no
third one to be had by rescaling.

**The straight-line impulse approximation.** The cross section assumes the
dark-matter particle's trajectory is undeflected by the sphere. Removing the
outer cutoff makes the integral reach large impact parameters, where that
approximation is *better*, not worse; it is the inner cutoff at `R_eff` that
holds the other end. What the removal does cost is the guarantee that every
counted flyby happened inside the laboratory, and that is what `m_cut` restores.

---

## 10. Integrity, provenance and environment

### Verify what you downloaded

`SHA256SUMS` carries the SHA-256 digest of every file in the release, under the
name it was distributed with. From the directory holding them:

```console
$ sha256sum -c SHA256SUMS
```

```
CITATION.cff: OK
LICENSE: OK
README.md: OK
exclusion_massless_mode1.png: OK
luhdm_contours_v10_A_f1_atm.json: OK
luhdm_contours_v10_B_f0p1_noatm.json: OK
luhdm_datarelease_v10_A_f1_atm.h5: OK
luhdm_datarelease_v10_B_f0p1_noatm.h5: OK
luhdm_lambda_scan_v10.npz: OK
luhdm_release.py: OK
provenance_luhdm_datarelease_v10_A_f1_atm.json: OK
provenance_luhdm_datarelease_v10_B_f0p1_noatm.json: OK
```

The digests of the datasets themselves are

```
db126b5852f97764674e02fda5e97d323e9f5b4207b72cdacc2aa2a9adfcb116  luhdm_datarelease_v10_A_f1_atm.h5
cad157a7244d66ba454361bb41bc8cb90a3eeb1285ad3c417bea270f22180f33  luhdm_datarelease_v10_B_f0p1_noatm.h5
```

If you renamed a file, compare its digest directly: the digest is what matters,
not the filename. On macOS use `shasum -a 256 -c SHA256SUMS`.

### How the released files were made

The scan runs one mediator range at a time (`scripts/build_release.py`), on a
many-core node, and the per-range outputs are assembled into a cube by
`scripts/assemble_release.py`. Both files here come from **one scan**: the same
shards, the same seed, the same inputs. The assembly step was run twice with
different `--select` arguments, once for each hypothesis, which is why the two
files share every detector dataset, every reference curve, every halo map and
every axis but the two that name the hypothesis, and differ only in `/results`.

Nothing was subset or post-processed afterwards. Unlike the v6 cube, which was
an axis subset of a larger internal one, these are the assembly outputs
themselves.

### Names, tags and what they refer to

The `version_tag` attribute is `v10.0-night-m0p356mg-q1TeV-nocap-wmargnight-a18iso-mudex0p002-vE245` and the
files are distributed as `luhdm_datarelease_v10_A_f1_atm.h5` and
`luhdm_datarelease_v10_B_f0p1_noatm.h5`. The tag and the filenames were fixed at
different moments and do not match word for word:

* `-wmargnight` in the tag records which efficiency the cube was built with: the
  curves averaged over the impulse arrival phase and over the **night-selected**
  segments ([§7.1](#71-detector-inputs)), rather than the fixed-phase ones that
  earlier cubes used or the full-run average of v8. It is not a statement about
  fidelity: the Monte-Carlo settings are the production ones, `n_mc` = 10 000
  in the bulk and `n_mc_hi` = 100 000 at the boundary, and they are in the
  `fid_*` attributes.
* `-a18iso` records the projection kernel, `isotropic-folded`, which is also in
  the `projection_kernel` attribute ([§5.5](#55-the-sidecar-files)).
* `-mudex0p002` records the Monte-Carlo calibration granularity,
  `fid_mu_dex` = 0.002 dex, ten times finer than the 0.02 every earlier cube
  used ([§9](#9-known-limitations)).
* `-vE245` records the halo frame: `v_earth_km_s` = 245.0, the laboratory frame
  of [§3](#the-halo-is-evaluated-in-the-laboratory-frame). Its absence from a
  tag, and of the attribute from a file, means the Galactic rest frame.
* `-mc2tier` is **gone from the tag but not from the build**. The two-tier Monte
  Carlo `v9.1` introduced is still on here — `fid_n_mc_hi` = 100 000 trials for
  any cell whose base extremeness landed in `[fid_p_hi_lo, 1)` = `[0.9, 1)` — it
  is simply no longer the newest thing about the cube, and the tag names what
  changed. Read the `fid_*` attributes, not the tag, for what the build did.
* the filenames carry the hypothesis (`A_f1_atm`, `B_f0p1_noatm`), which the tag
  does not, because one tag covers both files. They also shorten `v10.0` to
  `v10`.

**The attributes are the authority**, and they are not edited after a file is
built: renaming a released file, or rewriting a tag to make it prettier, would
break the digests in `SHA256SUMS` and every provenance record that quotes them.
Cite the tag and the digest together ([§12](#12-how-to-cite)).

### Provenance in the file

The root attributes are a complete record. You do not need the `provenance_*.json`
files to know what produced the numbers:

```python
import h5py, json

with h5py.File("luhdm_datarelease_v10_A_f1_atm.h5", "r") as f:
    a = dict(f.attrs)
for k in ("version_tag", "created", "git_commit", "git_dirty", "seed",
          "t_exposure_s", "q_thresh_gev", "b_constrained_max_m",
          "v_earth_km_s", "m_cut_10cm_f1_gev", "m_cut_n_transits_required",
          "m_cut_applied_to_stored_surfaces", "df", "fid_mu_cap", "fid_mu_dex",
          "events_mode1_sha256", "efficiency_npz_sha256"):
    print(f"{k:34s} {a[k]}")
print("packages", json.loads(a["packages_json"]))
```

```
version_tag                        v10.0-night-m0p356mg-q1TeV-nocap-wmargnight-a18iso-mudex0p002-vE245
created                            2026-08-18T14:01:23.802961+00:00
git_commit                         4193b74c0439b13d5fbea7f31df940dea0b3ccdf
git_dirty                          True
seed                               20260702
t_exposure_s                       790778.0
q_thresh_gev                       1000.0
b_constrained_max_m                nan
v_earth_km_s                       245.0
m_cut_10cm_f1_gev                  840123306518896.1
m_cut_n_transits_required          3.0
m_cut_applied_to_stored_surfaces   False
df                                 3
fid_mu_cap                         85.0
fid_mu_dex                         0.002
events_mode1_sha256                9bdc69c90b6f9e80db114821e1af363157a1a55c260907e2d4ebfc0641c1f5b6
efficiency_npz_sha256              6505e5e39621f094fc5b902e24f9ad4cd802c53c2a1c2930d6e5b7921202d497
packages {'numpy': '2.5.1', 'scipy': '1.18.0', 'h5py': '3.16.0', 'optimum_interval': '0.3.0', 'luhdm': '0.1.0', 'matplotlib': '3.11.1', 'pandas': '3.0.3', 'python': '3.14.7'}
```

`b_constrained_max_m` is `NaN`, which is how an uncapped cross section is
recorded; a finite value there would mean the impact-parameter integral had been
truncated at that radius.

`v_earth_km_s` and `fid_mu_dex` are **new in `v10`** and have no counterpart in
any earlier file. Both are stated conventions rather than measurements, and both
have a defined meaning by absence: no `v_earth_km_s` means the Galactic rest
frame ([§3](#the-halo-is-evaluated-in-the-laboratory-frame)), no `fid_mu_dex`
means 0.02 dex ([§9](#9-known-limitations)). They also appear inside
`fid_json`, the fidelity block the refiner and the verifier read back.

The SHA-256 of every input (event lists, efficiency table) is recorded, along
with the git commit of the analysis code in
<https://github.com/PolonaiseExperiment/luhdm>, the RNG seed, the Monte-Carlo
fidelity settings (`fid_*`, including the expected-count cap `fid_mu_cap`, the
calibration granularity `fid_mu_dex` and
the two-tier pair `fid_n_mc_hi` / `fid_p_hi_lo`) and
the versions of every package used (`packages_json`). Exclusion limits were
computed with `optimum_interval` 0.3.0.

`git_dirty` tells you honestly whether the working tree had uncommitted changes
at build time, and for these files it is **`True`**: the scan ran from a tree
that carried edits on top of commit `4193b74`. The commit identifies the code
that the tree was based on; the physics settings that the edits touched — the
1 TeV threshold, the absent cap, the mediator-range set, the halo frame, the
Monte-Carlo fidelity — are all recorded as attributes in their own right, so the numbers
remain reproducible from what is in the file even where the commit alone would
not pin them. The efficiency table is pinned the same way, by
`efficiency_npz_sha256` rather than by the commit.

**Physics fiducials** are attributes too, so a reinterpretation does not have to
guess them: `rho_dm_gev4`, `f_x`, `n_neutrons`, `r_eff_m`, `q_thresh_gev`,
`q_hi_ref_gev`, `m_planck_gev`, `t_exposure_s`, `v_earth_km_s`, and the
`m_cut_*` block of
[§5.4](#54-the-mass-cut-m_cut-and-how-to-apply-it).

Some attributes (`efficiency_npz`, `events_dir`, `inputs_json`) and many entries
in the `provenance_*.json` files name the input files the build read, each
paired with its SHA-256. **The digest, not the path, is what identifies an
input**: nothing in the release needs any of these paths to resolve on your
machine.

The build timestamps in the `provenance_*.json` files are a few milliseconds
later than the `created` attributes in the corresponding cubes because the two
records are written one after the other in the same run. They refer to the same
build.

### The environment it was built in

Python 3.14.7 with numpy 2.5.1, scipy 1.18.0, h5py 3.16.0, matplotlib 3.11.1,
pandas 3.0.3, luhdm 0.1.0 and optimum_interval 0.3.0, as recorded in
`packages_json`. **To read the release you need none of that**: any Python with
`numpy` and `h5py` will do, and the examples here were checked on the versions
above. `pandas` is needed only for `to_dataframe()`, `matplotlib` only for the
figure in [§6](#6-worked-example-the-published-limit), and the `h5ls` and
`h5dump` command-line tools come with the HDF5 C library rather than with
`pip install h5py`.

### How it was checked

Every cell of both assembled files was compared back, element by element and
`NaN` against `NaN`, with the per-range shard outputs it was built from, so a
number read here is the number the scan produced. Cells sampled across the
evaluation regimes (Monte Carlo, deterministic shortcut, finite ranges and the
massless limit) were recomputed from scratch and reproduced the stored values on
the toolchain used to build the release. The two files were checked against each
other on everything they are supposed to share — the shared axes, `/detector`,
`/reference_curves` and `/halo` — and found identical. Both cubes' halo pass and
results pass were checked to carry the same `v_earth_km_s`, since a frame
mismatch between them is invisible in the numbers themselves. The 200 m
validation slice was checked against the massless slice
([§9](#9-known-limitations)). The worked
example of [§6](#6-worked-example-the-published-limit) reproduces the published
endpoint numbers and the shipped figure.

---

## 11. Versions

The released files are identified by their `version_tag` attribute, here
`v10.0-night-m0p356mg-q1TeV-nocap-wmargnight-a18iso-mudex0p002-vE245`, which
reads component
by component as: cube version 10.0; `night`, the night selection of
[§3](#3-what-the-data-is); `m0p356mg`, a 0.356 mg sphere; `q1TeV`, the 1 TeV
analysis window; `nocap`, no impact-parameter cap; `wmargnight`, the
arrival-phase-averaged night-selection efficiency of
[§7.1](#71-detector-inputs); `a18iso`, the A18 isotropic-folded projection kernel
of [§5.5](#55-the-sidecar-files); `mudex0p002`, the ten-times-finer Monte-Carlo
calibration granularity; and `vE245`, the laboratory-frame halo. The last two
are what `v10` adds, and are described just below. `mc2tier` has dropped out of
the tag even though the two-tier Monte Carlo is still on
([§10](#names-tags-and-what-they-refer-to)): the tag names the newest changes,
the `fid_*` attributes name everything.

**The laboratory-frame halo** is the reason for this release. Every cube up to
and including `v9.1` evaluated the standard halo model in the *Galactic* rest
frame: an isotropic truncated Maxwellian, `v_0` = 220 km/s, cut off at
`v_esc` = 544 km/s, with the detector at rest in it. `v10` puts the detector
where it is, moving through the halo at `v_earth_km_s` = 245 km/s, and the
arrival-speed distribution becomes the boosted, direction-integrated form of
Lewin & Smith (1996) — the convention of Monteiro (2020) and Tseng (2025), the
two levitated-sensor results this analysis overlays, so all three curves are now
read in one frame. [§3](#the-halo-is-evaluated-in-the-laboratory-frame) sets out
what it means; what it *moves* is:

* **the halo's support**, from `v_esc` = 544 km/s to `v_esc + v_E` = 789 km/s,
  which is the ceiling of every speed integral and of the rate integral;
* **the kinematic wall**, `q_thresh / v_max`, down from 5.51 × 10⁵ to
  3.80 × 10⁵ GeV, because the wall goes as `1/v_max`;
* **the flux-averaged speed `<v>`**, up from 245 972 to 338 173 m/s, a factor
  1.3748, and with it the flux cut `m_cut`, from 6.11 × 10¹⁴ to
  8.40 × 10¹⁴ GeV at `f_DM` = 1 and from 6.11 × 10¹³ to 8.40 × 10¹³ GeV at
  `f_DM` = 0.1;
* **every surface in the cube**, through a faster and differently shaped arrival
  flux. The refined massless mode-1 floor moves from `alpha_n` = 2.548 × 10⁻⁹ at
  7.08 × 10⁶ GeV to 3.207 × 10⁻⁹ at 5.38 × 10⁶ GeV, and the composite
  cross-section floor of the `f_DM` = 0.1 20 µm surface from 3.86 × 10⁻²⁸ to
  4.48 × 10⁻²⁸ cm² at 1.56 × 10⁶ GeV. The mediator-range scan's exclusion
  pinches off at 7.96 µm (24.8 meV) rather than 6.32 µm (31.2 meV);
* **`/halo` and `/reference_curves`**, which are computed in the same frame as
  `/results` and so are not the curves `v9.1` shipped.

A new root attribute carries it, `v_earth_km_s` = 245.0, repeated inside the
`fid_json` fidelity block. It exists in no earlier file, and **absence means the
Galactic rest frame** — that is how to read a pre-`v10` cube.

**Finer Monte-Carlo calibration.** The optimum-interval calibration is sharded
by the expected count rounded onto a log grid, which is what makes the cube
bit-reproducible and also quantizes `extremeness` into steps in `mu`. `v10`
takes that step, `attrs['fid_mu_dex']`, from 0.02 dex to **0.002 dex**. It is
not a statistics upgrade — no number of toy trials touches this floor — but a
resolution one: on the bare-halo plane at high mass, where the sensitivity of
`mu` to the coupling is weakest, the old 0.02-dex bin was up to 0.4 dex wide in
`alpha_n`, and the new one is a tenth of that.
[§9](#9-known-limitations) has the mechanism.

**Two-tier Monte Carlo** was what `v9.1` added, and it is still on here. The
extremeness at a grid point is the fraction of background-free pseudo-experiments
that look less extreme than the data, so it is a Monte Carlo quantile and carries
a Monte Carlo error. Away from the boundary that error is irrelevant — the cell
is at `extremeness` 0.02 or 0.999 and no plausible fluctuation moves it across
0.95. Near the boundary it is the whole answer. So the build evaluates every cell
on a table of `n_mc` = 10 000 trials as before, and any cell whose result lands
in `[p_hi_lo, 1)` with `p_hi_lo` = 0.9 — the band the 95% level lives in — is
re-evaluated on a second table of `n_mc_hi` = 100 000 trials, seeded one step off
the first. Both numbers are stored in the fidelity block
(`attrs['fid_json']`, and `fid_n_mc_hi` / `fid_p_hi_lo` individually), and the
rule is deterministic: the same cell takes the same tier every run, so the cube
and anything that recomputes against it still agree bit for bit. The correction
it applies is a quantile bias of the smaller table, and it is small but not
nothing — within the Galactic-rest-frame `v9` campaign it moved the composite
cross-section floor of the `f_DM` = 0.1 20 µm surface from 4.49 × 10⁻²⁸ to
3.86 × 10⁻²⁸ cm².

`v9.0` was the first public version, `v9.1` superseded it, and `v10` supersedes
`v9.1`. Earlier cubes (`v1`
to `v8`) were internal analysis products and were never distributed; three of
them, `v6.0-night-m0p356mg-bcap10cm-lam4`,
`v7.0-quick-night-m0p356mg-q1TeV-nocap` and
`v8.0-night-m0p356mg-q1TeV-nocap-wmarg`, were prepared for release and superseded
before publication. What changed along the way:

| | v6 (unpublished) | v7.0 (unpublished) | v8.0 (unpublished) | v9.0 | v9.1 | v10 (this release) |
|---|---|---|---|---|---|---|
| analysis window | 0.1 TeV | **1 TeV** | 1 TeV | 1 TeV | 1 TeV | 1 TeV |
| impact-parameter integral | truncated at 10 cm | **uncapped** | uncapped | uncapped | uncapped | uncapped |
| halo frame | Galactic rest (`v_E` = 0, unrecorded) | Galactic rest (unrecorded) | Galactic rest (unrecorded) | Galactic rest (unrecorded) | Galactic rest (unrecorded) | **laboratory, `v_earth_km_s` = 245 km/s**; support to `v_esc + v_E` = 789 km/s |
| kinematic wall | `q_thresh / v_esc`, at the 0.1 TeV window | 5.51 × 10⁵ GeV | 5.51 × 10⁵ GeV | 5.51 × 10⁵ GeV | 5.51 × 10⁵ GeV | **3.80 × 10⁵ GeV**, `q_thresh / (v_esc + v_E)` |
| right-hand edge in mass | closure of the capped cross section, 2.7 × 10¹⁴ GeV | **explicit flux cut `m_cut`**, 6.11 × 10¹⁴ GeV at `N_req = 3` | the same cut, unchanged | the same cut, unchanged | the same cut, unchanged | the same cut, **moved to 8.40 × 10¹⁴ GeV** by the lab-frame `<v>` |
| layout | one file, 2 × 2 hypotheses | **two files, one hypothesis each** | two files | two files | two files | two files |
| mediator ranges | 2 mm, 200 µm, 20 µm + massless | the same three + a **200 m validation slice** + massless | the same five | the same five | the same five | the same five |
| detection efficiency | impulse arrival phase fixed | the same fixed-phase curves | **averaged over the arrival phase**, full run | **the same average over the night-selected segments** | unchanged | unchanged |
| projection kernel | planar-signed (unrecorded) | planar-signed (unrecorded) | planar-signed (unrecorded) | **`isotropic-folded`, recorded in `projection_kernel`** | unchanged | unchanged |
| Monte Carlo | single tier, `n_mc` = 10 000 | single tier | single tier | single tier | **two tiers: 10⁵ trials for cells in `[0.9, 1)`** | unchanged, and no longer named in the tag |
| MC calibration granularity | `mu_dex` = 0.02 dex (unrecorded) | 0.02 (unrecorded) | 0.02 (unrecorded) | 0.02 (unrecorded) | 0.02 (unrecorded) | **0.002 dex, recorded in `fid_mu_dex`** |
| grid | 44 couplings × 119 masses | 44 × 119 | 44 × 119 | 22 × 60 | **44 × 119 restored** (0.233 dex in `alpha_n`, 0.119 dex in mass) | 44 × 119 |
| refined contours | — | — | shipped | shipped, and **truncated at `m_cut`** where the surface does not close | the same, and **the `f_DM` = 0.1 plane at all three sensor modes** | **mode 1 only**: eight surfaces, four per cube, against eleven in `v9.1` |

The halo-frame row is the step from `v9.1` to this release, and unlike anything
above it, it is a change of *physics convention* rather than of resolution or
scope: it moves the wall, the flux cut and every surface at once, and there is
no factor that converts one frame's cube into the other's
([§9](#9-known-limitations)). The finer `mu_dex` row rides along with it,
lifting a quantization floor that no amount of Monte Carlo could. The kernel
row was the substantive change of `v9.0` and is still a live convention: it is a
statement about how the three-dimensional impulse is projected onto the measured
axis, fixed for the whole cube, and it moves every cross section by a constant
factor in the Coulomb limit ([§5.5](#55-the-sidecar-files)).

The last row is a **reduction in scope, and it is the one thing here that a
`v9.1` user loses**. `v10` ships refined contours for sensor mode 1 only —
`luhdm_contours_v10_A_f1_atm.json` and `luhdm_contours_v10_B_f0p1_noatm.json`,
four surfaces each. There are no `_mode2` and `_mode3` contour files, and the
mediator-range sidecar likewise carries mode 1 alone (`modes` = `[1]`). Mode 1
is the mode the paper reports; modes 2 and 3 are still present in both cubes,
at full grid resolution, and are contoured per
[§5.3](#53-the-exclusion-convention) as any other surface is
([§5.5](#55-the-sidecar-files) says how to refine them yourself). Nothing was
removed from the cubes.

For orientation, the massless mode-1 floor taken from the refined boundary is
`alpha_n` = 3.207 × 10⁻⁹ at 5.38 × 10⁶ GeV, against 2.548 × 10⁻⁹ at
7.08 × 10⁶ GeV in `v9.1`; on the grid it is 3.993 × 10⁻⁹ at 8.13 × 10⁶ GeV,
against 2.826 × 10⁻⁹ at the same mass.

**Published files are never edited in place.** Any correction or extension is
issued as a new version with a new tag, new digests and its own DOI, and this
section will carry the changelog. Cite the version you used.

---

## 12. How to cite

Please cite **both** the paper and the dataset.

The paper:

```bibtex
@article{Uitenbroek:2026uhdm,
    author  = "Uitenbroek, Dennis G. and Amaral, Dorian W. P. and Qin, Juehang
               and Langendorff, Jurriaan and Gingerich, Andrew and
               Oosterkamp, Tjerk H. and Tunnell, Christopher D.",
    title   = "{First Search for Ultraheavy Dark Matter Using a Magnetically
               Levitated Particle}",
    year    = "2026",
    note    = "arXiv identifier and journal reference to be assigned"
}
```

The dataset:

```bibtex
@misc{luhdm_datarelease,
    author = "Uitenbroek, Dennis G. and Amaral, Dorian W. P. and Qin, Juehang
              and Langendorff, Jurriaan and Gingerich, Andrew and
              Oosterkamp, Tjerk H. and Tunnell, Christopher D.",
    collaboration = "POLONAISE",
    title  = "{Data release for: First Search for Ultraheavy Dark Matter
              Using a Magnetically Levitated Particle}",
    year   = "2026",
    note   = "Version v10.0-night-m0p356mg-q1TeV-nocap-wmargnight-a18iso-mudex0p002-vE245. DOI to be assigned"
}
```

`@misc` rather than `@dataset` because classic BibTeX, which REVTeX uses, does
not define a `dataset` entry type.

The same metadata is in [`CITATION.cff`](CITATION.cff) in machine-readable
form, which GitHub, Zenodo and Zotero read directly.

The dataset DOI is not yet minted. Until it is, cite the version tag
`v10.0-night-m0p356mg-q1TeV-nocap-wmargnight-a18iso-mudex0p002-vE245` and the digests in
[§10](#10-integrity-provenance-and-environment), which identify the files
unambiguously. This section and `CITATION.cff` will be updated with the DOI when
it is assigned.

The optimum-interval implementation used to compute the limits is a separate
package, <https://github.com/tunnell/optimum_interval>, version 0.3.0. The
apparatus is described in D. G. Uitenbroek, J. Langendorff and T. H. Oosterkamp,
*Picometer control of a levitating milligram gravity sensor*,
[arXiv:2605.28479](https://arxiv.org/abs/2605.28479).

---

## 13. License and contact

Two licences, because this release is part data and part code.

**The data is CC BY 4.0.** `luhdm_datarelease_v10_A_f1_atm.h5`,
`luhdm_datarelease_v10_B_f0p1_noatm.h5`, and the `provenance_*.json`,
`SHA256SUMS`, `CITATION.cff`, `exclusion_massless_mode1.png` and this `README.md`
that travel with them, are released under the Creative Commons Attribution 4.0
International licence. The full text is in [`LICENSE`](LICENSE) in this
directory; the human-readable summary is at
<https://creativecommons.org/licenses/by/4.0/>. You may share and adapt the data
for any purpose, including commercially, provided you give attribution.
**Attribution here means citing the paper and the dataset as set out in
[§12](#12-how-to-cite).**

**The code is GPL-3.0-or-later.** `luhdm_release.py` in this directory, and
everything in the analysis repository, are released under the GNU General Public
License, version 3 or (at your option) any later version. The full text is in
[`LICENSE` at the repository root](https://github.com/PolonaiseExperiment/luhdm/blob/main/LICENSE).
The `LICENSE` file next to `luhdm_release.py` is the CC BY 4.0 text and covers
the data; it does not cover the reader, which carries its own
`SPDX-License-Identifier: GPL-3.0-or-later` header.

Nothing you compute *from* the data is a derivative of the code, so the copyleft
does not reach your analysis; using `luhdm_release.py` to read the cubes leaves
your own work entirely yours.

**Contact.** Dorian W. P. Amaral, <damaral@ifae.es>. For anything about the
files themselves, open an issue at
<https://github.com/PolonaiseExperiment/luhdm/issues>, which is the route most
likely to outlive any individual affiliation.

---

## 14. Glossary

Terms that appear in dataset names and attributes, and are not standard usage.

| term | meaning |
|---|---|
| **blip** | A reconstructed transient impulse in the sensor readout, before the analysis quality selection. `all_blips_mode{n}` is the pre-selection list; `events_mode{n}` is what survives and what the limit is set on. |
| **night selection** | The data selection this release encodes: 19:00 to 07:00 local time, excluding 19 and 20 January 2026 and the periods when the calibration drive was on. See [§3](#3-what-the-data-is). |
| **mode** | One of the three translational eigenmodes of the levitated sphere (51.2365, 59.4663, 94.86 Hz), demodulated from the same readout stream and analysed independently. Indexed 1, 2, 3 by ascending frequency. |
| **extremeness** | The optimum-interval statistic: the probability that a background-free pseudo-experiment under a hypothesis looks less extreme than the observed data. Exclusion at confidence `C` is `extremeness >= C`, inside the mass window. |
| **massless sentinel** | The final `inf` entry of `axes/lambda_m`, holding the analytic Coulomb-like limit of a massless mediator, where `m_phi_gev` is exactly 0. |
| **kinematic wall** | The lowest mass that can deliver a threshold impulse at any coupling, `q_thresh / v_max` = 3.80 × 10⁵ GeV, where `v_max` is the top of the halo's support. The left edge of every excluded region here. It sits lower than the 5.51 × 10⁵ GeV of the Galactic-rest-frame releases because `v_max` is larger in the laboratory frame. See [§3](#the-analysis-window-starts-at-1-tev). |
| **halo frame**, **lab frame** (`v_earth_km_s`, `vE245`) | Which frame the standard halo model is evaluated in. `v_earth_km_s` = 245.0 here: the Earth's speed through the halo, so the arrival-speed distribution is the Galactic Maxwellian boosted into the detector frame and integrated over arrival direction (Lewin & Smith 1996; the convention of Monteiro 2020 and Tseng 2025). Its support runs to `v_esc + v_E` = 789 km/s instead of `v_esc` = 544 km/s. **Absence of the attribute means 0, the Galactic rest frame**, which is what every release before `v10` carried. See [§3](#the-halo-is-evaluated-in-the-laboratory-frame). |
| **`<v>`** | The flux-weighted first moment of the halo speed distribution, 338 173 m/s here, the same convention the `n_transit` surface uses. It is what normalises the transit rate, and therefore the flux cut `m_cut`. See [§5.4](#54-the-mass-cut-m_cut-and-how-to-apply-it). |
| **`m_cut`** | The flux cut: the largest mass for which the halo delivers at least `N_req` = 3 transits within a 10 cm aperture during the exposure. The right edge of every excluded region here. **Not applied to the stored surfaces**; see [§5.4](#54-the-mass-cut-m_cut-and-how-to-apply-it). |
| **`b_constrained_max`** | The outer cutoff of the impact-parameter integral in earlier versions of this analysis. `NaN` here, meaning uncapped. |
| **validation slice** | The `lambda` = 200 m entry of the mediator-range axis: a convergence check of the finite-range cross section against the analytic massless limit, not a physics result. See [§4.1](#41-axes). |
| **`df`** | Degrees of freedom of the detection-efficiency fit. Two hypotheses are shipped; the analysis used `df = 3`. |
| **`m0p356mg`, `q1TeV`, `nocap`, `wmarg`** | Version-tag fragments: the 0.356 mg sphere mass, the 1 TeV analysis window, the absence of an impact-parameter cap, and the efficiency curves averaged over the impulse arrival phase. |
| **MC calibration granularity** (`mu_dex`, `mudex0p002`) | The log step, in dex, onto which the expected count `mu` is rounded before the seeded optimum-interval table is drawn. It is what makes a cell's `extremeness` a pure function of `(rounded mu, seed, n_mc)` — hence bit-reproducible regardless of worker count — and also what quantizes a boundary into steps. `attrs['fid_mu_dex']` = 0.002 here, ten times finer than the 0.02 of every earlier cube; **absence of the attribute means 0.02**. See [§9](#9-known-limitations). |
| **two-tier Monte Carlo** (`mc2tier`) | The rule `v9.1` introduced and this release keeps: every cell is evaluated on a table of `fid_n_mc` = 10 000 optimum-interval trials, and any cell landing in `[fid_p_hi_lo, 1)` = `[0.9, 1)` — the band the 95% boundary lives in — is re-evaluated on a second, independently seeded table of `fid_n_mc_hi` = 100 000. It buys the boundary a factor of ten in Monte-Carlo resolution and changes nothing about determinism: the tier a cell takes is a function of the cell. The fragment is no longer in the version tag; the `fid_*` attributes are the authority. See [§11](#11-versions). |
