# POLONAISE ultra-heavy dark matter — data release

> **This document describes the release tagged `v5.0-night-m0p356mg-bcap10cm`.**
> Every version tag, axis length, event count and physics constant quoted below
> is also stored inside the HDF5 file itself and is read back from it, so
> **trust the file's own attributes over the text**:
> `python luhdm_release.py luhdm_datarelease_v5.h5` prints all of them, and
> [§2](#2-you-do-not-need-to-install-anything) shows how to read them with five
> lines of `h5py`. Every worked example below was executed against the released
> file and its output pasted verbatim.

Data release DOI to be assigned.

---

## 1. What this is

A search for **ultra-heavy dark matter** with a **levitated micro-sensor**. A
magnetically levitated microsphere (0.356 mg, effective radius 0.26 mm,
N ≈ 1.07 × 10²⁰ neutrons) is monitored for sudden momentum impulses. A
dark-matter particle of mass `m_DM` flying past the sphere with impact
parameter `b` transfers momentum through a new Yukawa-type interaction with the
sphere's neutrons, of strength `alpha_n` per neutron and range `lambda`
(mediator mass `m_phi = 1/lambda` in natural units; `lambda → ∞` is the massless
/ Coulomb-like limit). Because the dark-matter number density falls as `1/m_DM`
while the momentum kick grows with it, a single sphere is sensitive to masses far
above the WIMP range — here 10⁵ to 1.22 × 10¹⁹ GeV. Strongly coupled candidates
also lose energy crossing the atmosphere and overburden before reaching the
detector, which both attenuates and reshapes the arrival-velocity distribution;
this release carries the analysis **with and without** that propagation.

The dataset is the **whole limit-setting calculation**, not just the final
contour: for every point of a
(`f_DM` × `atmosphere` × `sensor mode` × `alpha_n` × `m_DM` × `lambda`) grid it
stores the expected signal, the expected number of in-reach transits, and the
optimum-interval extremeness from which the exclusion at any confidence level is
a level set. You can therefore re-derive the published limit, quote it at a
different confidence level, reinterpret it for a different dark-matter fraction,
or compare against it, without rerunning any of the physics.

### The analysis selection it encodes

The release describes the **night data selection**: total live time
`T_obs = 790 778 s` (219.66 h), recorded in `attrs['t_exposure_s']` and in
`detector/exposure_s`. After the full selection above the 100 GeV analysis
threshold, the surviving impulse candidates number **8, 26 and 126** for sensor
modes 1, 2 and 3. Those lists are shipped in `detector/events_mode{1,2,3}`, and
the larger pre-selection impulse lists (66, 99 and 443 entries) are shipped
alongside as `detector/all_blips_mode{1,2,3}` so the selection can be inspected
or varied.

### The interaction model, in one paragraph

The differential cross section for a flyby is built from the impulse delivered
along a straight-line trajectory. Two cutoffs bound the impact-parameter
integral. The **outer** cutoff is `b_constrained_max = 0.1 m`
(`attrs['b_constrained_max_m']`), applied identically in the cross section and
in the geometric transit count. The **inner** cutoff is the sphere's own
effective radius: no trajectory approaches closer than `R_eff`
(`attrs['r_eff_m']` = 2.6 × 10⁻⁴ m), so in the massless limit the impulse
saturates at

```
q_max = 2 alpha_n / (v R_eff)
```

and `dsigma/dq` vanishes identically above it. This is the massless limit of the
finite-range cutoff the Yukawa branch already applies, and it removes the
contribution of arbitrarily close approaches.

---

## 2. You do not need to install anything

The release is plain HDF5 with dimension scales and per-dataset `units` /
`description` attributes. `h5py`, `h5ls`, `h5dump`, MATLAB, Julia, IDL and
xarray (via h5netcdf) all open it. There is nothing to `pip install` beyond
whatever HDF5 binding you already have.

```python
import h5py, numpy as np

with h5py.File("luhdm_datarelease_v5.h5", "r") as f:
    print(f.attrs["version_tag"], f["results/extremeness"].shape)
    print({d: len(f["axes"][d]) for d in f["axes"]})
    p = f["results/extremeness"][0, 0, 0, :, :, -1]   # f_DM idx0, atm idx0, mode idx0, massless
    print("excluded cells at 95% CL:", int((p >= 0.95).sum()), "of", p.size)
```

```
v5.0-night-m0p356mg-bcap10cm (2, 2, 3, 44, 119, 55)
{'alpha_halo_n': 64, 'alpha_n': 44, 'atmosphere': 2, 'f_dm': 2, 'lambda_m': 55, 'm_phi_gev': 55, 'mass_gev': 119, 'mass_halo_gev': 64, 'mode': 3}
excluded cells at 95% CL: 1517 of 5236
```

Note the hard-coded indices in that snippet — **do not do that in real code.**
Resolve them from the axes instead, as in [§6](#6-the-exclusion-convention). An
optional single-file reader that does it for you is described in
[§7](#7-optional-the-standalone-reader).

From the shell:

```console
$ h5ls -r luhdm_datarelease_v5.h5 | head
```

```
/                        Group
/axes                    Group
/axes/alpha_halo_n       Dataset {64}
/axes/alpha_n            Dataset {44}
/axes/atmosphere         Dataset {2}
/axes/f_dm               Dataset {2}
/axes/lambda_m           Dataset {55}
/axes/m_phi_gev          Dataset {55}
/axes/mass_gev           Dataset {119}
/axes/mass_halo_gev      Dataset {64}
```

---

## 3. File layout

Every dataset under `/results` carries HDF5 **dimension scales**, so its axes
are self-identifying: `ds.attrs['DIMENSION_LABELS']` names them in order and
each is attached to the matching `/axes/<name>` dataset. Read the axis order
from there rather than assuming it.

Symbolic lengths, with this release's values in the last column:

| symbol | meaning | value |
|---|---|---|
| `n_f` | dark-matter fraction hypotheses | 2 |
| `n_atm` | atmosphere hypotheses | 2 |
| `n_mode` | sensor modes | 3 |
| `n_alpha` | coupling grid points | 44 |
| `n_mass` | dark-matter mass grid points | 119 |
| `n_lam` | mediator ranges, **finite + 1 massless sentinel** | 55 (54 + 1) |
| `n_halo` | coupling / mass points of the halo diagnostic maps | 64 |

### `/axes` — the coordinate arrays

| dataset | shape | dtype | units | meaning |
|---|---|---|---|---|
| `f_dm` | (`n_f`,) | f8 | 1 | fraction of the local DM density carried by this species. A **pure flux normalisation**: `mu` and `n_transit` are exactly linear in it. Values `[0.1, 1.0]`. |
| `atmosphere` | (`n_atm`,) | i1 | bool | `1` = propagation through atmosphere/overburden applied; `0` = bare halo flux. Stored in the order `[1, 0]` — **index 0 is atmosphere ON.** |
| `mode` | (`n_mode`,) | u1 | 1 | sensor mode label (1, 2, 3). Modes differ in threshold and efficiency, hence in their event lists. |
| `alpha_n` | (`n_alpha`,) | f8 | 1 | per-neutron coupling, log-spaced 10⁻¹⁰ … 1 (0.2326 dex per step). This is the parameter the limit is set on. |
| `mass_gev` | (`n_mass`,) | f8 | GeV | dark-matter mass, log-spaced 10⁵ … 1.22 × 10¹⁹ (the Planck mass). |
| `lambda_m` | (`n_lam`,) | f8 | m | mediator range: finite values **ascending**, then `inf` last (see below). |
| `m_phi_gev` | (`n_lam`,) | f8 | GeV | mediator mass `1/lambda` in natural units, parallel to `lambda_m`; exactly `0.0` at the `inf` entry. |
| `alpha_halo_n` | (`n_halo`,) | f8 | 1 | coupling axis of the `/halo` maps (extends down to 2 × 10⁻¹¹). |
| `mass_halo_gev` | (`n_halo`,) | f8 | GeV | mass axis of the `/halo` maps. |

**The massless slice.** The last element of `lambda_m` is `inf`, with
`m_phi_gev = 0.0`: the analytic, Coulomb-like limit of a massless mediator.
The number of finite entries is the `n_finite` attribute of `axes/lambda_m`
(54 here), so the finite part is `lambda_m[:n_finite]` and the massless index
is the one where `~np.isfinite(lambda_m)`. Never assume it is `-1` by
arithmetic on a hard-coded length — read `n_finite`.

**Named ranges.** `axes/lambda_m` carries a `tags_json` attribute mapping short
names to exact axis values, so tag-based slicing is an integer index with no
tolerance games:

```python
import h5py, json
with h5py.File("luhdm_datarelease_v5.h5", "r") as f:
    tags = json.loads(f["axes/lambda_m"].attrs["tags_json"])
print(tags)
```

```
{'2m': 2.0, '20cm': 0.2, '2cm': 0.02, '2mm': 0.002, '200um': 0.0002, '20um': 2e-05, '10um': 1e-05, '2um': 2e-06}
```

The eight named ranges span 2 m down to 2 µm. The 0.2 m point sits just above
the 0.1 m impact-parameter cap and is useful for seeing where the cap, rather
than the mediator range, starts to set the reach — see
[§10](#10-known-limitations).

### `/results` — the analysis cube

| dataset | shape | dtype | units | meaning |
|---|---|---|---|---|
| `extremeness` | (`n_f`, `n_atm`, `n_mode`, `n_alpha`, `n_mass`, `n_lam`) | f4 | 1 | optimum-interval extremeness: the probability that a background-free pseudo-experiment under this hypothesis looks *less* extreme than the data. **Exclusion at confidence `C` is the level set `extremeness >= C`.** `NaN` where `status == 1`. |
| `mu` | same | f4 | counts | expected detected signal events (efficiency folded in). `NaN` where `status == 1`. Exactly linear in `f_DM`. |
| `status` | same | u1 | enum | how the cell was obtained — see [§8](#8-status-codes-and-nan-policy). |
| `n_transit` | (`n_f`, `n_atm`, `n_alpha`, `n_mass`, `n_lam`) | f4 | counts | expected number of dark-matter transits within threshold reach. **No `mode` axis** — the flyby rate does not depend on which sensor mode you read out. Clipped at ≥ 0 (`clipped_nonnegative` attribute); exactly linear in `f_DM`. `NaN` for cells where the calculation raised. |

### `/detector` — the analysis inputs

| dataset | shape | dtype | units | meaning |
|---|---|---|---|---|
| `exposure_s` | scalar | f8 | s | total live time (790 778 s). |
| `events_mode{1,2,3}` | (8,) (26,) (126,) | f8 | GeV | **the analysis event lists** — momentum kicks after the full selection. |
| `all_blips_mode{1,2,3}` | (66,) (99,) (443,) | f8 | eV | all reconstructed blip momenta before selection; context only. |
| `q_gev_{1,2,3}` | (400,) | f8 | GeV | momentum grid of the measured efficiency curves. |
| `eff_{1,2,3}_df{2,3}` | (400,) | f8 | 1 | measured detection efficiency ε(q) per mode, for the two degrees-of-freedom hypotheses. The analysis used `df` = `attrs['df']` (3). |

Event-list lengths are data, not schema: they change with the selection.
[§7.1](#71-the-detector-inputs-efficiency-exposure-and-events) works through
reading these.

### `/halo` — flux diagnostics (optional)

| dataset | shape | dtype | units | meaning |
|---|---|---|---|---|
| `n_transit` | (`n_halo`, `n_halo`, `n_lam`) | f4 | counts | unattenuated-halo expected transits on the coarser diagnostic grid. |
| `bmax` | (`n_halo`, `n_halo`, `n_lam`) | f4 | m | flux-averaged threshold reach √⟨b²⟩ — how far out a transit can still push the sensor over threshold. |

These are an **independently sampled, coarser** map (own `alpha_halo_n` /
`mass_halo_gev` axes) used for intuition and figures, and they are stored at
the **baseline `f_DM`** (`attrs['f_dm_default']` = 0.1) with no `f_dm` axis —
multiply by `f_DM / 0.1` for another fraction. Where their grid meets the main
cube they agree with `/results/n_transit` at `atmosphere = 0` to ≈ 0.2%, the
residual being Monte-Carlo sampling noise between the two passes. For anything
quantitative use `/results`.

### `/reference_curves` — showcase spectra

Single-point illustrations at `m_DM = 10⁸ GeV`, `alpha_n = 10⁻³`, for figures:
`v` + `fv_shm` / `fv_<tag>` (500,) are arrival-speed distributions (each
`fv_<tag>` carries a `survival_fraction` attribute), and `q_gev` +
`drdq_<tag>` (160,) are raw differential rates `dR/dq` in s⁻¹ GeV⁻¹ with no
efficiency applied. There is one curve per named range, including `20cm`. Not
needed to use the limits.

---

## 4. The 2 × 2 hypothesis structure

The two leading axes of every `/results` dataset are *hypotheses*, not
measurements. Every combination is fully computed, so the cube contains four
parallel analyses:

|  | `atmosphere = 1` (attenuated) | `atmosphere = 0` (bare halo) |
|---|---|---|
| **`f_dm = 0.1`** | the baseline published result | what the limit would be with no overburden |
| **`f_dm = 1.0`** | this species is all of the dark matter | ditto, no overburden |

* **`f_dm`** is the fraction of the local dark-matter density carried by this
  species. It enters only as a flux normalisation, so `mu` and `n_transit`
  scale exactly linearly with it (the `extremeness` does not — it is a
  non-linear function of `mu`). The baseline `f_DM = 0.1` is the conventional
  choice that evades self-interaction constraints; `attrs['f_dm_default']`
  records which one the headline numbers use.
* **`atmosphere`** selects whether the arrival flux has been propagated through
  the atmosphere and overburden. With attenuation ON, strongly coupled
  candidates are slowed or stopped before reaching the sensor, so the exclusion
  **closes from above** and becomes a two-sided band in `alpha_n`. With it OFF
  there is no such ceiling — see [§10](#10-known-limitations).

Select them by value, never by position:

```python
import h5py, numpy as np

with h5py.File("luhdm_datarelease_v5.h5", "r") as f:
    i_f   = int(np.flatnonzero(f["axes/f_dm"][:] == 0.1)[0])
    i_atm = int(np.flatnonzero(f["axes/atmosphere"][:] == 1)[0])   # 1 = attenuation ON
    i_mod = int(np.flatnonzero(f["axes/mode"][:] == 2)[0])
    print("indices:", i_f, i_atm, i_mod)
    print("axes/atmosphere is stored as", f["axes/atmosphere"][:], "-> ON is index", i_atm)
```

```
indices: 0 0 1
axes/atmosphere is stored as [1 0] -> ON is index 0
```

`axes/atmosphere` is `[1, 0]`, so **`atmosphere = 0` (bare halo) is index 1, not
index 0.** This is the single easiest thing to get wrong in this file.

---

## 5. Selecting a mediator range

`lambda_m` values are exact floats: the named tags and the round decades are
exact axis members, so `==` works for them. For anything else, snap to the
nearest point in log space and *check* what you got.

```python
import h5py, numpy as np

with h5py.File("luhdm_datarelease_v5.h5", "r") as f:
    lam = f["axes/lambda_m"][:]
    n_finite = int(f["axes/lambda_m"].attrs["n_finite"])

i_massless = int(np.flatnonzero(~np.isfinite(lam))[0])
i_200um    = int(np.flatnonzero(lam == 2e-4)[0])
i_20cm     = int(np.flatnonzero(lam == 0.2)[0])
i_near     = int(np.argmin(np.abs(np.log10(lam[:n_finite]) - np.log10(3.7e-5))))

print(f"massless -> index {i_massless} (lambda={lam[i_massless]})")
print(f"200 um   -> index {i_200um}")
print(f"0.2 m    -> index {i_20cm}")
print(f"3.7e-5 m -> nearest index {i_near}, lambda = {lam[i_near]:.6g} m")
```

```
massless -> index 54 (lambda=inf)
200 um   -> index 29
0.2 m    -> index 47
3.7e-5 m -> nearest index 24, lambda = 3.0831e-05 m
```

---

## 6. The exclusion convention

`extremeness` is the probability that a background-free pseudo-experiment under
a given hypothesis looks *less* extreme than the observed data, computed with
Yellin's optimum-interval method. A grid point is **excluded at confidence
`C`** when

```
extremeness >= C          # C = 0.95 for the published 95% CL limits
```

`attrs['confidence_recommended']` records the level the release is quoted at
(0.95). A 2-D exclusion region is just that level set — contour it directly.

For a **boundary in the coupling** the project quotes the *interpolated*
crossing rather than the last excluded grid point: at each mass, take the run of
`alpha_n` indices with `extremeness >= C` and find where the level is crossed by
linear interpolation in `log10(alpha_n)` between the two bracketing points. If
the excluded run already starts at the first (or ends at the last) scanned
coupling there is nothing to bracket, and the edge **saturates** at that end of
the grid — such an edge is a property of the scan range, not a measurement.

This is the complete implementation, in plain numpy:

```python
import h5py, numpy as np

with h5py.File("luhdm_datarelease_v5.h5", "r") as f:
    alpha = f["axes/alpha_n"][:]
    mass  = f["axes/mass_gev"][:]
    i_f   = int(np.flatnonzero(f["axes/f_dm"][:] == 0.1)[0])
    i_atm = int(np.flatnonzero(f["axes/atmosphere"][:] == 1)[0])
    i_mod = int(np.flatnonzero(f["axes/mode"][:] == 1)[0])
    i_lam = int(np.flatnonzero(f["axes/lambda_m"][:] == 2e-4)[0])
    p = f["results/extremeness"][i_f, i_atm, i_mod, :, :, i_lam]   # (alpha, mass)

C = 0.95
lo = np.full(mass.size, np.nan)
hi = np.full(mass.size, np.nan)
for j in range(mass.size):
    above = np.flatnonzero(p[:, j] >= C)              # NaN compares False -> not excluded
    if above.size == 0:
        continue                                      # nothing excluded at this mass
    a, b = above[0], above[-1]
    lo[j] = alpha[0] if a == 0 else 10 ** np.interp(
        C, p[a - 1:a + 1, j], np.log10(alpha[a - 1:a + 1]))
    hi[j] = alpha[-1] if b == alpha.size - 1 else 10 ** np.interp(
        -C, -p[b:b + 2, j], np.log10(alpha[b:b + 2]))

ok = np.isfinite(lo)
print(f"excluded at {ok.sum()} of {mass.size} masses, "
      f"m_DM = {mass[ok].min():.4g} .. {mass[ok].max():.4g} GeV")
j = np.flatnonzero(ok)[0]
print(f"first excluded mass {mass[j]:.6g} GeV: alpha_n in [{lo[j]:.6g}, {hi[j]:.6g}]")
```

```
excluded at 33 of 119 masses, m_DM = 5.203e+05 .. 3.438e+09 GeV
first excluded mass 520302 GeV: alpha_n in [8.84157e-08, 0.000116647]
```

The negations in the `hi` branch exist only to make the descending side
increasing for `np.interp`, which requires increasing `x`. This snippet
reproduces the published boundary exactly — it is the same arithmetic the
analysis code runs, and the standalone reader's `excluded_band()` is a wrapper
around it.

---

## 7. Optional: the standalone reader

[`luhdm_release.py`](luhdm_release.py) is a **single self-contained file**.
Copy it next to the HDF5 and import it — there is no package to install, no
relative imports, no data files, and it never imports the analysis code. Its
only requirements are `numpy` and `h5py`; `pandas` is imported lazily inside
`to_dataframe()` and is optional. It is also readable as reference: everything
it does is a few lines of numpy.

What it adds over raw `h5py`:

* **selection by physical value** — `mode=2`, `lam='200um'` or `lam=20e-6` or
  `lam='massless'`, `f_dm=0.1`, `atmosphere=True`, `mass=1e8`, `alpha=1e-3` —
  with errors that list the available values when a request misses;
* slices returned **with their axes attached**, and the resolved grid values
  echoed back so you can see what `mass=1e8` snapped to;
* `excluded_band()` implementing [§6](#6-the-exclusion-convention), including
  saturation flags and an explicit count of undefined (status-1) cells;
* `efficiency_curve()`, `events()`, `all_blips()` and `exposure_s` for the
  detector inputs — see [§7.1](#71-the-detector-inputs-efficiency-exposure-and-events);
* `summary()` — everything in this README, read out of the file you actually
  have;
* `to_dataframe()` — tidy long-format table for a chosen hypothesis;
* it is **schema-driven**: axis names, lengths, units, tags and the massless
  sentinel come from the file, so it keeps working across cube versions.

```console
$ python luhdm_release.py luhdm_datarelease_v5.h5      # print the summary
```

```
==============================================================================
POLONAISE UHDM data release   v5.0-night-m0p356mg-bcap10cm
==============================================================================
file            : luhdm_datarelease_v5.h5
format          : luhdm-datarelease version 2 (schema 1)
created         : 2026-08-08T12:10:42.831616+00:00
exposure        : 790,778 s  (219.66 h)
impact-param cap: 0.1 m
recommended CL  : 0.95

hypothesis axes
  f_dm          : [0.1, 1.0]  (default 0.1) -- DM fraction, pure flux scale
  atmosphere    : [1, 0] -> [True, False]  (1/True = attenuation applied)
  mode          : [1, 2, 3]
```

```python
import numpy as np
import luhdm_release

with luhdm_release.open_release("luhdm_datarelease_v5.h5") as rel:
    print(rel.version_tag, "exposure", rel.exposure_s, "s, cap", rel.b_constrained_max_m, "m")

    sl = rel.get("extremeness", mode=1, lam="200um", f_dm=0.1, atmosphere=True)
    print(sl)                                   # axes travel with the array
    print("excluded cells:", int((sl.values >= 0.95).sum()))

    band = rel.excluded_band(mode=1, lam="200um")          # 95% CL, per mass
    print("mass range:", band.mass_range)
    j = int(np.flatnonzero(np.isfinite(band.alpha_lo))[0])
    print("at m =", band.mass_gev[j], "alpha_n in",
          (band.alpha_lo[j], band.alpha_hi[j]))

    cell = rel.cell(mode=2, alpha=1e-3, mass=1e8, lam="200um")
    print({k: cell[k] for k in ("mass_gev", "alpha_n", "mu", "extremeness", "status")})

    df = rel.to_dataframe(mode=1, lam="massless", mass=1e12)   # needs pandas
    print(df.shape, list(df.columns)[:7])
```

```
v5.0-night-m0p356mg-bcap10cm exposure 790778.0 s, cap 0.1 m
<Slice extremeness [1] (alpha_n=44, mass_gev=119) at f_dm=0.1, atmosphere=True, mode=1, lambda_m=0.0002>
excluded cells: 587
mass range: (520302.1070058871, 3437837694.994715)
at m = 520302.1070058871 alpha_n in (np.float64(8.841573381446489e-08), np.float64(0.00011664691612510738))
{'mass_gev': 96471330.72266711, 'alpha_n': 0.0009478599776522384, 'mu': 148.25978088378906, 'extremeness': 1.0, 'status': 3}
(44, 13) ['f_dm', 'atmosphere', 'mode', 'alpha_n', 'mass_gev', 'lambda_m', 'm_phi_gev']
```

The reader's docstrings carry the full API; `help(luhdm_release)` and
`help(luhdm_release.Release)` work offline.

### 7.1 The detector inputs: efficiency, exposure and events

The measured detection efficiency travels **inside the file**, so a
reinterpretation never has to reconstruct it. Each mode has its own momentum
grid `detector/q_gev_{mode}` and its efficiency `detector/eff_{mode}_df{2,3}`,
where `df` is the degrees-of-freedom hypothesis of the efficiency fit; the
analysis used `attrs['df']` = 3. The live time is `detector/exposure_s` and the
candidate lists are `detector/events_mode{n}`.

With the standalone reader (`df` defaults to the file's own `attrs['df']`):

```python
import numpy as np
import luhdm_release

with luhdm_release.open_release("luhdm_datarelease_v5.h5") as rel:
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
mode 1:   8 candidates, q =  1520.7 ..   12790.7 GeV | eff grid 400 pts, 50% at q =   958.1 GeV
mode 2:  26 candidates, q =   554.2 ..    8473.1 GeV | eff grid 400 pts, 50% at q =   749.4 GeV
mode 3: 126 candidates, q =  1569.0 ..   17234.7 GeV | eff grid 400 pts, 50% at q =  4386.5 GeV
mode 1 efficiency at q = 2000 GeV: df=2 0.9963, df=3 0.9963
```

The analysis package exposes the same three quantities through the same method
names, so code written against either reader ports over unchanged:

```python
import numpy as np
from luhdm import release

rel = release.open_release("luhdm_datarelease_v5.h5")
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
mode 1:   8 candidates, q =  1520.7 ..   12790.7 GeV | eff grid 400 pts, 50% at q =   958.1 GeV
mode 2:  26 candidates, q =   554.2 ..    8473.1 GeV | eff grid 400 pts, 50% at q =   749.4 GeV
mode 3: 126 candidates, q =  1569.0 ..   17234.7 GeV | eff grid 400 pts, 50% at q =  4386.5 GeV
```

Or straight from `h5py`, with no reader at all:

```python
import h5py

with h5py.File("luhdm_datarelease_v5.h5", "r") as f:
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
  mode 1: q_gev_1 (400,), eff_1_df3 (400,), events_mode1 (8,), eff_max 1.0000
  mode 2: q_gev_2 (400,), eff_2_df3 (400,), events_mode2 (26,), eff_max 1.0000
  mode 3: q_gev_3 (400,), eff_3_df3 (400,), events_mode3 (126,), eff_max 1.0000
```

The efficiency is already folded into `results/mu`; you need these curves only
if you are recomputing rates yourself or folding a different spectrum.

---

## 8. Status codes and NaN policy

`/results/status` records how each cell was obtained. Its `description`
attribute is the authoritative short form; the long form:

| code | meaning | `extremeness` |
|---|---|---|
| `0` | the optimum-interval Monte Carlo ran | MC value in (0, 1) |
| `1` | **the cell raised an exception** | `NaN` (and `mu`, `n_transit` are `NaN` too) |
| `2` | `mu` below the MC floor (0.2) — nothing to expect | exactly `0.0` |
| `3` | `mu` above the MC cap (85, `attrs['fid_mu_cap']`) — asserted excluded | exactly `1.0` |
| `4` | the spectrum has no support, `mu == 0` | exactly `0.0` |

**Codes 2, 3 and 4 are deterministic shortcuts, not failures.** They are how
the scan avoids Monte Carlo where the answer is taken to be known, and they are
the majority of the cube. Code 3 in particular is an *assertion* rather than a
computation — see [§10](#10-known-limitations).

```python
import h5py, numpy as np

with h5py.File("luhdm_datarelease_v5.h5", "r") as f:
    st  = f["results/status"][:]
    ext = f["results/extremeness"][:]

codes, counts = np.unique(st, return_counts=True)
print({int(c): int(n) for c, n in zip(codes, counts)})
print("NaN extremeness exactly where status==1:",
      np.array_equal(np.isnan(ext), st == 1))
print("status-1 cells:", int((st == 1).sum()),
      f"({100 * (st == 1).mean():.2f}% of the cube)")
```

```
{0: 435797, 1: 52548, 2: 2226349, 3: 432844, 4: 308222}
NaN extremeness exactly where status==1: True
status-1 cells: 52548 (1.52% of the cube)
```

That is 12.6% code 0, 1.5% code 1, 64.4% code 2, 12.5% code 3, 8.9% code 4.

**Code 1 is a failure and needs care.** `extremeness` is `NaN` there, and
`NaN >= 0.95` is `False`, so **a failed cell silently reads as "not excluded"**
in any naive level-set. That is the published convention and this release does
not change it, but you should know where those cells are — see
[§10](#10-known-limitations) for how far they sit from any contour.

The failures are a cross-section interpolant underflowing its tabulation floor
at the strongest couplings and heaviest masses in the few-µm mediator range.
A cell-level failure marks all three modes, so `n_transit` (which has no mode
axis) is `NaN` at exactly the cells where any mode has `status == 1`.

The standalone reader's `excluded_band()` counts them for you
(`band.n_undefined` per mass) and warns unless you pass `nan_policy='ignore'`.

---

## 9. Reproducibility

**Integrity.** `SHA256SUMS` carries the digest of the cube under the name it
was distributed with. If your copy still has that name, `sha256sum -c
SHA256SUMS` checks it in one step; if you renamed it, compare the digest
directly — the digest is what matters, not the filename.

```console
$ cat SHA256SUMS
$ sha256sum luhdm_datarelease_v5.h5
```

```
1d591f096db1a900639f288baab371dd4edccc2543148059a6b005dde2705733  luhdm_datarelease_v5.h5
1d591f096db1a900639f288baab371dd4edccc2543148059a6b005dde2705733  luhdm_datarelease_v5.h5
```

**Provenance in the file.** The root attributes are a complete record — you do
not need `provenance.json` to know what produced the numbers:

```python
import h5py, json

with h5py.File("luhdm_datarelease_v5.h5", "r") as f:
    a = dict(f.attrs)
for k in ("version_tag", "created", "git_commit", "git_dirty", "seed",
          "t_exposure_s", "b_constrained_max_m", "df", "fid_mu_cap",
          "events_mode1_sha256", "efficiency_npz_sha256"):
    print(f"{k:22s} {a[k]}")
print("packages", json.loads(a["packages_json"]))
```

```
version_tag            v5.0-night-m0p356mg-bcap10cm
created                2026-08-08T12:10:42.831616+00:00
git_commit             1ca2828dddaa859c5ceee3be74fc81b388748ce1
git_dirty              False
seed                   20260702
t_exposure_s           790778.0
b_constrained_max_m    0.1
df                     3
fid_mu_cap             85.0
events_mode1_sha256    9bdc69c90b6f9e80db114821e1af363157a1a55c260907e2d4ebfc0641c1f5b6
efficiency_npz_sha256  451e6ca10c759ecbe4620672796f3571538914dd7c7dba63fd679710d04183b3
packages {'numpy': '2.5.1', 'scipy': '1.18.0', 'h5py': '3.16.0', 'optimum_interval': '0.3.0', 'luhdm': '0.1.0', 'matplotlib': '3.11.1', 'pandas': '3.0.3', 'python': '3.14.6'}
```

The SHA256 of every input (event lists, efficiency table) is recorded, along
with the git commit of the analysis code, the RNG seed, the Monte-Carlo
fidelity settings (`fid_*`, including the expected-count cap `fid_mu_cap`) and
the versions of every package used (`packages_json`). Exclusion limits were
computed with `optimum_interval` 0.3.0. `git_dirty` tells you honestly whether
the working tree had uncommitted changes at build time; for this release it is
`False`, so `git_commit` identifies the code exactly.

**`provenance.json`** (shipped alongside) adds the build-side detail: the
assembly command line, the per-input records, and the impact-parameter cap
block.

**Physics fiducials** are attributes too, so a reinterpretation does not have to
guess them: `rho_dm_gev4`, `f_x`, `n_neutrons`, `r_eff_m`, `q_thresh_gev`,
`q_hi_ref_gev`, `m_planck_gev`, `t_exposure_s`.

**How it was checked.** Cells sampled across the evaluation regimes (Monte
Carlo, deterministic shortcut, off-tag mediator ranges and the massless limit)
were recomputed from scratch and reproduced the stored values bit-for-bit on
the toolchain used to build the release; the assembled cube was checked back
against the per-range inputs it was built from; and the standalone reader was
compared against the analysis package on 250 randomly drawn cells spanning
every mode, both `f_DM` planes, both atmosphere states and both finite and
massless ranges, with `extremeness`, `mu`, `n_transit` and `status` agreeing
exactly.

### Numerical notes

The retained fraction of the outer impact-parameter shell is evaluated in a
cancellation-stable form with a small-argument series branch, which matters
when the ratio of the flyby reach to the 0.1 m cap is large (≳ 10⁴).

---

## 10. Known limitations

**Cells above the expected-count cap are asserted excluded, not computed.**
Where `mu` exceeds `attrs['fid_mu_cap']` = 85 the optimum-interval Monte Carlo
is skipped and `extremeness` is set to exactly `1.0` (`status == 3`, 12.5% of
the cube). The assertion is that such a hypothesis is overwhelmingly excluded.
It is validated for **modes 1 and 2**: no Monte-Carlo cell with `p < 0.95` has
`mu` above 39, leaving a wide margin below the cap. For **mode 3** it is not:
Monte-Carlo cells with `mu` up to ≈ 85 still show `p < 0.95`, so the assertion
is applied right where the computed answer can still fall below threshold.
**Mode-3 exclusion boundaries near high expected counts therefore carry
additional uncertainty** beyond the quoted Monte-Carlo noise. Mask on
`status == 3` if you need a purely computed boundary.

**The mode-3 massless floor is resolved to about one coupling grid step.** The
`alpha_n` axis is 0.2326 dex per step, and for the massless slice in mode 3 the
lower edge of the excluded band is determined to roughly that spacing. Treat
the mode-3 massless floor as grid-resolution-limited rather than as a precisely
located number.

**Undefined (status-1) cells.** 52 548 cells (1.52% of the cube) raised during
evaluation and carry `NaN` in `extremeness`, `mu` and `n_transit`; `NaN` is
exactly coincident with `status == 1`. They sit deep in the non-excluded corner
— at strong coupling and heavy mass in the few-µm range — and are **at least 7
grid steps from any 95% contour** measured as a Chebyshev distance in
(`alpha_n`, `mass`, `lambda`) index space, and at least 18 grid steps when the
distance is confined to the (`alpha_n`, `mass`) plane a contour is actually
drawn in. The published limits are therefore unaffected. If you contour a
different confidence level, or work in that corner, mask on `status == 1`
explicitly rather than trusting the `NaN` comparison.

**Atmosphere-off upper edges are grid artefacts, not ceilings.** With
attenuation ON, strong couplings are stopped by the overburden and the exclusion
closes from above: a genuine two-sided band in `alpha_n`. With attenuation OFF
there is no such mechanism, so the exclusion simply continues past the top of
the scanned coupling grid (`alpha_n = 1`) and the upper edge is wherever the
scan stopped:

```python
import h5py, numpy as np

with h5py.File("luhdm_datarelease_v5.h5", "r") as f:
    ext = f["results/extremeness"][:]
    atm_axis = f["axes/atmosphere"][:]

above = ext >= 0.95                              # (f, atm, mode, alpha, mass, lam)
for i, on in enumerate(atm_axis):
    a = above[:, i]                              # alpha is axis 2 after slicing atm
    any_exc, top_exc = a.any(axis=2), a[:, :, -1]
    n = int(any_exc.sum())
    print(f"atmosphere={int(on)}: {n} excluded columns, "
          f"{100 * (any_exc & top_exc).sum() / n:.1f}% saturate at alpha_n = 1")
```

```
atmosphere=1: 11611 excluded columns, 1.3% saturate at alpha_n = 1
atmosphere=0: 12118 excluded columns, 94.7% saturate at alpha_n = 1
```

Quote atmosphere-off results as one-sided lower bounds (`alpha_n > lo`), not as
bands. The reader's `band.saturated_hi` flags this per mass.

**The impact-parameter cap.** The outer limit of the impact-parameter integral
is `min(b_constrained_max, b_max(q))` with `b_constrained_max` = 0.1 m
(`attrs['b_constrained_max_m']`; `NaN` would mean uncapped). Flybys further than
10 cm from the sensor are not counted, in both the differential cross section
and the geometric transit reach. This is deliberate — it keeps the calculation
inside the regime where the straight-line-impulse approximation and the
laboratory geometry are trustworthy — but it means results for long mediator
ranges are **cap-limited, not range-limited**: once `lambda` is well past 10 cm
the reach stops growing and `n_transit` becomes independent of `lambda`. The
2 m and 0.2 m slices are both in that regime. Do not read the flattening at
large `lambda` as physics.

**Monte-Carlo granularity.** Cells with `status == 0` carry toy noise at the
`1/n_mc` = 10⁻⁴ level. Contours near 0.95 wobble at that scale. Cubes are
stored as float32, whose spacing near 0.95 is three orders of magnitude finer
than the MC noise, so the precision loss is irrelevant.

**Modes are separate measurements.** The three sensor modes have different
thresholds, efficiencies and event lists. Combining them is an analysis choice
the release does not make for you; figures in the paper that show a single
curve take the per-mode maximum of the extremeness (the most constraining mode
at each point), which is a valid but conservative choice.

**`f_DM` is not a free knob on `extremeness`.** `mu` and `n_transit` scale
exactly linearly with `f_DM`, but `extremeness` does not — it is a non-linear
function of `mu`. Use the stored `f_dm = 1.0` surface rather than rescaling the
`f_dm = 0.1` one.

---

## 11. Citation

Placeholder — the citation will be added when the release is published
alongside the paper. Data release DOI to be assigned.
