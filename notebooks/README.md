# Notebooks

Seven notebooks. Six of them are one narrative arc: **what the instrument
delivered → the limit → how the limit is made → why it stops where it stops →
the atmosphere-off benchmark → how to use the released files yourself.** Read
those in order and nothing is referenced before it is explained. The seventh,
06, is not part of the arc: it is the derivation of the halo-frame convention
`v10` adopts, and it compares that frame against the Galactic rest frame of
`v9.1` and earlier.

In notebooks 00–05 every figure and number is read from a data release cube,
through `luhdm.release`, so a notebook run and a cluster run cannot drift apart.
The only recomputation there is the handful of one-minute inline calculations
that *illustrate* a convention — notebook 02's two projection kernels, notebook
03's replication of the cube's own `mu` cells — and each of those is checked
against the released numbers in the same cell. Notebook 06 is the exception by
design: it computes both frames itself, because a cube exists in only one of
them, and it opens a released file for its projection-kernel and cutoff
conventions rather than for numbers.

| # | notebook | in one line |
|---|---|---|
| 00 | [`00_data_and_selection`](00_data_and_selection.ipynb) | the measured efficiency, the exposure, every blip, the night candidate lists — and why the selection is the night one |
| 01 | [`01_the_limit`](01_the_limit.ipynb) | the result: excluded (mass, coupling) regions, the refined boundary, all three modes, and the mediator-range planes |
| 02 | [`02_how_the_limit_is_made`](02_how_the_limit_is_made.ipynb) | the pipeline: halo → atmosphere → cross section → **projection kernel** → rate → optimum interval |
| 03 | [`03_the_edges`](03_the_edges.ipynb) | why the region stops where it does: kinematic wall on the left, flux cut on the right, transit/reach maps in between |
| 04 | [`04_composite_benchmark`](04_composite_benchmark.ipynb) | file B: three modes combined, atmosphere off, f_DM = 0.1 — the surfaces the composite benchmark is recast from |
| 05 | [`05_using_the_data_release`](05_using_the_data_release.ipynb) | a guided tour of the released files, from raw `h5py` to the loader, ending in the worked example the release ships a figure of |
| 06 | [`06_halo_frame_and_rates`](06_halo_frame_and_rates.ipynb) | the halo frame: the Galactic rest frame against the laboratory frame `v10` adopts, and the two moments through which the change reaches the rates |

**Which cube.** The release is two files, one hypothesis each, both tracked in
[`release/`](../release). Notebooks 00, 01, 02 and 03 open
`luhdm_datarelease_v10_A_f1_atm.h5` (f_DM = 1, atmospheric attenuation on);
notebook 04 opens `luhdm_datarelease_v10_B_f0p1_noatm.h5` (f_DM = 0.1,
attenuation off), which is its whole subject; notebooks 02, 03, 04 and 05 open
both — 02 and 03 because their replication checks need the bare-halo pass, 04
for its side-by-side figure, 05 because it is the guided tour. Notebook 06 opens
file B too, but only to inherit its projection kernel and inner cutoff; it reads
no surface from it. Three sidecars sit beside the cubes:
`luhdm_contours_v10_A_f1_atm.json`, the root-found 95% boundary notebook 01
publishes; `luhdm_contours_v10_B_f0p1_noatm.json`, the same boundary for the
bare-halo plane; and `luhdm_lambda_scan_v10.npz`, a mediator-range sidecar
holding sensor mode 1's (coupling, range) plane on a 54-point range axis from
0.1 µm to 2 m at that mode's best dark matter mass. Notebook 01 opens the
file-A contours and the lambda scan and checks their provenance records against
the cube before drawing, and reproduces a piece of the file-B contours live.
`release/SHA256SUMS` has 13 entries: every file of the current release except
`SHA256SUMS` itself — the two cubes, the three sidecars, the two provenance
records, `luhdm_release.py`, `README.md`, `CITATION.cff`, `LICENSE`,
`exclusion_massless_mode1.png` and `aux/null_calibration_toymc.json`. No
superseded file is listed; earlier cubes and their sidecars are not carried
forward.

**Two conventions of the release** show up in every notebook that draws a
mass axis. The analysis window starts at `config.Q_THRESH` = 1 TeV, which puts a
hard kinematic wall at 3.80 × 10⁵ GeV on the left. And the impact-parameter
integral is uncapped, so nothing in the stored surfaces closes the massless
contour on the right: the region is closed instead by the halo flux cut `m_cut`
from the files' attributes (8.40 × 10¹⁴ GeV at f_DM = 1, 8.40 × 10¹³ at
f_DM = 0.1), which assumes N_req = 3 expected transits within 10 cm during the
exposure. That assumption is stated wherever the cut is drawn, and the cut is
drawn as a dashed brown line rather than applied to the surfaces. Notebook 03
takes both edges apart. Both numbers moved in `v10`, and they moved in opposite
directions, for the reason the fourth convention below gives.

**A third convention, and the one a referee will ask about.** The released cubes
have been built, from `v9.0` on, with the **isotropic-folded** projection kernel
(the A18 convention:
coefficient 8π/3, shell fraction x³), while `luhdm.cross_section.KERNEL_DEFAULT`
is still the historical `planar-signed` one (2π, arcsine). Every file records
its own choice in the `projection_kernel` root attribute, and
`Release.make_xsec` threads it into a recomputation handle. Recomputing a
released cell through the module default instead is a silent physics error worth
12% in μ at the exclusion boundary. Notebook 02 has the argument, the two
kernels side by side, and the demonstration.

**A fourth convention, new in `v10`: the halo frame.** The standard halo model
is an isotropic truncated Maxwellian, v₀ = 220 km/s, v_esc = 544 km/s, and every
release up to and including `v9.1` evaluated it in the **Galactic rest frame**,
with the detector at rest in the halo. `v10` evaluates it in the **laboratory
frame** instead: the same Maxwellian boosted by the Earth's motion through the
halo and integrated over arrival direction (Lewin & Smith 1996), at
`attrs['v_earth_km_s']` = 245 km/s — the convention of the two levitated-sensor
results this analysis overlays. Two consequences run through every figure. The
support of the distribution now ends at v_esc + v_E = 789 km/s rather than at
v_esc, which is the ceiling of every speed integral and which moves the
kinematic wall down to 3.80 × 10⁵ GeV. And the arriving flux is faster on
average, which moves `m_cut` out. Earlier cubes carry no `v_earth_km_s`
attribute at all, and **absence of it means the Galactic rest frame**. The
convention is fixed per cube, so nothing you *read* out of a file needs to know
about it; anything you *recompute* does, exactly as with the projection kernel,
and with the analysis code that means setting `LUHDM_V_EARTH` before importing
`luhdm`. Notebook 06 is the whole argument, and it is where the numbers quoted
here come from.

**Refined contours cover mode 1 only.** `v10` ships eight refined surfaces, four
per cube — massless, 2 mm, 200 µm and 20 µm at mode 1 — against the eleven of
`v9.1`, which also refined all three sensor modes on the bare-halo plane in one
file per mode. Mode 1 is the mode the paper reports. Modes 2 and 3 are still in
both cubes at full grid resolution and are contoured off the grid as before, so
a notebook that draws them draws the grid contour, not a root-found boundary.

## What each notebook is for

### 00 — the data and the selection

The measured inputs, before any physics model touches them: the per-mode
detection efficiency ε(q) marginalised over the impulse arrival phase, the three
efficiency products against each other (night / full run / fixed w = 1), the
exposure and duty cycle, every reconstructed blip in the run, the night-selected
candidate lists, and the argument for the night selection. It reproduces the
measured products behind Letter Fig. 2; the Letter figure itself is drawn by
`scripts/paper_fig_data_spectrum.py`.

Figures: `00_efficiency_curves`, `00_efficiency_products`,
`00_blip_momentum_spectrum`.

### 01 — the limit

The paper's result. Per-mode excluded regions at the reference range and for a
massless mediator; the whole released range family on one plane; the **refined
boundary** from `luhdm_contours_v10_A_f1_atm.json` against the contour of the
stored grid — at 0.233 dex per coupling cell the grid contour is a drawing
convention and the Letter quotes the root-found boundary; the three sensor modes
compared on the mass plane; and the excluded band swept over the mediator range
from the lambda-scan sidecar, down to mode 1's pinch-off. Backs the left
panel of the Letter's results figure and the Supplemental Material's
single-mode limit figures.

Figures: `01_excluded_region_200um`, `01_excluded_massless`,
`01_all_mediator_ranges`, `01_refined_vs_grid`, `01_sensitivity_vs_range`,
`04_exclusion_modes123`, `04_exclusion_modes123_ranges`,
`04_mediator_vs_coupling`, `04_mediator_vs_coupling_zoom`.

### 02 — how the limit is made

One section per stage of the pipeline: the SHM speed distribution, what the
atmosphere does to it per mediator range, the finite-range cross section, the
**projection kernel** (what "dσ/dq" means when the read-out measures one
component of an impulse, why v9 changed the convention, and why the massless
floor barely moved while everything bulk-dominated gained 4/3), the rate
assembly, the optimum-interval statistic, and the validation appendix. This is
the Supplemental Material companion; its spectra panel is redrawn at
single-column size by `scripts/paper_fig_sm_spectra.py`.

Figures: `02_arrival_speed_distributions`, `02_spectra`, `02_projection_kernel`.

### 03 — the edges

Both ends of the mass window and the map between them, in one story. Sections
1–4: the kinematic wall q_thresh/v_max, where the measured edge lands on the grid,
the halo-tail onset just above it, and a retrospective on why the superseded
0.1 TeV window had a *soft* left edge instead. Section 5: transit-count and
reach maps from the cube's `/halo` group, which is where every edge of the
region can be read off at once. Sections 6–9: the transit curve N(m), the
uncapped contour against the flux cut, what N_req = 3 versus 6.8 buys, and the
shell-of-validity argument for the 10 cm aperture. Section 10 re-checks all
twelve numerical claims.

Figures: `10_kinematic_wall`, `10_contour_left_edge`, `10_halo_tail_onset`,
`10_soft_vs_hard_edge`, `03_transit_reach_maps`, `11_transit_curve`,
`11_contour_vs_cut`, `11_capped_vs_flux_cut`, `11_shell_of_validity`.

### 04 — the composite benchmark

File B: the three sensor modes combined (a point is excluded if *any* mode
excludes it) with atmospheric attenuation switched off, at f_DM = 0.1. Those are
the surfaces the Letter's composite-dark-matter benchmark is recast from — the
20 µm, m_φ ≈ 10 meV slice — although the recast into a nucleon cross section is
done by `scripts/paper_fig_limits.py` and not here. The notebook also puts the
two released hypotheses on one plane, and is explicit that they differ in *both*
f_DM and atmosphere, so the gap between them is not "what the atmosphere costs".

Figures: `05_composite_noatm`, `04_released_hypotheses`.

### 05 — using the data release

The guided tour: the whole tree straight from `h5py` with no `luhdm` code, then
the standalone single-file reader, then the package loader; the two-files-one-
hypothesis-each layout; the exclusion-band convention and the mass cut that is
not in the surfaces; status codes and the NaN policy; detector inputs and how to
verify what you downloaded; and the worked example that reproduces the headline
number in plain numpy and writes
[`release/exclusion_massless_mode1.png`](../release/exclusion_massless_mode1.png),
the figure the release ships so a reader can check their copy against ours.
Backs the Supplemental Material's Analysis Code and Data Release section.

Writes no figure into `png/`; its one output goes to `release/`.

### 06 — the lab-frame halo

Where the `v10` halo convention comes from, and what it does to the rates. The
Galactic rest frame of `v9.1` and earlier against the laboratory frame this
release adopts: the same isotropic truncated Maxwellian, v₀ = 220 km/s and
v_esc = 544 km/s unchanged, but boosted by the Earth's motion through the halo
and integrated over arrival direction, at v_E = 245 km/s. The notebook evaluates
both frames in one process — `config.set_v_earth_km_s()` moves the boost and the
ceiling of every speed integral, `config.V_MAX` = v_esc + v_E, together — and
follows the change through the two moments it acts by. ⟨v⟩, what a *fixed*
aperture sees, rises by a factor 1.375 and carries the flux cut `m_cut` out with
it; ⟨1/v⟩, what a *threshold reach* sees, falls to 0.703 of its rest-frame
value, because a Coulomb impulse
needs slow particles. So the massless slice loses about a third of its expected
count while finite-range slices of the same cube gain: the two move in opposite
directions, and no single factor converts a rest-frame cube into a lab-frame
one. It then does the same for dR/dq and for μ in the analysis window, where the
ratio is a function of mass, coupling and mediator range rather than a number,
and closes on the kinematic wall — q_thresh/v_max moves from 5.51 × 10⁵ to
3.80 × 10⁵ GeV, a factor 1.45, but the wall is not the exclusion edge, and the
notebook shows both reasons why the measured left edge will not follow it.
Backs the Supplemental Material's halo-model section.

Section 4 is a cross-check against the `halo` pass of two build campaigns that
differ only in `--v-earth`; those shards are local build artefacts and are not
shipped, so see [Running them](#running-them) for what that section needs.

Figures: `06_speed_distributions`, `06_moment_weightings`,
`06_shard_transit_ratios`, `06_spectra_two_frames`, `06_mu_vs_mass_two_frames`,
`06_kinematic_wall`.

## Figure names, and where the paper's figures come from

Figures are written to `png/`, `svg/` and `pdf/`, and each is stamped with the
version tag of the cube it came from.

**Figure basenames are stable identifiers, not notebook numbers.** They were
assigned when the notebooks were numbered differently and are deliberately kept
across the restructure, because the manuscript and its Supplemental Material
reference several of them by name. So notebook 01 writes the `04_*`
mode-comparison figures it absorbed, and notebook 03 writes the `03_*`, `10_*`
and `11_*` figures of the three notebooks it merges. The per-notebook lists
above are the authoritative mapping.

The Letter's data-derived figures are drawn by scripts rather than notebooks:
`scripts/paper_fig_data_spectrum.py --stem efficiency` (Fig. 2, the impulse
spectrum with the mode-1 efficiency overlaid) and `scripts/paper_fig_limits.py`
(Fig. 3, the two-panel result); `scripts/paper_fig_efficiency.py` draws the
three-mode efficiency comparison, which is not a Letter figure, and
`scripts/paper_fig_sm_spectra.py` the single-column spectra panel of the
Supplemental Material. Each takes `--release` and defaults to
`luhdm.release.DEFAULT_PATH`, which is file A of the released pair, so they read
the same cube the notebooks do.

Figures `07_impact_parameter_cap` and `08_mass_coupling_degeneracy` also come
from `scripts/` rather than from a notebook. `07` is a standalone geometry
explainer for the retired impact-parameter cap; it opens no cube and reads no
efficiency table, so it still redraws exactly as committed. `08` is not retired
wholesale: the α²/m degeneracy of the massless mediator and the sensor-radius
cutoff that breaks it — panel A and the ratio strip — are the cross section the
release is built on, and panel B's arithmetic, what share of μ is carried by
flybys beyond 10 cm, is the geometric argument behind `m_cut`. Only its ending,
which closes the exclusion island with the cap itself, describes the retired
scheme. Its committed renders are frozen: the figure was measured with the
earlier fixed-arrival-phase efficiency table and its validation gates are pinned
to that table, so re-running it against the canonical one now stops at a gate
rather than redrawing. Regenerating it means pinning the old table back through
`LUHDM_EFFICIENCY_NPZ`, which the script's docstring spells out. One label in
that figure to read with care: the parameter box along its bottom edge prints
`q_th = 100 GeV`, which is where its momentum grid starts, not the analysis
window — the window opens at 1 TeV.

## Dark matter fraction

Which fraction a figure uses is a property of the file it opens, not a read
option: each released cube carries one `f_dm` value. Notebooks 00, 01, 02 and 03
read file A at `f_dm=1.0`, the fraction the Letter quotes its `alpha_n` limits
at, and pass it explicitly on every call — the loader's fallback is the
build-side baseline 0.1, which is not on file A's axis and raises if it is used.
Notebook 04 reads file B at `f_dm=0.1`, the composite-benchmark plane. Notebooks
02, 03, 04 and 05 open both. Notebook 06 opens file B as well, but takes only
its projection kernel and inner cutoff from it and computes its own rates, so no
fraction is read from the file there.

The `/halo` diagnostic maps are the one exception in either file: they are stored
once at the baseline `f_dm` = 0.1 with no fraction axis, so notebook 03's
section-5 transit counts are a factor of ten below the `/results` ones at the
same point.

## Running them

From a fresh checkout, with Python 3.10 or newer:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebooks]"
pip install "optimum_interval @ git+https://github.com/tunnell/optimum_interval"
jupyter lab
```

Notebooks 00 to 05 then run start to finish with no environment variables set
and no file from outside the repository, in a few minutes for the set; they read
released files rather than recomputing surfaces, which is why they are quick.
Every assertion in them is meant to pass on a clean checkout — including the
kernel-consistency assertion in 02 and the twelve self-consistency checks in 03.
Notebook 06 is different in kind: it evaluates the halo, the cross section and
the rate integral in both frames itself, since only one frame exists as a cube.

**Two blocks need files that are not distributed.** Notebook 00's full-run
exposure and duty cycle come from the raw instrument file
`data/fit_data_temp_lockin_transients_selected.hdf5`, which is far too large for
git. That block is guarded on the file's presence: with it, the notebook
additionally re-derives every blip momentum from the raw transients and checks
them against the release; without it, the block prints one line and skips. Every
other product notebook 00 shows — the efficiency table, the candidate lists, all
the reconstructed blips — is read from the release's `/detector` group.

Notebook 06's section 4 is the other. It checks the two frames' transit counts
against the `halo` pass of two build campaigns that differ only in `--v-earth`,
and those shard directories are local build artefacts, shipped with neither the
repository nor the release. Point `LUHDM_SHARDS_REST` and `LUHDM_SHARDS_LAB` at
them to run that section. Nothing else in the notebook depends on them: the
moments, the spectra, the μ scans and the kinematic wall are all computed from
the package.
