# Notebooks

Every figure and number below is read from a data release cube, through
`luhdm.release`. The notebooks do not recompute any physics, so a notebook run
and a cluster run cannot drift apart. Three are outside that rule: notebook 00
reads the raw instrument file, notebook 09 is closed-form pedagogy that reads no
cube at all, and notebook 04 reads the internal full-lambda cube rather than a
released one.

**Which cube.** The release is two files, one hypothesis each, both tracked in
[`release/`](../release). Notebooks 01, 02 and 03 open
`luhdm_datarelease_v7_A_f1_atm.h5` (f_DM = 1, atmospheric attenuation on);
notebook 05 opens `luhdm_datarelease_v7_B_f0p1_noatm.h5` (f_DM = 0.1,
attenuation off), which is its whole subject; notebook 06 opens both, since it
is the guided tour.

**Two conventions of the v7 release** show up in every notebook that draws a
mass axis. The analysis window starts at `config.Q_THRESH` = 1 TeV, which puts a
hard kinematic wall at 5.51 × 10⁵ GeV on the left. And the impact-parameter
integral is uncapped, so nothing in the stored surfaces closes the massless
contour on the right: the region is closed instead by the halo flux cut `m_cut`
from the files' attributes (6.11 × 10¹⁴ GeV at f_DM = 1, 6.11 × 10¹³ at
f_DM = 0.1), which assumes N_req = 3 expected transits within 10 cm during the
exposure. That assumption is stated wherever the cut is drawn, and the cut is
drawn as a dashed brown line rather than applied to the surfaces.

## What each notebook is for

| notebook | what it reproduces in the paper | figures written |
|---|---|---|
| [`00_efficiency_and_blips`](00_efficiency_and_blips.ipynb) | the measured detection efficiency and candidate lists behind Letter Fig. 2 | `00_efficiency_curves`, `00_blip_momentum_spectrum` |
| [`01_limit_contour`](01_limit_contour.ipynb) | the surfaces behind the left panel of the Letter results figure; per-mode exclusion views beyond what the paper prints | `01_excluded_massless`, `01_sensitivity_vs_range`\*, `01_excluded_region_200um`, `01_all_mediator_ranges` |
| [`02_methodology`](02_methodology.ipynb) | SM, The Transferred Momentum: the halo, attenuation, cross-section and statistics chain (the SM's single-column spectra panel itself is drawn by `scripts/paper_fig_sm_spectra.py`) | `02_spectra`, `02_arrival_speed_distributions` |
| [`03_understanding`](03_understanding.ipynb) | nothing directly; background on why the excluded region has the shape it does | `03_transit_reach_maps` |
| [`04_mode_comparison`](04_mode_comparison.ipynb) | nothing directly; the per-mode cross-check behind the single-mode SM figures | `04_exclusion_modes123`, `04_mediator_vs_coupling`\*, `04_mediator_vs_coupling_zoom`\* |
| [`05_composite`](05_composite.ipynb) | the atmosphere-off companion scan, a release extra beyond what the paper prints | `05_composite_noatm` |
| [`06_datarelease`](06_datarelease.ipynb) | SM, Analysis Code and Data Release: a guided tour of the two released cubes | none |
| [`09_projection_peak`](09_projection_peak.ipynb) | nothing directly; **historical** — why the *capped* spectrum had a plateau and a horn. The v7 cross section has no cap, so the peak is not in the released spectra; the projection argument behind it still explains the endpoint at `q_max`. | `09_projection_peak`, `09_projection_peak_galaxy` |

Of those, only the spectra panel appears in the paper (as the Supplemental
Material's `02_spectra`, redrawn at single-column size by
`scripts/paper_fig_sm_spectra.py`); the per-mode and atmosphere-off views are
release extras. Figures are written to `png/`, `svg/` and `pdf/`. Each
figure *derived from the cube* is stamped with that cube's version tag;
notebooks 00 and 09 write unstamped figures, since neither reads a cube.

\* The figures marked with an asterisk scan the mediator range continuously and
need the internal full-lambda cube (54 finite ranges, 0.1 µm to 2 m); the
released cubes carry the 2 mm, 200 µm and 20 µm slices, a 200 m convergence
check and the massless limit, so against them those cells report a skip instead
of drawing. Every other figure in the table regenerates from the released files
— with the exception of notebook 04's, which still read the internal cube, and
notebook 00's, which need the raw instrument file.

The Letter's data-derived figures are drawn by scripts rather than notebooks:
`scripts/paper_fig_data_spectrum.py --stem efficiency` (Fig. 2, the impulse
spectrum with the mode-1 efficiency overlaid) and `scripts/paper_fig_limits.py`
(Fig. 3, the two-panel result); `scripts/paper_fig_efficiency.py` draws the
three-mode efficiency comparison, which is not a Letter figure. **Those scripts
have not yet been moved to the v7 two-file layout and are under review; they
still read the internal cube and its 0.1 TeV window.** Figures
`07_impact_parameter_cap` and `08_mass_coupling_degeneracy` likewise come from
`scripts/`, and both are explainers for the retired impact-parameter cap: like
notebook 09 they are kept as background on the capped scheme, not as
descriptions of the current release.

## Dark matter fraction

Which fraction a figure uses is now a property of the file it opens, not a read
option: each released cube carries one `f_dm` value. Notebooks 01, 02 and 03 read
file A at `f_dm=1.0`, the fraction the Letter quotes its `alpha_n` limits at, and
pass it explicitly on every call — the loader's fallback is
`attrs['f_dm_default']` = 0.1, the build-side baseline, which is not on file A's
axis and raises if it is used. Notebook 05 reads file B at `f_dm=0.1`, the
composite-benchmark plane. Notebook 06 opens both and shows them side by side.
Notebook 04 still reads the internal full-lambda cube, which carries both planes.
(Notebook 09 reads no extremeness plane, so the question does not arise there.)

The `/halo` diagnostic maps are the one exception in either file: they are stored
once at the baseline `f_dm` = 0.1 with no fraction axis, so notebook 03's transit
counts are a factor of ten below the `/results` ones at the same point.

## Running them

From a fresh checkout, with Python 3.10 or newer:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebooks]"
pip install "optimum_interval @ git+https://github.com/tunnell/optimum_interval"
jupyter lab
```

Notebooks 01, 02, 03, 05, 06 and 09 then run start to finish with no environment
variables set and no file from outside the repository, in about a minute for all
six together. The first five read a released cube rather than recomputing it,
which is why they are quick; notebook 09 recomputes closed-form cross sections
and a small Monte Carlo, and is quick for its own reasons.

**Notebook 04 does not run from a fresh checkout either.** It calls
`open_release()` with no argument, which resolves to
`luhdm.release.DEFAULT_PATH`, the internal full-lambda cube. Only the two v7
release files are tracked in git; the internal cube is local- and Zenodo-only,
so without it the first cell stops with a `FileNotFoundError` naming that path.
Its outputs are committed.

**Notebook 00 does not run from a fresh checkout.** It is the only one that reads
the raw instrument file `data/fit_data_temp_lockin_transients_selected.hdf5`,
which is too large for git and is not distributed. Without it the first cell stops
with a `FileNotFoundError` naming that path. Its outputs are committed, so the
figures and numbers it produced can still be read.
