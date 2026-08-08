# Notebooks

Every figure and number below is read from the data release cube tracked in
[`release/`](../release), through `luhdm.release`. The notebooks do not recompute
any physics, so a notebook run and a cluster run cannot drift apart. The one
exception is notebook 00, which reads the raw instrument file.

## What each notebook is for

| notebook | what it reproduces in the paper | figures written |
|---|---|---|
| [`00_efficiency_and_blips`](00_efficiency_and_blips.ipynb) | the measured detection efficiency and candidate lists behind Letter Fig. 2 | `00_efficiency_curves`, `00_blip_momentum_spectrum` |
| [`01_limit_contour`](01_limit_contour.ipynb) | SM, Supplementary Limit Figures; also the surfaces behind the left panel of the Letter results figure | `01_excluded_massless`, `01_sensitivity_vs_range`, `01_excluded_region_200um`, `01_all_mediator_ranges` |
| [`02_methodology`](02_methodology.ipynb) | SM, The Transferred Momentum: the halo, attenuation, cross-section and statistics chain | `02_spectra`, `02_arrival_speed_distributions` |
| [`03_understanding`](03_understanding.ipynb) | nothing directly; background on why the excluded region has the shape it does | `03_transit_reach_maps` |
| [`04_mode_comparison`](04_mode_comparison.ipynb) | nothing directly; the per-mode cross-check behind the single-mode SM figures | `04_exclusion_modes123`, `04_mediator_vs_coupling`, `04_mediator_vs_coupling_zoom` |
| [`05_composite`](05_composite.ipynb) | SM, Supplementary Limit Figures: the atmosphere-off companion scan | `05_composite_noatm` |
| [`06_datarelease`](06_datarelease.ipynb) | SM, Analysis Code and Data Release: a guided tour of the released cube | none |
| [`09_projection_peak`](09_projection_peak.ipynb) | nothing directly; background on the impact-parameter cap | `09_projection_peak`, `09_projection_peak_galaxy` |

Of those, four are the ones the Supplemental Material actually includes:
`01_excluded_massless`, `01_sensitivity_vs_range`, `02_spectra` and
`05_composite_noatm`. Figures are written to `png/`, `svg/` and `pdf/`, each
stamped with the version tag of the cube that produced it.

The three Letter figures are drawn by scripts rather than notebooks:
`scripts/paper_fig_efficiency.py` (impulse spectrum and efficiency),
`scripts/paper_fig_limits.py` (the two-panel result) and
`scripts/paper_fig_data_spectrum.py`. Figures `07_impact_parameter_cap` and
`08_mass_coupling_degeneracy` likewise come from `scripts/`.

## Dark matter fraction

The cube stores both `f_dm` planes, so which one a figure uses is a read option,
not a recompute. Notebook 01 asks for `f_dm=1.0`, the fraction the Letter quotes
its `alpha_n` limits at. Notebooks 04 and 05 do not pass `f_dm=` and therefore get
the loader default, 0.1. Each notebook says which one it is using.

## Running them

From a fresh checkout, with Python 3.10 or newer:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebooks]"
pip install "optimum_interval @ git+https://github.com/tunnell/optimum_interval"
jupyter lab
```

Notebooks 01 to 06 and 09 then run start to finish with no environment variables
set and no file from outside the repository. Notebook 01 is the slow one, around
forty minutes; the rest take seconds.

**Notebook 00 does not run from a fresh checkout.** It is the only one that reads
the raw instrument file `data/fit_data_temp_lockin_transients_selected.hdf5`,
which is too large for git and is not distributed. Without it the first cell stops
with a `FileNotFoundError` naming that path. Its outputs are committed, so the
figures and numbers it produced can still be read.
