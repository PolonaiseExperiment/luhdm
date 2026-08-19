# luhdm

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22018169.svg)](https://doi.org/10.5281/zenodo.22018169)

Levitated-sensor ultraheavy dark matter (POLONAISE): the physics models, the
limit setting, and the `v10` data release behind *First Search for Ultraheavy
Dark Matter Using a Magnetically Levitated Particle*. Everything the paper's
data-derived figures are drawn from is in this repository, and the released
files can be read without installing any of this code.

The spectrum code — the physics modules (`config`, `halo`, `atmosphere`,
`cross_section`) — was written initially by Dorian Amaral ([@dwpamaral](https://github.com/dwpamaral)); it is packaged here with its notation unchanged. New here:

- `units` — unit conversions (lengths ↔ GeV⁻¹, rate GeV → s⁻¹);
- `cross_section` additionally holds the straight-line-impulse (K1) machinery
  that previously lived in the limit notebook;
- `limits` — the bridge to the public
  [`optimum_interval`](https://github.com/tunnell/optimum_interval) statistics
  package: rate → (μ, spectrum CDF) → optimum-interval extremeness, plus band
  helpers. Because our spectra depend on the coupling being limited
  (finite-range cross section, attenuation), limits come from scanning the
  coupling and taking level sets of the extremeness — see
  [`notebooks/02_how_the_limit_is_made.ipynb`](notebooks/02_how_the_limit_is_made.ipynb).

**Main product:** the [data release](release/README.md) in `release/`, two
self-describing HDF5 cubes (0.56 and 0.58 MB) holding the whole limit-setting
calculation for the paper: extremeness, expected signal and transit counts over
the (mode, alpha_n, m_DM, lambda) grid, plus the measured efficiency curves,
candidate lists and live time. The two files are the two hypotheses the paper
uses — `luhdm_datarelease_v10_A_f1_atm.h5` is f_DM = 1 with atmospheric
propagation, `luhdm_datarelease_v10_B_f0p1_noatm.h5` is f_DM = 0.1 without — and
share every axis and every detector input. Five lines of `h5py` get you the
excluded region, and [`release/README.md`](release/README.md) is the front door:
quickstart, data dictionary, worked reproduction of the published limit, known
limitations and how to cite. Built by `scripts/build_release.py` and
`scripts/assemble_release.py` on a many-core node; everything else here reads
them.

Beside the two cubes: `luhdm_contours_v10_A_f1_atm.json` and
`luhdm_contours_v10_B_f0p1_noatm.json`, the root-found 95% boundaries of the
mode-1 surfaces the paper draws; `luhdm_lambda_scan_v10.npz`, the
mediator-range sidecar; `luhdm_release.py`, an optional standalone reader that
needs only `numpy` and `h5py`; `exclusion_massless_mode1.png`, the worked
example's figure; the two `provenance_*.json` build records;
`aux/null_calibration_toymc.json`, the noise-only null calibration of the
transient statistic; and `SHA256SUMS`, `README.md`, `CITATION.cff` and
`LICENSE`. `SHA256SUMS` covers the current release only — superseded cubes and
their sidecars are not carried forward.

Three conventions of the release that a reader has to know, all spelled out in
[`release/README.md`](release/README.md). The analysis window starts at
`config.Q_THRESH` = 1 TeV. The impact-parameter integral is **uncapped**, so the
excluded region is closed on the right not by the surfaces but by a post-hoc
halo flux cut `m_cut` stored in the files' attributes, which assumes N_req = 3
expected transits within 10 cm during the exposure; nothing applies that cut for
you. And from `v10` the halo is evaluated in the **laboratory frame**, at
`attrs['v_earth_km_s']` = 245 km/s, so its support ends at v_esc + v_E =
789 km/s; `v9.1` and earlier carried the Galactic rest-frame distribution and no
such attribute, and absence of it means the rest frame.

**Notebooks:** [`notebooks/`](notebooks/README.md) has the index, including which
paper figure each notebook reproduces.
[`notebooks/01_the_limit.ipynb`](notebooks/01_the_limit.ipynb) is the one
to start with: excluded (m_DM, α_n) contours read straight off the cube. Figures
are written to `notebooks/{png,svg,pdf}/NN_description.*`. The physics that gets
tweaked lives in `luhdm/rate.py` and `luhdm/config.py`, shared by the notebooks
and the scripts.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,notebooks]"
pip install "optimum_interval @ git+https://github.com/tunnell/optimum_interval"
pytest
```

## License

Code is [GPL-3.0-or-later](LICENSE); the data release in `release/` is
[CC BY 4.0](release/LICENSE), with attribution meaning the citation in
[`release/README.md` §12](release/README.md#12-how-to-cite).

Cite the paper and, if you used the software, this repository:
[`CITATION.cff`](CITATION.cff) carries the machine-readable metadata for the
code, and [`release/CITATION.cff`](release/CITATION.cff) the separate record for
the dataset.
