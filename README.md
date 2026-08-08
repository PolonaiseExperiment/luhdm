# luhdm

**Private.** Levitated-sensor ultraheavy dark matter (POLONAISE): physics models
and limit setting.

The spectrum code — the physics modules (`config`, `halo`, `atmosphere`,
`cross_section`) — was written by Dorian Amaral; it is packaged here with its
notation unchanged. New here:

- `units` — unit conversions (lengths ↔ GeV⁻¹, rate GeV → s⁻¹);
- `cross_section` additionally holds the straight-line-impulse (K1) machinery
  that previously lived in the limit notebook;
- `limits` — the bridge to the public
  [`optimum_interval`](https://github.com/tunnell/optimum_interval) statistics
  package: rate → (μ, spectrum CDF) → optimum-interval extremeness, plus band
  helpers. Because our spectra depend on the coupling being limited
  (finite-range cross section, attenuation), limits come from scanning the
  coupling and taking level sets of the extremeness — see the notebook.

**Main product:** the [data release](release/README.md) in `release/`, a single
self-describing HDF5 cube (14.8 MB) holding the whole limit-setting calculation
for the paper: extremeness, expected signal and transit counts over the
(f_DM, atmosphere, mode, alpha_n, m_DM, lambda) grid, plus the measured
efficiency curves, candidate lists and live time. Five lines of `h5py` get you
the excluded region, and [`release/README.md`](release/README.md) is the front
door: quickstart, data dictionary, worked reproduction of the published limit,
known limitations and how to cite. Built by `scripts/build_release.py` and
`scripts/assemble_release.py` on a many-core node; everything else here reads
it.

**Notebooks:** [`notebooks/`](notebooks/README.md) has the index, including which
paper figure each notebook reproduces.
[`notebooks/01_limit_contour.ipynb`](notebooks/01_limit_contour.ipynb) is the one
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
