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

**Main product:** the data release in [`release/`](release/README.md), a single
self-describing HDF5 cube holding the whole limit-setting calculation, built by
`scripts/build_release.py` and `scripts/assemble_release.py` on a many-core node.
Everything else reads it.

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
