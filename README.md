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

**Main product:** [`notebooks/01_limit_contour.ipynb`](notebooks/01_limit_contour.ipynb)
— event list in, excluded (m_DM, α_n) contours out, plus the machine-readable
`notebooks/computation_cache/limit_contour_data.npz`. Companions:
[`notebooks/02_methodology.ipynb`](notebooks/02_methodology.ipynb) (the
physics and statistics pipeline, validations) and
[`notebooks/03_understanding.ipynb`](notebooks/03_understanding.ipynb) (why
the exclusion regions look the way they do). Figures are written to
`notebooks/{png,svg,pdf}/NN_description.*`; the npz caches live in
`notebooks/computation_cache/` (regenerate with `scripts/` on a many-core
node). The physics that gets tweaked lives in `luhdm/rate.py` and
`luhdm/config.py`, shared by the notebooks and the scripts.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ~/code/optimum_interval   # statistics dependency
pip install -e ".[dev]"
pytest
```
