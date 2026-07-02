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

**Main product:** [`notebooks/limit_contour.ipynb`](notebooks/limit_contour.ipynb)
— event list in, excluded (m_DM, α_n) contour out. The physics that gets tweaked
(rate formula, halo/attenuation choices, grids, exposure) lives in the notebook;
conversions and statistics plumbing live in the package.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ~/code/optimum_interval   # statistics dependency
pip install -e ".[dev]"
pytest
```
