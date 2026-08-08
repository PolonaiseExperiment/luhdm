#!/usr/bin/env python3
"""PRL figure: SM differential-rate spectra at the showcase point, one column.

Single-column replacement for the two-column ``02_spectra`` panel drawn by
``notebooks/02_methodology.ipynb``: the differential rate of momentum transfers
dR/dq at the showcase point (m_DM = 1e8 GeV, alpha_n = 1e-3), one solid curve
per mediator range carried through the analysis (the sequential viridis ramp of
the SM figure family, ordered in lambda), the analytic massless result dashed,
and the observed mode-1 candidate impulses as thin red verticals.  Everything is
read from the data release; nothing is recomputed here.

Style: PRL single column at final printed size; see ``scripts/paper_style.py``.
``\\includegraphics`` in ``main.tex`` carries no ``width=`` for this figure.

Usage
-----
    python scripts/paper_fig_sm_spectra.py                 # default release
    python scripts/paper_fig_sm_spectra.py --release <h5>
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402

import paper_style as ps  # noqa: E402
from luhdm import release  # noqa: E402

MODE = 1  # the paper's channel

# Ramp shared with the notebook-02 family: seven ordered ranges, 2um hidden,
# so the six drawn curves keep the exact colours of the wide variant.
_TAGS = ["2m", "2cm", "2mm", "200um", "20um", "10um", "2um"]
_LABELS = {"2m": r"$2\,$m", "2cm": r"$2\,$cm", "2mm": r"$2\,$mm",
           "200um": r"$200\,\mu$m", "20um": r"$20\,\mu$m", "10um": r"$10\,\mu$m"}
_HIDE = {"2um"}
_RAMP = plt.cm.viridis(np.linspace(0.02, 0.78, len(_TAGS)))


def build(rel):
    ps.apply_prl_style()
    fig, ax = plt.subplots(figsize=(ps.COLUMN_W_IN, ps.COLUMN_W_IN / 1.35))

    for q_ev in rel.events(MODE):
        ax.axvline(q_ev, color="#d62728", lw=0.55, zorder=0)

    q, r = rel.raw_spectrum("massless")
    ax.loglog(q, r, color="k", ls="--", lw=1.0, label="massless")
    curves = [ax.lines[-1]]
    for tag, color in zip(_TAGS, _RAMP):
        if tag in _HIDE:
            continue
        q, r = rel.raw_spectrum(tag)
        ax.loglog(q, r, color=tuple(color), lw=0.9, label=_LABELS[tag])
        curves.append(ax.lines[-1])

    ax.set_ylim(bottom=1e-13)
    ax.set_xlabel(r"$q$ [GeV/$c$]")
    ax.set_ylabel(r"$dR/dq$  [s$^{-1}$ (GeV/$c$)$^{-1}$]")

    note = ax.text(0.03, 0.05,
                   r"$m_{\rm DM} = 10^{8}\,$GeV$/c^{2}$, $\alpha_n = 10^{-3}$",
                   transform=ax.transAxes, ha="left", va="bottom", fontsize=6.5,
                   bbox=dict(fc="white", ec="0.7", lw=0.4, alpha=0.9))

    leg = ax.legend(fontsize=6.5, ncol=2, loc="upper right",
                    handlelength=1.6, columnspacing=0.9, labelspacing=0.25,
                    borderpad=0.3, framealpha=0.9)

    tag = ps.add_preliminary_tag(ax, ps.preliminary_tag_text(rel.version_tag))
    fig.tight_layout(pad=0.15)

    ps.assert_legend_complete(leg, ["massless"] +
                              [_LABELS[t] for t in _TAGS if t not in _HIDE], [ax])
    ps.assert_inside_figure(fig, {"legend": leg, "note": note,
                                  **({"tag": tag} if tag else {})})
    ps.assert_legend_clear_of(fig, leg, curves[:1])  # dashed massless on top
    return fig


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--release", type=Path, default=release.DEFAULT_PATH,
                   help="data-release HDF5 (default: %(default)s)")
    p.add_argument("--outdir", type=Path,
                   default=Path(__file__).resolve().parents[1]
                   / "ignore" / "overleaf" / "figs",
                   help="output directory (default: %(default)s)")
    p.add_argument("--stem", default="02_spectra",
                   help="output basename without extension")
    args = p.parse_args(argv)

    with release.open_release(args.release) as rel:
        print(f"release: {args.release}  ({rel.version_tag})")
        fig = build(rel)

    pdf = args.outdir / f"{args.stem}.pdf"
    png = args.outdir / f"{args.stem}.png"
    ps.savefig_exact(fig, pdf, png)
    plt.close(fig)
    print(ps.report_size(fig, pdf))
    print(f"preview: {png}")


if __name__ == "__main__":
    main()
