#!/usr/bin/env python3
"""PRL figure: SM differential-rate spectra at the showcase point, one column.

Single-column replacement for the two-column ``02_spectra`` panel drawn by
``notebooks/02_methodology.ipynb``: the differential rate of momentum transfers
dR/dq at the showcase point (m_DM = 1e8 GeV, alpha_n = 1e-3), one solid curve
per mediator range carried through the analysis (the sequential viridis ramp of
the SM figure family, ordered in lambda), the analytic massless result dashed,
and the observed mode-1 candidate impulses as thin vertical guides.  Everything
is read from the data release; nothing is recomputed here.

Direct labels, no legend
------------------------
The mediator-range family is a fan of near-parallel curves, so every curve is
named *on itself* instead of through a legend key: each label is written in its
curve's own colour, offset a couple of points off the line and rotated onto the
line's local slope (measured in display space, after the layout is final), so a
label runs parallel to -- and never crosses -- the curve it names.  ``halo()``
is deliberately not used: it is incompatible with ``text.usetex`` (see its
docstring), so the labels instead live in the empty wedges of the fan.

Anchors sit near the left spine where the fan is widest *and* clear of the
candidate verticals (the first one is at q ~ 1.5e3).  Two exceptions:

* ``10um`` runs only ~0.4 decades under ``20um`` -- far too little room for
  6.5 pt type -- so it is labelled from *below*, out to the right of the
  candidate group where nothing else is drawn.
* the candidate verticals stop short of the top of the axes, which reserves a
  clean band for their single shared ``observed candidates`` label, and they
  start just above the boxed parameter note, whose white ``bbox`` would
  otherwise notch them.

Style: PRL single column at final printed size; see ``scripts/paper_style.py``.
``\\includegraphics`` in ``main.tex`` carries no ``width=`` for this figure.

Usage
-----
    python scripts/paper_fig_sm_spectra.py                 # default release
    python scripts/paper_fig_sm_spectra.py --release <h5>
    python scripts/paper_fig_sm_spectra.py --talk          # slide SVG only
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402

import paper_style as ps  # noqa: E402
from luhdm import release  # noqa: E402

MODE = 1  # the paper's channel

# Ramp shared with the notebook-02 family: seven ordered ranges, 2um hidden,
# so the six drawn curves keep the exact colours of the wide variant.
_TAGS = ["2m", "2cm", "2mm", "200um", "20um", "10um", "2um"]
#: Direct-label strings.  The topmost curve carries the "lambda =" so the reader
#: learns what the family varies without a legend; the rest are bare values.
_LABELS = {"2m": r"$\lambda = 2\,$m", "2cm": r"$2\,$cm", "2mm": r"$2\,$mm",
           "200um": r"$200\,\mu$m", "20um": r"$20\,\mu$m", "10um": r"$10\,\mu$m"}
_HIDE = {"2um"}
_RAMP = plt.cm.viridis(np.linspace(0.02, 0.78, len(_TAGS)))

#: Where each direct label is anchored: (q in GeV/c, side of its own curve).
_ANCHORS = {"2m": (1.15e2, "above"), "2cm": (1.15e2, "above"),
            "2mm": (1.15e2, "above"), "200um": (1.15e2, "above"),
            "20um": (1.15e2, "above"), "10um": (2.0e4, "below")}
_MASSLESS_ANCHOR = (3.5e2, "above")

#: Axes fraction at which the candidate verticals stop, leaving a clear band for
#: their shared label just above them.  The band is kept clear of the very top
#: of the axes on purpose: that is where a pre-v3 release parks the PRELIMINARY
#: corner tag, and the two labels collided there.
_EVENT_TOP = 0.82
_EVENT_LABEL_Y = 0.885

# Print vs slide: the slide variant scales every hardcoded size ~x1.9, except
# the in-plot type, which is pinned to the talk tick-label size (13 pt) so that
# nothing on the plot renders smaller than the surrounding slide body text.
_PAPER = dict(figsize=(ps.COLUMN_W_IN, ps.COLUMN_W_IN / 1.35),
              fs_label=6.5, fs_note=6.5, fs_tag=6.0, dy_pt=1.6,
              lw_curve=0.9, lw_ref=1.0, lw_event=0.55)
_TALK = dict(figsize=ps.TALK_FIGSIZE,
             fs_label=13.0, fs_note=13.0, fs_tag=11.0, dy_pt=3.0,
             lw_curve=2.2, lw_ref=2.4, lw_event=1.2)


def _curve_label(ax, line, x, text, *, above, fontsize, dy_pt):
    """Write ``text`` in ``line``'s colour, ``dy_pt`` points off the curve at ``x``.

    Returns ``(Text, line, x)``; feed the tuple to :func:`_align_to_curves` once
    the layout is final to rotate the label onto the curve's slope.
    """
    xd, yd = np.asarray(line.get_xydata(), dtype=float).T
    lx, ly = np.log10(xd), np.log10(yd)
    y = 10.0 ** np.interp(np.log10(x), lx, ly)
    # A label hung *below* a curve already carries the font's ascent as dead
    # space between the anchor and its cap line, so it needs a smaller nudge
    # than one sitting above; without this the two sides look unbalanced.
    dy = dy_pt if above else -0.35 * dy_pt
    tr = mtransforms.offset_copy(ax.transData, ax.figure, y=dy, units="points")
    txt = ax.text(x, y, text, transform=tr, color=line.get_color(),
                  fontsize=fontsize, ha="left",
                  va="bottom" if above else "top",
                  rotation_mode="anchor", zorder=4)
    return txt, line, x


def _align_to_curves(ax, pending, dq_dec=0.04):
    """Rotate each direct label onto its curve's slope, measured in *display* space.

    Must run after the final layout: the on-screen angle of a curve depends on
    the rendered axes aspect, so an angle computed before ``tight_layout`` would
    leave the labels visibly off-parallel.
    """
    for txt, line, x in pending:
        xd, yd = np.asarray(line.get_xydata(), dtype=float).T
        lx, ly = np.log10(xd), np.log10(yd)
        xs = x * 10.0 ** np.array([-dq_dec, dq_dec])
        ys = 10.0 ** np.interp(np.log10(xs), lx, ly)
        (x0, y0), (x1, y1) = ax.transData.transform(np.column_stack([xs, ys]))
        txt.set_rotation(np.degrees(np.arctan2(y1 - y0, x1 - x0)))


def _retract_above(ax, vlines, txt, pad_pt=2.0, top=_EVENT_TOP):
    """Cut the guide verticals off above ``txt``'s box, measured on the renderer.

    The parameter note is boxed, and a white ``bbox`` drawn over a line notches
    it (charter F-rule; see :func:`paper_style.halo`).  Rather than pick a magic
    axes fraction that a font change would silently invalidate, the retraction
    is read back from the note's rendered patch, so it still holds after a blind
    re-run against a new cube.
    """
    fig = ax.figure
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    patch = txt.get_bbox_patch()
    tb = (patch if patch is not None else txt).get_window_extent(renderer=r)
    ab = ax.get_window_extent(renderer=r)
    bottom = (tb.y1 + pad_pt * fig.dpi / 72.0 - ab.y0) / ab.height
    bottom = float(np.clip(bottom, 0.0, 0.5))
    for line in vlines:
        line.set_ydata([bottom, top])


def build(rel, talk=False):
    st = _TALK if talk else _PAPER
    (ps.apply_talk_style if talk else ps.apply_prl_style)()
    fig, ax = plt.subplots(figsize=st["figsize"])

    q_ev = np.asarray(rel.events(MODE), dtype=float)
    vlines = [ax.axvline(q1, ymax=_EVENT_TOP, color=ps.CANDIDATE_COLOR,
                         lw=st["lw_event"], zorder=0) for q1 in q_ev]

    q, r = rel.raw_spectrum("massless")
    ax.loglog(q, r, color="k", ls="--", lw=st["lw_ref"], label="massless")
    massless = ax.lines[-1]
    lines = {}
    for tag, color in zip(_TAGS, _RAMP):
        if tag in _HIDE:
            continue
        q, r = rel.raw_spectrum(tag)
        ax.loglog(q, r, color=tuple(color), lw=st["lw_curve"], label=_LABELS[tag])
        lines[tag] = ax.lines[-1]

    ax.set_ylim(bottom=1e-13)
    ax.set_xlabel(r"$q$ [GeV/$c$]")
    ax.set_ylabel(r"$dR/dq$  [s$^{-1}$ (GeV/$c$)$^{-1}$]")

    # -- direct labels, one per curve, in the curve's own colour -------------
    pending = []
    x_ml, side_ml = _MASSLESS_ANCHOR
    pending.append(_curve_label(ax, massless, x_ml, "massless",
                                above=(side_ml == "above"),
                                fontsize=st["fs_label"], dy_pt=st["dy_pt"]))
    for tag, line in lines.items():
        x_a, side = _ANCHORS[tag]
        pending.append(_curve_label(ax, line, x_a, _LABELS[tag],
                                    above=(side == "above"),
                                    fontsize=st["fs_label"], dy_pt=st["dy_pt"]))
    texts = {"massless": pending[0][0]}
    texts.update({tag: t for tag, (t, _, _) in zip(lines, pending[1:])})

    # One shared label for the candidate verticals, in the band they stop below.
    texts["candidates"] = ax.text(
        float(np.sqrt(q_ev.min() * q_ev.max())), _EVENT_LABEL_Y,
        "observed candidates", transform=ax.get_xaxis_transform(),
        ha="center", va="center", fontsize=st["fs_label"],
        color=ps.CANDIDATE_COLOR, zorder=4)

    texts["note"] = ax.text(
        0.03, 0.05, r"$m_{\rm DM} = 10^{8}\,$GeV$/c^{2}$, $\alpha_n = 10^{-3}$",
        transform=ax.transAxes, ha="left", va="bottom",
        fontsize=st["fs_note"],
        bbox=dict(fc="white", ec="0.7", lw=0.4, alpha=0.9))

    tag_txt = ps.add_preliminary_tag(ax, ps.preliminary_tag_text(rel.version_tag))
    if tag_txt is not None:
        tag_txt.set_fontsize(st["fs_tag"])
        texts["preliminary tag"] = tag_txt

    fig.tight_layout(pad=0.15)
    _align_to_curves(ax, pending)   # angles are only meaningful once laid out
    _retract_above(ax, vlines, texts["note"])

    checked = dict(texts)
    checked["xlabel"] = ax.xaxis.label
    checked["ylabel"] = ax.yaxis.label
    ps.assert_text_clearance(fig, checked)
    ps.assert_inside_figure(fig, checked)
    return fig


#: Slide asset consumed by the COSMO deck (Slidev wants SVG).
TALK_OUT = Path("/home/tunnell/code/luhdm/ignore/talks/talks-main/2026 COSMO"
                "/public/assets/luhdm/talk_spectra")
TALK_PNG_DPI = 200


def save_talk(fig, stem=TALK_OUT, dpi=TALK_PNG_DPI):
    """Write ``stem``.svg and ``stem``.png at the exact figure size.

    Same contract as the other talk figures: Slidev consumes the SVG and the
    PNG is the raster fallback; the talk asset tree carries no PDFs, so the PDF
    ``savefig_exact`` always writes is sent to a scratch directory and
    discarded.  Returns ``(svg, png)``.
    """
    stem.parent.mkdir(parents=True, exist_ok=True)
    svg, png = stem.with_suffix(".svg"), stem.with_suffix(".png")
    with tempfile.TemporaryDirectory() as tmp:
        ps.savefig_exact(fig, Path(tmp) / f"{stem.name}.pdf", png,
                         png_dpi=dpi, svg_path=svg)
    return svg, png


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
    p.add_argument("--talk", action="store_true",
                   help="slide variant: talk style/size, written as SVG to "
                        f"{TALK_OUT}.svg (the paper outputs are left alone)")
    args = p.parse_args(argv)

    with release.open_release(args.release) as rel:
        print(f"release: {args.release}  ({rel.version_tag})")
        fig = build(rel, talk=args.talk)

    if args.talk:
        svg, png = save_talk(fig)
        plt.close(fig)
        print(ps.report_size(fig, svg))
        print(f"preview: {png}")
        return

    pdf = args.outdir / f"{args.stem}.pdf"
    png = args.outdir / f"{args.stem}.png"
    ps.savefig_exact(fig, pdf, png)
    plt.close(fig)
    print(ps.report_size(fig, pdf))
    print(f"preview: {png}")


if __name__ == "__main__":
    main()
