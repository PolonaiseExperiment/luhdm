#!/usr/bin/env python3
"""PRL figure: detection efficiency eps(q) for the three sensor modes (df = 3).

One panel, three curves.  This is a supporting three-mode comparison, not a
single-channel figure: the Letter's limit is set on **mode 1** (see
paper_fig_limits.py and paper_fig_data_spectrum.py, both MODE = 1).

Colour follows the charter's per-mode contract (``paper_style.MODE_COLORS``):
mode 1 blue, mode 2 orange, mode 3 purple, the *same* hue the mode carries in
every other figure it appears in.  Hue is never the only cue -- each mode also
has its own dash pattern, so the three stay separable in grayscale and under
any colour vision deficiency (charter F9/F10).  There is no legend: each curve
is named in place, in its own colour, beside its rise.

The momentum at which each curve crosses eps = 0.5 is *measured* from the
tabulated curve at run time by :func:`q_at_efficiency` -- never hard-coded --
marked on the curve itself and reported on stdout, so the figure cannot drift
from the release it was built from.  The in-place labels are anchored the same
way, to the point where each curve crosses :data:`LABEL_EPS`, so they follow the
curves if a future cube moves them.

Curves are evaluated with the analysis' own extrapolation convention
(``luhdm.efficiency.make_efficiency``): eps = 0 below the calibrated table and
held at the saturated value above it.  The plotted window stops at 3e4 GeV,
inside every mode's calibrated range, so no curve is drawn past the top of its
table; the only extrapolated stretch is the eps = 0 run below each table's
lower edge, where the tabulated value is already < 1e-18.

Style: PRL single column at final printed size; see ``scripts/paper_style.py``.
``--talk`` re-renders the same content at slide scale (``apply_talk_style``)
and writes only the SVG (plus a PNG preview) that the COSMO deck consumes.
Figures built from a pre-v3 cube carry a red PRELIMINARY corner tag; the tag
disappears by itself when ``--release`` points at a v3 file.

Usage
-----
    python scripts/paper_fig_efficiency.py                # default release
    python scripts/paper_fig_efficiency.py --release <h5> # e.g. the v3 cube
    python scripts/paper_fig_efficiency.py --talk         # slide SVG only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

import paper_style as ps  # noqa: E402
from luhdm import release  # noqa: E402

MODES = (1, 2, 3)
DF = 3                   # dof hypothesis

REPO = Path(__file__).resolve().parents[1]

#: Slide asset consumed by the COSMO deck (SVG + PNG preview, no PDF).
TALK_DIR = (REPO / "ignore" / "talks" / "talks-main" / "2026 COSMO"
            / "public" / "assets" / "luhdm")
TALK_STEM = "talk_efficiency_modes"

# Plotted window: left edge matches the data-spectrum figure so the two panels
# stack; right edge sits inside every mode's calibrated table (mode 1 ends at
# 3.16e4 GeV), so nothing is drawn beyond calibration.
X_LO, X_HI = 3.0e1, 3.0e4
# Headroom above eps = 1 keeps the saturated curves clear of the corner tag,
# and the small negative floor keeps the eps = 0 runs off the bottom spine.
Y_LO, Y_HI = -0.05, 1.20

#: Efficiency at which the in-place mode labels are anchored.  Chosen on the
#: steep part of every sigmoid -- where the curves are farthest apart and the
#: label sits unambiguously next to one of them -- and away from the eps = 0.5
#: reference rule, which would otherwise run through the text.
LABEL_EPS = 0.62

#: (colour, dash, z, marker) per mode.  Colour is the per-mode contract; the
#: dashes differ for every mode so the figure survives grayscale printing and
#: CVD (charter F9/F10).  Line width is left to the rcParams so the print and
#: talk styles each get their own weight, and matplotlib scales the dash
#: patterns with it.
#:
#: Modes 1 and 3 sit at *higher* zorder than mode 2 on purpose: below ~400 GeV
#: and above ~10 TeV all three curves coincide, and a broken line drawn over
#: the solid one lets the reader see that they overlap instead of hiding two
#: series under the third.
STYLE = {
    1: dict(color=ps.MODE_COLORS[1], ls=(0, (4.0, 1.7)), zorder=4, marker="s"),
    2: dict(color=ps.MODE_COLORS[2], ls="-", zorder=3, marker="o"),
    3: dict(color=ps.MODE_COLORS[3], ls=(0, (1.0, 1.4)), zorder=5, marker="^"),
}


# --------------------------------------------------------------------------- #
# Efficiency-table numerics.  Shared with paper_fig_data_spectrum.py, which
# overlays the mode-1 curve on the candidate histogram.
# --------------------------------------------------------------------------- #
def efficiency_interp(q_tab, eff_tab):
    """eps(q) with the analysis' extrapolation: 0 below the table, held above.

    Mirrors ``luhdm.efficiency.make_efficiency`` so the drawn curve is the same
    function the limit actually used, including outside the calibrated range.
    """
    q_tab = np.asarray(q_tab, dtype=float)
    eff_tab = np.asarray(eff_tab, dtype=float)
    e_hi = float(eff_tab[-1])
    return lambda q: np.interp(np.asarray(q, dtype=float), q_tab, eff_tab,
                               left=0.0, right=e_hi)


def q_at_efficiency(q_tab, eff_tab, level=0.5):
    """Momentum where the tabulated efficiency crosses ``level``.

    Linear in log q between the two bracketing table points.  Asserts that the
    table is monotone and that ``level`` is genuinely bracketed, so a future
    release with a different curve fails loudly instead of silently returning
    an edge value.
    """
    q = np.asarray(q_tab, dtype=float)
    e = np.asarray(eff_tab, dtype=float)
    assert np.all(np.diff(e) >= -1e-12), "efficiency table is not monotone in q"
    assert e[0] < level < e[-1], (
        f"efficiency level {level} not bracketed by the table "
        f"[{e[0]:.3g}, {e[-1]:.3g}]")
    i = int(np.searchsorted(e, level))
    e0, e1 = e[i - 1], e[i]
    assert e1 > e0, "degenerate bracket around the requested efficiency"
    t = (level - e0) / (e1 - e0)
    return float(10.0 ** (np.log10(q[i - 1])
                          + t * (np.log10(q[i]) - np.log10(q[i - 1]))))


def build(rel, talk=False):
    q_plot = np.geomspace(X_LO, X_HI, 1200)
    curves, q50, q_lab = {}, {}, {}
    for m in MODES:
        q_tab, eff_tab = rel.efficiency_curve(m, DF)
        assert np.all((eff_tab >= 0.0) & (eff_tab <= 1.0)), \
            f"mode {m} efficiency outside [0,1]"
        curves[m] = efficiency_interp(q_tab, eff_tab)(q_plot)
        q50[m] = q_at_efficiency(q_tab, eff_tab, 0.5)
        q_lab[m] = q_at_efficiency(q_tab, eff_tab, LABEL_EPS)
        print(f"  mode {m}: table {q_tab[0]:.1f}-{q_tab[-1]:.1f} GeV, "
              f"eps_max = {eff_tab[-1]:.4f}, q50 = {q50[m]:.1f} GeV")
        assert X_LO < q50[m] < X_HI, \
            f"mode {m} q50 = {q50[m]:.1f} GeV is outside the plotted window"

    if talk:
        ps.apply_talk_style()
    else:
        ps.apply_prl_style()
    fig, ax = plt.subplots(figsize=ps.TALK_FIGSIZE if talk else ps.FIGSIZE)
    ax.tick_params(which="both", top=True, right=True)

    # -- eps = 0.5 reference level, under the data --------------------------
    # A plain light rule, not a dashed one: a dashed guide would read as a
    # fourth series next to the dashed mode-1 curve.
    ax.axhline(0.5, color="#BBBBBB", ls="-", lw=1.0 if talk else 0.5, zorder=1)

    lines = []
    for m in MODES:
        st = dict(STYLE[m])
        marker = st.pop("marker")
        (ln,) = ax.plot(q_plot, curves[m], label=f"mode {m}",
                        solid_capstyle="round", dash_capstyle="round", **st)
        # 50% crossing, marked on the curve itself
        ax.plot([q50[m]], [0.5], marker=marker, ms=6.0 if talk else 3.0,
                ls="none", mfc="white", mec=st["color"],
                mew=1.4 if talk else 0.8, zorder=6)
        lines.append(ln)

    ax.set_xscale("log")
    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_xlabel(r"Impulse momentum $q$ (GeV)")
    ax.set_ylabel(r"Detection efficiency $\varepsilon(q)$")
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1), numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())

    # -- direct labels instead of a legend -----------------------------------
    # A legend would cost a key lookup per curve and, at three well-separated
    # sigmoids, buys nothing.  ``halo()`` cannot be used to protect the glyphs
    # (it breaks under text.usetex, see its docstring), so each label goes in
    # genuinely empty space: the mode whose curve rises first is named to its
    # *left*, out over the eps = 0 run where nothing is drawn; the others are
    # named to the right of their own rise, in the gap before the next curve.
    # Which mode is leftmost is read off the release, not assumed.
    order = sorted(MODES, key=lambda m: q_lab[m])
    pad_pt = 6.0 if talk else 3.5
    labels = {}
    for m in MODES:
        left_of_curve = (m == order[0])
        labels[f"label mode {m}"] = ax.annotate(
            f"mode {m}", xy=(q_lab[m], LABEL_EPS),
            xytext=(-pad_pt if left_of_curve else pad_pt, 0),
            textcoords="offset points",
            ha="right" if left_of_curve else "left", va="center",
            color=STYLE[m]["color"], zorder=7, annotation_clip=False)

    tag = ps.add_preliminary_tag(ax, ps.preliminary_tag_text(rel.version_tag))

    # -- self-gating ---------------------------------------------------------
    gated = dict(labels)
    gated.update({
        "preliminary tag": tag,
        "x label": ax.xaxis.label,
        "y label": ax.yaxis.label,
    })
    ps.assert_text_clearance(fig, gated, min_gap_pt=1.0)
    ps.assert_inside_figure(fig, gated)
    # The clearance check above is text-vs-text only; a label parked on a curve
    # is the other way this figure can rot when the cube changes.
    _assert_clear_of_curves(fig, dict(labels, **{"preliminary tag": tag}),
                            lines, pad_pt=1.0)
    return fig, q50


def _assert_clear_of_curves(fig, named_texts, lines, pad_pt=1.0):
    """No in-axes annotation may sit on top of any curve."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for name, txt in named_texts.items():
        if txt is None:
            continue
        tb = txt.get_window_extent(renderer=r).padded(
            pad_pt * fig.dpi / 72.0 / 2.0)
        for ln in lines:
            pts = ln.axes.transData.transform(np.asarray(ln.get_xydata(), float))
            pts = pts[np.isfinite(pts).all(axis=1)]
            hit = ((pts[:, 0] >= tb.x0) & (pts[:, 0] <= tb.x1)
                   & (pts[:, 1] >= tb.y0) & (pts[:, 1] <= tb.y1))
            if hit.any():
                raise AssertionError(
                    f"{name!r} overlaps curve {ln.get_label()!r} at "
                    f"{int(hit.sum())} sampled points")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--release", type=Path, default=release.DEFAULT_PATH,
                   help="data-release HDF5 (default: %(default)s)")
    p.add_argument("--outdir", type=Path,
                   default=REPO / "ignore" / "overleaf" / "figs",
                   help="output directory (default: %(default)s)")
    # NOT "efficiency": that stem is the Letter's Fig. 2, written by
    # paper_fig_data_spectrum.py --stem efficiency, and a default run here
    # would silently overwrite it.
    p.add_argument("--stem", default="efficiency_modes123",
                   help="output basename without extension")
    p.add_argument("--talk", action="store_true",
                   help="render at slide scale and write only the talk SVG "
                        f"(+ PNG preview) to {TALK_DIR}; --outdir/--stem are "
                        "ignored")
    args = p.parse_args(argv)

    with release.open_release(args.release) as rel:
        print(f"release: {args.release}  ({rel.version_tag})")
        fig, q50 = build(rel, talk=args.talk)

    if args.talk:
        # Not savefig_exact(): that helper always writes a PDF, and the deck
        # wants the SVG alone.  The bbox rule is the same -- no tight bbox, so
        # the canvas stays exactly TALK_FIGSIZE.
        TALK_DIR.mkdir(parents=True, exist_ok=True)
        svg = TALK_DIR / f"{TALK_STEM}.svg"
        png = TALK_DIR / f"{TALK_STEM}.png"
        fig.savefig(svg, format="svg")
        fig.savefig(png, format="png", dpi=plt.rcParams["savefig.dpi"])
        plt.close(fig)
        print(ps.report_size(fig, svg))
        print(f"preview: {png}")
        return q50

    pdf = args.outdir / f"{args.stem}.pdf"
    png = args.outdir / f"{args.stem}.png"
    ps.savefig_exact(fig, pdf, png)
    plt.close(fig)
    print(ps.report_size(fig, pdf))
    print(f"preview: {png}")
    return q50


if __name__ == "__main__":
    main()
