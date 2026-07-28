#!/usr/bin/env python3
"""PRL figure: detection efficiency eps(q) for the three sensor modes (df = 3).

One panel, three curves.  Mode 2 -- the channel the Letter's limit is set in --
is drawn solid and in colour; modes 1 and 3 are muted greys with distinct dash
patterns, so the three are separable in print grayscale and under any colour
vision deficiency (no series is distinguished by hue alone).

The momentum at which each curve crosses eps = 0.5 is *measured* from the
tabulated curve at run time by :func:`q_at_efficiency` -- never hard-coded --
and reported both on stdout and in the legend, so the figure cannot drift from
the release it was built from.

Curves are evaluated with the analysis' own extrapolation convention
(``luhdm.efficiency.make_efficiency``): eps = 0 below the calibrated table and
held at the saturated value above it.  The plotted window stops at 3e4 GeV,
inside every mode's calibrated range, so no curve is drawn past the top of its
table; the only extrapolated stretch is the eps = 0 run below each table's
lower edge, where the tabulated value is already < 1e-18.

Style: PRL single column at final printed size; see ``scripts/paper_style.py``.
Figures built from a pre-v3 cube carry a red PRELIMINARY corner tag; the tag
disappears by itself when ``--release`` points at a v3 file.

Usage
-----
    python scripts/paper_fig_efficiency.py                # default release
    python scripts/paper_fig_efficiency.py --release <h5> # e.g. the v3 cube
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
PAPER_MODE = 2           # solid + coloured; the Letter's channel
DF = 3                   # dof hypothesis

# Plotted window: left edge matches the data-spectrum figure so the two panels
# stack; right edge sits inside every mode's calibrated table (mode 1 ends at
# 3.16e4 GeV), so nothing is drawn beyond calibration.
X_LO, X_HI = 3.0e1, 3.0e4
# Headroom above eps = 1 keeps the saturated curves clear of the corner tag,
# and the small negative floor keeps the eps = 0 runs off the bottom spine.
Y_LO, Y_HI = -0.05, 1.20

#: (colour, dash, linewidth, z) per mode.  Dashes differ for every mode, so the
#: figure survives grayscale printing and CVD (charter F9/F10).
#:
#: The muted modes sit at *higher* zorder than mode 2 on purpose: below ~400 GeV
#: and above ~10 TeV all three curves coincide, and a broken line drawn over the
#: solid one lets the reader see that they overlap instead of hiding two series
#: under the third.  Mode 2 stays prominent through colour and line width.
STYLE = {
    1: dict(color="#7C7C7C", ls=(0, (4.0, 1.7)), lw=0.8, zorder=4, marker="s"),
    2: dict(color=ps.OI_BLUE, ls="-", lw=1.1, zorder=3, marker="o"),
    3: dict(color=ps.GREY_DARK, ls=(0, (1.0, 1.4)), lw=0.85, zorder=5,
            marker="^"),
}


# --------------------------------------------------------------------------- #
# Efficiency-table numerics.  Shared with paper_fig_data_spectrum.py, which
# overlays the same mode-2 curve on the candidate histogram.
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


def build(rel):
    q_plot = np.geomspace(X_LO, X_HI, 1200)
    curves, q50 = {}, {}
    for m in MODES:
        q_tab, eff_tab = rel.efficiency_curve(m, DF)
        assert np.all((eff_tab >= 0.0) & (eff_tab <= 1.0)), \
            f"mode {m} efficiency outside [0,1]"
        curves[m] = efficiency_interp(q_tab, eff_tab)(q_plot)
        q50[m] = q_at_efficiency(q_tab, eff_tab, 0.5)
        print(f"  mode {m}: table {q_tab[0]:.1f}-{q_tab[-1]:.1f} GeV, "
              f"eps_max = {eff_tab[-1]:.4f}, q50 = {q50[m]:.1f} GeV")
        assert X_LO < q50[m] < X_HI, \
            f"mode {m} q50 = {q50[m]:.1f} GeV is outside the plotted window"

    ps.apply_prl_style()
    fig, ax = plt.subplots(figsize=ps.FIGSIZE)
    ax.tick_params(which="both", top=True, right=True)

    # -- eps = 0.5 reference level, under the data --------------------------
    # A plain light rule, not a dashed one: a dashed guide would read as a
    # fourth series next to the dashed mode-1 curve.
    ax.axhline(0.5, color="#BBBBBB", ls="-", lw=0.5, zorder=1)

    lines, labels = [], []
    for m in MODES:
        st = dict(STYLE[m])
        marker = st.pop("marker")
        lbl = rf"Mode {m}, $q_{{50}} = {q50[m]:.0f}$ GeV"
        (ln,) = ax.plot(q_plot, curves[m], label=lbl, solid_capstyle="round",
                        dash_capstyle="round", **st)
        # 50% crossing, marked on the curve itself
        ax.plot([q50[m]], [0.5], marker=marker, ms=3.0, ls="none",
                mfc="white", mec=st["color"], mew=0.8, zorder=6)
        lines.append(ln)
        labels.append(lbl)

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

    # Upper left is the only region all three sigmoids leave empty; the corner
    # tag takes the upper right, above the saturated plateau.
    leg = ax.legend(lines, labels, loc="upper left", borderaxespad=0.45)
    leg.set_zorder(8)

    tag = ps.add_preliminary_tag(ax, ps.preliminary_tag_text(rel.version_tag))

    # -- self-gating ---------------------------------------------------------
    ps.assert_text_clearance(fig, {
        "preliminary tag": tag,
        "x label": ax.xaxis.label,
        "y label": ax.yaxis.label,
    }, min_gap_pt=1.0)
    ps.assert_inside_figure(fig, {
        "preliminary tag": tag,
        "legend": leg,
        "x label": ax.xaxis.label,
        "y label": ax.yaxis.label,
    })
    ps.assert_legend_complete(leg, labels, [ax])
    ps.assert_legend_clear_of(fig, leg, lines, pad_pt=1.0)
    if tag is not None:
        _assert_tag_clear_of(fig, tag, lines, pad_pt=1.0)
    return fig, q50


def _assert_tag_clear_of(fig, tag, lines, pad_pt=1.0):
    """The PRELIMINARY tag must not sit on top of any curve."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tb = tag.get_window_extent(renderer=r).padded(pad_pt * fig.dpi / 72.0 / 2.0)
    for ln in lines:
        pts = ln.axes.transData.transform(np.asarray(ln.get_xydata(), float))
        pts = pts[np.isfinite(pts).all(axis=1)]
        hit = ((pts[:, 0] >= tb.x0) & (pts[:, 0] <= tb.x1)
               & (pts[:, 1] >= tb.y0) & (pts[:, 1] <= tb.y1))
        if hit.any():
            raise AssertionError(
                f"PRELIMINARY tag overlaps curve {ln.get_label()!r} at "
                f"{int(hit.sum())} sampled points")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--release", type=Path, default=release.DEFAULT_PATH,
                   help="data-release HDF5 (default: %(default)s)")
    p.add_argument("--outdir", type=Path,
                   default=Path(__file__).resolve().parents[1]
                   / "ignore" / "overleaf" / "figs",
                   help="output directory (default: %(default)s)")
    p.add_argument("--stem", default="efficiency",
                   help="output basename without extension")
    args = p.parse_args(argv)

    with release.open_release(args.release) as rel:
        print(f"release: {args.release}  ({rel.version_tag})")
        fig, q50 = build(rel)

    pdf = args.outdir / f"{args.stem}.pdf"
    png = args.outdir / f"{args.stem}.png"
    ps.savefig_exact(fig, pdf, png)
    plt.close(fig)
    print(ps.report_size(fig, pdf))
    print(f"preview: {png}")
    return q50


if __name__ == "__main__":
    main()
