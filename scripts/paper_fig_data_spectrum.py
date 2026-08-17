#!/usr/bin/env python3
"""PRL figure: the mode-1 impulse data the UHDM search runs on.

Reconstructed impulse momenta of the mode-1 candidate sample -- mode 1 is the
paper's channel, the night-selected search reported in the Letter -- the observed
data, with no signal model on top (a signal curve would need a coupling, and the
normalisation is coupling dependent, so drawing one would be a choice, not a
measurement).  The panel carries three things:

1. a histogram of the surviving impulse candidates in log-spaced momentum bins
   (0.25 dex), read from ``/detector/events_mode1`` in the data release, which is
   the release's copy of ``notebooks/data_mode1.txt`` converted from eV to GeV;
2. the sub-threshold blip population in muted grey, *if* the release's
   ``/detector/all_blips_mode1`` dataset actually extends below the selection.
   In the night-selection cube it does not -- ``all_blips_mode1`` is the whole
   run's 66 up-crossings, in eV, a wider selection than the 8 night candidates
   -- so nothing grey is drawn and the script says so on stdout.  A later
   release that adds sub-threshold blips lights this up with no edit here;
3. the mode-1 detection efficiency eps(q) for the df = 3 dof hypothesis on a
   right-hand axis, using the analysis' own extrapolation convention
   (``luhdm.efficiency.make_efficiency``: zero below the calibrated table, held
   at the saturated value above it).

Two vertical guides are drawn: the selection edge at q_thresh (0.1 TeV, taken
from the release attributes) and the momentum at which eps = 0.5, which is
*measured* from the efficiency table at run time rather than hard-coded, so the
label tracks the release.

Style: PRL single column at final printed size; see ``scripts/paper_style.py``.
Figures built from a pre-v3 cube carry a red PRELIMINARY corner tag; the tag
disappears by itself when ``--release`` points at a v3 or later file.

Usage
-----
    python scripts/paper_fig_data_spectrum.py                # default release
    python scripts/paper_fig_data_spectrum.py --release <h5> # e.g. the night cube
    python scripts/paper_fig_data_spectrum.py --stem efficiency   # Letter Fig. 2
    python scripts/paper_fig_data_spectrum.py --talk        # slide-scale SVG/PNG

``--talk`` draws the SAME panel -- same histogram, efficiency curve, guides and
legend -- at presentation scale (see :func:`set_scale`) and writes it to the
talk asset tree as ``talk_<stem>.svg``/``.png``. It never touches the paper's
``figs/``.
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
from matplotlib.ticker import LogLocator, NullFormatter  # noqa: E402

import paper_style as ps  # noqa: E402
from luhdm import release  # noqa: E402
# Same eps(q) evaluation and eps = 0.5 solver as the standalone efficiency
# figure, so the two figures cannot disagree about the mode-1 curve.  Both
# helpers take the (q, eps) tables themselves, so they carry no mode of their own.
from paper_fig_efficiency import efficiency_interp, q_at_efficiency  # noqa: E402

MODE = 1                 # the paper's channel: the night-selected mode-1 search
DF = 3                   # dof hypothesis for the efficiency curve
DEX = 0.25               # histogram bin width in decades of q
EV_PER_GEV = 1.0e9

X_LO, X_HI = 3.0e1, 3.0e5    # plotted momentum window (GeV)
HEADROOM = 1.30              # top of both y axes / the data maximum

# --------------------------------------------------------------------------- #
# Talk variant (--talk): the same panel at slide scale
# --------------------------------------------------------------------------- #
#: Talk scale factors for the pt-valued literals in this file, matching the
#: jump ``paper_style.apply_talk_style`` makes in the rcParams (8 -> 15 pt type,
#: 0.9 -> 2.2 pt lines). Every hardcoded fontsize/linewidth/markersize below is
#: written as ``literal * S_FONT`` or ``literal * S_LINE``: a 7 pt guide label
#: left behind would be unreadable from the back of a room even though the
#: axis labels around it had grown.
FONT_SCALE = 1.9
LINE_SCALE = 2.2

#: 200 dpi over 7 in is 1400 px: sharp on a projector, small enough to ship.
TALK_PNG_DPI = 200

#: Where each variant is written. The talk tree is the Slidev asset directory;
#: --talk writes there and only there, never into the paper's figs/.
PAPER_OUTDIR = Path(__file__).resolve().parents[1] / "ignore" / "overleaf" / "figs"
TALK_OUTDIR = (Path(__file__).resolve().parents[1] / "ignore" / "talks"
               / "talks-main" / "2026 COSMO" / "public" / "assets" / "luhdm")

#: Live scale factors; 1.0 outside --talk, so every literal below evaluates to
#: exactly the number it was written as and the paper figure is unchanged.
S_FONT = 1.0
S_LINE = 1.0


def set_scale(talk):
    """Point the pt-valued literals at print scale (``talk=False``) or slide."""
    global S_FONT, S_LINE
    S_FONT = FONT_SCALE if talk else 1.0
    S_LINE = LINE_SCALE if talk else 1.0


def scaled_dash(dash):
    """A ``(offset, (on, off, ...))`` dash pattern scaled with the line weight.

    Dash runs are in points, so leaving them alone under --talk would put the
    print figure's 1.8 pt gaps on a 2.2 pt line and read as solid.
    """
    offset, seq = dash
    if not seq:                      # (0, ()) -- solid, and stays solid
        return dash
    return (offset * S_LINE, tuple(v * S_LINE for v in seq))


def save_talk(fig, outdir, stem, dpi=TALK_PNG_DPI):
    """Write ``stem``.svg and ``stem``.png at the exact figure size.

    Slidev consumes the SVG and the PNG is the raster fallback; the talk asset
    tree carries no PDFs, so the PDF ``savefig_exact`` always writes is sent to
    a scratch directory and discarded. Returns ``(svg, png)``.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    svg, png = outdir / f"{stem}.svg", outdir / f"{stem}.png"
    with tempfile.TemporaryDirectory() as tmp:
        ps.savefig_exact(fig, Path(tmp) / f"{stem}.pdf", png, png_dpi=dpi,
                         svg_path=svg)
    return svg, png


# --------------------------------------------------------------------------- #
# Numbers (measured, never assumed)
# --------------------------------------------------------------------------- #
def log_bin_edges(q_thresh, q_min, q_max, dex=DEX):
    """Bin edges on a ``dex``-wide log grid anchored exactly on ``q_thresh``.

    Anchoring on the threshold puts a bin edge on the selection line, so the
    histogram never straddles the cut, and lets a sub-threshold population
    extend the same grid downwards.
    """
    k_lo = int(np.floor((np.log10(q_min) - np.log10(q_thresh)) / dex))
    k_hi = int(np.ceil((np.log10(q_max) - np.log10(q_thresh)) / dex))
    k = np.arange(min(k_lo, 0), k_hi + 1)
    return q_thresh * 10.0 ** (dex * k)


def tev(q_gev):
    """'0.96' style TeV string for a momentum in GeV."""
    return f"{q_gev / 1.0e3:.2f}".rstrip("0").rstrip(".")


def count_tick_step(y_max, n_ticks=3):
    """A 1/2/5 x 10^n count step giving about ``n_ticks`` labelled ticks.

    The night selection leaves a two-count tallest bin where the whole-run
    sample had forty, so a fixed step of ten labels the left axis with nothing
    but the zero.  Never returns less than one, since the axis counts events.
    """
    target = max(float(y_max) / n_ticks, 1.0)
    decade = 10.0 ** np.floor(np.log10(target))
    for mult in (1.0, 2.0, 5.0):
        if mult * decade >= target:
            return mult * decade
    return 10.0 * decade


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def build(rel, talk=False):
    q_thresh = float(rel.attrs.get("q_thresh_gev", 100.0))
    events = np.asarray(rel.events(MODE), dtype=float)          # GeV
    blips = np.asarray(rel.all_blips(MODE), dtype=float) / EV_PER_GEV
    q_tab, eff_tab = rel.efficiency_curve(MODE, DF)

    assert events.size > 0, f"no mode-{MODE} candidates in the release"
    assert np.all(np.isfinite(events)), "non-finite candidate momenta"
    assert np.all((eff_tab >= 0.0) & (eff_tab <= 1.0)), "efficiency outside [0,1]"

    # ``all_blips_modeN`` is written straight from the raw analysis file and is
    # *not* re-cut with the analysis selection, so it can describe a wider run
    # than the candidates do (in the night-selection cube it is the whole run:
    # 66 mode-1 blips against 8 night candidates). Drawing its sub-threshold
    # tail beside a differently-selected histogram would put two exposures in
    # one panel. The tell is an above-threshold blip that is not a candidate:
    # in a same-selection list every one of them is, by construction.
    above = np.sort(blips[blips >= q_thresh])
    same_selection = (above.size == events.size
                      and np.allclose(above, np.sort(events), rtol=1e-9))
    sub = np.sort(blips[blips < q_thresh]) if same_selection else blips[:0]
    has_sub = sub.size > 0
    eps = efficiency_interp(q_tab, eff_tab)
    q50 = q_at_efficiency(q_tab, eff_tab, 0.5)

    print(f"  candidates (mode {MODE}): {events.size} "
          f"in {events.min():.1f}-{events.max():.1f} GeV")
    print(f"  all_blips_mode{MODE}: {blips.size} entries, "
          f"{blips.min():.1f}-{blips.max():.1f} GeV, "
          f"{int(above.size)} above the {q_thresh:g} GeV selection")
    if same_selection:
        print(f"  -> same selection as the candidates; {sub.size} "
              f"sub-threshold blips drawn")
    else:
        print(f"  -> a *different* (wider) selection than the {events.size} "
              f"candidates: {int(above.size - events.size)} above-threshold "
              f"blips are not candidates. Not drawn; the panel shows the "
              f"candidate sample only.")
    print(f"  eps(q) = 0.5 at q = {q50:.1f} GeV = {q50 / 1e3:.3f} TeV")

    q_lo = min(events.min(), sub.min()) if has_sub else events.min()
    edges = log_bin_edges(q_thresh, q_lo, events.max())
    counts_sel, _ = np.histogram(events, bins=edges)
    counts_sub, _ = np.histogram(sub, bins=edges) if has_sub else (None, None)
    y_max = float(counts_sel.max() if not has_sub
                  else max(counts_sel.max(), counts_sub.max()))
    y_top = HEADROOM * y_max
    print(f"  {len(edges) - 1} bins of {DEX} dex, tallest bin {int(y_max)}")

    set_scale(talk)
    (ps.apply_talk_style if talk else ps.apply_prl_style)()
    # Same single panel with its twinned efficiency axis, at slide size.
    fig, ax = plt.subplots(figsize=ps.TALK_FIGSIZE if talk else ps.FIGSIZE)
    axr = ax.twinx()
    ax.tick_params(which="both", top=True, right=False)
    axr.tick_params(which="both", top=False, right=True, direction="in")

    # -- selection region: everything to the left of the cut is discarded ----
    ax.axvspan(X_LO, q_thresh, facecolor=ps.GREY_BAND, alpha=0.45,
               edgecolor="none", zorder=0)

    # -- vertical guides (below the data: zorder < the histogram) -----------
    for xq in (q_thresh, q50):
        ax.axvline(xq, color=ps.GREY_GUIDE, ls=scaled_dash((0, (3, 2))),
                   lw=0.55 * S_LINE, zorder=1.5)

    # -- histograms ----------------------------------------------------------
    lbl_sel = rf"Candidates, mode {MODE}"
    lbl_sub = r"Sub-threshold blips"
    lbl_eff = rf"Efficiency $\varepsilon(q)$, df $= {DF}$"

    # The sub-threshold population, when a release carries one, is separated
    # from the candidates by fill tone *and* by a dashed outline, so the two
    # histograms stay distinct in grayscale as well as in colour (F9).
    if has_sub:
        ax.stairs(counts_sub, edges, fill=True, facecolor="#BFBFBF", alpha=0.55,
                  edgecolor=ps.GREY_DARK, ls=scaled_dash((0, (2.6, 1.4))),
                  lw=0.7 * S_LINE, zorder=2, label=lbl_sub)
    ax.stairs(counts_sel, edges, fill=True, facecolor=ps.OI_BLUE, alpha=0.38,
              edgecolor=ps.OI_BLUE, lw=0.8 * S_LINE, zorder=3, label=lbl_sel)

    # -- efficiency on the right axis ---------------------------------------
    q_curve = np.geomspace(X_LO, X_HI, 900)
    (eff_line,) = axr.plot(q_curve, eps(q_curve), color=ps.OI_VERMILLION,
                           ls=scaled_dash((0, (4.5, 1.8))), lw=1.0 * S_LINE,
                           zorder=4, label=lbl_eff)
    axr.plot([q50], [0.5], marker="o", ms=2.6 * S_FONT, mfc="white",
             mec=ps.OI_VERMILLION, mew=0.7 * S_LINE, ls="none", zorder=5)

    # -- axes ----------------------------------------------------------------
    ax.set_xscale("log")
    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(0.0, y_top)
    ax.set_xlabel(r"Impulse momentum $q$ (GeV)")
    ax.set_ylabel(rf"Candidates per {DEX} dex")
    ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.xaxis.set_minor_locator(
        LogLocator(base=10.0, subs=tuple(np.arange(2, 10) * 0.1), numticks=12))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(np.arange(0.0, y_top, count_tick_step(y_max)))

    axr.set_ylim(0.0, HEADROOM)
    axr.set_yticks([0.0, 0.5, 1.0])
    axr.set_ylabel(r"Detection efficiency $\varepsilon(q)$")
    # Match the frame the style sheet draws (0.5 pt in print, 1.0 pt in --talk)
    # rather than a literal, so the twin's spine cannot end up a different
    # weight from the three it joins.
    axr.spines["right"].set_linewidth(plt.rcParams["axes.linewidth"])

    # -- guide labels, in the headroom band reserved above the data ---------
    # No halo: HEADROOM keeps this band empty, and ps.halo() cannot be combined
    # with text.usetex in matplotlib 3.11 (see paper_style.halo).
    t_sel = ax.text(q_thresh, 0.985, "%s TeV\nselection" % tev(q_thresh),
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=7.0 * S_FONT, color="black", linespacing=1.15,
                    zorder=5)
    t_50 = ax.text(q50, 0.985, "%s TeV\n$\\varepsilon = 0.5$" % tev(q50),
                   transform=ax.get_xaxis_transform(), ha="center", va="top",
                   fontsize=7.0 * S_FONT, color="black", linespacing=1.15,
                   zorder=5)

    # -- legend --------------------------------------------------------------
    handles, labels = ax.get_legend_handles_labels()
    hr, lr = axr.get_legend_handles_labels()
    handles, labels = handles + hr, labels + lr
    order = [labels.index(lbl_sel)]
    if has_sub:
        order.append(labels.index(lbl_sub))
    order.append(labels.index(lbl_eff))
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]
    # Parked in the mid-right void: the efficiency curve has already saturated
    # above it and the candidate tail is a few counts tall below it.
    leg = axr.legend(handles, labels, loc="upper right",
                     bbox_to_anchor=(0.982, 0.57), borderaxespad=0.0)
    leg.set_zorder(6)

    tag = ps.add_preliminary_tag(ax, ps.preliminary_tag_text(rel.version_tag))
    if tag is not None:
        # add_preliminary_tag fixes the tag at 6 pt, which is print scale.
        tag.set_fontsize(6.0 * S_FONT)

    # -- self-gating ---------------------------------------------------------
    ps.assert_text_clearance(fig, {
        "selection guide label": t_sel,
        "50% efficiency guide label": t_50,
        "preliminary tag": tag,
        "x label": ax.xaxis.label,
        "left y label": ax.yaxis.label,
        "right y label": axr.yaxis.label,
    }, min_gap_pt=1.0)
    ps.assert_inside_figure(fig, {
        "selection guide label": t_sel,
        "50% efficiency guide label": t_50,
        "preliminary tag": tag,
        "legend": leg,
        "x label": ax.xaxis.label,
        "left y label": ax.yaxis.label,
        "right y label": axr.yaxis.label,
    })
    ps.assert_legend_complete(leg, labels, [ax, axr])
    ps.assert_legend_clear_of(fig, leg, [eff_line], pad_pt=1.0)
    return fig, {"q_thresh": q_thresh, "q50": q50, "n_events": int(events.size),
                 "n_sub": int(sub.size)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--release", type=Path, default=release.DEFAULT_PATH,
                   help="data-release HDF5 (default: %(default)s)")
    p.add_argument("--outdir", type=Path, default=None,
                   help="output directory (default: ignore/overleaf/figs, or "
                        "the talk asset tree under --talk)")
    p.add_argument("--stem", default="data_spectrum",
                   help="output basename without extension "
                        "(--talk writes talk_<stem>)")
    p.add_argument("--talk", action="store_true",
                   help="slide variant of the same panel: ~15 pt type, heavier "
                        f"lines, {ps.TALK_FIGSIZE[0]:.1f}x{ps.TALK_FIGSIZE[1]:.1f}"
                        " in, written as SVG+PNG to the talk asset tree. Never "
                        "writes to the paper's figs/.")
    args = p.parse_args(argv)

    with release.open_release(args.release) as rel:
        print(f"release: {args.release}  ({rel.version_tag})")
        fig, info = build(rel, talk=args.talk)

    if args.talk:
        outdir = args.outdir or TALK_OUTDIR
        main_out, png = save_talk(fig, outdir, f"talk_{args.stem}")
    else:
        outdir = args.outdir or PAPER_OUTDIR
        main_out = outdir / f"{args.stem}.pdf"
        png = outdir / f"{args.stem}.png"
        ps.savefig_exact(fig, main_out, png)
    plt.close(fig)
    print(ps.report_size(fig, main_out))
    print(f"preview: {png}")
    return info


if __name__ == "__main__":
    main()
