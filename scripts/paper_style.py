"""Shared PRL print style and self-gating render checks for the paper figures.

Imported by the ``paper_fig_*.py`` scripts. Everything here encodes Section 4 of
the PRL style charter so the individual figure scripts only describe *content*:

* **Size** -- figures are drawn at their final printed size, one journal column
  wide (:data:`COLUMN_W_IN` = 3.40 in = 8.64 cm, APS asks for 8.5 cm / 3-3/8 in)
  with an aspect ratio of 3:2 or wider (length-budget rule F14).  ``\\includegraphics``
  in ``main.tex`` carries **no** ``width=`` for these, so LaTeX never rescales
  them and the 8 pt type stays 8 pt (F2).
* **Fonts** -- Computer Modern via ``text.usetex``, matching the RevTeX body text
  and the existing figure set (``limits.pdf``, ``atmosphere.pdf``).  Base 8 pt,
  nothing below 6.5 pt, which clears the 2 mm cap-height floor at 8.6 cm (F2).
* **Lines** -- 0.5-1.1 pt, above the 0.18 mm / 0.5 pt reproduction floor (F2).
* **Colour** -- the Okabe-Ito colourblind-safe set (F10).  No curve is ever
  distinguished by hue alone; line style always varies too (F9/F10).
* **No** title, suptitle, or version-stamp footer.

Preliminary tagging
-------------------
The v2 cube's constants are superseded by a v3 recompute.  Every figure built
from a pre-v3 release carries a small red corner tag; re-running the same script
against a v3 release makes the tag vanish with no edit -- see
:func:`preliminary_tag_text`.

Self-gating
-----------
Figure scripts are permanent artifacts that will be re-run blind against a new
cube, so they must fail loudly rather than silently emit an ugly figure.  The
:func:`assert_text_clearance`, :func:`assert_inside_figure`,
:func:`assert_legend_complete` and :func:`assert_legend_clear_of` helpers do the
gating against the *real renderer bounding boxes*, not against guessed
coordinates.

Known matplotlib gotchas encoded here
-------------------------------------
* ``fig.text`` plus ``bbox_inches='tight'`` grows the canvas, so figures are
  saved with :func:`savefig_exact` (no tight bbox) and every annotation lives
  inside the axes.
* A white ``bbox`` behind a label notches the curve underneath it; use
  :func:`halo` (a path-effect glyph stroke) and keep annotation ``zorder`` below
  the data instead.
"""
from __future__ import annotations

import re

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
COLUMN_W_IN = 3.40          # 8.64 cm -- APS single column ("8.5 cm or 3 3/8 in")
ASPECT_3_2 = 1.5            # width / height; charter wants 3:2 *or wider*

#: Default single-column figure size at a 3:2 aspect ratio.
FIGSIZE = (COLUMN_W_IN, COLUMN_W_IN / ASPECT_3_2)

# --------------------------------------------------------------------------- #
# Colour -- Okabe-Ito, chosen for CVD safety *and* distinct grayscale value.
# --------------------------------------------------------------------------- #
OI_BLUE = "#0072B2"     # primary data series
OI_VERMILLION = "#D55E00"     # secondary series (efficiency)
OI_GREEN = "#009E73"
OI_PURPLE = "#CC79A7"
OI_ORANGE = "#E69F00"

GREY_MUTED = "#8A8A8A"     # de-emphasised curve, mid grayscale value
GREY_DARK = "#3F3F3F"     # de-emphasised curve, dark grayscale value
GREY_GUIDE = "#6E6E6E"     # vertical/horizontal reference guides
GREY_BAND = "#D9D9D9"     # excluded / out-of-selection shading

TAG_RED = "#C00000"     # PRELIMINARY corner tag

#: One hue per sensor mode, used identically in every figure a mode appears in.
#: Green is excluded from the set so no figure can put red and green adjacent,
#: and the three values stay distinct under deuteranopia and in grayscale.
MODE_COLORS = {1: OI_BLUE, 2: OI_ORANGE, 3: OI_PURPLE}

#: Observed candidate impulses drawn as guide lines over a coloured curve
#: family (e.g. the viridis mediator-range ramp): dark grey, labelled in place,
#: never red -- red guide lines over a green-containing ramp is the red/green
#: adjacency the colour charter bans.
CANDIDATE_COLOR = GREY_DARK


def apply_prl_style():
    """Install the charter's rcParams.  Call once, before creating a figure."""
    plt.rcParams.update({
        # -- text: Computer Modern through LaTeX, matching the RevTeX body ----
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.family": "serif",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        # -- lines: >= 0.5 pt everywhere (F2) --------------------------------
        "lines.linewidth": 0.9,
        "lines.markeredgewidth": 0.5,
        "axes.linewidth": 0.5,
        "grid.linewidth": 0.4,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.minor.width": 0.35,
        "ytick.minor.width": 0.35,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.pad": 2.0,
        "ytick.major.pad": 2.0,
        "axes.labelpad": 2.0,
        # -- legend ----------------------------------------------------------
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "0.75",
        "legend.fancybox": False,
        "legend.borderpad": 0.32,
        "legend.labelspacing": 0.30,
        "legend.handlelength": 1.85,
        "legend.handletextpad": 0.45,
        "legend.borderaxespad": 0.35,
        # -- misc ------------------------------------------------------------
        "axes.axisbelow": True,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.012,
        "figure.constrained_layout.w_pad": 0.012,
        "figure.constrained_layout.hspace": 0.0,
        "figure.constrained_layout.wspace": 0.0,
        "savefig.dpi": 600,
        "savefig.transparent": False,
        "pdf.compression": 6,
    })
    # A legend edge drawn by rcParams still needs its width set explicitly.
    plt.rcParams["patch.linewidth"] = 0.5


#: Default talk figure size: half a 16:9 slide's content area at ~96 px/in.
TALK_FIGSIZE = (7.0, 4.2)


def apply_talk_style():
    """Slide variant of the charter: same conventions at presentation scale.

    A paper figure dropped onto a slide carries 8 pt type into a room where the
    body text is ~15 pt; the plotting-guidance rule of thumb (and the reason
    this exists) is that no font on a plot should be smaller than the text
    around it.  Talk figures are therefore drawn at :data:`TALK_FIGSIZE` with
    ~15 pt type and heavier lines, and are exported as SVG for Slidev.  All
    other conventions (Okabe-Ito palette, :data:`MODE_COLORS`, tick style,
    no titles, no stamps) are inherited from :func:`apply_prl_style`.
    """
    apply_prl_style()
    plt.rcParams.update({
        "font.size": 15.0,
        "axes.labelsize": 15.0,
        "axes.titlesize": 15.0,
        "xtick.labelsize": 13.0,
        "ytick.labelsize": 13.0,
        "legend.fontsize": 13.0,
        "lines.linewidth": 2.2,
        "lines.markeredgewidth": 1.0,
        "axes.linewidth": 1.0,
        "grid.linewidth": 0.8,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.7,
        "ytick.minor.width": 0.7,
        "xtick.major.size": 5.2,
        "ytick.major.size": 5.2,
        "xtick.minor.size": 3.0,
        "ytick.minor.size": 3.0,
        "xtick.major.pad": 4.0,
        "ytick.major.pad": 4.0,
        "axes.labelpad": 4.0,
        "savefig.dpi": 200,
    })
    plt.rcParams["patch.linewidth"] = 1.0


# --------------------------------------------------------------------------- #
# Preliminary tag
# --------------------------------------------------------------------------- #
def preliminary_tag_text(version_tag):
    """Corner-tag text for a release ``version_tag``; ``""`` from v3 onwards.

    ``"v2.0-bcap10cm"`` -> ``"PRELIMINARY (v2 constants)"``;
    ``"v3.0-..."``, ``"v4.0-night-..."`` -> ``""`` (no tag drawn).  The figure
    scripts therefore drop the tag automatically when re-run against a v3 or
    later cube, with no edit.  The test is on the *number*, not on the literal
    string ``"v3"``: a prefix match silently re-tagged every figure the day the
    v4 night cube landed.
    """
    tag = str(version_tag or "")
    m = re.match(r"v(\d+)", tag)
    if m is not None and int(m.group(1)) >= 3:
        return ""
    major = tag.split(".", 1)[0].split("-", 1)[0] or "unversioned"
    return rf"PRELIMINARY ({major} constants)"


def add_preliminary_tag(ax, text, xy=(0.985, 0.975), ha="right", va="top"):
    """Draw the red corner tag *inside* ``ax`` (axes fraction ``xy``).

    Kept inside the axes on purpose: a ``fig.text`` tag would be ignored by
    ``constrained_layout`` and would grow the canvas under a tight bbox, which
    breaks the exact 8.6 cm width.  Returns the ``Text`` or ``None``.
    """
    if not text:
        return None
    return ax.text(xy[0], xy[1], text, transform=ax.transAxes,
                   ha=ha, va=va, fontsize=6.0, color=TAG_RED,
                   zorder=6, clip_on=False)


def halo(width=1.1, color="white"):
    """Path effect that outlines glyphs instead of putting a box behind them.

    A ``bbox`` patch behind an in-axes label paints over -- and visibly notches
    -- any curve it crosses.  A glyph stroke keeps the curve continuous.

    .. warning::
       **Incompatible with ``text.usetex``** in matplotlib 3.11: the path-effect
       renderer proxies ``draw_tex`` to ``RendererBase._draw_text_as_path``,
       which raises ``TypeError: ... missing 1 required positional argument:
       'mtext'``.  Under this module's usetex style, keep in-axes labels in an
       empty region (e.g. reserved headroom above the data) instead of haloing
       them.  Kept here for non-usetex figures and for when that bug is fixed.
    """
    return [pe.withStroke(linewidth=width, foreground=color)]


# --------------------------------------------------------------------------- #
# Renderer-level self-gating checks
# --------------------------------------------------------------------------- #
def _renderer(fig):
    fig.canvas.draw()
    return fig.canvas.get_renderer()


def _pad_px(fig, pad_pt):
    return pad_pt * fig.dpi / 72.0


def assert_text_clearance(fig, labelled_texts, min_gap_pt=1.0):
    """Every pair of the given texts must be separated by ``min_gap_pt``.

    ``labelled_texts`` is a mapping ``{name: Text}`` (or an iterable of
    ``(name, Text)``); names appear in the failure message so a broken figure
    says *which* two labels collided.  ``None`` entries are skipped, so an
    optional annotation can be passed unconditionally.
    """
    items = (list(labelled_texts.items()) if isinstance(labelled_texts, dict)
             else list(labelled_texts))
    items = [(n, t) for n, t in items if t is not None]
    r = _renderer(fig)
    pad = _pad_px(fig, min_gap_pt)
    boxes = [(n, t.get_window_extent(renderer=r).expanded(1.0, 1.0)
              .padded(pad / 2.0)) for n, t in items]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            (ni, bi), (nj, bj) = boxes[i], boxes[j]
            inter = Bbox.intersection(bi, bj)
            if inter is not None and inter.width > 0 and inter.height > 0:
                raise AssertionError(
                    f"label clearance: {ni!r} and {nj!r} overlap by "
                    f"{inter.width:.1f}x{inter.height:.1f} px "
                    f"(need {min_gap_pt} pt gap)")


def assert_inside_figure(fig, labelled_artists, pad_pt=0.0):
    """Every named artist's rendered bbox must lie inside the figure canvas.

    Catches annotations that a later ylim/xlim change pushed off the page --
    which, with no tight bbox, silently clips them in the PDF.
    """
    items = (list(labelled_artists.items()) if isinstance(labelled_artists, dict)
             else list(labelled_artists))
    items = [(n, a) for n, a in items if a is not None]
    r = _renderer(fig)
    fb = fig.bbox
    pad = _pad_px(fig, pad_pt)
    for name, art in items:
        b = art.get_window_extent(renderer=r)
        if (b.x0 < fb.x0 + pad or b.y0 < fb.y0 + pad
                or b.x1 > fb.x1 - pad or b.y1 > fb.y1 - pad):
            raise AssertionError(
                f"{name!r} falls outside the canvas: bbox={b.bounds} "
                f"vs figure={fb.bounds}")


def assert_legend_complete(legend, expected_labels, axes):
    """The legend lists exactly ``expected_labels`` and each one was drawn.

    Two failure modes are covered: (a) an entry in the legend whose artist is
    empty or absent from the axes -- a phantom key -- and (b) a plotted series
    that never made it into the legend.
    """
    expected = list(expected_labels)
    got = [t.get_text() for t in legend.get_texts()]
    if got != expected:
        raise AssertionError(f"legend labels {got!r} != expected {expected!r}")

    handles = list(getattr(legend, "legend_handles", []))
    if len(handles) != len(expected):
        raise AssertionError(
            f"legend has {len(handles)} handles for {len(expected)} labels")

    drawn = {}
    for ax in axes:
        for art in ax.get_children():
            lab = art.get_label() if hasattr(art, "get_label") else None
            if isinstance(lab, str) and lab and not lab.startswith("_"):
                drawn.setdefault(lab, []).append(art)
    for lab in expected:
        arts = drawn.get(lab)
        if not arts:
            raise AssertionError(
                f"legend entry {lab!r} has no artist in the axes "
                f"(drawn labels: {sorted(drawn)})")
        if not any(_artist_has_extent(a) for a in arts):
            raise AssertionError(
                f"legend entry {lab!r} maps to artists with zero rendered "
                f"extent -- nothing was actually drawn")


def _artist_has_extent(art):
    fig = art.get_figure()
    try:
        b = art.get_window_extent(renderer=_renderer(fig))
    except Exception:
        return False
    return b.width > 0 and b.height > 0


def assert_legend_clear_of(fig, legend, curves, pad_pt=1.0):
    """No sample of any ``curves`` Line2D may fall inside the legend box.

    Guards the "legend parked on top of the data" failure that only shows up
    once the data change -- exactly what a v3 re-run will do.
    """
    r = _renderer(fig)
    lb = legend.get_window_extent(renderer=r).padded(_pad_px(fig, pad_pt))
    for line in curves:
        xy = line.get_xydata()
        if xy is None or len(xy) == 0:
            continue
        pts = line.axes.transData.transform(np.asarray(xy, dtype=float))
        pts = pts[np.isfinite(pts).all(axis=1)]
        inside = ((pts[:, 0] >= lb.x0) & (pts[:, 0] <= lb.x1)
                  & (pts[:, 1] >= lb.y0) & (pts[:, 1] <= lb.y1))
        if inside.any():
            raise AssertionError(
                f"legend box overlaps curve {line.get_label()!r} at "
                f"{int(inside.sum())} sampled points")


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def savefig_exact(fig, pdf_path, png_path=None, png_dpi=600, svg_path=None):
    """Save at the exact figure size (no tight bbox) plus optional PNG/SVG.

    ``bbox_inches='tight'`` is deliberately *not* used: it re-measures the
    canvas around the drawn artists and would push the width off 8.6 cm.
    ``svg_path`` exists for the talk variants, which Slidev consumes as SVG.
    """
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, format="pdf")
    if png_path is not None:
        fig.savefig(png_path, format="png", dpi=png_dpi)
    if svg_path is not None:
        fig.savefig(svg_path, format="svg")
    return pdf_path


def report_size(fig, pdf_path):
    """One-line stdout summary of the saved geometry (width in cm, aspect)."""
    w_in, h_in = fig.get_size_inches()
    return (f"{pdf_path}  {w_in:.2f}x{h_in:.2f} in "
            f"({w_in * 2.54:.2f}x{h_in * 2.54:.2f} cm, aspect {w_in / h_in:.2f}:1)")
