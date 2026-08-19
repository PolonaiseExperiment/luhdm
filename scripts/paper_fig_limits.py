"""PRL money plot (``figs/results.pdf``): the two-panel Letter result.

The figure is a ``figure*`` spanning both columns and is placed by ``main.tex``
with a bare ``\\includegraphics{figs/results.pdf}`` -- no ``width=`` -- so it is
drawn at exactly the size it is printed at (:data:`FIGSIZE`, 499 x 192 pt, the
size of the draft version this replaces). Rescaling it in LaTeX would take the
8 pt type off 8 pt.

LEFT -- the coupling limit
--------------------------
The 95% CL excluded region in the (m_DM, alpha_n) plane for the headline mode
at ``f_DM = 1`` with the attenuating atmosphere, for the massless mediator and
the three finite ranges of :data:`LAMBDA_FAMILY`. For each range we take the
optimum-interval extremeness plane out of the release cube and reduce it, mass
column by mass column, to the excluded coupling interval with
``luhdm.limits.excluded_band`` -- the same level-set helper the release exposes
as ``Release.excluded_alpha_band`` and that notebooks 01/04 use, so this figure
cannot drift from them.

Those intervals are two-sided wherever the search has a ceiling: it has no
sensitivity below the floor (too few expected impulses above the threshold) and
loses it again above the ceiling, where the atmospheric overburden decelerates
the incoming flux below threshold. Building the band directly rather than
contouring the plane is what makes that ceiling explicit and yields one polygon
per range.

    HOW AN EDGE OF THE SCAN IS DRAWN. A boundary the data measured is outlined;
    a boundary that is merely where the cube stops is **not**. The fill runs to
    the edge of the scanned grid, but no line is stroked along it, so a region
    that continues past the scan reads as open rather than as a measured
    closure. That distinction is the whole point of drawing the band by hand,
    and :func:`scan_edges` / :func:`island_polygons` enforce it: every drawn
    stroke is a crossing of the confidence level.

At v10 two edges are open and both are open for a stated reason. (i) The
massless band runs to the top of the scanned coupling axis, ``alpha_n = 1``,
above ``m_DM ~ 2e12`` GeV: alpha_n = 1 is where we stopped scanning, not a
ceiling the atmosphere put there, and the panel's y axis ends at exactly that
value, so the fill runs into the top spine with no line across it. (ii) The
same band runs to the last mass column, which is ``m_Pl`` -- the deliberate end
of the mass grid, already marked in the panel by the dotted ``m_Pl`` guide, so
the fill runs into that guide with no line down it.

Two edges are *not* negotiable and remain hard errors: a floor sitting on the
bottom of the coupling grid (the quoted limit would then be the smallest
coupling computed rather than a measured crossing) and a band reaching the
low-mass end of the grid (the threshold turn-on the Letter describes would be
unmeasured). See :func:`scan_edges`.

The islands are nested (longer range => larger island), so they are painted
largest-first and their translucent fills stack. Ranges are separated by colour
*and* dash pattern (charter F9/F10: no curve distinguished by hue alone); the
colours are an ordinal viridis ramp, the right encoding for an ordered variable
like lambda and monotone in grayscale value. The massless slice is the headline
result and gets the solid, heaviest edge. Each island is named in place by its
mediator mass rather than through a legend key: the nested annuli are the only
large empty areas in the frame, a six-key legend does not fit in what is left,
and the caption speaks of "the mediator masses shown".

RIGHT -- the composite-DM cross-section recast
----------------------------------------------
The same exclusion at the paper's composite benchmark, recast as a limit on the
DM-neutron cross section. It is the SAME MODE as the left panel -- the Letter
reports mode 1 and states that the three modes are searched independently and
are not combined, so nothing here may be a composite over modes. Two things
change with respect to the left panel and both are stated in the caption: the
mediator is the 20 um / 10 meV slice only, and the abundance is the benchmark
``f_DM = 0.1``. The recast is the Monteiro convention,

    sigma_chi-n = 4 pi (hbar c)^2 mu_chi-n^2 alpha_n^2 / q0^4,   q0 = mu_chi-n v0,

a fixed multiple of alpha_n^2 (:data:`SIGMA_PER_ALPHA2`), so it is monotone and
the excluded coupling band maps to an excluded cross-section band unchanged.

Overlays
--------
*Left.* The massless-mediator limits of the two optically levitated-sphere
searches, Monteiro et al. PRL 125, 181102 (2020) and Tseng et al.
arXiv:2508.00815, read from CSVs written by
``scripts/digitize_reference_limits.py`` (exact curve vertices lifted from the
vector figures in the arXiv source packages; neither paper has a HEPData
record). Both are published at f_chi = 1, which is also this panel's
hypothesis, so they are drawn as published with no rescaling.

*Right.* The short-range fifth-force bound at the benchmark range, from
``luhdm/reference_data/fifthforce_alpha_tilde.csv``. It is carried through the
draft's own chain -- ``g_n = sqrt(16 pi u^2 G_N alpha_tilde / hbar c)``, then
``alpha_n <= g_d g_n m_DM / (4 pi m_d)`` at ``g_d = 1`` and the benchmark
constituent mass ``m_d = 1 keV`` -- and then through the same cross-section
recast as our own result, so the two curves in the panel are directly
comparable. Because alpha_n is linear in m_DM the bound is a slope-2 straight
line in this plane, and the region above it is already excluded.

    NOTE ON ATTRIBUTION. At the 20 um benchmark the envelope of that
    compilation is owned by Eot-Wash 2020 (Lee:2020zjt, alpha_tilde <~ 21), not
    by the HUST torsion balance (Tan:2020vpf), which owns 200 um. The owner is
    carried in the CSV and printed at every run; whoever writes the caption
    should cite the key this script reports.

If any overlay CSV is absent the overlay is skipped with a log line and the
figure still builds.

Usage
-----
    python scripts/paper_fig_limits.py
    python scripts/paper_fig_limits.py --release release/luhdm_datarelease_v10_A_f1_atm.h5
    python scripts/paper_fig_limits.py --talk      # slide-scale SVG/PNG

Re-running against a newer release regenerates the figure with no edit, and the
red PRELIMINARY corner tag stays absent from v3 on (see
``paper_style.preliminary_tag_text``).

``--talk`` draws the SAME figure -- same panels, colours, curves and labels --
at presentation scale (see :data:`FIGSIZE_TALK` and :func:`set_scale`) and
writes it to the talk asset tree. It never touches the paper's ``figs/``.
"""
from __future__ import annotations

import argparse
import csv
import sys
import tempfile
from collections import namedtuple
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.ticker import FixedLocator, LogLocator, NullFormatter  # noqa: E402

import paper_style as ps  # noqa: E402
from luhdm import limits, release  # noqa: E402

# --------------------------------------------------------------------------- #
# What is drawn
# --------------------------------------------------------------------------- #
MODE = 1                 # the paper's headline channel, in BOTH panels; --mode
                         # selects another. The modes are never combined.
F_DM_LEFT = 1.0          # presentation convention: everything at f_DM = 1 ...
F_DM_RIGHT = 0.1         # ... except the composite benchmark
# Atmosphere per panel. The composite benchmark has always been quoted from the
# BARE-halo surface (20um_f0p1_noatm), so the right panel must ask for it: the
# release splits the two planes across files, and no cube carries
# f_DM = 0.1 WITH atmosphere -- requesting that combination raises.
ATM_LEFT = True
ATM_RIGHT = False
BENCH_LAM = "20um"       # benchmark mediator range, m_phi ~ 10 meV
BENCH_M_D_GEV = 1.0e-6   # composite constituent mass, 1 keV/c^2
BENCH_G_D = 1.0          # dark-sector coupling, the draft's stated choice

XLIM_A = (1e4, 3e19)     # GeV/c^2; the top is one partial decade past m_Pl
YLIM_A = (1e-10, 1.0)    # alpha_n: the full scanned coupling axis
XLIM_B = (4e5, 1e8)      # GeV/c^2; the benchmark island plus breathing room
YLIM_B = (1e-29, 2e-21)  # cm^2

#: Mediator ranges, longest first -- which is also largest-island first, so the
#: nested translucent fills stack from the outside in. Labels are the mediator
#: MASS, which is what the caption names; the range each corresponds to is
#: printed at every run. Dash patterns are explicit (on, off) point runs so
#: they stay crisp at print size.
LAMBDA_FAMILY = [
    # tag,       in-figure label,      colour,    dash,                      lw
    ("massless", r"$0$ eV",            "#3B0F70", (0, ()),                   1.0),
    ("2mm",      r"$0.1$ meV",         "#3D5A8F", (0, (3.4, 1.3)),           0.75),
    ("200um",    r"$1$ meV",           "#218F8B", (0, (1.3, 1.1)),           0.75),
    ("20um",     r"$10$ meV",          "#5DC863", (0, (3.2, 1.1, 0.8, 1.1)), 0.75),
]
#: Island fill opacity. The left panel stacks four nested translucent fills, so
#: each must stay faint; the right panel draws one island over the fifth-force
#: shading and can be more solid.
FILL_ALPHA = 0.17
FILL_ALPHA_SOLO = 0.30

#: Ceiling smoothing scale, in mass-grid cells (0.12 dex each). Rendering only,
#: and applied to the CEILING alone: the atmospheric cutoff is nearly a step in
#: coupling, so the raw ceiling climbs in one-cell risers that are grid
#: resolution rather than physics. The floor is left exactly where the data put
#: it, because its minimum *is* the number the Letter quotes and a reader must
#: be able to measure it off the figure.
SMOOTH_SIGMA_CELLS = 1.1

#: Where the ``m_Pl`` guide is named: this far, in decades of alpha_n, below
#: whatever the m_Pl column already carries, and never higher than
#: :data:`M_PL_LABEL_TOP_DEX` below the top of the axis (the slot it occupied
#: when no band reached m_Pl, kept so that case is drawn exactly as before).
M_PL_LABEL_PAD_DEX = 0.20
#: Mass extent, in decades, that the rotated name occupies to the left of the
#: guide at print scale (measured: 0.40 dex, plus margin). Scaled by S_REL
#: under --talk, where the name is longer relative to the panel.
M_PL_LABEL_SPAN_DEX = 0.70
M_PL_LABEL_TOP_DEX = 0.25

#: Prior massless-mediator limits. Neutral greys separated by dash pattern, so
#: they read as context rather than as our result and survive grayscale.
REFERENCES = [
    ("monteiro2020", "Monteiro (2020)", ps.GREY_DARK,
     (0, (5.0, 1.7)), 0.75),
    # Both are drawn dashed, as the caption calls them, and told apart by
    # dash length and grayscale value rather than by hue.
    ("tseng2025", "Tseng (2025)", ps.GREY_MUTED,
     (0, (2.0, 1.5)), 0.75),
]

#: 499 x 192 pt: the natural size of the draft this replaces, and the size
#: main.tex places it at. Two column-width panels side by side.
FIGSIZE = (499.0 / 72.0, 192.0 / 72.0)

# --------------------------------------------------------------------------- #
# Talk variant (--talk): the same figure at slide scale
# --------------------------------------------------------------------------- #
#: Slide size: the printed figure's own aspect ratio and panel arrangement,
#: widened to fill the content width of a 16:9 slide. Only the scale changes;
#: nothing about *what* is drawn depends on --talk.
#:
#: The width sets the height through the fixed aspect, and the height is what
#: the choice is really about: the right panel's y label
#: ("DM--neutron cross section ...") is 3.3 in of set type at 15 pt, so a panel
#: shorter than that cannot hold its own axis label -- at 9.5 in wide the label
#: runs off the top of the canvas. 11 in leaves ~0.3 in of slack.
TALK_WIDTH_IN = 11.0
FIGSIZE_TALK = (TALK_WIDTH_IN, TALK_WIDTH_IN * FIGSIZE[1] / FIGSIZE[0])

#: Talk scale factors for the pt-valued literals in this file, matching the
#: jump ``paper_style.apply_talk_style`` makes in the rcParams (8 -> 15 pt type,
#: 0.9 -> 2.2 pt lines). Every hardcoded fontsize/linewidth below is written as
#: ``literal * S_FONT`` or ``literal * S_LINE``: an annotation left at 6.5 pt
#: would be unreadable from the back of a room even though everything the style
#: sheet controls had grown around it.
FONT_SCALE = 1.9
LINE_SCALE = 2.2

#: Label-placement clearances (:data:`LABEL_REF_CLEAR_DEX` and friends) are in
#: *data* units and were tuned against the printed type size, so what they must
#: track is how large the type is **relative to the panel** -- the type grows by
#: ``FONT_SCALE`` while the panel grows only by the figure widening.
REL_SCALE = FONT_SCALE * FIGSIZE[0] / FIGSIZE_TALK[0]

#: 200 dpi over 9.5 in is ~1900 px: sharp on a projector, small enough to ship.
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
S_REL = 1.0


def set_scale(talk):
    """Point the pt-valued literals at print scale (``talk=False``) or slide."""
    global S_FONT, S_LINE, S_REL
    S_FONT = FONT_SCALE if talk else 1.0
    S_LINE = LINE_SCALE if talk else 1.0
    S_REL = REL_SCALE if talk else 1.0


def scaled_dash(dash):
    """A ``(offset, (on, off, ...))`` dash pattern scaled with the line weight.

    Dash runs are in points, so leaving them alone under --talk would put the
    print figure's 1.3 pt gaps on a 1.65 pt line and read as solid.
    """
    offset, seq = dash
    if not seq:                      # (0, ()) -- solid, and stays solid
        return dash
    return (offset * S_LINE, tuple(v * S_LINE for v in seq))

# Z-order ladder. The Planck marker and the fifth-force band stay under the data.
Z_MARKER, Z_FILL, Z_EDGE, Z_REF, Z_TEXT = 1.0, 2.0, 3.0, 4.0, 6.0

# --------------------------------------------------------------------------- #
# The cross-section recast (Monteiro convention) and the fifth-force chain.
# Both are written out from their constants rather than pasted as a number, so
# a change of convention shows up as a code change.
# --------------------------------------------------------------------------- #
HBAR_C_GEV_CM = 1.9732698040e-14   # GeV cm
MU_CHI_N_GEV = 0.9395              # chi-n reduced mass -> m_n for heavy DM
V0_OVER_C = 1.0e-3                 # halo velocity scale entering q0

#: sigma_chi-n [cm^2] at alpha_n = 1, i.e. sigma = SIGMA_PER_ALPHA2 * alpha_n^2.
SIGMA_PER_ALPHA2 = (4.0 * np.pi * HBAR_C_GEV_CM ** 2
                    / (MU_CHI_N_GEV ** 2 * V0_OVER_C ** 4))


def to_sigma(alpha_n):
    """The recast, as the monotone ordinate transform the right panel plots."""
    return SIGMA_PER_ALPHA2 * np.asarray(alpha_n, float) ** 2

#: g_n = GN_PER_SQRT_ALPHA_TILDE * sqrt(alpha_tilde); the draft's eq:g_n_alpha
#: prefactor sqrt(16 pi u^2 G_N / hbar c) with CODATA 2018 constants and the
#: neutron-counting convention N_n = m/(2u) that the draft states.
GN_PER_SQRT_ALPHA_TILDE = 5.409269e-19


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_reference(key, ref_dir):
    """(mass_gev, alpha_n) of a digitised prior limit, or None if unavailable."""
    path = Path(ref_dir) / f"{key}_alpha_n_massless.csv"
    if not path.exists():
        return None
    rows = []
    with open(path) as fh:
        for row in csv.reader(fh):
            if not row or row[0].startswith("#") or row[0] == "mass_gev":
                continue
            rows.append((float(row[0]), float(row[1])))
    if len(rows) < 2:
        return None
    arr = np.asarray(rows, dtype=float)
    return arr[:, 0], arr[:, 1]


def load_fifth_force(ref_dir, lambda_m, rtol=1e-3):
    """``(alpha_tilde, owner_key, uncertainty)`` at ``lambda_m``, or None.

    Only an exact tabulated range is accepted. The compilation's envelope
    switches owner between experiments and has kinks where it does, so
    interpolating between the three frozen points would silently invent both a
    value and an attribution.
    """
    path = Path(ref_dir) / "fifthforce_alpha_tilde.csv"
    if not path.exists():
        return None
    for row in csv.reader(open(path)):
        if not row or row[0].startswith("#") or row[0] == "lambda_m":
            continue
        if abs(float(row[0]) / lambda_m - 1.0) < rtol:
            return float(row[1]), row[2], row[3]
    return None


def fifth_force_sigma(mass_gev, alpha_tilde, m_d_gev=BENCH_M_D_GEV,
                      g_d=BENCH_G_D):
    """``(alpha_n, sigma_cm2)`` of the fifth-force bound along the mass axis."""
    g_n = GN_PER_SQRT_ALPHA_TILDE * np.sqrt(alpha_tilde)
    alpha_n = g_d * g_n / (4.0 * np.pi) * np.asarray(mass_gev, float) / m_d_gev
    return alpha_n, SIGMA_PER_ALPHA2 * alpha_n ** 2


def excluded_band(plane, alphas, confidence):
    """Per-mass 95% CL coupling band: ``(floor, ceiling, n_nan, n_holes)``.

    Column by column in mass this is exactly ``luhdm.limits.excluded_band``
    (= ``optimum_interval.scanning.excluded_interval``), which returns the
    *log-interpolated* crossings of the confidence level -- finer than snapping
    to grid cells. Columns with no exclusion come back as NaN.

    Two conventions worth stating:

    * NaN extremeness (status == 1, the MC threw) is read as "not excluded".
      That is conservative: it can only ever shrink an island.
    * If a column's level set is not simply connected -- rare, and caused by MC
      noise one cell wide -- the band spans lowest to highest crossing. That is
      the published two-sided region: a single closed band.
    """
    plane = np.asarray(plane, float)
    n_nan = int(np.isnan(plane).sum())
    plane = np.nan_to_num(plane, nan=0.0)
    lo = np.full(plane.shape[1], np.nan)
    hi = np.full(plane.shape[1], np.nan)
    holes = 0
    for im in range(plane.shape[1]):
        column = plane[:, im]
        lo[im], hi[im] = limits.excluded_band(alphas, column, level=confidence)
        if np.isfinite(lo[im]):
            idx = np.where(column >= confidence)[0]
            holes += int(np.any(np.diff(idx) != 1))
    return lo, hi, n_nan, holes


def contiguous_runs(mask):
    """Index arrays of the maximal True runs in a 1-D boolean array."""
    idx = np.where(mask)[0]
    if not idx.size:
        return []
    return np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)


def smooth_log(y_log, sigma, cap):
    """Gaussian-weighted moving mean of ``y_log``, clipped to +/- ``cap`` of it.

    Purely a rendering step. A moving mean along the (fine) mass axis turns the
    ceiling's one-cell risers into ramps. The clip is what makes it safe: no
    drawn point moves further than ``cap`` from the crossing the data gave, so
    genuinely sharp features -- above all the tip of an island, where floor and
    ceiling close on each other within one mass step -- keep their shape instead
    of being smeared. The kernel is truncated and renormalised at the ends of
    the run, so the endpoints stay put and the island keeps its mass extent.
    """
    y_log = np.asarray(y_log, dtype=float)
    if sigma <= 0 or len(y_log) < 3:
        return y_log.copy()
    i = np.arange(len(y_log), dtype=float)
    w = np.exp(-0.5 * ((i[:, None] - i[None, :]) / sigma) ** 2)
    return np.clip((w @ y_log) / w.sum(axis=1), y_log - cap, y_log + cap)


#: One connected piece of an excluded region: the per-mass floor and ceiling
#: over a contiguous run of mass columns, plus which of its sides are the edge
#: of the scanned grid rather than a measured boundary. ``open_top`` is per
#: column (the ceiling can hit the scan cap over part of a run only);
#: ``open_right`` is a single flag for the high-mass end of the run.
Piece = namedtuple("Piece", "mass floor ceiling open_top open_right")


def scan_edges(mass, floor, ceiling, mass_axis, alpha_axis, tol=1e-9):
    """Which sides of a band are the edge of the scan rather than a boundary.

    Returns ``(open_left, open_right, open_bottom, open_top)``; ``open_top`` is
    a per-column boolean mask, the rest are scalars. A side is "open" when the
    band reaches the first/last node of the scanned grid on that side, i.e.
    where the cube stops computing rather than where the confidence level was
    crossed. Nothing here decides what is *allowed* -- see
    :func:`island_polygons` -- it only says what the data did.
    """
    return (bool(mass.min() <= mass_axis[0] * (1.0 + tol)),
            bool(mass.max() >= mass_axis[-1] * (1.0 - tol)),
            bool(floor.min() <= alpha_axis[0] * (1.0 + tol)),
            np.asarray(ceiling) >= alpha_axis[-1] * (1.0 - tol))


def island_polygons(rel, plane, confidence, cell, label):
    """``[Piece, ...]`` -- the smoothed pieces of the excluded region.

    Raises on the two edges that would make the published number unsafe (see
    the module docstring): a floor on the bottom of the coupling grid, or a
    band reaching the low-mass end of the mass grid. A band that runs off the
    TOP of the coupling axis, or off the high-mass end at ``m_Pl``, is drawn
    open -- filled to the edge of the scan, not stroked along it -- and said so
    on stdout.
    """
    ms, alphas = rel.axes.mass_gev, rel.axes.alpha_n
    lo, hi, n_nan, holes = excluded_band(plane, alphas, confidence)
    inside = np.isfinite(lo)
    if not inside.any():
        print(f"  [skip] {label}: no mass column reaches p >= {confidence:g}; "
              f"nothing drawn")
        return [], 0.0
    print(f"  [ok]   {label}: {int(inside.sum())} mass columns, "
          f"m {ms[inside].min():.3g}-{ms[inside].max():.3g} GeV, "
          f"floor {np.nanmin(lo):.3g}"
          + (f", {n_nan} NaN cells read as not-excluded" if n_nan else "")
          + (f", {holes} column(s) with a one-cell hole spanned" if holes else ""))

    out, shift = [], 0.0
    for run in contiguous_runs(inside):
        m_run = ms[run]
        # Which columns are capped by the scan is read off the RAW crossings,
        # before any cosmetic smoothing can move them.
        op_l, op_r, op_b, op_t = scan_edges(m_run, lo[run], hi[run], ms, alphas)
        if op_b or op_l:
            side = ("its floor sits on the bottom of the scanned coupling grid "
                    f"(alpha_n = {alphas[0]:g}), so the quoted limit would be "
                    "where the scan stops, not a measured crossing" if op_b else
                    "it reaches the low-mass end of the scanned mass grid "
                    f"(m_DM = {ms[0]:g} GeV), so the threshold turn-on would be "
                    "unmeasured")
            raise AssertionError(
                f"{label}: the {confidence:.0%} band cannot be drawn honestly "
                f"without extending the cube -- {side}")
        hi_raw = np.log10(hi[run])
        hi_s = smooth_log(hi_raw, SMOOTH_SIGMA_CELLS, 0.5 * cell)
        # A capped ceiling is a hard edge of the scan, not a measured curve:
        # smoothing must not dimple it below the cap.
        hi_s[op_t] = np.log10(alphas[-1])
        shift = max(shift, float(np.abs(hi_s - hi_raw).max()))
        out.append(Piece(m_run, lo[run], 10.0 ** hi_s, op_t, op_r))
        if op_t.any() or op_r:
            m_top = m_run[op_t]
            print(f"         open at the edge of the scan, drawn unstroked "
                  f"there: " + ", ".join(
                      ([f"ceiling at alpha_n = {alphas[-1]:g} over "
                        f"{int(op_t.sum())}/{len(run)} columns "
                        f"(m {m_top.min():.3g}-{m_top.max():.3g} GeV)"]
                       if op_t.any() else [])
                      + ([f"high-mass end at the last mass column "
                          f"({m_run.max():.3g} GeV)"] if op_r else [])))
            if op_t.any():
                print(f"         CAPTION MUST SAY that the region is bounded "
                      f"above there by the top of the scanned coupling axis, "
                      f"alpha_n = {alphas[-1]:g}, and not by a measured ceiling")
    return out, shift


def as_polygon(piece, transform=None):
    """Close a (floor, ceiling) band into one polygon: floor out, ceiling back.

    This is the FILLED region -- it always runs to wherever the band reaches,
    including the edge of the scan -- and is what the label-placement and
    collision gates test containment against. What is *stroked* is
    :func:`outline_segments`, which is not the same path once a side is open.

    ``transform`` maps the coupling ordinate to whatever the panel plots; the
    right panel passes the cross-section recast. It must be monotone, which the
    recast is (a positive multiple of alpha_n^2), or the band's two edges would
    cross.
    """
    m_run, floor, ceiling = piece.mass, piece.floor, piece.ceiling
    px = np.concatenate([m_run, m_run[::-1], m_run[:1]])
    py = np.concatenate([floor, ceiling[::-1], floor[:1]])
    return px, (py if transform is None else transform(py))


def outline_segments(piece):
    """``[(x, y), ...]`` -- only the parts of the boundary the data measured.

    A closed piece comes back as the single ring :func:`as_polygon` builds, so
    a fully measured island is stroked exactly as it always was, in one stroke
    with no dash-phase restart. When a side is the edge of the scan the ring is
    broken there instead: the floor and the mass ends are still stroked (they
    are crossings), the capped ceiling is not, and the fill alone carries the
    region into the top of the panel.
    """
    m_run, ceiling = piece.mass, piece.ceiling
    floor, open_top, open_right = piece.floor, piece.open_top, piece.open_right
    if not open_top.any() and not open_right:
        return [as_polygon(piece)]

    segs = []
    # The floor, plus whichever vertical ends are real boundaries. The left end
    # always is: island_polygons refuses a band that reaches the low-mass edge.
    x = np.concatenate([m_run[:1], m_run, m_run[-1:]] if not open_right
                       else [m_run[:1], m_run])
    y = np.concatenate([ceiling[:1], floor, ceiling[-1:]] if not open_right
                       else [ceiling[:1], floor])
    segs.append((x, y))
    # ... and the ceiling only where it is a measured crossing.
    for run in contiguous_runs(~open_top):
        if run.size >= 2:
            segs.append((m_run[run], ceiling[run]))
    return segs


def draw_island(ax, pieces, colour, dash, lw, label, transform=None,
                fill_alpha=FILL_ALPHA):
    """Fill each piece and stroke its measured boundary; returns the Line2Ds."""
    lines = []
    for k, piece in enumerate(pieces):
        px, py = as_polygon(piece, transform)
        ax.fill(px, py, color=colour, alpha=fill_alpha, lw=0, zorder=Z_FILL)
        for j, (sx, sy) in enumerate(outline_segments(piece)):
            line, = ax.plot(sx, sy if transform is None else transform(sy),
                            color=colour, lw=lw, ls=dash, zorder=Z_EDGE,
                            solid_joinstyle="round", dash_capstyle="butt",
                            label=label if (k == 0 and j == 0) else "_nolegend_")
            lines.append(line)
    return lines


#: Fraction of a free strip's mass extent that a label may be placed in,
#: centred. Both ends are excluded on purpose: an island pinches shut at its own
#: tip, and its other end abuts the strip of the next island out, whose label
#: would then sit right beside this one.
LABEL_STRIP_CORE = 0.6

#: Heights within the free annulus, as fractions of its log extent, that an
#: island's name may be written at. Several, so that a name pushed off one
#: height by a prior-limit curve has somewhere else to go in the same column.
LABEL_HEIGHT_FRACS = (0.30, 0.45, 0.60, 0.75)

#: A candidate column is only considered if its annulus is at least this
#: fraction as tall as the tallest one on offer. Without it the spreading rule
#: below would happily park a name in a pinched sliver at an island's tip
#: purely because that is the point furthest from its neighbours.
LABEL_MIN_HEIGHT_FRAC = 0.6

#: Vertical clearance, in decades of alpha_n, that an island's name must keep
#: from a prior-limit overlay running through the same annulus. 0.7 dex is
#: about 9 pt at this figure's scale, against a 6.5 pt label. Scaled by
#: :data:`S_REL` under --talk, where the label is larger relative to the panel.
LABEL_REF_CLEAR_DEX = 0.7

#: Half the width of a rendered island name, in decades of mass at this panel's
#: scale. Clearance from a prior-limit curve is tested right across that span,
#: not just under the anchor: the overlays climb steeply and a curve that clears
#: the centre of a name can still run through its right-hand end. Scaled by
#: :data:`S_REL` under --talk, where the name is wider relative to the panel.
LABEL_HALF_WIDTH_DEX = 0.45

#: Clearance, as a fraction of the panel diagonal, that an island's name must
#: keep from ANY island outline, its own included. The nested islands converge
#: on the same low-mass edge, so without this a name lands on a boundary even
#: when it is nowhere near that island's interior.
#:
#: NOT scaled under --talk, unlike the two clearances above. This one is capped
#: by the geometry rather than by the type size: the widest point of the 0.1
#: meV, 1 meV and 10 meV annuli is only 0.062 of the diagonal from an outline,
#: so a --talk threshold of 0.045 x S_REL = 0.062 leaves three of the four
#: islands with nowhere to put their name at all. At 0.045 the guard is still
#: 16 pt on a 12 pt slide label, i.e. more than a label height of white space.
LABEL_EDGE_CLEAR_FRAC = 0.045


def label_candidates(pieces, outer_mass_hi, refs=(), avoid=None,
                     xlim=None, ylim=None):
    """``[(mass, alpha), ...]`` -- every place an island's name could go.

    The islands are nested, so the only part of island k that is not painted
    over by island k+1 is the strip to the right of k+1's tip (and a sliver to
    its left, always narrower). Candidates are the core columns of that strip,
    each at the heights of :data:`LABEL_HEIGHT_FRACS`.  ``outer_mass_hi`` is
    the next-smaller island's right tip; pass ``None`` for the innermost
    island, whose own body is free.

    ``refs`` are the drawn prior-limit curves as ``(mass, alpha)`` arrays.
    Those cross the annuli diagonally, so they are avoided point by point:
    a candidate is dropped when a curve passes within
    :data:`LABEL_REF_CLEAR_DEX` of it at that mass. If that leaves a column
    with nothing, the column simply contributes no candidate.

    ``avoid`` is an ``(n, 2)`` array of (mass, alpha) points -- the other
    islands' outlines -- kept :data:`LABEL_EDGE_CLEAR_FRAC` of the panel
    diagonal away, measured in the log axes ``xlim``/``ylim`` define.
    """
    if avoid is not None and len(avoid):
        avoid = np.asarray(avoid, float)
        sx = np.log10(xlim[1] / xlim[0])
        sy = np.log10(ylim[1] / ylim[0])
        av = np.column_stack([np.log10(avoid[:, 0]) / sx,
                              np.log10(avoid[:, 1]) / sy])
    else:
        av = None
    out = []
    for piece in pieces:
        m_run, floor, ceiling = piece.mass, piece.floor, piece.ceiling
        free = np.ones_like(m_run, dtype=bool) if outer_mass_hi is None \
            else m_run > outer_mass_hi
        idx = np.where(free)[0]
        if not idx.size:
            continue
        # keep the central LABEL_STRIP_CORE of the strip, but never nothing
        drop = int(idx.size * (1.0 - LABEL_STRIP_CORE) / 2.0)
        core = idx[drop:idx.size - drop] if idx.size - 2 * drop >= 1 else idx
        height = np.log10(ceiling[core] / floor[core])
        keep = core[height >= LABEL_MIN_HEIGHT_FRAC * height.max()]
        for j in keep:
            m = float(m_run[j])
            half_w = LABEL_HALF_WIDTH_DEX * S_REL
            probe = np.logspace(np.log10(m) - half_w, np.log10(m) + half_w, 7)
            near = []
            for mr, ar in refs:
                inside = (probe >= mr.min()) & (probe <= mr.max())
                if inside.any():
                    near.extend(10.0 ** np.interp(
                        np.log10(probe[inside]), np.log10(mr), np.log10(ar)))
            for frac in LABEL_HEIGHT_FRACS:
                y = 10.0 ** (np.log10(floor[j])
                             + frac * np.log10(ceiling[j] / floor[j]))
                if any(abs(np.log10(y / a)) < LABEL_REF_CLEAR_DEX * S_REL
                       for a in near):
                    continue
                if av is not None:
                    d = np.hypot(av[:, 0] - np.log10(m) / sx,
                                 av[:, 1] - np.log10(y) / sy)
                    if d.min() < LABEL_EDGE_CLEAR_FRAC:
                        continue
                out.append((m, float(y)))
    return out


def spread_anchor(candidates, placed, xlim, ylim):
    """The candidate furthest (in axes fractions) from every ``placed`` anchor.

    The nested annuli mostly run parallel, so the naive "tallest column" pick
    lands every name at a similar height and the outer two collide once
    rendered. Choosing for separation instead is what keeps the four names
    legible without hand-placing any of them; with nothing placed yet it just
    takes the middle of the list.
    """
    if not candidates:
        return None
    if not placed:
        return candidates[len(candidates) // 2]
    sx = np.log10(xlim[1] / xlim[0])
    sy = np.log10(ylim[1] / ylim[0])

    def gap(c):
        return min(np.hypot(np.log10(c[0] / q[0]) / sx,
                            np.log10(c[1] / q[1]) / sy) for q in placed)

    return max(candidates, key=gap)


#: Clearance, in decades of the ordinate, kept between a curve's name and the
#: curve itself, the excluded region, any other annotation and the frame.
#: 0.1 dex of the right panel is ~2 pt against a 6.5 pt name.
CURVE_LABEL_PAD_DEX = 0.10


def place_under_curve(text, ax, curve, floors, keep_out, xlim, ylim,
                      pad=CURVE_LABEL_PAD_DEX, n_try=97):
    """Move ``text`` to the closest point *under* ``curve`` that stays legible.

    The name of an overlay has to touch its own curve to name it, and this
    panel's free space depends on the cube: with the atmosphere off the excluded
    region has no ceiling, so it is painted from its floor to the top of the
    frame and the only clear ground is *below* both the floor and the overlay.
    Rather than trust a hand-picked anchor -- which is what the v9 -> v10 change
    invalidated -- the anchor is searched for.

    ``text``'s own rendered size sets the box, so this is measured in the type
    that will be printed. For each candidate mass the box hangs from as high as
    it may: just under the lowest of ``curve`` and every band ``floors`` puts
    over its width. Candidates that fall off the bottom of ``ylim`` or touch a
    ``keep_out`` annotation are dropped, and of the rest the winner is the one
    whose box top is *closest to the curve*, i.e. the most clearly attached.

    Returns the drop, in decades, from the curve to the top of the name.
    """
    fig = ax.get_figure()
    fig.canvas.draw()
    box = text.get_window_extent(renderer=fig.canvas.get_renderer())
    inv = ax.transData.inverted()
    (bx0, by0), (bx1, by1) = inv.transform([(box.x0, box.y0), (box.x1, box.y1)])
    half_w = 0.5 * np.log10(bx1 / bx0) + pad
    height = np.log10(by1 / by0) + 2.0 * pad

    blocks = []
    for other in keep_out:
        ob = other.get_window_extent(renderer=fig.canvas.get_renderer())
        (ox0, oy0), (ox1, oy1) = inv.transform([(ob.x0, ob.y0), (ob.x1, ob.y1)])
        blocks.append((np.log10(ox0), np.log10(ox1),
                       np.log10(oy0), np.log10(oy1)))

    lm_c, ls_c = np.log10(curve[0]), np.log10(curve[1])
    best = None
    for lm in np.linspace(np.log10(xlim[0]) + half_w,
                          np.log10(xlim[1]) - half_w, n_try):
        span = np.linspace(lm - half_w, lm + half_w, 9)
        top = np.interp(span, lm_c, ls_c).min()          # under its own curve
        for f_m, f_y in floors:                          # ... and under the data
            f_m, f_y = np.log10(f_m), np.log10(f_y)
            hit = (span >= f_m.min()) & (span <= f_m.max())
            if hit.any():
                top = min(top, np.interp(span[hit], f_m, f_y).min())
        top -= pad
        if top - height < np.log10(ylim[0]) + pad:
            continue
        if any(lm - half_w < ox1 and lm + half_w > ox0
               and top > oy0 and top - height < oy1
               for ox0, ox1, oy0, oy1 in blocks):
            continue
        drop = float(np.interp(lm, lm_c, ls_c) - top)
        if best is None or drop < best[0]:
            best = (drop, lm, top)
    if best is None:
        raise AssertionError(
            f"nowhere left to write {text.get_text()!r}: every column of the "
            f"panel is covered by the excluded region, the curve itself or "
            f"another annotation")
    drop, lm, top = best
    text.set_position((10.0 ** lm, 10.0 ** (top - pad)))
    return drop


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def build_left(ax, rel, ref_dir, confidence, mode):
    """The (m_DM, alpha_n) excluded regions. Returns an artist bookkeeping dict."""
    alphas = rel.axes.alpha_n
    m_planck = float(rel.attrs.get("m_planck_gev", 1.22e19))
    cell = float(np.diff(np.log10(alphas)).mean())   # coupling grid step, dex

    curves, texts, islands, shifts = [], {}, [], []
    drawn = []                                       # (tag, label, pieces)
    for tag, label, colour, dash, lw in LAMBDA_FAMILY:
        plane = rel.mass_plane("extremeness", mode=mode, lam=tag,
                               atmosphere=ATM_LEFT, f_dm=F_DM_LEFT)
        pieces, shift = island_polygons(
            rel, plane, confidence, cell,
            f"lambda={tag} ({label.replace('$', '')})")
        if not pieces:
            continue
        shifts.append(shift)
        # No legend key: each island is named in place below, and a six-entry
        # legend does not fit in what the nested islands leave empty.
        curves += draw_island(ax, pieces, colour, scaled_dash(dash),
                              lw * S_LINE, "_nolegend_")
        islands += [as_polygon(p) for p in pieces]
        drawn.append((tag, label, colour, pieces))

    max_shift = max(shifts) if shifts else 0.0
    print(f"  smoothing: max edge shift {max_shift:.3f} dex "
          f"(coupling grid cell {cell:.3f} dex)")

    # -- prior massless-mediator limits -------------------------------------- #
    # Drawn before the islands are named so the naming can steer clear of the
    # mass range they occupy.
    labels, ref_curves, ref_xy = [], [], []
    for key, label, colour, dash, lw in REFERENCES:
        ref = load_reference(key, ref_dir)
        if ref is None:
            # Overlay stub: drop a CSV of this name into the reference-data
            # directory (scripts/digitize_reference_limits.py writes them) and
            # it is picked up automatically, with no edit to this file.
            print(f"  [skip] overlay {key}: "
                  f"{Path(ref_dir) / (key + '_alpha_n_massless.csv')} not found; "
                  f"run scripts/digitize_reference_limits.py")
            continue
        m_ref, a_ref = ref
        window = ((m_ref >= XLIM_A[0]) & (m_ref <= XLIM_A[1])
                  & (a_ref >= YLIM_A[0]) & (a_ref <= YLIM_A[1]))
        if window.sum() < 2:
            print(f"  [skip] overlay {key}: its published domain does not "
                  f"intersect the plotted window")
            continue
        first = True
        for run in contiguous_runs(window):
            if run.size < 2:
                continue
            # Separate runs: a curve that leaves and re-enters the frame must
            # not be bridged by a straight segment.
            line, = ax.plot(m_ref[run], a_ref[run], color=colour,
                            lw=lw * S_LINE, ls=scaled_dash(dash),
                            zorder=Z_REF, solid_capstyle="butt",
                            label=label if first else "_nolegend_")
            curves.append(line)
            ref_curves.append(line)
            first = False
        if first:
            continue
        labels.append(label)
        ref_xy.append((m_ref[window], a_ref[window]))
        print(f"  [ok]   overlay {key}: {int(window.sum())} points in window, "
              f"m {m_ref[window].min():.3g}-{m_ref[window].max():.3g} GeV")

    # -- name each island in its own widest annulus -------------------------- #
    placed = []
    for i, (tag, label, colour, pieces) in enumerate(drawn):
        inner = drawn[i + 1][3] if i + 1 < len(drawn) else None
        outer_hi = None if inner is None \
            else max(float(p.mass.max()) for p in inner)
        # Every island outline, this one's included: a name written along its
        # own boundary is as hard to read as one written along a neighbour's.
        others = [np.column_stack(as_polygon(p))
                  for _t, _l, _c2, pcs in drawn for p in pcs]
        cands = label_candidates(
            pieces, outer_hi, ref_xy,
            avoid=np.vstack(others) if others else None,
            xlim=XLIM_A, ylim=YLIM_A)
        anchor = spread_anchor(cands, placed, XLIM_A, YLIM_A)
        if anchor is None:
            raise AssertionError(f"no free annulus to name island {tag!r}")
        placed.append(anchor)
        if inner is not None and anchor[0] <= outer_hi:
            raise AssertionError(
                f"the {tag!r} label would be written over the next island in")
        texts[f"island {tag}"] = ax.text(
            anchor[0], anchor[1], label, color=colour, fontsize=6.5 * S_FONT,
            ha="center", va="center", zorder=Z_TEXT)
        print(f"  label {tag:>9s} at m = {anchor[0]:.3g} GeV, "
              f"alpha_n = {anchor[1]:.3g}")

    # -- axes ---------------------------------------------------------------- #
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*XLIM_A)
    ax.set_ylim(*YLIM_A)
    ax.set_xlabel(r"Dark matter mass, $m_{\mathrm{DM}}$ (GeV/$c^2$)")
    ax.set_ylabel(r"DM--neutron coupling, $\alpha_n$")
    # Label decades, but not every decade: 15 x-decades and 10 y-decades cannot
    # carry 25 legible labels at 7 pt (charter F13 against F2).
    ax.xaxis.set_major_locator(FixedLocator([1e4, 1e7, 1e10, 1e13, 1e16, 1e19]))
    ax.yaxis.set_major_locator(FixedLocator([1e-9, 1e-7, 1e-5, 1e-3, 1e-1]))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(1.0,), numticks=40))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(1.0,), numticks=20))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    # -- Planck-mass endpoint marker ----------------------------------------- #
    # The mass grid stops at m_Pl, and at v10 the massless band runs into that
    # stop, so the guide is also what tells a reader that the region's high-mass
    # end is the end of the scan rather than a measured edge.
    #
    # Its name hangs from just under whatever occupies the m_Pl column, found
    # from the bands actually drawn: the old fixed slot at the top of the axis
    # is inside the massless fill once that fill reaches alpha_n = 1. With
    # nothing at m_Pl the expression returns the top slot unchanged. No halo
    # stroke: path effects are unusable under usetex (see paper_style.halo).
    ax.axvline(m_planck, color=ps.GREY_GUIDE, lw=0.6 * S_LINE, ls=":",
               zorder=Z_MARKER)
    # Rotated, the name runs back down the mass axis from the guide, so the
    # clearance is the LOWEST floor anywhere under its length, not the floor in
    # the m_Pl column alone -- the floors rise steeply with mass out here.
    probe = np.geomspace(m_planck * 10.0 ** (-M_PL_LABEL_SPAN_DEX * S_REL),
                         m_planck, 9)
    over = []
    for _t, _l, _c, pcs in drawn:
        for p in pcs:
            hit = (probe >= p.mass.min()) & (probe <= p.mass.max())
            if hit.any():
                over.append(float(np.interp(np.log10(probe[hit]),
                                            np.log10(p.mass),
                                            np.log10(p.floor)).min()))
    y_pl = min(10.0 ** (min(over) - M_PL_LABEL_PAD_DEX) if over else YLIM_A[1],
               YLIM_A[1] * 10.0 ** -M_PL_LABEL_TOP_DEX)
    print(f"  m_Pl guide at {m_planck:.4g} GeV; name at alpha_n = {y_pl:.3g} "
          + (f"(lowest floor over its length {10.0 ** min(over):.3g})"
             if over else "(nothing drawn in that column)"))
    texts["m_Pl label"] = ax.text(
        m_planck, y_pl, r"$m_{\mathrm{Pl}}$", color=ps.GREY_GUIDE,
        fontsize=6.5 * S_FONT, ha="right", va="top", rotation=90,
        zorder=Z_MARKER)

    texts["hypothesis note"] = ax.text(
        0.028, 0.955, "$f_{\\mathrm{DM}} = 1$, 95\\% CL",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.0 * S_FONT,
        zorder=Z_TEXT)

    # Empty from v3 on, so nothing is drawn for the released cube; kept so the
    # stamp reappears by itself if the figure is ever rebuilt from an older one.
    tag = ps.add_preliminary_tag(
        ax, ps.preliminary_tag_text(rel.version_tag).replace(" (", "\n("),
        xy=(0.972, 0.975))
    if tag is not None:
        # add_preliminary_tag fixes the tag at 6 pt, which is print scale.
        tag.set_fontsize(6.0 * S_FONT)
        texts["PRELIMINARY tag"] = tag

    leg = ax.legend(loc="lower right", bbox_to_anchor=(0.985, 0.02))
    leg.set_zorder(Z_TEXT)

    return dict(legend=leg, labels=labels, texts=texts, islands=islands,
                curves=curves, ref_curves=ref_curves, max_shift=max_shift,
                alpha_cell=cell)


def build_right(ax, rel, ref_dir, confidence, mode):
    """The composite-DM sigma_chi-n recast at the benchmark. Bookkeeping dict.

    Same ``mode`` as the left panel: the modes are searched independently and
    the Letter reports one of them, so this panel must not be a per-mode
    maximum.
    """
    alphas = rel.axes.alpha_n
    cell = float(np.diff(np.log10(alphas)).mean())
    tag, label, colour, dash, lw = next(
        e for e in LAMBDA_FAMILY if e[0] == BENCH_LAM)

    plane = rel.mass_plane("extremeness", mode=mode, lam=BENCH_LAM,
                           atmosphere=ATM_RIGHT, f_dm=F_DM_RIGHT)
    pieces, shift = island_polygons(
        rel, plane, confidence, cell,
        f"mode {mode} lambda={BENCH_LAM}, f_DM={F_DM_RIGHT}")
    if not pieces:
        raise AssertionError(
            f"mode {mode} excludes nothing at the benchmark lambda={BENCH_LAM}; "
            f"there is no right panel to draw")
    curves = draw_island(ax, pieces, colour, (0, ()), 1.0 * S_LINE,
                         "_nolegend_", transform=to_sigma,
                         fill_alpha=FILL_ALPHA_SOLO)
    islands = [as_polygon(p, to_sigma) for p in pieces]
    sig_floor, m_at_floor = min(
        ((float(to_sigma(p.floor[j])), float(p.mass[j]))
         for p in pieces for j in [int(np.argmin(p.floor))]))
    print(f"  recast: sigma = {SIGMA_PER_ALPHA2:.6e} cm^2 x alpha_n^2 "
          f"(q0 = mu_chi-n v0, mu = {MU_CHI_N_GEV} GeV, v0/c = {V0_OVER_C})")
    print(f"  deepest cross-section limit {sig_floor:.4g} cm^2 "
          f"at m_DM = {m_at_floor:.4g} GeV")

    # -- axes, set before anything is placed against them --------------------- #
    # Label placement below measures rendered boxes in data coordinates, which
    # only mean anything once the panel's limits and scales are final.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*XLIM_B)
    ax.set_ylim(*YLIM_B)
    ax.set_xlabel(r"Dark matter mass, $m_{\mathrm{DM}}$ (GeV/$c^2$)")
    ax.set_ylabel(r"DM--neutron cross section, $\sigma_{\chi n}$ (cm$^2$)")
    ax.xaxis.set_major_locator(FixedLocator([1e6, 1e7, 1e8]))
    ax.yaxis.set_major_locator(FixedLocator([1e-28, 1e-26, 1e-24, 1e-22]))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(1.0,), numticks=20))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(1.0,), numticks=20))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    texts = {}
    texts["benchmark note"] = ax.text(
        0.972, 0.045,
        "composite DM\n"
        f"$m_\\phi = 10$ meV\n"
        f"$m_d = 1$ keV\n"
        f"$f_{{\\mathrm{{DM}}}} = {F_DM_RIGHT:g}$",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5 * S_FONT,
        linespacing=1.35, zorder=Z_TEXT)

    # -- fifth-force bound, recast the same way ------------------------------ #
    ff_line = None
    lam_m = float(rel.axes.lambda_m[rel.at_lambda(BENCH_LAM)])
    ff = load_fifth_force(ref_dir, lam_m)
    if ff is None:
        print(f"  [skip] fifth-force overlay: no tabulated alpha_tilde at "
              f"lambda = {lam_m:.3g} m in "
              f"{Path(ref_dir) / 'fifthforce_alpha_tilde.csv'}")
    else:
        alpha_tilde, owner, unc = ff
        m_ff = np.geomspace(*XLIM_B, 64)
        a_ff, s_ff = fifth_force_sigma(m_ff, alpha_tilde)
        ff_line, = ax.plot(m_ff, s_ff, color=ps.GREY_DARK, lw=0.75 * S_LINE,
                           ls=scaled_dash((0, (4.0, 1.6))), zorder=Z_REF,
                           label="_nolegend_")
        # Under the island fill, and faint: the bound covers most of the frame
        # and would otherwise read as the panel's subject.
        ax.fill_between(m_ff, s_ff, YLIM_B[1], color=ps.GREY_BAND, alpha=0.40,
                        lw=0, zorder=Z_MARKER)
        curves.append(ff_line)
        print(f"  [ok]   fifth-force overlay: alpha_tilde = {alpha_tilde:.4g} "
              f"({unc}) at lambda = {lam_m * 1e6:.0f} um, owner {owner}; "
              f"g_n = {GN_PER_SQRT_ALPHA_TILDE * np.sqrt(alpha_tilde):.4g}, "
              f"m_d = {BENCH_M_D_GEV * 1e6:.0f} keV, g_d = {BENCH_G_D:g}")
        print(f"         CAPTION MUST CITE {owner} for this curve")
        # Named on the curve rather than through a legend key, which would have
        # to be told apart from the island outline by dash pattern alone. It
        # goes UNDER the curve, in the only region this panel leaves free: with
        # no atmosphere there is no ceiling, so the excluded region is open
        # above and everything over its floor is painted. Which mass column has
        # room is a property of the cube, so it is measured rather than fixed --
        # a hard-coded anchor is exactly what the v9 -> v10 change broke.
        txt = ax.text(np.sqrt(XLIM_B[0] * XLIM_B[1]), np.sqrt(YLIM_B[0] * YLIM_B[1]),
                      "fifth force", color=ps.GREY_DARK, fontsize=6.5 * S_FONT,
                      ha="center", va="top", zorder=Z_TEXT)
        drop = place_under_curve(
            txt, ax, (m_ff, s_ff),
            floors=[(p.mass, to_sigma(p.floor)) for p in pieces],
            keep_out=[texts["benchmark note"]], xlim=XLIM_B, ylim=YLIM_B)
        texts["fifth-force label"] = txt
        print(f"         name written {drop:.2f} dex under the bound at "
              f"m_DM = {txt.get_position()[0]:.3g} GeV")

    return dict(legend=None, labels=[], texts=texts, islands=islands,
                curves=curves, ref_curves=[], max_shift=shift,
                alpha_cell=cell)


# --------------------------------------------------------------------------- #
# Self-gating checks
# --------------------------------------------------------------------------- #
def _assert_clear_of_data(fig, ax, artist, art, name, edges=True, pad_pt=1.0):
    """``artist``'s rendered box must neither cross nor sit inside any island.

    ``edges`` also samples the drawn curves, which catches a box that clips a
    boundary without containing a vertex. Containment is tested against the
    filled polygons, which catches a box parked *within* a region.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    box = artist.get_window_extent(renderer=r).padded(pad_pt * fig.dpi / 144.0)
    probes = [(box.x0, box.y0), (box.x1, box.y0), (box.x0, box.y1),
              (box.x1, box.y1), ((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2)]
    for px, py in art["islands"]:
        poly = MplPath(ax.transData.transform(np.column_stack([px, py])))
        assert not any(poly.contains_point(pt) for pt in probes), \
            f"the {name} sits inside a filled island; move it"
    if not edges:
        return
    for line in art["curves"]:
        pts = line.axes.transData.transform(
            np.asarray(line.get_xydata(), dtype=float))
        pts = pts[np.isfinite(pts).all(axis=1)]
        hit = ((pts[:, 0] >= box.x0) & (pts[:, 0] <= box.x1)
               & (pts[:, 1] >= box.y0) & (pts[:, 1] <= box.y1))
        assert not hit.any(), (
            f"the {name} overlaps curve {line.get_label()!r} at "
            f"{int(hit.sum())} sampled points")


def _assert_clear_of_curves(fig, artist, curves, name, pad_pt=1.0):
    """``artist``'s rendered box must not touch any of ``curves``."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    box = artist.get_window_extent(renderer=r).padded(pad_pt * fig.dpi / 144.0)
    for line in curves:
        pts = line.axes.transData.transform(
            np.asarray(line.get_xydata(), dtype=float))
        pts = pts[np.isfinite(pts).all(axis=1)]
        hit = ((pts[:, 0] >= box.x0) & (pts[:, 0] <= box.x1)
               & (pts[:, 1] >= box.y0) & (pts[:, 1] <= box.y1))
        assert not hit.any(), (
            f"the {name} is written over curve {line.get_label()!r} at "
            f"{int(hit.sum())} sampled points")


def verify(fig, axes, arts, figsize=FIGSIZE, talk=False):
    """Renderer gates: exact size, nothing clipped or colliding, legend honest.

    ``figsize`` is the size the figure was drawn for -- :data:`FIGSIZE` in
    print, :data:`FIGSIZE_TALK` under --talk. Only the two gates below that are
    written against the *printed* geometry read it and ``talk``; every gate
    after them measures rendered artists and runs unchanged at either scale.
    """
    w_in, h_in = fig.get_size_inches()
    assert abs(w_in - figsize[0]) * 72.0 < 0.5, (
        f"figure is {w_in * 72:.1f} pt wide, not the {figsize[0] * 72:.1f} pt "
        f"it is drawn for (in print, main.tex places it unscaled)")
    assert w_in / h_in >= ps.ASPECT_3_2, \
        "aspect ratio must be 3:2 or wider (charter F14)"
    # Neither panel may be wider than the drawing width its type sizes were
    # chosen for: one journal column in print, half the slide under --talk.
    max_panel_cm = (figsize[0] / 2.0 if talk else ps.COLUMN_W_IN) * 2.54
    for ax in axes:
        w_panel_cm = ax.get_position().width * w_in * 2.54
        assert w_panel_cm <= max_panel_cm + 0.05, (
            f"a panel is {w_panel_cm:.2f} cm of drawing area, wider than the "
            f"{max_panel_cm:.2f} cm the type sizes are chosen for")

    for ax, art in zip(axes, arts):
        named = dict(art["texts"])
        named["x label"] = ax.xaxis.label
        named["y label"] = ax.yaxis.label
        ticks = {f"xtick {t.get_text()}": t
                 for t in ax.get_xticklabels() if t.get_text()}
        ticks.update({f"ytick {t.get_text()}": t
                      for t in ax.get_yticklabels() if t.get_text()})

        # Nothing may fall off the canvas: with no tight bbox that is silent
        # clipping.
        ps.assert_inside_figure(fig, {**named, **ticks})
        # Axis labels, tick labels and in-axes annotations must not collide.
        ps.assert_text_clearance(fig, {**named, **ticks}, min_gap_pt=0.5)
        # Every in-axes annotation must be legible, i.e. off the data.
        for name, txt in art["texts"].items():
            if name.startswith("island"):
                # Deliberately inside its own annulus, so the island polygons
                # cannot be a collider -- but it must still not be written over
                # somebody else's published limit.
                _assert_clear_of_curves(fig, txt, art["ref_curves"], name)
            else:
                _assert_clear_of_data(fig, ax, txt, art, name)

        if art["legend"] is not None:
            leg = art["legend"]
            # The legend lists exactly what was drawn, each key a real curve.
            ps.assert_legend_complete(leg, art["labels"], [ax])
            # ... and the legend must not be parked on the data.
            ps.assert_legend_clear_of(fig, leg, art["curves"], pad_pt=1.0)
            # assert_legend_clear_of samples the outlines; the legend could
            # still sit wholly *inside* a filled island without touching one.
            _assert_clear_of_data(fig, ax, leg, art, "legend", edges=False)

        # The cosmetic edge smoothing must stay inside the grid's resolution.
        assert art["max_shift"] < art["alpha_cell"], (
            f"edge smoothing moved a boundary by {art['max_shift']:.3f} dex, "
            f"more than one coupling-grid cell ({art['alpha_cell']:.3f} dex); "
            f"lower SMOOTH_SIGMA_CELLS")

        # Every plotted line must survive reproduction (charter F2: >= 0.5 pt).
        for line in art["curves"]:
            assert line.get_linewidth() >= 0.5, (
                f"curve {line.get_label()!r} is {line.get_linewidth()} pt, "
                f"under the 0.5 pt floor")

    # The two panels must be telling the same story about the benchmark: the
    # right panel is the left panel's own 10 meV island, at f_DM = 0.1 and
    # recast, so its cross-section floor must be the recast of a coupling this
    # search actually reaches.
    assert arts[1]["islands"], "the right panel drew no island"


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


def report_benchmark(rel, confidence, mode):
    """Print the numbers a reader of the caption would want to check."""
    alphas = rel.axes.alpha_n
    for lam_tag, _lab, _c, _d, _lw in LAMBDA_FAMILY:
        il = rel.at_lambda(lam_tag)
        lam_m = float(rel.axes.lambda_m[il])
        m_phi_ev = float(rel.axes.m_phi_gev[il]) * 1e9
        print(f"  {lam_tag:>9s}: lambda = {lam_m:.4g} m, "
              f"m_phi = {m_phi_ev * 1e3:.4g} meV")
    plane = rel.mass_plane("extremeness", mode=mode, lam=BENCH_LAM,
                           atmosphere=ATM_RIGHT, f_dm=F_DM_RIGHT)
    lo, _hi, _n, _h = excluded_band(plane, alphas, confidence)
    j = int(np.nanargmin(lo))
    inside = np.isfinite(lo)
    ms = rel.axes.mass_gev
    print(f"  mode-{mode} alpha_n floor {lo[j]:.6g} at "
          f"m_DM = {ms[j]:.6g} GeV "
          f"-> sigma_chi-n = {SIGMA_PER_ALPHA2 * lo[j] ** 2:.6g} cm^2")
    print(f"  mode-{mode} benchmark island spans m_DM = "
          f"{ms[inside].min():.6g}-{ms[inside].max():.6g} GeV "
          f"(sigma from {SIGMA_PER_ALPHA2 * np.nanmin(lo) ** 2:.6g} cm^2 up)")


#: How the released pair names its two hypotheses. The release splits the two
#: surfaces the panels need across two files -- f_DM = 1 attenuated (A) and
#: f_DM = 0.1 bare-halo (B) -- and NO cube carries f_DM = 0.1 with the
#: atmosphere, so the right panel simply cannot be read out of the left panel's
#: file. Deriving B from A keeps ``--release`` a single knob: point it at any
#: version's A cube and its own B cube is used for the benchmark panel.
HYPOTHESIS_TAG_LEFT = "_A_f1_atm"
HYPOTHESIS_TAG_RIGHT = "_B_f0p1_noatm"


def benchmark_release_path(path):
    """The bare-halo f_DM = 0.1 companion of an attenuated f_DM = 1 cube."""
    path = Path(path)
    name = path.name.replace(HYPOTHESIS_TAG_LEFT, HYPOTHESIS_TAG_RIGHT)
    if name == path.name:
        raise SystemExit(
            f"cannot tell which cube carries f_DM = {F_DM_RIGHT:g} without the "
            f"atmosphere: {path.name!r} does not name the hypothesis "
            f"{HYPOTHESIS_TAG_LEFT!r}, so its companion cannot be derived. "
            f"Pass --release-benchmark explicitly.")
    companion = path.with_name(name)
    if not companion.exists():
        raise SystemExit(
            f"the right panel needs the f_DM = {F_DM_RIGHT:g} bare-halo cube "
            f"{companion}, which does not exist; pass --release-benchmark")
    return companion


def assert_same_release(rel_left, rel_right):
    """The two panels must come from one release, or they are one figure of two.

    The pair is two files, so nothing on disk stops a stale B cube being drawn
    beside a fresh A cube. They must agree on the release version and on the
    axes both panels index, which is what makes "the right panel is the left
    panel's own 10 meV island, recast" a true statement.
    """
    left = str(rel_left.version_tag or "").split("-", 1)[0]
    right = str(rel_right.version_tag or "").split("-", 1)[0]
    if left != right:
        raise AssertionError(
            f"the two panels would come from different releases: "
            f"{rel_left.version_tag!r} (left) vs {rel_right.version_tag!r} "
            f"(right)")
    for name in ("mass_gev", "alpha_n", "lambda_m"):
        a = np.asarray(getattr(rel_left.axes, name), float)
        b = np.asarray(getattr(rel_right.axes, name), float)
        if a.shape != b.shape or not np.allclose(a, b, rtol=1e-12,
                                                 equal_nan=True):
            raise AssertionError(
                f"the two cubes disagree on the {name} axis, so the panels "
                f"are not two views of one scan")


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--release", type=Path, default=release.DEFAULT_PATH,
                   help="data-release HDF5 for the LEFT panel, f_DM = 1 with "
                        "the atmosphere (default: %(default)s)")
    p.add_argument("--release-benchmark", type=Path, default=None,
                   help="data-release HDF5 for the RIGHT panel, f_DM = 0.1 on "
                        "the bare halo (default: the companion of --release)")
    p.add_argument("--outdir", type=Path, default=None,
                   help="output directory (default: ignore/overleaf/figs, or "
                        "the talk asset tree under --talk)")
    p.add_argument("--stem", default="results",
                   help="output basename without extension "
                        "(--talk writes talk_<stem>)")
    p.add_argument("--talk", action="store_true",
                   help="slide variant of the same figure: ~15 pt type, "
                        f"heavier lines, {FIGSIZE_TALK[0]:.1f}x"
                        f"{FIGSIZE_TALK[1]:.1f} in, written as SVG+PNG to the "
                        "talk asset tree. Never writes to the paper's figs/.")
    p.add_argument("--mode", type=int, choices=(1, 2, 3), default=MODE,
                   help="sensor mode whose exclusion BOTH panels draw "
                        "(default: %(default)s). Give a distinct --stem per "
                        "mode so the variants do not overwrite each other.")
    p.add_argument("--refdir", type=Path,
                   default=Path(__file__).resolve().parents[1]
                   / "luhdm" / "reference_data",
                   help="digitised external-limit tables (default: %(default)s)")
    args = p.parse_args(argv)

    set_scale(args.talk)
    figsize = FIGSIZE_TALK if args.talk else FIGSIZE
    (ps.apply_talk_style if args.talk else ps.apply_prl_style)()
    bench_path = args.release_benchmark or benchmark_release_path(args.release)
    with release.open_release(args.release) as rel, \
            release.open_release(bench_path) as rel_bench:
        assert_same_release(rel, rel_bench)
        conf = float(rel.attrs.get("confidence_recommended", 0.95))
        conf_bench = float(rel_bench.attrs.get("confidence_recommended", conf))
        assert conf_bench == conf, (
            f"the two cubes recommend different confidence levels "
            f"({conf:g} left, {conf_bench:g} right)")
        print(f"release: {args.release}  ({rel.version_tag})")
        print(f"left:  mode {args.mode}, f_DM = {F_DM_LEFT:g}, atmosphere "
              f"{'on' if ATM_LEFT else 'off'}, confidence {conf:g}")
        print(f"release: {bench_path}  ({rel_bench.version_tag})")
        print(f"right: mode {args.mode}, f_DM = {F_DM_RIGHT:g}, atmosphere "
              f"{'on' if ATM_RIGHT else 'off'}, lambda = {BENCH_LAM}")

        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=figsize)
        art_a = build_left(ax_a, rel, args.refdir, conf, args.mode)
        art_b = build_right(ax_b, rel_bench, args.refdir, conf, args.mode)
        verify(fig, (ax_a, ax_b), (art_a, art_b), figsize=figsize,
               talk=args.talk)
        print(f"  legend (left): {art_a['labels']}")
        tag = ps.preliminary_tag_text(rel.version_tag)
        print(f"  preliminary tag: {tag or '(none: v3+ cube)'}")
        report_benchmark(rel_bench, conf, args.mode)

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
