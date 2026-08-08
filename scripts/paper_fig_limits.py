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

Those intervals are two-sided, and the region they sweep out is a **closed
island**: the search has no sensitivity below the floor (too few expected
impulses above the 0.1 TeV threshold), loses it again above the ceiling, where
the atmospheric overburden decelerates the incoming flux below threshold, and
is closed on the right by the 10 cm impact-parameter cap. We publish the closed
region, so the fill runs between floor and ceiling and the outline is drawn all
the way round. Building the band directly rather than contouring the plane is
what makes the ceiling explicit and yields exactly one closed polygon per
range. :func:`island_is_closed` asserts every island is strictly interior to
the scanned grid, i.e. that no edge is merely where the cube stops.

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
    python scripts/paper_fig_limits.py --release .../luhdm_datarelease_v5.h5

Re-running against a newer release regenerates the figure with no edit, and the
red PRELIMINARY corner tag stays absent from v3 on (see
``paper_style.preliminary_tag_text``).
"""
from __future__ import annotations

import argparse
import csv
import sys
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


def island_is_closed(mass, floor, ceiling, mass_axis, alpha_axis):
    """True if the drawn band is strictly inside the scanned grid.

    A band that reaches the edge of the cube is not a measured boundary, it is
    where we stopped computing, and the closed two-sided claim would be unsafe.
    """
    return bool(mass.min() > mass_axis[0] and mass.max() < mass_axis[-1]
                and floor.min() > alpha_axis[0] and ceiling.max() < alpha_axis[-1])


def island_polygons(rel, plane, confidence, cell, label):
    """``[(mass, floor, ceiling), ...]`` -- the smoothed, closed island pieces."""
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
        hi_raw = np.log10(hi[run])
        hi_s = smooth_log(hi_raw, SMOOTH_SIGMA_CELLS, 0.5 * cell)
        shift = max(shift, float(np.abs(hi_s - hi_raw).max()))
        floor, ceiling = lo[run], 10.0 ** hi_s
        if not island_is_closed(m_run, floor, ceiling, ms, alphas):
            raise AssertionError(
                f"{label}: the {confidence:.0%} band reaches the edge of the "
                f"scanned grid, so it is not a closed island; the two-sided "
                f"region cannot be drawn honestly without extending the cube")
        out.append((m_run, floor, ceiling))
    return out, shift


def as_polygon(m_run, floor, ceiling, transform=None):
    """Close a (floor, ceiling) band into one polygon: floor out, ceiling back.

    ``transform`` maps the coupling ordinate to whatever the panel plots; the
    right panel passes the cross-section recast. It must be monotone, which the
    recast is (a positive multiple of alpha_n^2), or the band's two edges would
    cross.
    """
    px = np.concatenate([m_run, m_run[::-1], m_run[:1]])
    py = np.concatenate([floor, ceiling[::-1], floor[:1]])
    return px, (py if transform is None else transform(py))


def draw_island(ax, pieces, colour, dash, lw, label, transform=None,
                fill_alpha=FILL_ALPHA):
    """Fill + outline each closed piece; returns the drawn Line2D artists."""
    lines = []
    for k, (m_run, floor, ceiling) in enumerate(pieces):
        px, py = as_polygon(m_run, floor, ceiling, transform)
        ax.fill(px, py, color=colour, alpha=fill_alpha, lw=0, zorder=Z_FILL)
        line, = ax.plot(px, py, color=colour, lw=lw, ls=dash, zorder=Z_EDGE,
                        solid_joinstyle="round", dash_capstyle="butt",
                        label=label if k == 0 else "_nolegend_")
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
#: about 9 pt at this figure's scale, against a 6.5 pt label.
LABEL_REF_CLEAR_DEX = 0.7

#: Half the width of a rendered island name, in decades of mass at this panel's
#: scale. Clearance from a prior-limit curve is tested right across that span,
#: not just under the anchor: the overlays climb steeply and a curve that clears
#: the centre of a name can still run through its right-hand end.
LABEL_HALF_WIDTH_DEX = 0.45

#: Clearance, as a fraction of the panel diagonal, that an island's name must
#: keep from ANY island outline, its own included. The nested islands converge
#: on the same low-mass edge, so without this a name lands on a boundary even
#: when it is nowhere near that island's interior.
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
    for m_run, floor, ceiling in pieces:
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
            probe = np.logspace(np.log10(m) - LABEL_HALF_WIDTH_DEX,
                                np.log10(m) + LABEL_HALF_WIDTH_DEX, 7)
            near = []
            for mr, ar in refs:
                inside = (probe >= mr.min()) & (probe <= mr.max())
                if inside.any():
                    near.extend(10.0 ** np.interp(
                        np.log10(probe[inside]), np.log10(mr), np.log10(ar)))
            for frac in LABEL_HEIGHT_FRACS:
                y = 10.0 ** (np.log10(floor[j])
                             + frac * np.log10(ceiling[j] / floor[j]))
                if any(abs(np.log10(y / a)) < LABEL_REF_CLEAR_DEX for a in near):
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
                               atmosphere=True, f_dm=F_DM_LEFT)
        pieces, shift = island_polygons(
            rel, plane, confidence, cell,
            f"lambda={tag} ({label.replace('$', '')})")
        if not pieces:
            continue
        shifts.append(shift)
        # No legend key: each island is named in place below, and a six-entry
        # legend does not fit in what the nested islands leave empty.
        curves += draw_island(ax, pieces, colour, dash, lw, "_nolegend_")
        islands += [as_polygon(m, f, c) for m, f, c in pieces]
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
            line, = ax.plot(m_ref[run], a_ref[run], color=colour, lw=lw, ls=dash,
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
            else max(float(m.max()) for m, _f, _c in inner)
        # Every island outline, this one's included: a name written along its
        # own boundary is as hard to read as one written along a neighbour's.
        others = [np.column_stack(as_polygon(m, f, c))
                  for _t, _l, _c2, pcs in drawn for m, f, c in pcs]
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
            anchor[0], anchor[1], label, color=colour, fontsize=6.5,
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
    # The mass grid stops at m_Pl. No halo stroke: path effects are unusable
    # under usetex (see paper_style.halo), and none is needed because the label
    # sits in the empty right edge, clear of every island.
    ax.axvline(m_planck, color=ps.GREY_GUIDE, lw=0.6, ls=":", zorder=Z_MARKER)
    texts["m_Pl label"] = ax.text(
        m_planck, 0.975, r"$m_{\mathrm{Pl}}$", color=ps.GREY_GUIDE, fontsize=6.5,
        ha="right", va="top", rotation=90, zorder=Z_MARKER,
        transform=ax.get_xaxis_transform())

    texts["hypothesis note"] = ax.text(
        0.028, 0.955, "$f_{\\mathrm{DM}} = 1$, 95\\% CL",
        transform=ax.transAxes, ha="left", va="top", fontsize=7.0,
        zorder=Z_TEXT)

    # Empty from v3 on, so nothing is drawn for the released cube; kept so the
    # stamp reappears by itself if the figure is ever rebuilt from an older one.
    tag = ps.add_preliminary_tag(
        ax, ps.preliminary_tag_text(rel.version_tag).replace(" (", "\n("),
        xy=(0.972, 0.975))
    if tag is not None:
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
                           atmosphere=True, f_dm=F_DM_RIGHT)
    pieces, shift = island_polygons(
        rel, plane, confidence, cell,
        f"mode {mode} lambda={BENCH_LAM}, f_DM={F_DM_RIGHT}")
    if not pieces:
        raise AssertionError(
            f"mode {mode} excludes nothing at the benchmark lambda={BENCH_LAM}; "
            f"there is no right panel to draw")
    curves = draw_island(ax, pieces, colour, (0, ()), 1.0, "_nolegend_",
                         transform=to_sigma, fill_alpha=FILL_ALPHA_SOLO)
    islands = [as_polygon(m, f, c, to_sigma) for m, f, c in pieces]
    sig_floor, m_at_floor = min(
        ((float(to_sigma(f[j])), float(m[j]))
         for m, f, _c in pieces for j in [int(np.argmin(f))]))
    print(f"  recast: sigma = {SIGMA_PER_ALPHA2:.6e} cm^2 x alpha_n^2 "
          f"(q0 = mu_chi-n v0, mu = {MU_CHI_N_GEV} GeV, v0/c = {V0_OVER_C})")
    print(f"  deepest cross-section limit {sig_floor:.4g} cm^2 "
          f"at m_DM = {m_at_floor:.4g} GeV")

    # -- fifth-force bound, recast the same way ------------------------------ #
    texts, ff_line = {}, None
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
        ff_line, = ax.plot(m_ff, s_ff, color=ps.GREY_DARK, lw=0.75,
                           ls=(0, (4.0, 1.6)), zorder=Z_REF,
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
        # to be told apart from the island outline by dash pattern alone. The
        # only part of this panel the island leaves free is the strip beyond
        # its high-mass tip, so the name goes there, just above the line.
        # Right-aligned at the frame edge and lifted clear of the line: the
        # bound rises as m_DM^2, so a centred label would be overrun by its own
        # curve within half a label width.
        m_lab = XLIM_B[1] * 0.97
        _a, s_lab = fifth_force_sigma(m_lab, alpha_tilde)
        texts["fifth-force label"] = ax.text(
            m_lab, s_lab * 3.0, "fifth force", color=ps.GREY_DARK,
            fontsize=6.5, ha="right", va="bottom", zorder=Z_TEXT)

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

    texts["benchmark note"] = ax.text(
        0.972, 0.045,
        "composite DM\n"
        f"$m_\\phi = 10$ meV\n"
        f"$m_d = 1$ keV\n"
        f"$f_{{\\mathrm{{DM}}}} = {F_DM_RIGHT:g}$",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5,
        linespacing=1.35, zorder=Z_TEXT)

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


def verify(fig, axes, arts):
    """Renderer gates: exact size, nothing clipped or colliding, legend honest."""
    w_in, h_in = fig.get_size_inches()
    assert abs(w_in * 72.0 - FIGSIZE[0] * 72.0) < 0.5, \
        f"figure is {w_in * 72:.1f} pt wide; main.tex places it unscaled"
    assert w_in / h_in >= ps.ASPECT_3_2, \
        "aspect ratio must be 3:2 or wider (charter F14)"
    for ax in axes:
        w_panel_cm = ax.get_position().width * w_in * 2.54
        assert w_panel_cm <= ps.COLUMN_W_IN * 2.54 + 0.05, (
            f"a panel is {w_panel_cm:.2f} cm of drawing area, wider than the "
            f"single-column width the type sizes are chosen for")

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
                           atmosphere=True, f_dm=F_DM_RIGHT)
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


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--release", type=Path, default=release.DEFAULT_PATH,
                   help="data-release HDF5 (default: %(default)s)")
    p.add_argument("--outdir", type=Path,
                   default=Path(__file__).resolve().parents[1]
                   / "ignore" / "overleaf" / "figs",
                   help="output directory (default: %(default)s)")
    p.add_argument("--stem", default="results",
                   help="output basename without extension")
    p.add_argument("--mode", type=int, choices=(1, 2, 3), default=MODE,
                   help="sensor mode whose exclusion BOTH panels draw "
                        "(default: %(default)s). Give a distinct --stem per "
                        "mode so the variants do not overwrite each other.")
    p.add_argument("--refdir", type=Path,
                   default=Path(__file__).resolve().parents[1]
                   / "luhdm" / "reference_data",
                   help="digitised external-limit tables (default: %(default)s)")
    args = p.parse_args(argv)

    ps.apply_prl_style()
    with release.open_release(args.release) as rel:
        conf = float(rel.attrs.get("confidence_recommended", 0.95))
        print(f"release: {args.release}  ({rel.version_tag})")
        print(f"left:  mode {args.mode}, f_DM = {F_DM_LEFT:g}, atmosphere on, "
              f"confidence {conf:g}")
        print(f"right: mode {args.mode}, f_DM = {F_DM_RIGHT:g}, "
              f"atmosphere on, lambda = {BENCH_LAM}")

        fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=FIGSIZE)
        art_a = build_left(ax_a, rel, args.refdir, conf, args.mode)
        art_b = build_right(ax_b, rel, args.refdir, conf, args.mode)
        verify(fig, (ax_a, ax_b), (art_a, art_b))
        print(f"  legend (left): {art_a['labels']}")
        tag = ps.preliminary_tag_text(rel.version_tag)
        print(f"  preliminary tag: {tag or '(none: v3+ cube)'}")
        report_benchmark(rel, conf, args.mode)

    pdf = args.outdir / f"{args.stem}.pdf"
    png = args.outdir / f"{args.stem}.png"
    ps.savefig_exact(fig, pdf, png)
    plt.close(fig)
    print(ps.report_size(fig, pdf))
    print(f"preview: {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
