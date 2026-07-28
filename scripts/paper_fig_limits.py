"""PRL money plot: the mode-2 excluded regions in the (m_DM, alpha_n) plane.

What is drawn
-------------
For each mediator range in :data:`LAMBDA_FAMILY` we take the mode-2
optimum-interval extremeness plane out of the atmosphere-attenuated release cube
and reduce it, mass column by mass column, to the 95% CL excluded coupling
interval with ``luhdm.limits.excluded_band`` -- the same level-set helper the
release exposes as ``Release.excluded_alpha_band`` and that notebooks 01/04 use,
so this figure cannot drift from them.

Those intervals are two-sided, and the region they sweep out is a **closed
island**: the search has no sensitivity below the floor (too few expected
impulses above the 0.1 TeV threshold) and loses it again above the ceiling,
where the atmospheric overburden decelerates the incoming flux below threshold.
We publish the closed two-sided region, so the fill runs between floor and
ceiling and the outline is drawn all the way round -- upper edge included.
Building the band directly rather than contouring the plane is what makes the
ceiling explicit and yields exactly one closed polygon per range.
:func:`island_is_closed` asserts every island is strictly interior to the
scanned grid, i.e. that no edge is merely where the cube stops.

The islands are nested (longer range => larger island), so they are painted
largest-first and their translucent fills stack. Ranges are separated by colour
*and* dash pattern (charter F9/F10: no curve distinguished by hue alone); the
colours are an ordinal viridis ramp, the right encoding for an ordered variable
like lambda and monotone in grayscale value. The massless slice is the headline
result and gets the solid, heaviest edge.

The one cosmetic liberty is a capped moving mean along the mass axis
(:data:`SMOOTH_SIGMA_CELLS`): the coupling grid is 0.23 dex coarse and the
extremeness is nearly a step function across the ceiling, so the raw ceiling
climbs in one-cell risers that are grid resolution, not physics. No drawn point
may move more than half a coupling cell from the crossing the data gave, and
:func:`verify` asserts it.

Overlays
--------
The massless-mediator limits of the two optically levitated-sphere searches,
Monteiro et al. PRL 125, 181102 (2020) and Tseng et al. arXiv:2508.00815, are
read from CSVs written by ``scripts/digitize_reference_limits.py`` (exact curve
vertices lifted from the vector figures in the arXiv source packages; neither
paper has a HEPData record). Both assume f_chi = 1 where this work uses
f_DM = 0.1, and both are one-sided upper limits; the figure says so. If the CSVs
are absent the overlays are skipped with a log line and the figure still builds.

Usage
-----
    python scripts/paper_fig_limits.py
    python scripts/paper_fig_limits.py --release .../luhdm_datarelease_v3.h5

Re-running against a v3 release regenerates the figure with no edit, and the red
PRELIMINARY corner tag disappears on its own.
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

MODE = 2                 # the paper's channel
XLIM = (1e4, 3e19)       # GeV/c^2; the top is one partial decade past m_Pl
YLIM = (1e-10, 1.0)      # alpha_n: the full scanned coupling axis

#: Mediator ranges, longest first -- which is also largest-island first, so the
#: nested translucent fills stack from the outside in. Dash patterns are
#: explicit (on, off) point runs so they stay crisp at 8.6 cm.
LAMBDA_FAMILY = [
    # tag,       legend label,             colour,    dash,                      lw
    ("massless", "massless",               "#3B0F70", (0, ()),                   1.0),
    ("2mm",      r"$\lambda = 2$ mm",      "#3D5A8F", (0, (3.4, 1.3)),           0.75),
    ("200um",    r"$\lambda = 200\,\mu$m", "#218F8B", (0, (1.3, 1.1)),           0.75),
    ("20um",     r"$\lambda = 20\,\mu$m",  "#5DC863", (0, (3.2, 1.1, 0.8, 1.1)), 0.75),
    ("2um",      r"$\lambda = 2\,\mu$m",   "#9FD744", (0, (0.9, 1.1)),           0.75),
]
FILL_ALPHA = 0.17

#: Edge smoothing scale, in mass-grid cells (0.12 dex each). Rendering only.
SMOOTH_SIGMA_CELLS = 1.1

#: Prior massless-mediator limits. Neutral greys separated by dash pattern, so
#: they read as context rather than as our result and survive grayscale.
REFERENCES = [
    ("monteiro2020", "Monteiro et al.", ps.GREY_DARK,
     (0, (5.0, 1.7)), 0.75),
    ("tseng2025", "Tseng et al.", ps.GREY_MUTED,
     (0, (3.0, 1.2, 0.8, 1.2, 0.8, 1.2)), 0.75),
]

#: Wider than 3:2 -- aspect ratio is a length-budget decision (charter F14).
FIGSIZE = (ps.COLUMN_W_IN, ps.COLUMN_W_IN / 1.62)

# Z-order ladder. The Planck marker stays under the data.
Z_MARKER, Z_FILL, Z_EDGE, Z_REF, Z_TEXT = 1.0, 2.0, 3.0, 4.0, 6.0


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


def excluded_band(rel, tag, mode, confidence):
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
    plane = np.asarray(rel.mass_plane("extremeness", mode=mode, lam=tag), float)
    alphas = rel.axes.alpha_n
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


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def build(rel, ref_dir, confidence=0.95):
    ms = rel.axes.mass_gev
    alphas = rel.axes.alpha_n
    m_planck = float(rel.attrs.get("m_planck_gev", 1.22e19))
    cell = float(np.diff(np.log10(alphas)).mean())   # coupling grid step, dex

    fig, ax = plt.subplots(figsize=FIGSIZE)

    labels, islands, curves, shifts = [], [], [], []
    for tag, label, colour, dash, lw in LAMBDA_FAMILY:
        lo, hi, n_nan, holes = excluded_band(rel, tag, MODE, confidence)
        inside = np.isfinite(lo)
        if not inside.any():
            print(f"  [skip] lambda={tag}: no mass column reaches "
                  f"p >= {confidence:g} in mode {MODE}; nothing drawn and no "
                  f"legend entry (a legend key with no curve is a figure bug)")
            continue
        print(f"  [ok]   lambda={tag}: {int(inside.sum())} mass columns"
              + (f", {n_nan} NaN cells read as not-excluded" if n_nan else "")
              + (f", {holes} column(s) with a one-cell hole spanned" if holes else ""))

        for k, run in enumerate(contiguous_runs(inside)):
            m_run = ms[run]
            lo_raw, hi_raw = np.log10(lo[run]), np.log10(hi[run])
            lo_s = smooth_log(lo_raw, SMOOTH_SIGMA_CELLS, 0.5 * cell)
            hi_s = smooth_log(hi_raw, SMOOTH_SIGMA_CELLS, 0.5 * cell)
            shifts.append(max(np.abs(lo_s - lo_raw).max(),
                              np.abs(hi_s - hi_raw).max()))
            floor, ceiling = 10.0 ** lo_s, 10.0 ** hi_s
            if not island_is_closed(m_run, floor, ceiling, ms, alphas):
                raise AssertionError(
                    f"lambda={tag}: the {confidence:.0%} band reaches the edge "
                    f"of the scanned grid, so it is not a closed island; the "
                    f"two-sided region cannot be drawn honestly without "
                    f"extending the cube")
            # Closed polygon: floor left to right, then ceiling right to left.
            px = np.concatenate([m_run, m_run[::-1], m_run[:1]])
            py = np.concatenate([floor, ceiling[::-1], floor[:1]])
            ax.fill(px, py, color=colour, alpha=FILL_ALPHA, lw=0, zorder=Z_FILL)
            # Only the first piece carries the label, so the legend keeps one
            # key per mediator range however many pieces the island comes in.
            line, = ax.plot(px, py, color=colour, lw=lw, ls=dash, zorder=Z_EDGE,
                            solid_joinstyle="round", dash_capstyle="butt",
                            label=label if k == 0 else "_nolegend_")
            islands.append((px, py))
            curves.append(line)
        labels.append(label)

    max_shift = max(shifts) if shifts else 0.0
    print(f"  smoothing: max edge shift {max_shift:.3f} dex "
          f"(coupling grid cell {cell:.3f} dex)")

    # -- prior massless-mediator limits ------------------------------------- #
    for key, label, colour, dash, lw in REFERENCES:
        ref = load_reference(key, ref_dir)
        if ref is None:
            # Overlay stub: drop a CSV of this name into the reference-data
            # directory (scripts/digitize_reference_limits.py writes them) and
            # it is picked up automatically, with no edit to this file. Further
            # comparisons: append to REFERENCES with a matching CSV.
            print(f"  [skip] overlay {key}: "
                  f"{Path(ref_dir) / (key + '_alpha_n_massless.csv')} not found; "
                  f"run scripts/digitize_reference_limits.py")
            continue
        m_ref, a_ref = ref
        window = ((m_ref >= XLIM[0]) & (m_ref <= XLIM[1])
                  & (a_ref >= YLIM[0]) & (a_ref <= YLIM[1]))
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
            first = False
        if first:
            continue
        labels.append(label)
        print(f"  [ok]   overlay {key}: {int(window.sum())} points in window, "
              f"m {m_ref[window].min():.3g}-{m_ref[window].max():.3g} GeV")

    # -- axes --------------------------------------------------------------- #
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xlabel(r"Dark matter mass, $m_{\mathrm{DM}}$ (GeV/$c^2$)")
    ax.set_ylabel(r"Coupling per neutron, $\alpha_n$")
    # Label decades, but not every decade: 15 x-decades and 10 y-decades cannot
    # carry 25 legible labels at 7 pt (charter F13 against F2).
    ax.xaxis.set_major_locator(FixedLocator([1e4, 1e7, 1e10, 1e13, 1e16, 1e19]))
    ax.yaxis.set_major_locator(FixedLocator([1e-9, 1e-7, 1e-5, 1e-3, 1e-1]))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=(1.0,), numticks=40))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(1.0,), numticks=20))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_minor_formatter(NullFormatter())

    # -- Planck-mass endpoint marker (repo house style, scaled to print) ----- #
    # The mass grid stops at m_Pl. No halo stroke: path effects are unusable
    # under usetex (see paper_style.halo), and none is needed because the label
    # sits in the empty right third of the plane, clear of every island.
    ax.axvline(m_planck, color=ps.GREY_GUIDE, lw=0.6, ls=":", zorder=Z_MARKER)
    planck_txt = ax.text(m_planck, 0.820, r"$m_{\mathrm{Pl}}$",
                         color=ps.GREY_GUIDE, fontsize=6.5, ha="right", va="top",
                         rotation=90, zorder=Z_MARKER,
                         transform=ax.get_xaxis_transform())

    # -- annotations -------------------------------------------------------- #
    note = ax.text(0.028, 0.955,
                   "$f_{\\mathrm{DM}} = 0.1$, 95\\% CL\n"
                   "prior limits: $f_\\chi = 1$",
                   transform=ax.transAxes, ha="left", va="top", fontsize=7.0,
                   linespacing=1.4, zorder=Z_TEXT)

    # Set over two lines: on one line the stamp is wide enough to reach back
    # over the top of the massless island, and `verify` rejects that.
    tag = ps.add_preliminary_tag(
        ax, ps.preliminary_tag_text(rel.version_tag).replace(" (", "\n("),
        xy=(0.962, 0.975))

    # Lower right: the islands stop near m ~ 5e13 and their floor climbs
    # steeply, so this is the largest empty block in the frame.
    leg = ax.legend(loc="lower right", bbox_to_anchor=(0.968, 0.02))
    leg.set_zorder(Z_TEXT)

    return fig, ax, dict(legend=leg, note=note, tag=tag, planck=planck_txt,
                         labels=labels, islands=islands, curves=curves,
                         max_shift=max_shift, alpha_cell=cell)


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


def verify(fig, ax, art):
    """Renderer gates: exact size, nothing clipped or colliding, legend honest."""
    leg = art["legend"]

    w_cm = fig.get_size_inches()[0] * 2.54
    assert abs(w_cm - ps.COLUMN_W_IN * 2.54) < 0.02, \
        f"figure is {w_cm:.2f} cm wide; it must be drawn at the column width"
    assert fig.get_size_inches()[0] / fig.get_size_inches()[1] >= ps.ASPECT_3_2, \
        "aspect ratio must be 3:2 or wider (charter F14)"

    named = {"x label": ax.xaxis.label, "y label": ax.yaxis.label,
             "note": art["note"], "m_Pl label": art["planck"],
             "PRELIMINARY tag": art["tag"]}
    ticks = {f"xtick {t.get_text()}": t
             for t in ax.get_xticklabels() if t.get_text()}
    ticks.update({f"ytick {t.get_text()}": t
                  for t in ax.get_yticklabels() if t.get_text()})

    # Nothing may fall off the canvas: with no tight bbox that is silent clipping.
    ps.assert_inside_figure(fig, {**named, **ticks})
    # Axis labels, tick labels and in-axes annotations must not collide.
    ps.assert_text_clearance(fig, {**named, **ticks}, min_gap_pt=0.5)
    # The legend lists exactly what was drawn, and every key maps to a real curve.
    ps.assert_legend_complete(leg, art["labels"], [ax])
    # ... and the legend must not be parked on the data.
    ps.assert_legend_clear_of(fig, leg, art["curves"], pad_pt=1.0)

    # The red stamp must not sit on a result either.
    if art["tag"] is not None:
        _assert_clear_of_data(fig, ax, art["tag"], art, "PRELIMINARY tag")
    # assert_legend_clear_of samples the outlines; the legend could still sit
    # wholly *inside* a filled island without touching one, so also test
    # containment (the island polygons, not just their edges).
    _assert_clear_of_data(fig, ax, leg, art, "legend", edges=False)

    # The cosmetic edge smoothing must stay inside the grid's own resolution.
    assert art["max_shift"] < art["alpha_cell"], (
        f"edge smoothing moved a boundary by {art['max_shift']:.3f} dex, more "
        f"than one coupling-grid cell ({art['alpha_cell']:.3f} dex); lower "
        f"SMOOTH_SIGMA_CELLS")

    # Every plotted line must survive reproduction (charter F2: >= 0.5 pt).
    for line in art["curves"]:
        assert line.get_linewidth() >= 0.5, (
            f"curve {line.get_label()!r} is {line.get_linewidth()} pt, "
            f"under the 0.5 pt floor")


# --------------------------------------------------------------------------- #
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--release", type=Path, default=release.DEFAULT_PATH,
                   help="data-release HDF5 (default: %(default)s)")
    p.add_argument("--outdir", type=Path,
                   default=Path(__file__).resolve().parents[1]
                   / "ignore" / "overleaf" / "figs",
                   help="output directory (default: %(default)s)")
    p.add_argument("--stem", default="limits",
                   help="output basename without extension")
    p.add_argument("--refdir", type=Path,
                   default=Path(__file__).resolve().parents[1]
                   / "luhdm" / "reference_data",
                   help="digitised prior-limit CSVs (default: %(default)s)")
    args = p.parse_args(argv)

    ps.apply_prl_style()
    with release.open_release(args.release) as rel:
        conf = float(rel.attrs.get("confidence_recommended", 0.95))
        print(f"release: {args.release}  ({rel.version_tag}), mode {MODE}, "
              f"f_DM = {rel.attrs.get('f_x')}, confidence {conf:g}")
        fig, ax, art = build(rel, args.refdir, conf)
        verify(fig, ax, art)
        print(f"  legend: {art['labels']}")
        print("  tag:    "
              f"{ps.preliminary_tag_text(rel.version_tag) or '(none: v3 cube)'}")

    pdf = args.outdir / f"{args.stem}.pdf"
    png = args.outdir / f"{args.stem}.png"
    ps.savefig_exact(fig, pdf, png)
    plt.close(fig)
    print(ps.report_size(fig, pdf))
    print(f"preview: {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
