"""Extract the published massless-mediator alpha_n limits of the levitated-sphere
searches, exactly, from the vector figures in their arXiv source packages.

Why this exists
---------------
The two searches we compare against in the money plot,

    Monteiro et al., Phys. Rev. Lett. 125, 181102 (2020)   [arXiv:2007.12067]
    Tseng   et al., arXiv:2508.00815 (2025)

have no HEPData record (hepdata.net is behind Cloudflare and neither paper
declares a data release; the Tseng text advertises only analysis *code*). Their
limit figures are, however, matplotlib-produced **vector** PDFs shipped inside
the arXiv source tarballs. This script therefore reads the curve vertices out of
the PDF content stream and maps them through the axis calibration recovered from
the figures' own tick marks. No pixel tracing and no eyeballing is involved: the
numbers written here are the numbers the authors' plotting code emitted.

Validation
----------
Tseng et al. redraw Monteiro et al.'s massless limit as a dashed curve in their
own figure. Extracting *both* independently -- Monteiro's from Monteiro's PDF,
Tseng's redraw from Tseng's PDF -- and comparing them is a closed-loop check on
the whole pipeline (stream parse + axis calibration on two unrelated figures).
They agree to <1% over the shared mass range; ``--check`` asserts this.

Caveats carried into the CSV header (and which the figure script must state)
---------------------------------------------------------------------------
* Both published limits assume a fractional abundance f_chi = 1, whereas this
  work quotes f_DM = 0.1. The curves are written **as published**; they are not
  rescaled, because the coupling dependence of the spectrum shape makes a naive
  f^-1/2 rescale wrong in detail and it is not ours to make.
* Each curve is clipped to the axis limits of its source figure, i.e. to the
  domain over which its authors chose to show it.
* Both are one-sided upper limits (exclusion above the curve), unlike the closed
  two-sided regions of this work.

Usage
-----
    python scripts/digitize_reference_limits.py            # fetch + write CSVs
    python scripts/digitize_reference_limits.py --check    # + cross-validate
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "luhdm" / "reference_data"
CACHE = REPO / "ignore" / "eprint_cache"

# --------------------------------------------------------------------------- #
# Source descriptions. `box` is the axes clip rectangle (x0, y0, x1, y1) in PDF
# points; `xcal`/`ycal` are (pt_of_decade_anchor, log10_value_there, pt_per_dec)
# recovered from the figure's own major tick marks by `--check`. `lim` is the
# source figure's axis window, used to clip the curve to its published domain.
# `path_index` selects the stroked curve in draw order (verified by colour).
# --------------------------------------------------------------------------- #
SOURCES = {
    "monteiro2020": dict(
        arxiv="2007.12067",
        figure="combined_limit_plot.pdf",
        path_index=105,
        colour=(0.5796078431, 0.7701960784, 0.8737254902),
        box=(48.077054, 36.488, 350.893658, 217.584),
        xcal=(48.077054, 2.0, 50.469434),
        ycal=(57.424057, -8.0, 40.039993),
        lim=(1e2, 1e8, 3e-9, 1e-4),
        label="Monteiro et al. (2020)",
        cite="Phys. Rev. Lett. 125, 181102 (2020); arXiv:2007.12067",
        legend="m_phi = 0 eV",
    ),
    "tseng2025": dict(
        arxiv="2508.00815",
        figure="figures/combined_alpha_limits.pdf",
        path_index=2,
        colour=(0.3294117647, 0.8901960784, 0.5294117647),
        box=(63.5690850586, 53.076, 410.448834668, 313.2),
        xcal=(63.5690850586, -1.0, 43.3599812),
        ycal=(73.8018, -7.0, 68.8495),
        lim=(1e-1, 1e7, 5e-8, 3e-4),
        label="Tseng et al. (2025)",
        cite="arXiv:2508.00815",
        legend="0 eV/c^2",
    ),
}

# Tseng's dashed redraw of the Monteiro massless limit -- validation only.
_TSENG_MONTEIRO_PATH = 6


# --------------------------------------------------------------------------- #
# A very small PDF content-stream reader (enough for matplotlib output).
# --------------------------------------------------------------------------- #
def _mul(a, b):
    a1, b1, c1, d1, e1, f1 = a
    a2, b2, c2, d2, e2, f2 = b
    return (a1 * a2 + b1 * c2, a1 * b2 + b1 * d2,
            c1 * a2 + d1 * c2, c1 * b2 + d1 * d2,
            e1 * a2 + f1 * c2 + e2, e1 * b2 + f1 * d2 + f2)


def _apply(m, x, y):
    a, b, c, d, e, f = m
    return (a * x + c * y + e, b * x + d * y + f)


_NUM = re.compile(r"-?\d*\.?\d+(e-?\d+)?")


def page_paths(qdf_bytes):
    """[(vertices Nx2, style, paint-op)] for the single page of an uncompressed PDF.

    Beziers are reduced to their end points; matplotlib emits polylines for line
    plots, so no curvature is lost for the curves we read.
    """
    obj = int(re.search(rb"/Contents (\d+) 0 R", qdf_bytes).group(1))
    body = re.search((r"\n%d 0 obj\n" % obj).encode() + rb"(.*?)stream\n(.*?)\nendstream",
                     qdf_bytes, re.S).group(2)
    toks = body.decode("latin-1").split()
    ctm = (1, 0, 0, 1, 0, 0)
    stack, cur, paths, pend = [], [], [], []
    gs = {"lw": 1.0, "SC": (0.0, 0.0, 0.0), "FC": (0.0, 0.0, 0.0)}
    for tok in toks:
        if _NUM.fullmatch(tok):
            pend.append(float(tok))
            continue
        if tok == "q":
            stack.append((ctm, dict(gs)))
        elif tok == "Q":
            if stack:
                ctm, saved = stack.pop()
                gs = dict(saved)
        elif tok == "cm":
            ctm = _mul(tuple(pend[-6:]), ctm)
        elif tok == "m":
            if cur:
                paths.append((np.array(cur), dict(gs), "?"))
            cur = [_apply(ctm, pend[-2], pend[-1])]
        elif tok in ("l", "c", "v", "y"):
            cur.append(_apply(ctm, pend[-2], pend[-1]))
        elif tok in ("S", "s", "f", "f*", "B", "B*", "b", "n"):
            if cur:
                paths.append((np.array(cur), dict(gs), tok))
                cur = []
        elif tok == "w":
            gs["lw"] = pend[-1]
        elif tok == "RG":
            gs["SC"] = tuple(pend[-3:])
        elif tok == "rg":
            gs["FC"] = tuple(pend[-3:])
        if tok not in ("q", "Q"):
            pend = []
    if cur:
        paths.append((np.array(cur), dict(gs), "?"))
    return paths


def major_ticks(paths, box, axis):
    """Major-tick positions (PDF pt) on the left/bottom spine of the axes box.

    Matplotlib draws ticks as two-point paths of ~3.5 pt anchored on the spine;
    minor ticks are ~2 pt. Returned sorted and de-duplicated.
    """
    x0, y0, _, _ = box
    hits = []
    for pts, _gs, _op in paths:
        if len(pts) != 2:
            continue
        (ax_, ay), (bx, by) = pts
        length = float(np.hypot(bx - ax_, by - ay))
        if not 3.0 < length < 4.5:
            continue
        if axis == "x" and abs(ay - y0) < 0.1 and abs(bx - ax_) < 1e-6:
            hits.append(ax_)
        elif axis == "y" and abs(ax_ - x0) < 0.1 and abs(by - ay) < 1e-6:
            hits.append(ay)
    return sorted({round(v, 6) for v in hits})


# --------------------------------------------------------------------------- #
# Fetch / unpack
# --------------------------------------------------------------------------- #
def fetch_figure(arxiv_id, member):
    """Path to `member` inside the cached arXiv source tarball for `arxiv_id`."""
    CACHE.mkdir(parents=True, exist_ok=True)
    tgz = CACHE / f"{arxiv_id}.tar.gz"
    if not tgz.exists():
        url = f"https://arxiv.org/e-print/{arxiv_id}"
        print(f"  fetching {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "luhdm-figure-build"})
        with urllib.request.urlopen(req, timeout=120) as r, open(tgz, "wb") as fh:
            fh.write(r.read())
    root = CACHE / arxiv_id
    if not (root / member).exists():
        with tarfile.open(tgz) as tf:
            tf.extractall(root)
    return root / member


def uncompressed(pdf_path):
    """qpdf --qdf bytes: object streams disabled, content streams inflated."""
    out = pdf_path.with_suffix(".qdf.pdf")
    if not out.exists():
        subprocess.run(["qpdf", "--qdf", "--object-streams=disable",
                        str(pdf_path), str(out)], check=True)
    return out.read_bytes()


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def _to_data(pts, xcal, ycal):
    px0, lx0, dx = xcal
    py0, ly0, dy = ycal
    return (10.0 ** (lx0 + (pts[:, 0] - px0) / dx),
            10.0 ** (ly0 + (pts[:, 1] - py0) / dy))


def extract(key, path_index=None, verify_calibration=True):
    """(mass_gev, alpha_n) for one source, clipped to its published axis window."""
    src = SOURCES[key]
    pdf = fetch_figure(src["arxiv"], src["figure"])
    paths = page_paths(uncompressed(pdf))

    if verify_calibration:
        for axis, cal in (("x", src["xcal"]), ("y", src["ycal"])):
            ticks = major_ticks(paths, src["box"], axis)
            steps = np.diff(ticks)
            assert len(ticks) >= 4, f"{key}: found only {len(ticks)} {axis} major ticks"
            assert np.allclose(steps, cal[2], rtol=2e-4), (
                f"{key}: {axis} tick spacing {steps} disagrees with the stored "
                f"{cal[2]} pt/decade -- the source figure changed, recalibrate.")
            assert np.any(np.isclose(ticks, cal[0], atol=1e-3)), (
                f"{key}: calibration anchor {cal[0]} pt is not a major {axis} tick")

    idx = src["path_index"] if path_index is None else path_index
    pts, gs, op = paths[idx]
    if path_index is None:
        assert np.allclose(gs["SC"], src["colour"], atol=2e-3), (
            f"{key}: path {idx} has stroke colour {gs['SC']}, expected "
            f"{src['colour']} -- draw order changed, re-identify the curve.")
        assert op == "S", f"{key}: path {idx} is not a stroked curve (op={op})"

    m, a = _to_data(pts, src["xcal"], src["ycal"])
    xlo, xhi, ylo, yhi = src["lim"]
    keep = (m >= xlo * (1 - 1e-9)) & (m <= xhi * (1 + 1e-9)) & (a >= ylo) & (a <= yhi)
    m, a = m[keep], a[keep]
    order = np.argsort(m)
    return m[order], a[order]


def write_csv(key, mass, alpha):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    src = SOURCES[key]
    out = OUT_DIR / f"{key}_alpha_n_massless.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"# 95% CL upper limit on alpha_n, massless mediator "
                    f"({src['legend']}), from {src['label']}"])
        w.writerow([f"# source: {src['cite']}"])
        w.writerow([f"# extracted from the vector figure {src['figure']} in the "
                    f"arXiv:{src['arxiv']} source package by "
                    f"scripts/digitize_reference_limits.py"])
        w.writerow(["# ASSUMES f_chi = 1 (this work uses f_DM = 0.1); NOT rescaled"])
        w.writerow(["# one-sided: the region ABOVE the curve is excluded"])
        w.writerow([f"# clipped to the source figure's axes: "
                    f"m in [{src['lim'][0]:g}, {src['lim'][1]:g}] GeV/c^2, "
                    f"alpha_n in [{src['lim'][2]:g}, {src['lim'][3]:g}]"])
        w.writerow(["mass_gev", "alpha_n"])
        for mm, aa in zip(mass, alpha):
            w.writerow([f"{mm:.6e}", f"{aa:.6e}"])
    print(f"  wrote {out}  ({len(mass)} points, "
          f"m {mass.min():.3g}-{mass.max():.3g} GeV)")
    return out


def cross_validate():
    """Monteiro's own massless curve vs Tseng's redraw of it. Independent PDFs."""
    m_own, a_own = extract("monteiro2020")
    pts, gs, _op = page_paths(
        uncompressed(fetch_figure(SOURCES["tseng2025"]["arxiv"],
                                  SOURCES["tseng2025"]["figure"])))[_TSENG_MONTEIRO_PATH]
    assert np.allclose(gs["SC"], SOURCES["tseng2025"]["colour"], atol=2e-3), (
        "Tseng path %d is not the green (massless) redraw" % _TSENG_MONTEIRO_PATH)
    m_re, a_re = _to_data(pts, SOURCES["tseng2025"]["xcal"], SOURCES["tseng2025"]["ycal"])
    order = np.argsort(m_re)
    m_re, a_re = m_re[order], a_re[order]

    lo = max(m_own.min(), m_re.min()) * 1.01
    hi = min(m_own.max(), m_re.max()) * 0.99
    probe = np.geomspace(lo, hi, 40)
    f_own = np.interp(np.log10(probe), np.log10(m_own), np.log10(a_own))
    f_re = np.interp(np.log10(probe), np.log10(m_re), np.log10(a_re))
    dex = np.abs(f_own - f_re)
    print(f"  cross-check over {lo:.3g}-{hi:.3g} GeV: "
          f"max |Dlog10 alpha_n| = {dex.max():.4f} dex "
          f"({100 * (10 ** dex.max() - 1):.2f}%)")
    assert dex.max() < 0.02, (
        "Monteiro's own massless curve and Tseng's redraw of it disagree by "
        f"{dex.max():.3f} dex; the extraction or a calibration is wrong.")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true",
                    help="also cross-validate Monteiro against Tseng's redraw")
    args = ap.parse_args(argv)

    for key in SOURCES:
        print(f"{key}:")
        mass, alpha = extract(key)
        assert len(mass) > 20, f"{key}: only {len(mass)} points survived clipping"
        assert np.all(np.diff(mass) > 0), f"{key}: mass axis is not monotonic"
        write_csv(key, mass, alpha)
    if args.check:
        print("cross-validation:")
        cross_validate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
