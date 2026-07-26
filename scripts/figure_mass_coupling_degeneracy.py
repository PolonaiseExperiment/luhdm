"""Two-panel figure: the alpha^2/m degeneracy of the massless (1/r^2) mediator.

Why the v1 massless exclusion island ran unbounded in DM mass, and why the 10 cm
impact-parameter cap ends it.

For a massless mediator the projected cross section is Rutherford,
dsigma/dq = 2 pi alpha^2 / (v^2 q^3), and the halo number density scales as
1/m, so the detected spectrum carries the coupling and the mass only through the
combination alpha^2/m:

    dmu/dq  =  T * eps(q) * f_X * (rho/m) * <v * 2 pi alpha^2/(v^2 q^3)>
            propto (alpha^2/m) q^-3 eps(q).

Walking along alpha propto sqrt(m) therefore leaves the detector-level spectrum
-- and hence mu and the optimum-interval extremeness -- exactly unchanged. With
nothing in the analysis to break the degeneracy, the massless exclusion island
extended to arbitrarily large mass (panel A).

The geometry is *not* invariant. The massless flyby reach is
b_max(q) = 2 alpha/(q v), so along the same line the impact parameters that
deliver a given kick grow as sqrt(m): the same mu is assembled from flybys at
ever-larger distance (panel B). Once the impact-parameter integral is capped at
the production value b_constrained_max = 10 cm, the heavy/strong end of the line
loses essentially all of its rate, and the island closes.

The three anchor pairs are chosen so that the whole story is told inside the
laboratory-relevant range of b: at m = 1e10 GeV the flybys that carry mu sit at
b ~ 1 mm, comfortably outside the sensor radius R_eff = 200 um (the kept domain
of figure 07) and comfortably inside the 10 cm cap; at m = 1e14 GeV they sit at
b ~ 12 cm and the cap bites; at m = 1e18 GeV they sit at b ~ 12 m and the cap
removes all but ~1e-4 of mu.

Pipeline: bare-halo SHM (no atmosphere -- attenuation would break the degeneracy
at high alpha and is beside the geometric point), massless mediator, mode-1
measured efficiency, exposure config.T_EXPOSURE. Every number is produced by the
same luhdm.rate / luhdm.limits calls the data-release builder uses
(scripts/build_release.py, noatm massless slice), and is cross-checked against
the uncapped v1 cube at the exact grid points nearest the three anchor pairs
(skipped with a note when the release file is not present -- it is gitignored
and distributed via Zenodo).

One deliberate departure from the production settings: panel B *differentiates*
mu(<b) in log10 b, which divides the quadrature error of two neighbouring b
nodes by the node spacing (0.025 decades) and so amplifies it by ~40x. On the
production 240-point q grid that noise is large enough to put a visible
shoulder on the heaviest hump, so the cap scan (and only the cap scan) runs on
a refined q grid, N_Q_PANEL_B points over the identical q span. The cube
cross-check below stays on the exact production grid the cube was built with.
"""

import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from optimum_interval import scanning

from luhdm import config, efficiency, halo, rate

# ---------------------------------------------------------------------------
# Release-pipeline constants -- the scripts/build_release.py settings that this
# figure actually exercises (only the panel-B q grid departs, see below)
# ---------------------------------------------------------------------------
T_TOTAL = config.T_EXPOSURE            # 1_691_020.0 s
Q_HI_REF = 8.4e3                       # fixed qs upper-momentum reference
DF = 3                                 # efficiency dof hypothesis
# Only the q-grid entries of build_release.py's FID_PROD are live in this
# figure: the halo here is the analytic SHM, so the attenuation ODE (n_ode),
# the sampled halo (n_shm) and the optimum-interval Monte Carlo (n_mc) are
# never called. They are left out rather than carried as inert settings.
FID = dict(n_q=240, q_span=3e4)        # FID_PROD q grid
# Panel B's cap scan runs on a refined q grid (see the module docstring): the
# d/dlog10 b derivative amplifies q-quadrature noise by ~40x. Beyond ~1700
# points the residual x100-translate deviation stops falling -- it is then set
# by the fixed 500-point, m-dependent v grid inside differential_rate_trapz --
# so 8x the production grid buys all the smoothness that is available.
N_Q_PANEL_B = 8 * FID["n_q"]           # 1920
Q_MIN = 100.0                          # analysis threshold [GeV] = config.Q_THRESH
MODE = 1                               # measured efficiency mode

B_CAP = 0.1                            # b_constrained_max [m] = 10 cm (production)
R_EFF = config.R_EFF                   # sensor radius [m] (200 um)

MU_TARGET = 3.0                        # <N> = 3 zero-event benchmark (notebooks 01/04)

MASSES = np.array([1e10, 1e14, 1e18])  # degeneracy-line masses [GeV]

# plotted windows
Q_LO, Q_HI = 1e2, 3e4                  # panel A momentum window [GeV]
B_LO, B_HI = 2e-5, 1e3                 # panel B impact-parameter window [m]

# b scan (wider than the plot so the plateau is resolved), points per decade
B_SCAN_LO, B_SCAN_HI, B_PER_DEC = 1e-8, 1e6, 40

# panel-B curves are drawn only where they carry a meaningful share of mu.
# Below this the stroke has no resolvable height but still paints a solid line
# along y ~ 0, which misreads as a nonzero floor (the deepest tails of the three
# humps would otherwise tile the whole panel). The absolute floor is ~3 px at
# 140 dpi; together the two cuts keep +-1.2 decades around each peak, i.e.
# >99.8% of every hump's area.
B_FLOOR_FRAC = 3e-3
B_FLOOR_ABS = 0.05                     # [mu per decade], i.e. ~1% of the y axis

CUBE_V1 = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "release", "luhdm_datarelease_v1.h5")

# ---------------------------------------------------------------------------
# Pipeline: identical construction to build_release.py's noatm massless cell
# ---------------------------------------------------------------------------
_EFF_F = efficiency.make_efficiency(MODE, DF)

QS = np.geomspace(Q_MIN, FID["q_span"] * Q_HI_REF, FID["n_q"])
EFF_QS = _EFF_F(QS)

# same span, finer sampling -- panel B only (GATE 1 and panel A stay on QS)
QS_B = np.geomspace(Q_MIN, FID["q_span"] * Q_HI_REF, N_Q_PANEL_B)
EFF_QS_B = _EFF_F(QS_B)

# The noatm pass feeds the bare SHM straight into the rate (build_release.py:
# NO_ATM -> f_v_f = halo.standard_halo_model). The builder's seeded SHM sample
# only ever drives the attenuation ODE on the atm pass, so nothing random enters
# here and no seed is needed.
F_V_F = halo.standard_halo_model

XS_UNCAPPED = rate.make_xsec(None)      # massless, uncapped


def mu_of(alpha_n, m, xs, qs=QS, eff_qs=EFF_QS):
    """Expected detected counts mu, exactly as the cube stores it.

    ``rate.differential_rate_trapz(..., eff=None)`` then the mode efficiency
    post-multiply and ``spectrum_from_rate`` -- i.e. the mu returned by
    limits.extremeness_and_mu -> optimum_interval.scanning.scan_extremeness,
    without paying for the Monte-Carlo calibration we do not need here.

    ``qs``/``eff_qs`` default to the production q grid (what the cube used);
    panel B passes the refined pair QS_B/EFF_QS_B.
    """
    raw = rate.differential_rate_trapz(qs, alpha_n, m, F_V_F, xs, eff=None)
    spec = scanning.spectrum_from_rate(qs, raw * eff_qs, T_TOTAL)
    return 0.0 if spec is None else float(spec[0])


def dmu_dq(alpha_n, m, xs):
    """Detected differential counts dmu/dq [GeV^-1] on QS."""
    raw = rate.differential_rate_trapz(QS, alpha_n, m, F_V_F, xs, eff=None)
    return raw * EFF_QS * T_TOTAL


# ---------------------------------------------------------------------------
# Anchor the degeneracy line at mu = 3 for the lightest pair.
# mu propto alpha^2 exactly for the bare halo, so one trial call suffices.
# ---------------------------------------------------------------------------
_A_TRIAL = 1e-9
_MU_TRIAL = mu_of(_A_TRIAL, MASSES[0], XS_UNCAPPED)
ALPHA0_N = _A_TRIAL * np.sqrt(MU_TARGET / _MU_TRIAL)

# alpha propto sqrt(m): masses step by 1e4, so alpha_n steps by exactly 100
ALPHAS_N = ALPHA0_N * np.sqrt(MASSES / MASSES[0])
PAIRS = list(zip(MASSES, ALPHAS_N))

MU_PAIRS = np.array([mu_of(a, m, XS_UNCAPPED) for m, a in PAIRS])


# ---------------------------------------------------------------------------
# VALIDATION GATE 1 -- reproduce the uncapped-v1 grid points nearest the pairs
# ---------------------------------------------------------------------------
def cube_crosscheck(pairs):
    """[(alpha_grid, m_grid, mu_cube, mu_here, rel), ...], one per pair.

    Returns ``None`` when the release cube is not on disk: ``release/*.h5`` is
    gitignored and shipped through Zenodo, so a clean clone must still be able
    to render the figure. The check runs whenever the file is present.
    """
    if not os.path.exists(CUBE_V1):
        return None

    import h5py

    grid = []
    with h5py.File(CUBE_V1, "r") as f:
        alphas = f["axes/alpha_n"][:]
        ms = f["axes/mass_noatm_gev"][:]
        lam = f["axes/lambda_m"][:]
        il = lam.shape[0] - 1             # massless is the last lambda index
        assert not np.isfinite(lam[il]), "last lambda is not massless"
        for m, a in pairs:
            ia = int(np.argmin(np.abs(np.log10(alphas) - np.log10(a))))
            im = int(np.argmin(np.abs(np.log10(ms) - np.log10(m))))
            grid.append((float(alphas[ia]), float(ms[im]),
                         float(f["noatm/mu"][MODE - 1, ia, im, il])))

    out = []
    for a_pt, m_pt, mu_cube in grid:
        mu_here = mu_of(a_pt, m_pt, XS_UNCAPPED)
        rel = abs(mu_here / mu_cube - 1.0) if mu_cube > 0 else float("nan")
        out.append((a_pt, m_pt, mu_cube, mu_here, rel))
    return out


CUBE_CHECK = cube_crosscheck(PAIRS)
if CUBE_CHECK is not None:
    for _a_pt, _m_pt, _mu_c, _mu_h, _rel in CUBE_CHECK:
        assert _mu_c > 0 and _rel < 1e-2, (
            "cube cross-check failed at alpha_n=%.6e, m=%.6e: "
            "mu_here=%.6f vs cube mu=%.6f (rel %.3e)"
            % (_a_pt, _m_pt, _mu_h, _mu_c, _rel))

# ---------------------------------------------------------------------------
# PANEL A data -- dmu/dq for the three pairs
# ---------------------------------------------------------------------------
DMU_DQ = np.array([dmu_dq(a, m, XS_UNCAPPED) for m, a in PAIRS])

_sup = DMU_DQ[0] > 0                     # common support (lightest pair is tightest)
_q = QS[_sup]
_r = DMU_DQ[:, _sup]
_spread = np.max(_r, axis=0) / np.min(_r, axis=0) - 1.0

# cumulative mu fraction along q (lightest pair), to weight the coincidence test
_cum = np.concatenate([[0.0], np.cumsum(0.5 * (_r[0][1:] + _r[0][:-1]) * np.diff(_q))])
_cum /= _cum[-1]

_win = QS[_sup] <= Q_HI
SPREAD_PLOT = float(_spread[_win].max())            # over the plotted window
SPREAD_90 = float(_spread[_cum <= 0.90].max())      # over the central 90% of mu
SPREAD_50 = float(_spread[_cum <= 0.50].max())      # over the central 50% of mu
# all three pairs now sit far from their kinematic edge q <~ m v: pure alpha^2/m
SPREAD_HEAVY = float(np.max(np.abs(_r[1] / _r[2] - 1.0)))

# VALIDATION GATE (panel A): the alpha^2/m scaling is exact up to (i) the
# kinematic edge q < m v_esc, which for the lightest pair now sits at
# 1e10 * v_esc ~ 2e7 GeV, three decades above the plotted window, and (ii) the
# v-quadrature grid, whose lower edge v_min = q/m moves with m. Only (ii)
# survives inside the plot, at the 1e-5 level.
assert SPREAD_HEAVY < 1e-4, "m=1e14 vs 1e18 not degenerate: %.3e" % SPREAD_HEAVY
assert SPREAD_90 < 2e-4, "dmu/dq spread over 90%% of mu too large: %.3e" % SPREAD_90
assert SPREAD_PLOT < 2e-4, \
    "dmu/dq spread over the plotted window too large: %.3e" % SPREAD_PLOT

# ---------------------------------------------------------------------------
# PANEL B data -- mu(<b) from the CAPPED massless cross section, then d/dlog10 b
#
# Everything here (curves, b_50/b_95, retained mu) is computed on the refined
# q grid QS_B: the log-b derivative below divides neighbouring-node quadrature
# noise by 0.025 decades, so the production grid is too coarse for it.
# ---------------------------------------------------------------------------
N_B = int(round(np.log10(B_SCAN_HI / B_SCAN_LO) * B_PER_DEC)) + 1
B_SCAN = np.geomspace(B_SCAN_LO, B_SCAN_HI, N_B)
LB = np.log10(B_SCAN)

MU_CUM = np.array([[mu_of(a, m, rate.make_xsec(None, b_constrained_max=b),
                          QS_B, EFF_QS_B)
                    for b in B_SCAN] for m, a in PAIRS])
DMU_DLOGB = np.array([np.gradient(c, LB) for c in MU_CUM])

# uncapped mu on the same refined grid: the plateau must reproduce it, and the
# gap to MU_PAIRS (production grid) is the grid-convergence residual
MU_PAIRS_B = np.array([mu_of(a, m, XS_UNCAPPED, QS_B, EFF_QS_B) for m, a in PAIRS])
Q_GRID_SHIFT = np.abs(MU_PAIRS_B / MU_PAIRS - 1.0)

PLATEAU = MU_CUM[:, -1]
FRAC = MU_CUM / PLATEAU[:, None]

MU_RETAINED = np.array([np.interp(np.log10(B_CAP), LB, c) for c in MU_CUM])
RET_FRAC = MU_RETAINED / PLATEAU
B95 = np.array([10 ** np.interp(0.95, f, LB) for f in FRAC])
B50 = np.array([10 ** np.interp(0.50, f, LB) for f in FRAC])
# share of mu delivered inside the sensor radius (figure 07's inner cutoff):
# with the new triple this is small for every pair, so the "kept domain
# R_eff <= b" of figure 07 does not quietly discard the signal shown here.
FRAC_BELOW_REFF = np.array([float(np.interp(np.log10(R_EFF), LB, f)) for f in FRAC])

# the plotted extent of each hump: the panel draws the curve only where it is
# meaningfully nonzero (see B_FLOOR_*), and that drawn region is what the
# translate / smoothness gates below have to be evaluated on.
_pb = (B_SCAN >= B_LO * 0.5) & (B_SCAN <= B_HI * 2)
LIVE = np.array([_pb & (y > max(B_FLOOR_FRAC * y.max(), B_FLOOR_ABS))
                 for y in DMU_DLOGB])

# VALIDATION GATE 2a -- mu(<b) plateaus at the uncapped mu = 3
assert np.all(np.abs(PLATEAU / MU_TARGET - 1.0) < 1e-2), \
    "mu(<b) plateau off target: %s" % PLATEAU
assert np.all(np.abs(PLATEAU / MU_PAIRS_B - 1.0) < 1e-3), \
    "capped-at-infinity mu != uncapped mu: %s vs %s" % (PLATEAU, MU_PAIRS_B)
assert np.all(Q_GRID_SHIFT < 1e-2), \
    "refined q grid moves mu: %s vs %s" % (MU_PAIRS_B, MU_PAIRS)

# VALIDATION GATE 2b -- the three mu(<b) curves are horizontal translates by x100.
# The shift is an exact multiple of the log-grid spacing, so np.interp lands on
# nodes and adds no interpolation error. Compared over the body of the
# distribution (5%..99.9% of mu).
TRANSLATE_DEV = []
for k in (1, 2):
    shifted = np.interp(LB - 2.0 * k, LB, FRAC[0])
    sel = (shifted > 0.05) & (shifted < 0.999)
    dev = float(np.max(np.abs(FRAC[k][sel] / shifted[sel] - 1.0)))
    TRANSLATE_DEV.append(dev)
    assert dev < 1e-2, "mu(<b) not a x100 translate for pair %d: %.3e" % (k, dev)

# VALIDATION GATE 2b' -- the same statement on the curves that are actually
# DRAWN. The cumulative test above is blind to shape: it compares smooth,
# monotone functions, so quadrature noise that is invisible in mu(<b) can still
# be a visible shoulder once d/dlog10 b divides it by the 0.025-decade node
# spacing. This gate is therefore taken directly on the plotted derivative, in
# plot units, over the drawn (live) region of each hump.
DERIV_DEV = []
for k in (1, 2):
    shifted = np.interp(LB - 2.0 * k, LB, DMU_DLOGB[0])
    dev = float(np.max(np.abs(DMU_DLOGB[k][LIVE[k]] - shifted[LIVE[k]])))
    DERIV_DEV.append(dev)
    assert dev < 2e-2, \
        "drawn hump %d is not the x100 translate of hump 0: %.4f plot units" \
        % (k, dev)


def _slope_flips(y, live):
    """Sign changes of the slope along a drawn hump (1 = single clean crest)."""
    idx = np.where(live)[0]
    assert idx.size and np.all(np.diff(idx) == 1), \
        "the drawn region of a hump is not one contiguous run"
    s = np.sign(np.diff(y[idx]))
    s = s[s != 0]
    return int(np.count_nonzero(s[1:] != s[:-1]))


SLOPE_FLIPS = [_slope_flips(DMU_DLOGB[k], LIVE[k]) for k in range(3)]
PEAKS = [float(DMU_DLOGB[k][LIVE[k]].max()) for k in range(3)]
assert all(f == 1 for f in SLOPE_FLIPS), \
    "a drawn hump has a shoulder/kink (slope sign changes %s, want 1 each)" \
    % SLOPE_FLIPS

# VALIDATION GATE 2c -- the story the figure claims, asserted numerically
assert RET_FRAC[0] > 0.95, "pair 0 should be untouched by the cap: %.4f" % RET_FRAC[0]
assert 0.05 < RET_FRAC[1] < 0.80, "pair 1 should be partially cut: %.4f" % RET_FRAC[1]
assert RET_FRAC[2] < 1e-3, "pair 2 should be annihilated: %.4e" % RET_FRAC[2]
assert np.all(FRAC_BELOW_REFF < 0.05), \
    "too much mu below R_eff (fig-07 inner cutoff): %s" % FRAC_BELOW_REFF

# ---------------------------------------------------------------------------
# Style (match repo: viridis for physical curves, red for removed/cap-critical)
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "mathtext.fontset": "dejavusans",
    "axes.axisbelow": True,
})

# widened hue ladder: light green / mid teal / dark purple, so the three
# coincident panel-A curves read as three distinct strata at 100% zoom
C_PAIR = [plt.cm.viridis(0.78), plt.cm.viridis(0.55), plt.cm.viridis(0.03)]
C_RED = "#d62728"                  # cap line + removed region
C_GUIDE = "#555555"                # power-law guide / annotations
C_REFF = "#444444"                 # inner cutoff
C_NEUTRAL = "#777777"              # legend proxies / translate arrows

# panel A stacking: very thick light solid under medium dashed under thin dark
# dotted. At 140 dpi this leaves ~3 px of each stratum visible on either side,
# so the coincidence reads as three curves rather than one fat one.
A_LW = [7.6, 4.2, 1.8]
A_LS = ["-", "--", ":"]
A_Z = [3, 4, 5]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.4, 5.9))


def _mant_exp(x):
    e = int(np.floor(np.log10(x)))
    return x / 10 ** e, e


def _sci(x, nd=2):
    """mathtext 'a.bb x 10^c'."""
    m, e = _mant_exp(x)
    return r"%.*f\times10^{%d}" % (nd, m, e)


def _len_str(b):
    """Plain-text length with a human-scale unit."""
    if b < 1e-2:
        return "%.2f mm" % (b * 1e3)
    if b < 1.0:
        return "%.1f cm" % (b * 1e2)
    return "%.1f m" % b


# ===========================================================================
# PANEL A -- the detector sees the same thing
# ===========================================================================
for k, (m, a) in enumerate(PAIRS):
    am, ae = _mant_exp(a)
    lbl = (r"$m=10^{%d}$ GeV,  $\alpha_n=%.3f\times10^{%d}$,  $\mu=%.2f$"
           % (int(round(np.log10(m))), am, ae, MU_PAIRS[k]))
    axA.plot(QS, DMU_DQ[k], color=C_PAIR[k], lw=A_LW[k], ls=A_LS[k],
             zorder=A_Z[k], solid_capstyle="round", dash_capstyle="round",
             label=lbl)

# q^-3 guide through the falling tail
_iq = int(np.argmax(DMU_DQ[0]))
_qg = np.geomspace(QS[_iq] * 3.0, Q_HI, 50)
_yg = DMU_DQ[0][_iq] * (_qg / QS[_iq]) ** -3 * 2.6
axA.plot(_qg, _yg, color=C_GUIDE, lw=1.2, ls=(0, (1, 2)), zorder=2)
txt = axA.text(_qg[20], _yg[20] * 2.4, r"$q^{-3}$", color=C_GUIDE, fontsize=10,
               ha="center", va="bottom")
txt.set_path_effects([pe.Stroke(linewidth=2.4, foreground="w"), pe.Normal()])

# efficiency turn-on marker, anchored on the steep rise
_ymark = 1e-7
_pos = DMU_DQ[0] > 0
_qmark = 10 ** float(np.interp(np.log10(_ymark),
                               np.log10(DMU_DQ[0][_pos][:_iq]),
                               np.log10(QS[_pos][:_iq])))
axA.annotate("mode-1 efficiency\nturn-on  $\\varepsilon(q)$",
             xy=(_qmark, _ymark), xytext=(1.12e2, 4e-6),
             color=C_GUIDE, fontsize=9, ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=C_GUIDE, lw=1.0,
                             shrinkA=4, shrinkB=3))

axA.text(0.545, 0.975,
         r"$d\mu/dq \;\propto\; (\alpha^{2}/m)\;q^{-3}\,\varepsilon(q)$" "\n"
         r"invariant along $\alpha\propto\sqrt{m}$:" "\n"
         r"identical $\mu$  $\Rightarrow$  identical limit",
         transform=axA.transAxes, fontsize=9.5, ha="left", va="top",
         color="#222222",
         bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#cccccc",
                   alpha=0.95))

axA.set_xscale("log")
axA.set_yscale("log")
axA.set_xlim(Q_LO, Q_HI)
axA.set_ylim(1e-13, 1e-1)
axA.set_xlabel("momentum transfer  $q$  [GeV]")
axA.set_ylabel(r"$d\mu/dq$  [GeV$^{-1}$]   (detected, mode 1)")
axA.set_title("A.  The detector sees the same thing")
axA.legend(loc="lower right", fontsize=8.8, framealpha=0.94,
           title="three points on the degeneracy line "
                 "$\\alpha\\propto\\sqrt{m}$",
           title_fontsize=9)
axA.grid(True, which="major", color="#e9e9e9", lw=0.7)

# ===========================================================================
# PANEL B -- ... but from ever-larger impact parameters
# ===========================================================================
Y_TOP = 6.0

# removed-by-cap band
axB.axvspan(B_CAP, B_HI * 2, color=C_RED, alpha=0.09, lw=0, zorder=0,
            label=r"removed by the cap  ($b>b_{\mathrm{cap}}$)")

for k in range(3):
    y = DMU_DLOGB[k]
    # draw the hump only where it is meaningfully nonzero (LIVE, built with the
    # gates above); without this the last curve paints a spurious continuous
    # floor at y ~ 0 across the panel
    live = LIVE[k]
    axB.fill_between(B_SCAN, 0.0, y, where=live, color=C_PAIR[k],
                     alpha=0.20, lw=0, zorder=2, interpolate=True)
    axB.plot(B_SCAN, np.where(live, y, np.nan), color=C_PAIR[k], lw=2.4,
             zorder=4)
    # b_95 stem
    y95 = float(np.interp(np.log10(B95[k]), LB, y))
    axB.plot([B95[k], B95[k]], [0, y95], color=C_PAIR[k], lw=1.1, ls="-",
             zorder=5)
    axB.plot([B95[k]], [y95], marker="v", ms=6.5, color=C_PAIR[k], zorder=6)
    # mass tag above each hump, nudged sideways if the peak sits on a guide
    # line (the halo would otherwise punch a hole through R_eff or the cap)
    _ipk = int(np.argmax(y))
    _tx, _ha, _ty = B_SCAN[_ipk], "center", y[_ipk] + 0.22
    for _g in (R_EFF, B_CAP):
        if 0.25 < B_SCAN[_ipk] / _g < 4.0:
            # sideways so the halo cannot notch the guide, and a row higher so
            # the widened (units-carrying) tag clears the neighbouring hump's tag
            _tx, _ha, _ty = B_SCAN[_ipk] * 1.5, "left", y[_ipk] + 0.68
    t = axB.text(_tx, _ty,
                 r"$m=10^{%d}$ GeV" % int(round(np.log10(MASSES[k]))),
                 color=C_PAIR[k], fontsize=9.5, ha=_ha, va="bottom",
                 zorder=7)
    t.set_path_effects([pe.Stroke(linewidth=2.6, foreground="w"), pe.Normal()])

# neutral legend proxy for the b_95 markers (they are coloured per pair)
axB.plot([], [], marker="v", ms=6.5, ls="none", color=C_NEUTRAL,
         label=r"$b_{95}$ (95% of $\mu$ accumulated)")

# overlays: sensor radius and the production cap
axB.axvline(R_EFF, color=C_REFF, ls="--", lw=1.6, zorder=3,
            label=r"$R_{\mathrm{eff}}=200\,\mu$m  (sensor radius)")
axB.axvline(B_CAP, color=C_RED, ls="-.", lw=1.9, zorder=3,
            label=r"$b_{\mathrm{cap}}=b_{\mathrm{constrained\,max}}=10$ cm")

# x100 translation arrows, drawn in the empty valleys between the humps and
# lifted clear of every curve; the spans never touch the R_eff or cap guides
for k in (0, 1):
    lo, hi = B50[k] * 4.0, B50[k + 1] / 4.0
    assert not (lo <= R_EFF <= hi or lo <= B_CAP <= hi), \
        "translate arrow %d would cross a guide line" % k
    _seg = (B_SCAN >= lo) & (B_SCAN <= hi)
    y_arr = float(np.max(DMU_DLOGB[:, _seg])) + 0.50
    axB.annotate("", xy=(hi, y_arr), xytext=(lo, y_arr),
                 arrowprops=dict(arrowstyle="->", color=C_NEUTRAL, lw=1.2,
                                 shrinkA=0, shrinkB=0), zorder=7)
    txt = axB.text(np.sqrt(lo * hi), y_arr + 0.14, r"$\times100$",
                   color="#555555", fontsize=9.5, ha="center", va="bottom",
                   zorder=7)
    txt.set_path_effects([pe.Stroke(linewidth=2.6, foreground="w"), pe.Normal()])

# Read-out block: the common hump area (mu, identical for all three), then
# b_95 and the mu surviving the cap for each pair.
# It lives in the empty corridor BETWEEN the two full-height guides, so
# every line has to stay narrower than the ~200 px (at figure dpi) from the
# R_eff dashes to the b_cap line -- hence no "b_95 =" / "mu_kept =" repeated on
# each row and no parenthetical percentages: with all three areas equal to mu,
# the retained fraction is read straight off the column. Enforced by the
# renderer-bbox gate after tight_layout (the layout is only final there).
READOUT_X = 0.148                      # axes fraction, just right of R_eff
READOUT_Y, READOUT_DY = 0.975, 0.058
_lines = [(r"all $\mu=%.2f$;   $b_{95}$,  $\mu(<b_{\mathrm{cap}})$:"
           % PLATEAU.mean(), "#222222", 8.6)]
for k in range(3):
    _ret = MU_RETAINED[k]
    _mu_s = ("$%.2f$" % _ret) if _ret > 0.05 else ("$%s$" % _sci(_ret, 1))
    _lines.append((
        r"$10^{%d}$:   %s,   %s"
        % (int(round(np.log10(MASSES[k]))), _len_str(B95[k]), _mu_s),
        C_PAIR[k], 8.6))
READOUT_TEXTS = []
for k, (s, c, fs) in enumerate(_lines):
    READOUT_TEXTS.append(axB.text(
        READOUT_X, READOUT_Y - READOUT_DY * k, s, transform=axB.transAxes,
        ha="left", va="top", fontsize=fs, color=c, zorder=8,
        bbox=dict(boxstyle="square,pad=0.22", fc="white", ec="none",
                  alpha=0.9)))

axB.set_xscale("log")
axB.set_xlim(B_LO, B_HI)
axB.set_ylim(0.0, Y_TOP)
axB.set_xlabel("impact parameter  $b$  [m]")
axB.set_ylabel(r"$d\mu/d\log_{10} b$")
axB.set_title(r"B.  $\dots$ but assembled from ever-larger impact parameters")
axB.legend(loc="upper right", fontsize=8.4, framealpha=0.94,
           borderpad=0.5)
axB.grid(True, which="major", color="#e9e9e9", lw=0.7)

# ---------------------------------------------------------------------------
# parameter box, caption + suptitle
# ---------------------------------------------------------------------------
_am, _ae = _mant_exp(ALPHA0_N)
param_txt = (
    "$q_{\\mathrm{th}}=%g$ GeV      $T=%.3f\\times10^{6}$ s      "
    "$f_X=%g$      $R_{\\mathrm{eff}}=200\\,\\mu$m      "
    "anchor  $\\alpha_n=%.3f\\times10^{%d}$ at $m=10^{%d}$ GeV, "
    "$\\times100$ per $\\times10^{4}$ in $m$"
    % (Q_MIN, T_TOTAL / 1e6, config.F_X, _am, _ae,
       int(round(np.log10(MASSES[0]))))
)
fig.text(0.5, 0.052, param_txt, ha="center", va="bottom", fontsize=9,
         family="monospace", color="#222222")

fig.suptitle(r"The $\alpha^{2}/m$ degeneracy of the long-range ($1/r^{2}$) "
             r"interaction: identical signal from ever-larger impact parameters",
             fontsize=13.5, y=0.985)

fig.tight_layout(rect=(0, 0.045, 1, 0.955))

# ---------------------------------------------------------------------------
# VALIDATION GATE 6 (layout) -- the read-out block clears both panel-B guides.
#
# R_eff and b_cap are full-height axvlines: anything the read-out paints on top
# of them punches a hole in the guide, and the guide in turn eats the glyphs.
# Measured with the real renderer, after tight_layout (the axes only reach
# their final position there) and including each line's bbox patch -- same
# renderer/geometry style as the x100-arrow guard above.
# ---------------------------------------------------------------------------
fig.canvas.draw()                       # realise the bbox patches
_rend = fig.canvas.get_renderer()
_x_reff_px = float(axB.transData.transform((R_EFF, 0.0))[0])
_x_cap_px = float(axB.transData.transform((B_CAP, 0.0))[0])
READOUT_GAPS = []                       # (left of R_eff line, right of cap line)
for _i, _t in enumerate(READOUT_TEXTS):
    _e = _t.get_window_extent(renderer=_rend)
    _patch = _t.get_bbox_patch()
    if _patch is not None:
        _pe = _patch.get_window_extent(_rend)
        _e = _e.union([_e, _pe])
    _gap_lo, _gap_hi = _e.x0 - _x_reff_px, _x_cap_px - _e.x1
    READOUT_GAPS.append((_gap_lo, _gap_hi))
    assert _gap_lo > 2.0, (
        "read-out line %d starts on/left of the R_eff guide (gap %.1f px): %s"
        % (_i, _gap_lo, _lines[_i][0]))
    assert _gap_hi > 2.0, (
        "read-out line %d runs into the b_cap guide (gap %.1f px): %s"
        % (_i, _gap_hi, _lines[_i][0]))

# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
name = "08_mass_coupling_degeneracy"
outpaths = {}
for ext in ("png", "svg", "pdf"):
    d = os.path.join(_root, "notebooks", ext)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{name}.{ext}")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    outpaths[ext] = p

# ---------------------------------------------------------------------------
# validation numbers
# ---------------------------------------------------------------------------
print("=" * 74)
print("Q GRIDS   GATE 1 + panel A: production n_q = %d   |   panel-B cap scan:"
      " n_q = %d  (same span)" % (FID["n_q"], N_Q_PANEL_B))
print()
print("GATE 1  uncapped v1 cube cross-check (noatm, mode 1, massless slice)")
if CUBE_CHECK is None:
    print("  cross-check skipped (release file not present): %s" % CUBE_V1)
    print("  release/*.h5 is gitignored and distributed via Zenodo; the figure")
    print("  is unaffected -- every curve is computed from luhdm directly.")
else:
    for _a_pt, _m_pt, _mu_c, _mu_h, _rel in CUBE_CHECK:
        print("  grid point   alpha_n = %.9e   m = %.9e GeV" % (_a_pt, _m_pt))
        print("    mu cube = %.6f   mu pipeline = %.6f   rel diff = %.3e"
              "   (tol 1e-2)   PASS" % (_mu_c, _mu_h, _rel))
print()
print("ANCHOR  mu_trial(alpha_n=%.0e, m=%.0e) = %.6f  ->  alpha_n(%.0e) = %.6e"
      % (_A_TRIAL, MASSES[0], _MU_TRIAL, MASSES[0], ALPHA0_N))
print()
print("DEGENERACY-LINE POINTS")
for k, (m, a) in enumerate(PAIRS):
    print("  m = %.3e GeV   alpha_n = %.6e   alpha = %.4e   mu = %.6f"
          % (m, a, a * config.N_NEUTRONS, MU_PAIRS[k]))
print()
print("GATE 2  panel A coincidence of dmu/dq (max relative spread)")
print("  over q carrying 50%% of mu      : %.3e" % SPREAD_50)
print("  over q carrying 90%% of mu      : %.3e   (tolerance 2e-4)   PASS"
      % SPREAD_90)
print("  over the plotted window        : %.3e   (tolerance 2e-4)   PASS"
      % SPREAD_PLOT)
print("                                   lightest pair's kinematic edge"
      " q ~ m v_esc = %.2e GeV," % (MASSES[0] * config.VESC))
print("                                   three decades above the window")
print("  m=1e14 vs m=1e18 (edge-free)   : %.3e   (tolerance 1e-4)   PASS"
      % SPREAD_HEAVY)
print()
print("GATE 3  mu(<b) plateau (uncapped target mu = %.2f, panel-B q grid)"
      % MU_TARGET)
for k, (m, a) in enumerate(PAIRS):
    print("  m = %.0e   plateau mu(<inf) = %.6f   dev %.3e   "
          "uncapped mu (n_q=%d) = %.6f   PASS"
          % (m, PLATEAU[k], abs(PLATEAU[k] / MU_TARGET - 1), N_Q_PANEL_B,
             MU_PAIRS_B[k]))
print("  q-grid refinement moves mu by %s  (tolerance 1e-2)   PASS"
      % "  ".join("%.2e" % s for s in Q_GRID_SHIFT))
print()
print("GATE 4  mu(<b) curves are x100 horizontal translates")
for k, dev in zip((1, 2), TRANSLATE_DEV):
    print("  cumulative, pair %d vs pair 0 shifted by 10^%d, 5%%..99.9%% of mu :"
          " max rel dev %.3e  (tolerance 1e-2)   PASS" % (k, 2 * k, dev))
for k, dev in zip((1, 2), DERIV_DEV):
    print("  AS DRAWN,   hump %d vs hump 0 shifted by 10^%d, over the live"
          " region : max dev %.4f plot units  (tolerance 2e-2)   PASS"
          % (k, 2 * k, dev))
print("  single clean crest per hump (slope sign changes, want 1 each) : %s"
      "   PASS" % ", ".join("%d" % f for f in SLOPE_FLIPS))
print("  peak heights : %s  [mu per decade]"
      % ", ".join("%.4f" % p for p in PEAKS))
print()
print("GATE 5  the three-act story, asserted")
print("  pair 0 retained %.4f of mu   (> 0.95 : cap harmless)   PASS"
      % RET_FRAC[0])
print("  pair 1 retained %.4f of mu   (0.05..0.80 : cap bites)   PASS"
      % RET_FRAC[1])
print("  pair 2 retained %.3e of mu   (< 1e-3 : cap kills)   PASS"
      % RET_FRAC[2])
print("  mu fraction inside R_eff = %g m (fig-07 inner cutoff), per pair:"
      % R_EFF)
print("    %s   (all < 0.05)   PASS"
      % "  ".join("%.2e" % f for f in FRAC_BELOW_REFF))
print()
print("GATE 6  read-out block clears both full-height guides in panel B")
print("  clearance to the R_eff dashes / to the b_cap line, per line"
      " [px at figure dpi]:")
for _i, (_g_lo, _g_hi) in enumerate(READOUT_GAPS):
    print("    line %d :  %6.1f  |  %6.1f   (both > 2.0)   PASS"
          % (_i, _g_lo, _g_hi))
print()
print("IMPACT-PARAMETER SCALES AND CAP RETENTION  (b_cap = %g m)" % B_CAP)
for k, (m, a) in enumerate(PAIRS):
    print("  m = %.0e   b_50 = %.4e m (%s)   b_95 = %.4e m (%s)"
          % (m, B50[k], _len_str(B50[k]), B95[k], _len_str(B95[k])))
    print("             mu(<10cm) = %.4e  (%.4g%% of %.3f)"
          % (MU_RETAINED[k], 100 * RET_FRAC[k], PLATEAU[k]))
print()
print("WHERE THE CAP CLOSES THE ISLAND (retained fraction along the line,"
      " panel-B q grid)")
for m in (1e12, 1e13, 1e14, 1e15, 1e16):
    a = ALPHA0_N * np.sqrt(m / MASSES[0])
    mc = mu_of(a, m, rate.make_xsec(None, b_constrained_max=B_CAP),
               QS_B, EFF_QS_B)
    mun = mu_of(a, m, XS_UNCAPPED, QS_B, EFF_QS_B)
    print("  m = %.1e   alpha_n = %.4e   mu_uncapped = %.4f   "
          "mu_capped = %.4f   retained = %.4f" % (m, a, mun, mc, mc / mun))
print()
for ext, p in outpaths.items():
    print("%s: %s" % (ext, p))
print("=" * 74)
