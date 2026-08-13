"""Three-axes figure: the alpha^2/m degeneracy of the massless (1/r^2) mediator,
where it holds exactly, where the sensor radius breaks it, and how the 10 cm
impact-parameter cap ends the exclusion island.

HISTORICAL in one respect. Everything here about the degeneracy and about the
INNER cutoff at R_eff is current and unchanged. What has changed is the ending:
the shipped data release does NOT cap the impact-parameter integral, so the
massless island is not closed by the cap discussed in panel B -- it runs to the
Planck mass in the stored surfaces and is closed instead by the post-hoc flux
cut m_cut, which requires N_req = 3 halo transits within the same 10 cm aperture
during the exposure (release/README.md section 5.4). The arithmetic of panel B
is exactly the arithmetic behind that cut -- how much of mu comes from flybys
beyond 10 cm -- so the panel is still the right picture of why a hardware-scale
aperture bounds the claim. Read it as the geometric argument for m_cut rather
than as a description of what the released cross section does.

FROZEN AGAINST THE EFFICIENCY TABLE IT WAS MEASURED WITH. Every number here is
anchored through the mode-1 detection efficiency, loaded from the canonical
table at import (_EFF_F below), and the committed renders were measured with
the earlier fixed-arrival-phase (w = 1) curves. The canonical table now holds
the arrival-phase-marginalised curves, which are lower through the turn-on;
under them the mu = 3 anchor re-solves upward and three of the gates below stop
the script rather than let it redraw: the light-anchor edge suppression comes
out around 9.3x against the pinned 12-17x band, and the two spectrum validity
thresholds around 4.2e12 and 2.2e13 GeV against the pinned 6.54e12 and 3.52e13
GeV. To reproduce the committed figure, pin the table it was built with:

    LUHDM_EFFICIENCY_NPZ=<the w = 1 table> \\
        python scripts/figure_mass_coupling_degeneracy.py

That table is not in the tree any more; it is recoverable from git history --
sha256 451e6ca10c759ecbe4620672796f3571538914dd7c7dba63fd679710d04183b3,
`git show 834294b^:luhdm/reference_data/efficiency_curves.npz`. Re-basing the
figure on the canonical table instead is a separate decision, not a rerun: it
means re-measuring and re-pinning every gate band below, redrawing the caption
and the domain-of-validity line, and accepting that the current efficiency is
then folded into cap-era physics.

For a massless mediator the projected cross section is Rutherford,
dsigma/dq = 2 pi alpha^2 / (v^2 q^3), and the halo number density scales as
1/m, so the detected spectrum carries the coupling and the mass only through the
combination alpha^2/m:

    dmu/dq  =  T * eps(q) * f_X * (rho/m) * <v * 2 pi alpha^2/(v^2 q^3)>
            propto (alpha^2/m) q^-3 eps(q).

Walking along alpha propto sqrt(m) therefore leaves the detector-level spectrum
-- and hence mu and the optimum-interval extremeness -- unchanged, and with
nothing in the analysis to break the degeneracy the massless exclusion island
extended to arbitrarily large mass.

TWO things break that invariance, at opposite ends of the impact parameter, and
this figure shows both.

(i) The INNER cutoff, panel A + ratio strip. No flyby approaches closer than the
sensor radius, so the b-integral starts at R_eff and the impulse saturates at

    q_max(v) = 2 alpha / (v conv(R_eff))     [cross_section.coulomb_q_max]

above which dsigma/dq is identically zero. q_max carries a bare alpha, NOT the
invariant alpha^2/m: along alpha propto sqrt(m) the endpoint moves as sqrt(m),
so the LIGHTER the anchor the LOWER its endpoint. At the lightest anchor here
(m = 1e10 GeV) q_max(v_esc) lands inside the plotted 1e2-3e4 GeV window -- this
figure's own momentum grid, which starts a decade below the 1 TeV analysis
window the release is set in -- and the detected spectrum is visibly suppressed
towards the upper edge, taking ~1.9% off mu. The two heavy anchors have their
endpoints 2 and 4 decades above that window and stay degenerate at the 1e-5
level. The validity domain is computed, not asserted: DEG_M_SPECTRUM /
DEG_M_MU below are root-solved thresholds.

(ii) The OUTER cap, panel B. The massless flyby reach is b_max(q) = 2 alpha/(q v),
so along the same line the impact parameters that deliver a given kick grow as
sqrt(m): the same mu is assembled from flybys at ever-larger distance. Once the
impact-parameter integral is capped at the production value
b_constrained_max = 10 cm, the heavy/strong end of the line loses essentially all
of its rate, and the island closes. The breaking of (i) does NOT close the
island -- it is a percent-level effect confined to the light end -- so the cap
remains the reason the island is bounded.

The same physics appears in both panels and the figure gates that identity: the
mu the lightest anchor is missing in panel A (1.9%) is exactly the share of mu
the heavy anchors carry below the sqrt(m)-scaled sensor radius, i.e. the inner
flank that panel B shows amputated at the R_eff guide.

The three anchor pairs are unchanged from the pre-cutoff version of this figure,
deliberately: they are the pairs the paper's degeneracy discussion quotes, and
the whole story is told inside the laboratory-relevant range of b. At
m = 1e10 GeV the flybys that carry mu sit at b ~ 1.8 mm, just outside the sensor
radius R_eff = 260 um (so the inner cutoff bites) and comfortably inside the
10 cm cap; at m = 1e14 GeV they sit at b ~ 18 cm and the cap bites; at
m = 1e18 GeV they sit at b ~ 18 m and the cap removes all but ~3e-5 of mu.

Pipeline: bare-halo SHM (no atmosphere -- attenuation would break the degeneracy
at high alpha and is beside the geometric point), massless mediator, mode-1
measured efficiency, exposure config.T_EXPOSURE. Every number is produced by the
same luhdm.rate / luhdm.limits calls the data-release builder uses
(scripts/build_release.py, noatm massless slice), and is cross-checked against
the internal capped parent cube (see CUBE below) at that cube's own 10 cm cap,
at the exact grid points nearest the three anchor pairs. That cross-check is
dormant today: the cube it wants is the internal full-lambda build behind
luhdm.release.DEFAULT_PATH, which is not in the tree, so GATE 1 prints a skip.
The tracked release files cannot stand in for it -- they carry no cap
(b_constrained_max_m is NaN), which is the very quantity the gate compares.

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
from scipy.optimize import brentq

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from optimum_interval import scanning

from luhdm import config, cross_section, efficiency, halo, rate, release

# ---------------------------------------------------------------------------
# Release-pipeline constants -- the scripts/build_release.py settings that this
# figure actually exercises (only the panel-B q grid departs, see below)
# ---------------------------------------------------------------------------
T_TOTAL = config.T_EXPOSURE            # live-time [s], LUHDM_T_EXPOSURE-aware
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
# Where this figure's momentum grid starts. NOT the analysis threshold, and not
# config.Q_THRESH, which is 1 TeV: 100 GeV is the reconstruction threshold of
# the stored candidate lists and the window the capped scheme this figure
# documents was set in. The value stays at 100.0 so the script reproduces its
# committed renders; moving it to the 1 TeV window means re-pinning every gate
# band below and redrawing the figure.
Q_MIN = 100.0                          # q-grid floor [GeV]
MODE = 1                               # measured efficiency mode

B_CAP = 0.1                            # b_constrained_max [m] = 10 cm (production)
R_EFF = config.R_EFF                   # sensor radius [m] (260 um)

MU_TARGET = 3.0                        # <N> = 3 zero-event benchmark (notebook 01)

MASSES = np.array([1e10, 1e14, 1e18])  # degeneracy-line masses [GeV]
K_REF = 2                              # the degenerate reference anchor (heaviest)

# plotted windows
Q_LO, Q_HI = 1e2, 3e4                  # panel A momentum window [GeV]
B_LO, B_HI = 2e-5, 1e3                 # panel B impact-parameter window [m]

# b scan (wider than the plot so the plateau is resolved), points per decade.
# The scan STARTS at the sensor radius: no flyby passes closer, so the massless
# b-integral runs over the annulus R_eff <= b <= b_cap, mu(<b) is identically
# zero at b = R_eff, and a cap below R_eff is rejected outright by
# cross_section_rutherford_projection_capped. The span is an exact 14 decades so
# the node spacing stays exactly 0.025 dex and the x100-translate gate below
# still lands on grid nodes.
B_SCAN_LO, B_SCAN_HI, B_PER_DEC = R_EFF, R_EFF * 1e14, 40

# panel-B curves are drawn only where they carry a meaningful share of mu.
# Below this the stroke has no resolvable height but still paints a solid line
# along y ~ 0, which misreads as a nonzero floor (the deepest tails of the three
# humps would otherwise tile the whole panel). The absolute floor is ~3 px at
# 140 dpi; together the two cuts keep +-1.2 decades around each peak, i.e.
# >99.8% of every hump's area.
B_FLOOR_FRAC = 3e-3
B_FLOOR_ABS = 0.05                     # [mu per decade], i.e. ~1% of the y axis

# The cube used by the cross-check. luhdm.release.DEFAULT_PATH is the INTERNAL
# full-lambda parent cube -- capped at 10 cm, built in the 0.1 TeV window and on
# the fixed-arrival-phase efficiency, i.e. the scheme this figure documents. It
# is deliberately not the shipped release: the tracked files are uncapped
# (b_constrained_max_m = NaN), so they cannot serve a cap parity check. That
# parent cube is not in the tree either, so GATE 1 prints a skip and the parity
# check has been dormant since the release split into the current two files.
# When a matching cube is present, the parent is built WITH the R_eff inner
# cutoff, so the parity check runs through the figure's own pipeline (GATE 1a)
# and switching the cutoff off becomes the counterfactual whose size GATE 1b
# pins; against an older cube built without the cutoff the two roles are
# exchanged (see cube_crosscheck).
CUBE = str(release.DEFAULT_PATH)

# ---------------------------------------------------------------------------
# Pipeline: identical construction to build_release.py's noatm massless cell
# ---------------------------------------------------------------------------
# The one input that floats: this reads whatever efficiency table is canonical
# at import time (LUHDM_EFFICIENCY_NPZ if it is set, otherwise the committed
# luhdm/reference_data/efficiency_curves.npz). The committed renders were
# measured with the fixed-arrival-phase (w = 1) curves and the gates below are
# pinned to them, so regenerating this figure means setting
# LUHDM_EFFICIENCY_NPZ to that table first -- see the module docstring for the
# digest and how to recover it. Left unset, the canonical
# arrival-phase-marginalised curves are loaded and the script stops at a gate.
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

XS_UNCAPPED = rate.make_xsec(None)      # massless, no OUTER cap (inner cutoff stays)


def mu_of(alpha_n, m, xs, qs=QS, eff_qs=EFF_QS, r_eff=R_EFF):
    """Expected detected counts mu, exactly as the cube stores it.

    ``rate.differential_rate_trapz(..., eff=None)`` then the mode efficiency
    post-multiply and ``spectrum_from_rate`` -- i.e. the mu returned by
    limits.extremeness_and_mu -> optimum_interval.scanning.scan_extremeness,
    without paying for the Monte-Carlo calibration we do not need here.

    ``qs``/``eff_qs`` default to the production q grid (what the cube used);
    panel B passes the refined pair QS_B/EFF_QS_B. ``r_eff`` is the INNER edge
    of the b-integral; only GATE 1b departs from the sensor radius (r_eff = 0
    reproduces the pipeline as it stood before the inner cutoff landed).
    """
    raw = rate.differential_rate_trapz(qs, alpha_n, m, F_V_F, xs, R_eff=r_eff,
                                       eff=None)
    spec = scanning.spectrum_from_rate(qs, raw * eff_qs, T_TOTAL)
    return 0.0 if spec is None else float(spec[0])


def dmu_dq(alpha_n, m, xs=XS_UNCAPPED):
    """Detected differential counts dmu/dq [GeV^-1] on QS."""
    raw = rate.differential_rate_trapz(QS, alpha_n, m, F_V_F, xs, eff=None)
    return raw * EFF_QS * T_TOTAL


# ---------------------------------------------------------------------------
# Anchor the degeneracy line at mu = 3 for the lightest pair.
#
# mu is NOT exactly proportional to alpha^2: the massless b-integral starts at
# the sensor radius, so no flyby delivers more than
# q_max(v) = 2 alpha / (v conv(R_eff)) and the spectrum carries a hard endpoint
# that moves with alpha (not with alpha^2/m). A single alpha^2-rescaled trial
# call therefore misses the target badly whenever the trial's endpoint sits
# lower in the q grid than the anchor's, so the anchor is root-solved instead.
# ---------------------------------------------------------------------------
ALPHA0_N = 10.0 ** brentq(
    lambda la: mu_of(10.0 ** la, MASSES[0], XS_UNCAPPED) - MU_TARGET,
    -10.0, -5.0, xtol=1e-10)

# alpha propto sqrt(m): masses step by 1e4, so alpha_n steps by exactly 100
ALPHAS_N = ALPHA0_N * np.sqrt(MASSES / MASSES[0])
PAIRS = list(zip(MASSES, ALPHAS_N))

MU_PAIRS = np.array([mu_of(a, m, XS_UNCAPPED) for m, a in PAIRS])

# the endpoint each pair carries, at the slowest-decaying (escape) speed: this is
# the quantity that scales as alpha ~ sqrt(m) and so breaks the degeneracy
Q_MAX_VESC = np.array([cross_section.coulomb_q_max(a * config.N_NEUTRONS,
                                                   config.VESC)
                       for _m, a in PAIRS])
# absolute end of support: above sqrt(q_max(v_esc) * v_esc * m) even the slowest
# kinematically allowed flyby (v = q/m) is inside R_eff, and dmu/dq is exactly 0
Q_SUPPORT_END = np.sqrt(Q_MAX_VESC * config.VESC * MASSES)


# ---------------------------------------------------------------------------
# VALIDATION GATE 1 -- reproduce the shipped cube's grid points nearest the pairs
# ---------------------------------------------------------------------------
CUBE_TAG = None                        # filled in by cube_crosscheck()


def cube_crosscheck(pairs):
    """[(alpha_grid, m_grid, mu_cube, mu_nocut, rel_nocut, mu_cut, rel_cut), ...].

    Compares against the shipped cube at *its own* impact-parameter cap, read
    from the root attribute ``b_constrained_max_m`` rather than assumed: the
    production cube is capped at 10 cm, so an uncapped cross section would
    disagree by construction on the two heavy pairs (that disagreement is the
    whole point of panel B, not a pipeline error). Comparing at the cube's cap
    makes this a parity check on the code path production actually uses.

    Two mu values are returned per grid point, one for each side of the R_eff
    inner cutoff, so the pair of gates pins the cutoff from both directions:

    * ``mu_cut`` -- this pipeline as the figure uses it, inner edge at R_eff.
      From v5 on the release is built with the cutoff, so this is the one that
      must reproduce the cube to round-off; it is the parity check (GATE 1a).
    * ``mu_nocut`` -- the same pipeline with the inner edge switched off
      (``r_eff = 0``), i.e. the pipeline as it stood before the cutoff landed.
      Its deviation from the cube is the endpoint effect the figure displays,
      and GATE 1b pins that deviation's size: negligible for the heavy pairs,
      percent-level for the lightest.

    Against a pre-v5 cube, built without the cutoff, the two exchange roles --
    ``mu_nocut`` is then the parity check and ``mu_cut`` the displaced one. The
    magnitude of the displacement is the same either way, which is why one band
    (:data:`CUBE_CUT_BAND`) serves both and the gates below simply read it off
    the arm that is displaced.

    Read through ``luhdm.release`` rather than raw h5py so the check works on
    either cube layout (the v3 /atm+/noatm groups and the current /results axis
    layout name their mass axis differently).

    Returns ``None`` when the release cube is not on disk: ``release/*.h5`` is
    gitignored and shipped through Zenodo, so a clean clone must still be able
    to render the figure. The check runs whenever the file is present.
    """
    global CUBE_TAG
    if not os.path.exists(CUBE):
        return None

    with release.open_release(CUBE) as cube:
        CUBE_TAG = cube.version_tag
        cube_cap = cube.b_constrained_max
        grid = [cube.cell(m, a, "massless", mode=MODE, group="noatm")
                for m, a in pairs]

    xs_cube = rate.make_xsec(None, b_constrained_max=cube_cap)
    out = []
    for _c in grid:
        a_pt, m_pt, mu_cube = _c["alpha_n"], _c["mass_gev"], _c["mu"]
        mu_nocut = mu_of(a_pt, m_pt, xs_cube, r_eff=0.0)
        mu_cut = mu_of(a_pt, m_pt, xs_cube)
        rel_nocut = abs(mu_nocut / mu_cube - 1.0) if mu_cube > 0 else float("nan")
        rel_cut = abs(mu_cut / mu_cube - 1.0) if mu_cube > 0 else float("nan")
        out.append((a_pt, m_pt, mu_cube, mu_nocut, rel_nocut, mu_cut, rel_cut))
    return out


CUBE_CHECK = cube_crosscheck(PAIRS)

# Does the shipped cube already carry the R_eff inner cutoff? Decided from the
# file, not from its filename: whichever arm reproduces the cube better is the
# one the cube was built with. That keeps GATE 1 meaningful against a v5-or-
# later cube (cutoff in: parity through mu_cut) and against an archived pre-v5
# one (cutoff out: parity through mu_nocut) with no edit here.
#
# The vote is taken at the single pair that discriminates, the one whose two
# arms are furthest apart -- in practice the lightest, where the cutoff costs
# 2.7e-2 and the other arm agrees to 4e-8. At the heavy pairs both arms sit
# under GATE 1a's 1e-4 tolerance, so polling them would only let round-off
# decide.
def _cube_has_inner_cutoff(check):
    """True if ``check``'s cube reproduces the pipeline with the cutoff ON."""
    if check is None:
        return False
    # (..., rel_nocut, ..., rel_cut) = (_[4], _[6]); pick the discriminating pair
    _a_pt, _m_pt, _mu_c, _mu_n, rel_nocut, _mu_k, rel_cut = max(
        check, key=lambda c: abs(c[4] - c[6]))
    return rel_cut < rel_nocut


CUBE_HAS_INNER_CUTOFF = _cube_has_inner_cutoff(CUBE_CHECK)

# Expected size of the inner cutoff's effect at each pair: the light grid point
# gains/loses a few percent of mu to it, the two heavy ones are untouched.
# Pinned as a band so BOTH a regression that removes the cutoff and one that
# inflates it are caught. Re-measured against v5
# (v5.0-night-m0p356mg-bcap10cm): 2.67e-2, 8.69e-6, 6.02e-6, which are the same
# numbers the v4 comparison gave from the other side (2.60e-2, 8.7e-6, 6.0e-6)
# -- as they must be, since it is one and the same displacement. Like the GATE 2
# bands, these were measured with the fixed-arrival-phase efficiency table and
# would have to be re-measured alongside a cube built on a different one.
CUBE_CUT_BAND = [(1.5e-2, 4.0e-2), (0.0, 1e-4), (0.0, 1e-4)]
if CUBE_CHECK is not None:
    for _i, (_a_pt, _m_pt, _mu_c, _mu_n, _rel_n, _mu_k, _rel_k) in \
            enumerate(CUBE_CHECK):
        # The arm built into the cube, and the arm displaced away from it.
        if CUBE_HAS_INNER_CUTOFF:
            _mu_par, _rel_par, _mu_off, _rel_off = _mu_k, _rel_k, _mu_n, _rel_n
            _par_txt, _off_txt = "R_eff inner cutoff ON", "inner cutoff OFF"
        else:
            _mu_par, _rel_par, _mu_off, _rel_off = _mu_n, _rel_n, _mu_k, _rel_k
            _par_txt, _off_txt = "inner cutoff OFF", "R_eff inner cutoff ON"
        # GATE 1a -- parity with the build that produced the cube
        assert _mu_c > 0 and _rel_par < 1e-4, (
            "cube parity failed at alpha_n=%.6e, m=%.6e: mu(%s)"
            "=%.6f vs cube mu=%.6f (rel %.3e)"
            % (_a_pt, _m_pt, _par_txt, _mu_par, _mu_c, _rel_par))
        # GATE 1b -- switching the inner cutoff to its other state moves mu away
        # from the cube by the pinned amount. This is the effect the figure is
        # about, so its size is asserted rather than merely displayed.
        _lo, _hi = CUBE_CUT_BAND[_i]
        assert _lo <= _rel_off <= _hi, (
            "inner-cutoff shift vs the cube is outside its pinned band at "
            "alpha_n=%.6e, m=%.6e (%s): %.3e not in [%.3e, %.3e]"
            % (_a_pt, _m_pt, _off_txt, _rel_off, _lo, _hi))

# ---------------------------------------------------------------------------
# PANEL A data -- dmu/dq for the three pairs, and the ratio strip beneath it
# ---------------------------------------------------------------------------
DMU_DQ = np.array([dmu_dq(a, m) for m, a in PAIRS])

WIN = (QS >= Q_LO) & (QS <= Q_HI)                 # the plotted window
DREF = DMU_DQ[K_REF]                              # degenerate reference anchor
assert np.all(DREF[WIN] > 0), "reference anchor has no support in the window"
RATIO = DMU_DQ / DREF                             # 1 exactly for k = K_REF


def _ratio_at(k, q):
    """log-interpolated RATIO[k] at an arbitrary q (grid-node independent)."""
    return 10.0 ** float(np.interp(np.log10(q), np.log10(QS),
                                   np.log10(np.maximum(RATIO[k], 1e-300))))


def _q_at_ratio(k, target):
    """The q at which RATIO[k] first falls to ``target`` (log-interpolated)."""
    below = np.where((RATIO[k] < target) & WIN)[0]
    if below.size == 0:
        return float("nan")
    j = below[0]
    r0, r1 = RATIO[k][j - 1], RATIO[k][j]
    w = (r0 - target) / (r0 - r1)
    return 10.0 ** float(np.log10(QS[j - 1]) * (1 - w) + np.log10(QS[j]) * w)


# the two numbers the figure asserts: the heavy pair is degenerate, the light
# pair is suppressed at the window edge by the R_eff endpoint
DEV_HEAVY = float(np.max(np.abs(RATIO[1][WIN] - 1.0)))
EDGE_RATIO = _ratio_at(0, Q_HI)
EDGE_SUPPRESSION = 1.0 / EDGE_RATIO
# where the light anchor departs from the degenerate reference
Q_DEP = {t: _q_at_ratio(0, t) for t in (0.99, 0.90, 0.50)}
# integrated: mu relative to the degenerate reference
MU_DEV = MU_PAIRS / MU_PAIRS[K_REF] - 1.0


def _spectrum_dev(log_m):
    """max |dmu/dq (m) / dmu/dq (reference) - 1| over the plotted window."""
    m = 10.0 ** log_m
    d = dmu_dq(ALPHA0_N * np.sqrt(m / MASSES[0]), m)
    return float(np.max(np.abs((d / DREF)[WIN] - 1.0)))


def _mu_dev(log_m):
    """|mu(m) / mu(reference) - 1| on the degeneracy line."""
    m = 10.0 ** log_m
    return abs(mu_of(ALPHA0_N * np.sqrt(m / MASSES[0]), m, XS_UNCAPPED)
               / MU_PAIRS[K_REF] - 1.0)


def _threshold_mass(dev_fn, tol):
    """Lightest mass on the line whose deviation from exact degeneracy < tol."""
    return 10.0 ** brentq(lambda x: np.log(dev_fn(x) / tol), 10.0, 15.0,
                          xtol=2e-3)


# the domain of validity, root-solved rather than eyeballed. Two statements: the
# SPECTRUM (what the optimum interval sees) and the integrated mu.
DEG_M_SPECTRUM = _threshold_mass(_spectrum_dev, 1e-3)
DEG_M_SPECTRUM_4 = _threshold_mass(_spectrum_dev, 1e-4)
DEG_M_MU = _threshold_mass(_mu_dev, 1e-3)

# ---------------------------------------------------------------------------
# VALIDATION GATE 2 -- the degeneracy where it holds, its breaking where it does
# not. Both directions are pinned: a regression that restores perfect degeneracy
# (inner cutoff lost) fails the breaking gates, and one that deforms the heavy
# anchors fails the degeneracy gate.
#
# Every band below was measured with the fixed-arrival-phase (w = 1) efficiency
# table, and the anchor is root-solved through eps(q), so all of them move with
# the table: under the canonical arrival-phase-marginalised curves the anchor
# coupling rises and (b) and (c) fail. That is the intended behaviour -- these
# gates are what stops an unpinned rerun from quietly redrawing the figure with
# a different efficiency. See the module docstring for the pin.
# ---------------------------------------------------------------------------
# (a) it HOLDS for the heavy pair: alpha^2/m exact to the v-quadrature floor
assert DEV_HEAVY < 1e-4, \
    "m=1e14 vs 1e18 not degenerate over the window: %.3e" % DEV_HEAVY
assert abs(MU_DEV[1]) < 1e-4, \
    "m=1e14 vs 1e18 mu not degenerate: %.3e" % MU_DEV[1]

# (b) it is BROKEN for the light pair, by the amount the R_eff endpoint predicts
assert 12.0 < EDGE_SUPPRESSION < 17.0, (
    "light-anchor edge suppression at q = %g GeV outside its pinned band: "
    "%.3fx (want 12-17x)" % (Q_HI, EDGE_SUPPRESSION))
assert -2.3e-2 < MU_DEV[0] < -1.5e-2, (
    "light-anchor mu deviation outside its pinned band: %.4e (want -1.9e-2)"
    % MU_DEV[0])
# the departure sets in below q_max(v_esc), where the fast half of the halo has
# already been cut away: pin that ratio too, so a change in the halo or in
# coulomb_q_max cannot slide the two apart unnoticed
DEP_FRAC = Q_DEP[0.99] / Q_MAX_VESC[0]
assert 0.55 < DEP_FRAC < 0.80, (
    "the 1%% departure point is no longer just below q_max(v_esc): "
    "q_dep/q_max = %.3f" % DEP_FRAC)
assert Q_LO < Q_MAX_VESC[0] < Q_HI, (
    "the lightest anchor's endpoint has left the plotted window: %.3e GeV"
    % Q_MAX_VESC[0])
assert Q_MAX_VESC[1] > 10 * Q_HI and Q_MAX_VESC[2] > 10 * Q_HI, (
    "a heavy anchor's endpoint has entered the window: %s" % Q_MAX_VESC)

# (c) the validity domain itself, pinned to +-25% in mass. These three masses
# are the ones printed in the figure's domain-of-validity line, so they are
# pinned to the table the caption was written from; on the canonical table the
# two spectrum thresholds land at 4.18e12 and 2.22e13 GeV instead.
for _name, _val, _want in (("spectrum, 1e-3", DEG_M_SPECTRUM, 6.54e12),
                           ("spectrum, 1e-4", DEG_M_SPECTRUM_4, 3.52e13),
                           ("mu, 1e-3", DEG_M_MU, 1.97e11)):
    assert 0.75 < _val / _want < 1.25, (
        "degeneracy validity threshold (%s) moved: %.3e vs pinned %.3e"
        % (_name, _val, _want))

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

# THE INNER CUTOFF, SEEN IN PANEL B. The lightest hump is amputated at b = R_eff
# (the scan starts there and mu(<R_eff) = 0 by construction). The heavy anchors'
# equivalent inner region sits at 100^k R_eff, and the share of mu they carry
# below it is exactly the mu the light anchor is missing in panel A.
AMPUTATED = np.array([float(np.interp(np.log10(R_EFF * 100.0 ** k), LB, FRAC[k]))
                      for k in range(3)])

# the plotted extent of each hump: the panel draws the curve only where it is
# meaningfully nonzero (see B_FLOOR_*), and that drawn region is what the
# translate / smoothness gates below have to be evaluated on.
_pb = (B_SCAN >= B_LO * 0.5) & (B_SCAN <= B_HI * 2)
LIVE = np.array([_pb & (y > max(B_FLOOR_FRAC * y.max(), B_FLOOR_ABS))
                 for y in DMU_DLOGB])

# VALIDATION GATE 3a -- mu(<b) plateaus at the uncapped mu, and the three
# plateaus differ by exactly the inner-cutoff deficit measured in panel A
assert np.all(np.abs(PLATEAU / MU_PAIRS_B - 1.0) < 1e-6), \
    "capped-at-infinity mu != uncapped mu: %s vs %s" % (PLATEAU, MU_PAIRS_B)
assert np.all(Q_GRID_SHIFT < 1e-2), \
    "refined q grid moves mu: %s vs %s" % (MU_PAIRS_B, MU_PAIRS)
PLATEAU_DEV = PLATEAU / PLATEAU[K_REF] - 1.0
assert abs(PLATEAU_DEV[1]) < 1e-4, \
    "heavy plateaus not degenerate: %.3e" % PLATEAU_DEV[1]
assert abs(PLATEAU_DEV[0] / MU_DEV[0] - 1.0) < 1e-2, (
    "panel B plateau deficit (%.4e) disagrees with panel A mu deficit (%.4e)"
    % (PLATEAU_DEV[0], MU_DEV[0]))
assert abs(PLATEAU[0] / MU_TARGET - 1.0) < 1e-2, \
    "the anchored (lightest) plateau is off mu = 3: %.4f" % PLATEAU[0]

# VALIDATION GATE 3b -- the amputation identity: the mu the light anchor loses to
# the inner cutoff == the mu the heavy anchors carry below the scaled R_eff
for k in (1, 2):
    assert abs(AMPUTATED[k] / abs(MU_DEV[0]) - 1.0) < 2e-2, (
        "amputation identity fails for pair %d: mu below 100^%d R_eff = %.4e, "
        "light-anchor deficit = %.4e" % (k, k, AMPUTATED[k], abs(MU_DEV[0])))
assert AMPUTATED[0] == 0.0, \
    "the scan does not start at R_eff: mu(<R_eff) = %.3e" % AMPUTATED[0]

# VALIDATION GATE 4a -- the three mu(<b) curves are horizontal translates by x100
# ONCE THE AMPUTATION IS ACCOUNTED FOR. Compared in absolute mu (not normalised),
# which is the exact statement:
#     mu_k(<b) - mu_k(<100^k R_eff)  ==  mu_0(<b / 100^k)  ,
# i.e. the heavy anchors' distribution with its inner part removed IS the light
# anchor's. Restricted to b >= 100^k R_eff, where the light anchor exists at all.
# The shift is an exact multiple of the log-grid spacing, so np.interp lands on
# nodes and adds no interpolation error.
TRANSLATE_DEV = []
for k in (1, 2):
    valid = (LB - 2.0 * k) >= LB[0]
    offset = float(np.interp(np.log10(R_EFF * 100.0 ** k), LB, MU_CUM[k]))
    shifted = np.interp(LB - 2.0 * k, LB, MU_CUM[0])
    dev = float(np.max(np.abs(MU_CUM[k][valid] - offset - shifted[valid]))
                / MU_TARGET)
    TRANSLATE_DEV.append(dev)
    assert dev < 1e-3, \
        "mu(<b) not a x100 translate for pair %d: %.3e of mu" % (k, dev)

# VALIDATION GATE 4b -- the same statement on the curves that are actually
# DRAWN. The cumulative test above is blind to shape: it compares smooth,
# monotone functions, so quadrature noise that is invisible in mu(<b) can still
# be a visible shoulder once d/dlog10 b divides it by the 0.025-decade node
# spacing. This gate is therefore taken directly on the plotted derivative, in
# plot units, over the drawn (live) region of each hump that has a light-anchor
# counterpart (b >= 100^k R_eff; below that the light hump is amputated, which
# is the point of GATE 3b, not a shape error).
DERIV_DEV = []
for k in (1, 2):
    valid = LIVE[k] & ((LB - 2.0 * k) >= LB[0])
    shifted = np.interp(LB - 2.0 * k, LB, DMU_DLOGB[0])
    dev = float(np.max(np.abs(DMU_DLOGB[k][valid] - shifted[valid])))
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

# VALIDATION GATE 5 -- the three-act story, asserted
assert RET_FRAC[0] > 0.95, "pair 0 should be untouched by the cap: %.4f" % RET_FRAC[0]
assert 0.05 < RET_FRAC[1] < 0.80, "pair 1 should be partially cut: %.4f" % RET_FRAC[1]
assert RET_FRAC[2] < 1e-3, "pair 2 should be annihilated: %.4e" % RET_FRAC[2]

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
C_END = "#b5651d"                  # the R_eff endpoint q_max and its band

# panel A stacking: very thick light solid under medium dashed under thin dark
# dotted. At 140 dpi this leaves ~3 px of each stratum visible on either side,
# so the coincidence reads as three curves rather than one fat one.
A_LW = [7.6, 4.2, 1.8]
A_LS = ["-", "--", ":"]
A_Z = [3, 4, 5]

fig = plt.figure(figsize=(13.6, 6.6))
_gs = fig.add_gridspec(2, 2, height_ratios=[2.55, 1.0], width_ratios=[1.0, 1.0])
axA = fig.add_subplot(_gs[0, 0])
axR = fig.add_subplot(_gs[1, 0], sharex=axA)
axB = fig.add_subplot(_gs[:, 1])


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


def _white(t):
    t.set_path_effects([pe.Stroke(linewidth=2.6, foreground="w"), pe.Normal()])
    return t


# ===========================================================================
# PANEL A -- the same spectrum, until the sensor-radius endpoint
# ===========================================================================
# the band the lightest anchor's endpoint has already eaten into
for _ax in (axA, axR):
    _ax.axvspan(Q_MAX_VESC[0], Q_HI, color=C_END, alpha=0.07, lw=0, zorder=0)
    _ax.axvline(Q_MAX_VESC[0], color=C_END, ls=(0, (5, 2)), lw=1.6, zorder=1)

for k, (m, a) in enumerate(PAIRS):
    am, ae = _mant_exp(a)
    lbl = (r"$m=10^{%d}$ GeV,  $\alpha_n=%.3f\times10^{%d}$,  $\mu=%.2f$"
           % (int(round(np.log10(m))), am, ae, MU_PAIRS[k]))
    axA.plot(QS, DMU_DQ[k], color=C_PAIR[k], lw=A_LW[k], ls=A_LS[k],
             zorder=A_Z[k], solid_capstyle="round", dash_capstyle="round",
             label=lbl)

# q^-3 guide through the falling tail
_iq = int(np.argmax(DMU_DQ[K_REF]))
_qg = np.geomspace(QS[_iq] * 3.0, Q_HI, 50)
_yg = DMU_DQ[K_REF][_iq] * (_qg / QS[_iq]) ** -3 * 2.6
axA.plot(_qg, _yg, color=C_GUIDE, lw=1.2, ls=(0, (1, 2)), zorder=2)
# labelled near the FAR end of the guide: the explanation box now reaches down
# to ~2e-4 over the middle of the window and would swallow a mid-guide label
_white(axA.text(_qg[38], _yg[38] * 2.4, r"$q^{-3}$", color=C_GUIDE, fontsize=10,
                ha="center", va="bottom"))

# efficiency turn-on marker, anchored on the steep rise
_ymark = 1e-7
_pos = DMU_DQ[K_REF] > 0
_qmark = 10 ** float(np.interp(np.log10(_ymark),
                               np.log10(DMU_DQ[K_REF][_pos][:_iq]),
                               np.log10(QS[_pos][:_iq])))
axA.annotate("mode-1 efficiency\nturn-on  $\\varepsilon(q)$",
             xy=(_qmark, _ymark), xytext=(1.12e2, 4e-6),
             color=C_GUIDE, fontsize=9, ha="left", va="center",
             arrowprops=dict(arrowstyle="->", color=C_GUIDE, lw=1.0,
                             shrinkA=4, shrinkB=3))

# The endpoint itself. Placed low and to the right of its own guide line: the
# only part of panel A that is free of both the falling tail and the explanation
# box is the wedge under the q^-3 tail, right of q_max. The full-height dashed
# guide carries the association, so the label needs no leader.
_white(axA.text(Q_MAX_VESC[0] * 1.10, 1.2e-10,
                r"$q_{\max}(v_{\mathrm{esc}})=2\alpha/"
                r"(v_{\mathrm{esc}}R_{\mathrm{eff}})$"
                "\n"
                r"$=%s$ GeV  at  $m=10^{10}$" % _sci(Q_MAX_VESC[0], 2),
                color=C_END, fontsize=9, ha="left", va="center", zorder=6))

axA.text(0.52, 0.975,
         r"$d\mu/dq\propto(\alpha^{2}/m)\,q^{-3}\varepsilon(q)$:"
         r" invariant along $\alpha\propto\sqrt{m}$," "\n"
         r"but only below the endpoint"
         r" $q_{\max}(v)=2\alpha/(v R_{\mathrm{eff}})$," "\n"
         r"and $q_{\max}\propto\alpha\propto\sqrt{m}$, not $\alpha^{2}/m$."
         "\n"
         r"$\Rightarrow$ the lightest anchor's endpoint is in the window.",
         transform=axA.transAxes, fontsize=8.5, ha="left", va="top",
         color="#222222",
         bbox=dict(boxstyle="round,pad=0.40", fc="white", ec="#cccccc",
                   alpha=0.95))

axA.set_xscale("log")
axA.set_yscale("log")
axA.set_xlim(Q_LO, Q_HI)
axA.set_ylim(1e-13, 1e-1)
axA.tick_params(labelbottom=False)
axA.set_ylabel(r"$d\mu/dq$  [GeV$^{-1}$]   (detected, mode 1)")
axA.set_title("A.  The same detected spectrum -- until the "
              "$R_{\\mathrm{eff}}$ endpoint")
axA.legend(loc="lower left", fontsize=8.5, framealpha=0.94,
           title="three points on the degeneracy line "
                 "$\\alpha\\propto\\sqrt{m}$",
           title_fontsize=9)
axA.grid(True, which="major", color="#e9e9e9", lw=0.7)

# ===========================================================================
# RATIO STRIP -- the breaking, made explicit
# ===========================================================================
axR.axhline(1.0, color=C_PAIR[K_REF], lw=1.8, ls=":", zorder=4)
for k in (0, 1):
    axR.plot(QS, RATIO[k], color=C_PAIR[k], lw=A_LW[k] * 0.55, ls=A_LS[k],
             zorder=A_Z[k], solid_capstyle="round", dash_capstyle="round")

_white(axR.text(1.25e2, 0.30,
                r"$10^{14}$ vs $10^{18}$:  flat to $%s$"
                % _sci(DEV_HEAVY, 1),
                color=C_PAIR[1], fontsize=8.8, ha="left", va="center"))
axR.annotate(r"$\times\,1/%.0f$" % EDGE_SUPPRESSION,
             xy=(Q_HI * 0.985, EDGE_RATIO), xytext=(9.0e3, 0.115),
             color=C_PAIR[0], fontsize=10, ha="center", va="top",
             arrowprops=dict(arrowstyle="->", color=C_PAIR[0], lw=1.2,
                             shrinkA=3, shrinkB=2), zorder=7)

axR.set_xscale("log")
axR.set_yscale("log")
axR.set_xlim(Q_LO, Q_HI)
axR.set_ylim(0.05, 3.0)
axR.set_yticks([0.1, 1.0])
axR.set_yticklabels(["0.1", "1"])
axR.set_xlabel("momentum transfer  $q$  [GeV]")
axR.set_ylabel("ratio to\n$m=10^{18}$", fontsize=10)
axR.grid(True, which="major", color="#e9e9e9", lw=0.7)

# ===========================================================================
# PANEL B -- ... and assembled from ever-larger impact parameters
# ===========================================================================
Y_TOP = 6.0

# removed-by-cap band, and the region no flyby can reach
axB.axvspan(B_CAP, B_HI * 2, color=C_RED, alpha=0.09, lw=0, zorder=0,
            label=r"removed by the cap  ($b>b_{\mathrm{cap}}$)")
# labelled in the band rather than in the legend: a fifth legend entry pushes the
# legend box down onto the m = 1e14 mass tag, and the band is 1.1 empty decades
axB.axvspan(B_LO * 0.5, R_EFF, color=C_REFF, alpha=0.08, lw=0, zorder=0)
_white(axB.text(np.sqrt(B_LO * R_EFF), 3.05,
                r"no flyby:  $b<R_{\mathrm{eff}}$", color=C_REFF, fontsize=8.5,
                ha="center", va="center", rotation=90, zorder=6))

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
    _white(axB.text(_tx, _ty,
                    r"$m=10^{%d}$ GeV" % int(round(np.log10(MASSES[k]))),
                    color=C_PAIR[k], fontsize=9.5, ha=_ha, va="bottom",
                    zorder=7))

# neutral legend proxy for the b_95 markers (they are coloured per pair)
axB.plot([], [], marker="v", ms=6.5, ls="none", color=C_NEUTRAL,
         label=r"$b_{95}$ (95% of $\mu$ accumulated)")

# overlays: sensor radius and the production cap
axB.axvline(R_EFF, color=C_REFF, ls="--", lw=1.6, zorder=3,
            label=r"$R_{\mathrm{eff}}=%.0f\,\mu$m  (sensor radius)"
                  % (R_EFF * 1e6))
axB.axvline(B_CAP, color=C_RED, ls="-.", lw=1.9, zorder=3,
            label=r"$b_{\mathrm{cap}}=b_{\mathrm{constrained\,max}}=10$ cm")

# THE SAME BREAKING, SEEN HERE: the lightest hump is cut off at R_eff, and the
# slice it loses is the 1.9% of mu missing from panel A's light spectrum.
axB.annotate(r"$-%.1f\%%$ of $\mu$" % (100 * abs(MU_DEV[0])),
             xy=(R_EFF, float(DMU_DLOGB[0][0]) * 0.75),
             xytext=(R_EFF * 0.85, 1.55),
             color=C_PAIR[0], fontsize=9.0, ha="right", va="center",
             arrowprops=dict(arrowstyle="->", color=C_PAIR[0], lw=1.2,
                             shrinkA=3, shrinkB=2), zorder=8)

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
    _white(axB.text(np.sqrt(lo * hi), y_arr + 0.14, r"$\times100$",
                    color="#555555", fontsize=9.5, ha="center", va="bottom",
                    zorder=7))

# Read-out block: b_95 and the mu surviving the cap for each pair.
# It lives in the empty corridor BETWEEN the two full-height guides, so
# every line has to stay narrower than the ~200 px (at figure dpi) from the
# R_eff dashes to the b_cap line -- hence no "b_95 =" / "mu_kept =" repeated on
# each row and no parenthetical percentages. Enforced by the renderer-bbox gate
# after tight_layout (the layout is only final there).
# Offset from the R_eff guide rather than an absolute axes fraction: the guide
# sits at config.R_EFF, so a hard-coded x silently drifts into it when the sensor
# radius changes (it did, at 200 -> 260 um). 0.0181 is the clearance the layout
# was tuned with, now measured from wherever the guide actually is.
READOUT_X = (np.log10(R_EFF / B_LO) / np.log10(B_HI / B_LO)) + 0.0181
READOUT_Y, READOUT_DY = 0.975, 0.058
_lines = [(r"$b_{95}$,   $\mu(<b_{\mathrm{cap}})$:", "#222222", 8.6)]
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
axB.set_title(r"B.  $\dots$ assembled from ever-larger impact parameters")
axB.legend(loc="upper right", fontsize=8.4, framealpha=0.94,
           borderpad=0.5)
axB.grid(True, which="major", color="#e9e9e9", lw=0.7)

# ---------------------------------------------------------------------------
# parameter box, caption + suptitle
#
# The first slot is fed Q_MIN, so the committed renders print "q_th = 100 GeV".
# Read it as the q grid's lower end, not as the analysis threshold: the analysis
# window opens at 1 TeV (config.Q_THRESH), and 100 GeV is the reconstruction
# threshold of the stored candidate lists. The label is left as drawn because
# relabelling it means regenerating the three committed renders, which cannot
# be done without pinning the efficiency table (module docstring);
# notebooks/README.md flags the same thing for readers of the figure.
# ---------------------------------------------------------------------------
_am, _ae = _mant_exp(ALPHA0_N)
param_txt = (
    "$q_{\\mathrm{th}}=%g$ GeV      $T=%.3f\\times10^{6}$ s      "
    "$f_X=%g$      $R_{\\mathrm{eff}}=%.0f\\,\\mu$m      "
    "anchor  $\\alpha_n=%.3f\\times10^{%d}$ at $m=10^{%d}$ GeV, "
    "$\\times100$ per $\\times10^{4}$ in $m$"
    % (Q_MIN, T_TOTAL / 1e6, config.F_X, R_EFF * 1e6, _am, _ae,
       int(round(np.log10(MASSES[0]))))
)
fig.text(0.5, 0.062, param_txt, ha="center", va="bottom", fontsize=9,
         family="monospace", color="#222222")

domain_txt = (
    "DOMAIN OF VALIDITY along $\\alpha\\propto\\sqrt{m}$:  the spectra coincide "
    "to $<10^{-3}$ across $10^{2}$-$3\\times10^{4}$ GeV for "
    "$m\\gtrsim%s$ GeV ($\\mu$ to $<10^{-3}$ for $m\\gtrsim%s$ GeV); "
    "below that the $R_{\\mathrm{eff}}$ endpoint breaks the degeneracy "
    "-- by %.1f%% in $\\mu$ at $m=10^{10}$ GeV, which does not close the "
    "island; the $b_{\\mathrm{cap}}$ of panel B does."
    % (_sci(DEG_M_SPECTRUM, 1), _sci(DEG_M_MU, 1), 100 * abs(MU_DEV[0]))
)
fig.text(0.5, 0.014, domain_txt, ha="center", va="bottom", fontsize=8.6,
         color="#222222")

fig.suptitle(r"The $\alpha^{2}/m$ degeneracy of the long-range ($1/r^{2}$) "
             r"interaction: broken at low mass by the sensor radius, "
             r"closed at high mass by the impact-parameter cap",
             fontsize=13.5, y=0.985)

fig.tight_layout(rect=(0, 0.075, 1, 0.952), h_pad=0.6)

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
print("=" * 78)
print("Q GRIDS   GATE 1 + panel A: production n_q = %d   |   panel-B cap scan:"
      " n_q = %d  (same span)" % (FID["n_q"], N_Q_PANEL_B))
print()
print("GATE 1  %s cube cross-check at the cube's own 10 cm cap"
      " (noatm, mode 1, massless slice)" % (CUBE_TAG or "release"))
print("  cube: %s" % CUBE)
if CUBE_CHECK is None:
    # This is the path taken today: CUBE is the internal capped parent cube,
    # which is not in the tree. The printed line below predates the split into
    # the current tracked two-file release -- those files are here, but they are
    # uncapped and so cannot serve this gate. Left as printed, since the string
    # is part of the frozen run; see the CUBE comment above for what is true.
    print("  SKIPPED (release file not present).")
    print("  release/*.h5 is gitignored and distributed via Zenodo; the figure")
    print("  is unaffected -- every curve is computed from luhdm directly.")
else:
    _par_lbl = ("R_eff = %.0f um" % (R_EFF * 1e6)) if CUBE_HAS_INNER_CUTOFF \
        else "R_eff = 0"
    _off_lbl = "R_eff = 0" if CUBE_HAS_INNER_CUTOFF \
        else ("R_eff = %.0f um" % (R_EFF * 1e6))
    print("  this cube was built with the R_eff inner cutoff %s (detected from"
          % ("ON" if CUBE_HAS_INNER_CUTOFF else "OFF"))
    print("  the file: whichever arm reproduces it, not from its name)")
    print("  1a  parity: cube == this pipeline at %s" % _par_lbl)
    print("  1b  flipping the inner cutoff (this figure's subject) to %s then"
          % _off_lbl)
    print("      moves mu away from the cube by the pinned amount -- nonzero on")
    print("      the light point, negligible on the two heavy ones.")
    for _i, (_a_pt, _m_pt, _mu_c, _mu_n, _rel_n, _mu_k, _rel_k) in \
            enumerate(CUBE_CHECK):
        _lo, _hi = CUBE_CUT_BAND[_i]
        _mu_par, _rel_par = (_mu_k, _rel_k) if CUBE_HAS_INNER_CUTOFF \
            else (_mu_n, _rel_n)
        _mu_off, _rel_off = (_mu_n, _rel_n) if CUBE_HAS_INNER_CUTOFF \
            else (_mu_k, _rel_k)
        print("  grid point   alpha_n = %.9e   m = %.9e GeV" % (_a_pt, _m_pt))
        print("    1a  mu cube = %.6e   mu (%s) = %.6e   rel = %.3e"
              "   (tol 1e-4)   PASS" % (_mu_c, _par_lbl, _mu_par, _rel_par))
        print("    1b  mu (%s) = %.6e   rel to cube = %.3e"
              "   (band %.1e..%.1e)   PASS"
              % (_off_lbl, _mu_off, _rel_off, _lo, _hi))
print()
print("ANCHOR  brentq on log10 alpha_n at m = %.0e for mu = %.1f  ->"
      "  alpha_n = %.6e" % (MASSES[0], MU_TARGET, ALPHA0_N))
print("        (a single alpha^2 rescale is NOT usable: the R_eff endpoint moves"
      " with alpha)")
print()
print("DEGENERACY-LINE POINTS")
for k, (m, a) in enumerate(PAIRS):
    print("  m = %.3e GeV   alpha_n = %.6e   alpha = %.4e   mu = %.6f"
          % (m, a, a * config.N_NEUTRONS, MU_PAIRS[k]))
    print("      q_max(v_esc) = %.4e GeV   end of support = %.4e GeV   "
          "(window %.0e-%.0e GeV)"
          % (Q_MAX_VESC[k], Q_SUPPORT_END[k], Q_LO, Q_HI))
print()
print("GATE 2  the degeneracy where it holds, and its breaking where it does not")
print("  2a HOLDS   m=1e14 vs 1e18, max |ratio-1| over the window : %.3e"
      "   (tol 1e-4)   PASS" % DEV_HEAVY)
print("             m=1e14 vs 1e18, mu                            : %.3e"
      "   (tol 1e-4)   PASS" % MU_DEV[1])
print("  2b BROKEN  m=1e10 vs 1e18, dmu/dq at q = %.0e GeV        : ratio %.5f"
      "  = 1/%.2f   (band 12-17x)   PASS" % (Q_HI, EDGE_RATIO, EDGE_SUPPRESSION))
print("             m=1e10 vs 1e18, integrated mu                  : %.4e"
      "   (band -1.5e-2..-2.3e-2)   PASS" % MU_DEV[0])
print("             the light ratio falls to 0.99 / 0.90 / 0.50 at"
      " q = %.3e / %.3e / %.3e GeV"
      % (Q_DEP[0.99], Q_DEP[0.90], Q_DEP[0.50]))
print("             q(1%% departure) / q_max(v_esc) = %.3f"
      "   (band 0.55-0.80)   PASS" % DEP_FRAC)
print("  2c DOMAIN  spectra degenerate to <1e-3 for m > %.3e GeV"
      "   (pinned 6.54e12 +-25%%)   PASS" % DEG_M_SPECTRUM)
print("             spectra degenerate to <1e-4 for m > %.3e GeV"
      "   (pinned 3.52e13 +-25%%)   PASS" % DEG_M_SPECTRUM_4)
print("             mu      degenerate to <1e-3 for m > %.3e GeV"
      "   (pinned 1.97e11 +-25%%)   PASS" % DEG_M_MU)
print("             mechanism: q_max = 2 alpha/(v_esc R_eff) propto alpha propto"
      " sqrt(m),")
print("             so the endpoint slides INTO the window as m falls; it is not"
      " an alpha^2/m quantity.")
print()
print("GATE 3  mu(<b) plateau, and the amputation identity across the two panels")
for k, (m, a) in enumerate(PAIRS):
    print("  m = %.0e   plateau mu(<inf) = %.6f   == uncapped mu (n_q=%d) = %.6f"
          "   dev to reference %.3e" % (m, PLATEAU[k], N_Q_PANEL_B,
                                        MU_PAIRS_B[k], PLATEAU_DEV[k]))
print("  q-grid refinement moves mu by %s  (tolerance 1e-2)   PASS"
      % "  ".join("%.2e" % s for s in Q_GRID_SHIFT))
print("  panel-B plateau deficit %.4e  vs  panel-A mu deficit %.4e"
      "   (agree to %.2e, tol 1e-2)   PASS"
      % (PLATEAU_DEV[0], MU_DEV[0], abs(PLATEAU_DEV[0] / MU_DEV[0] - 1.0)))
print("  mu below the sqrt(m)-scaled sensor radius 100^k R_eff, per pair:")
print("    %s   (k=1,2 must equal the light anchor's %.4e deficit; k=0 is 0 by"
      " construction)   PASS"
      % ("  ".join("%.4e" % f for f in AMPUTATED), abs(MU_DEV[0])))
print()
print("GATE 4  mu(<b) curves are x100 horizontal translates of the light one,")
print("        once its amputated inner part is added back")
for k, dev in zip((1, 2), TRANSLATE_DEV):
    print("  cumulative, pair %d: max |mu_k(<b) - mu_k(<100^%d R_eff)"
          " - mu_0(<b/100^%d)| = %.3e of mu  (tolerance 1e-3)   PASS"
          % (k, k, k, dev))
for k, dev in zip((1, 2), DERIV_DEV):
    print("  AS DRAWN,  hump %d vs hump 0 shifted by 10^%d, over the live"
          " region with b >= 100^%d R_eff : max dev %.4f plot units"
          "  (tolerance 2e-2)   PASS" % (k, 2 * k, k, dev))
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
print("=" * 78)
