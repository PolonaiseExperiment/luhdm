"""Two-panel figure explaining the impact-parameter cap in the cross section.

HISTORICAL. The v7 data release has NO impact-parameter cap: the b-integral
runs to infinity and ``b_constrained_max_m`` is NaN in both released cubes. The
hardware scale this figure is about now enters only after the calculation, as
the 10 cm aperture of the post-hoc flux cut ``m_cut`` (release/README.md
section 5.4). Keep this figure as the explainer for the capped scheme used up
to v6 -- it is why that spectrum had a plateau, and notebook 09 is its
companion -- not as a description of the current release.

This uses the v6 production cap b_constrained_max = 0.1 m (10 cm) and the
MASSLESS (Coulomb) mediator slice, where the cap actually bites. The massless
flyby reach is the clean power law

    b_max(q) = 2 alpha / (q v) / conv_m2pGeV(1)      [m]   (slope -1 in log-log),

and the cap replaces the outer edge of the impact-parameter integral, b_max(q),
with min(b_max(q), b_constrained_max). Panel A draws the integration domain in
the (q, b) plane: the flyby reach b_max(q) falls with q, the sensor radius
R_eff is the inner cutoff, and the horizontal 10 cm cap slices off the large-b
wedge whenever b_max(q) > b_cap -- i.e. for q below the crossover q* where
b_max(q*) = b_cap. Panel B shows the consequence: dsigma/dq is untouched for
q > q* (the cap sits outside the reach) and is suppressed below q*, where the
discarded large-b flybys no longer contribute.

This is the massless/Coulomb slice at the real 10 cm cap. Finite-lambda slices
are affected only where the reach exceeds 10 cm (strong coupling / long range).
"""

import os
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from luhdm import cross_section as xs
from luhdm import config, units

# ---------------------------------------------------------------------------
# Massless (Coulomb) example at the real production cap.
# alpha_n and v chosen so the crossover q* (where b_max(q*) = 0.1 m) lands in
# the middle of the plotted q window -> q* ~ 1.7e3 GeV (log-midpoint of the
# decade range). q* = 2 alpha_n N_n / (v b_cap conv(1)) is linear in N_n, so
# alpha_n has to be retuned whenever the magnet neutron count changes; 4.13e-6
# is the v3 value (N_n = 1.07e20).
# ---------------------------------------------------------------------------
ALPHA_N = 4.13e-6                      # coupling per neutron
ALPHA = ALPHA_N * config.N_NEUTRONS    # full coupling alpha = alpha_n * N_neutrons
V = 1e-3                               # incoming speed [c]
R_EFF = config.R_EFF                   # sensor radius [m] (260 um), inner floor
B_CAP = 0.1                            # b_constrained_max [m] = 10 cm (production)

Q_LO, Q_HI = 1e2, 3e4                  # plotted q window [GeV]

_CONV1 = units.conv_m2pGeV(1.0)        # 1 m in GeV^-1


def b_max_coulomb(q):
    """Massless flyby reach b_max(q) = 2 alpha / (q v) / conv(1)  [m]."""
    return 2.0 * ALPHA / (np.asarray(q, dtype=float) * V) / _CONV1


# ---------------------------------------------------------------------------
# Lock the numbers
# ---------------------------------------------------------------------------
q_grid = np.geomspace(Q_LO, Q_HI, 600)
b_max = b_max_coulomb(q_grid)

# crossover q* where b_max(q*) = B_CAP (closed form for the power law)
Q_STAR = 2.0 * ALPHA / (V * B_CAP * _CONV1)

b_max_lo = float(b_max_coulomb(Q_LO))
b_max_hi = float(b_max_coulomb(Q_HI))

# ---------------------------------------------------------------------------
# Panel B: dsigma/dq, uncapped vs capped (massless/Coulomb projection)
# ---------------------------------------------------------------------------
dsig_uncap = xs.cross_section_rutherford_projection(q_grid, ALPHA, V)
dsig_cap = xs.cross_section_rutherford_projection_capped(q_grid, ALPHA, V, B_CAP)

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

C_REACH = plt.cm.viridis(0.18)     # b_max(q) reach curve
C_KEPT = plt.cm.viridis(0.62)      # kept-domain fill
C_UNCAP = plt.cm.viridis(0.20)     # dsigma uncapped
C_CAPPED = plt.cm.viridis(0.60)    # dsigma capped
C_REFF = "#444444"                 # inner cutoff
C_RED = "#d62728"                  # cap line + removed wedge (semantically apt)
C_GUIDE = "#555555"                # q* guide

fig, (axA, axB) = plt.subplots(1, 2, figsize=(12.6, 5.3))

# ===========================================================================
# PANEL A -- integration domain in the (q, b) plane
# ===========================================================================
b_inner = np.full_like(q_grid, R_EFF)
b_outer = np.minimum(b_max, B_CAP)   # min(b_max(q), b_constrained_max)

# kept integration domain: R_eff <= b <= min(b_max, b_cap)
axA.fill_between(q_grid, b_inner, b_outer, color=C_KEPT, alpha=0.28,
                 lw=0, zorder=1,
                 label=r"kept domain  $R_{\mathrm{eff}}\leq b\leq"
                       r"\min(b_{\max},\,b_{\mathrm{cap}})$")

# removed wedge: b_cap < b <= b_max(q), only where b_max > b_cap (q < q*)
removed = b_max > B_CAP
axA.fill_between(q_grid, np.full_like(q_grid, B_CAP), b_max, where=removed,
                 facecolor="none", edgecolor=C_RED, hatch="////", lw=0.0,
                 alpha=0.9, zorder=2,
                 label=r"removed by cap  $b_{\mathrm{cap}}<b\leq b_{\max}(q)$")

# reach curve b_max(q): clean power law, slope -1 in log-log
axA.plot(q_grid, b_max, color=C_REACH, lw=2.4, zorder=5,
         label=r"$b_{\max}(q)=2\alpha/(q\,v)$  (Coulomb reach)")

# inner cutoff R_eff (dashed) and cap (dash-dot, red)
axA.axhline(R_EFF, color=C_REFF, ls="--", lw=1.6, zorder=4,
            label=r"$R_{\mathrm{eff}}=260\,\mu$m  (inner cutoff)")
axA.axhline(B_CAP, color=C_RED, ls="-.", lw=1.8, zorder=4,
            label=r"$b_{\mathrm{cap}}=b_{\mathrm{constrained\,max}}=10$ cm")

# q* guide
for ax in (axA, axB):
    ax.axvline(Q_STAR, color=C_GUIDE, ls=":", lw=1.5, zorder=3)

txt = axA.text(0.30, 0.96, "cap engages\nfor $q<q_*$", transform=axA.transAxes,
               color=C_GUIDE, fontsize=10, ha="center", va="top")
txt.set_path_effects([pe.Stroke(linewidth=2.4, foreground="w"), pe.Normal()])
txt = axA.text(Q_STAR * 0.82, 0.36, "$q_*$", color=C_GUIDE,
               fontsize=11, ha="right", va="center")
txt.set_path_effects([pe.Stroke(linewidth=2.4, foreground="w"), pe.Normal()])

# code connection note (massless / Coulomb branch)
axA.text(0.035, 0.045,
         r"outer limit $=\min(b_{\max}(q),\,b_{\mathrm{cap}})$" "\n"
         r"massless:  $b_{\max}(q)=2\alpha/(q\,v)$,   $b_{\mathrm{cap}}=0.1$ m",
         transform=axA.transAxes, fontsize=8.5, ha="left", va="bottom",
         family="monospace", color="#333333",
         bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="#cccccc",
                   alpha=0.9))

axA.set_xscale("log")
axA.set_yscale("log")
axA.set_xlim(Q_LO, Q_HI)
axA.set_ylim(R_EFF * 0.8, b_max_lo * 1.6)
axA.set_xlabel("momentum transfer  $q$  [GeV]")
axA.set_ylabel("impact parameter  $b$  [m]")
axA.set_title("A.  Integration domain in the $(q,\\,b)$ plane")
axA.legend(loc="upper right", fontsize=8.6, framealpha=0.92)

# ===========================================================================
# PANEL B -- effect on dsigma/dq
# ===========================================================================
axB.fill_between(q_grid, dsig_cap, dsig_uncap, where=dsig_uncap > dsig_cap,
                 color=C_RED, alpha=0.20, lw=0, zorder=1,
                 label="suppressed by cap")
axB.plot(q_grid, dsig_uncap, color=C_UNCAP, lw=2.4, zorder=4,
         label=r"uncapped  ($b_{\mathrm{cap}}=$ None)")
axB.plot(q_grid, dsig_cap, color=C_CAPPED, lw=2.4, ls="--", zorder=5,
         label=r"capped  ($b_{\mathrm{cap}}=10$ cm)")

txt = axB.text(Q_STAR * 1.15, dsig_uncap.min() * 1.4, "$q_*$", color=C_GUIDE,
               fontsize=11, ha="left", va="bottom")
txt.set_path_effects([pe.Stroke(linewidth=2.4, foreground="w"), pe.Normal()])
axB.annotate("capped $=$ uncapped\nfor $q>q_*$",
             xy=(Q_STAR * 3, dsig_uncap[np.searchsorted(q_grid, Q_STAR * 3)]),
             xytext=(Q_STAR * 2.4, dsig_uncap.max() * 0.03),
             color=C_GUIDE, fontsize=10, ha="left", va="top")

axB.set_xscale("log")
axB.set_yscale("log")
axB.set_xlim(Q_LO, Q_HI)
axB.set_xlabel("momentum transfer  $q$  [GeV]")
axB.set_ylabel("$d\\sigma/dq$  [GeV$^{-3}$]")
axB.set_title("B.  Effect on $d\\sigma/dq$")
axB.legend(loc="upper right", fontsize=9, framealpha=0.92)

# ---------------------------------------------------------------------------
# parameter box, caption + suptitle
# ---------------------------------------------------------------------------
param_txt = (
    "$\\alpha=\\alpha_n N_n=%.2e$  ($\\alpha_n=%.2g$)      "
    "$v=%g\\,c$      $b_{\\mathrm{cap}}=10$ cm      $q_*=%.0f$ GeV"
    % (ALPHA, ALPHA_N, V, Q_STAR)
)
fig.text(0.5, 0.052, param_txt, ha="center", va="bottom", fontsize=9.5,
         family="monospace", color="#222222")

caption = (
    "Massless / Coulomb slice at the real 10 cm production cap "
    "$b_{\\mathrm{constrained\\,max}}=0.1$ m.  Finite-$\\lambda$ slices are "
    "affected only where the reach exceeds 10 cm (strong coupling / long range)."
)
fig.text(0.5, 0.006, caption, ha="center", va="bottom", fontsize=8.6,
         color="#555555", style="italic")

fig.suptitle("Impact-parameter cap: discarding the large-$b$ wedge "
             "($b>b_{\\mathrm{cap}}$) for $q<q_*$ suppresses $d\\sigma/dq$",
             fontsize=13.5, y=0.99)

fig.tight_layout(rect=(0, 0.06, 1, 0.965))

# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
name = "07_impact_parameter_cap"
outpaths = {}
for ext in ("png", "svg", "pdf"):
    d = os.path.join(_root, "notebooks", ext)
    os.makedirs(d, exist_ok=True)
    p = os.path.join(d, f"{name}.{ext}")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    outpaths[ext] = p

print("alpha_n   = %.3e" % ALPHA_N)
print("alpha     = %.4e" % ALPHA)
print("v         = %g c" % V)
print("q*        = %.2f GeV" % Q_STAR)
print("b_max(1e2)= %.4e m" % b_max_lo)
print("b_max(3e4)= %.4e m" % b_max_hi)
for ext, p in outpaths.items():
    print(f"{ext}: {p}")
