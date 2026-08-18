import os

C = 299792458
V0 = 220e3 / C  # Virial velocity in NU
VESC = 544e3 / C  # Escape velocity in NU
RHO_DM = 2.3e-42  # local DM mass density in Gev^4 (total)
F_X = 0.1  # fraction of local DM in this species (evades self-interaction constraints)
M_MAG = 0.356e-6  # Mass of magnet in kg (Run45 magnet; matches upstream K_n calibration ~3.6e-7 kg)
N_NEUTRONS = M_MAG / 1.67e-27 / 2  # Number of neutrons in magnet, N_n ~ m/(2 m_N) with m_N = 1.67e-27 kg the neutron mass: DM-neutron coupling convention as in the levitated-sphere searches (Monteiro 2020, Tseng 2025)
R_EFF = 0.26e-3  # Effective radius of magnet in m (260 microns): subcomponent (cube/sphere) volumes summed, treated as one equivalent sphere
Q_THRESH = 1e3  # Momentum / analysis threshold, GeV/c (natural units; 1 TeV). Raised from 1e2 for the v7 analysis window so the window edge is a stated physics threshold instead of a place where the limit is carried by a steeply-varying, poorly-constrained efficiency. Under the marginalised-w efficiency (v8) the mode-1 turn-on reaches 50% at 1.22 TeV, so the edge sits at eps ~ 0.16 (mode 1) / 0.58 (mode 2). All 8 mode-1 candidates (1.52-12.8 TeV) remain inside; the 2 sub-TeV mode-2 impulses (554, 946 GeV) fall outside and are dropped by the optimum-interval support window automatically. Sets the kinematic wall at m_wall = Q_THRESH / V_MAX = 5.51e5 GeV in the Galactic rest frame (V_MAX = VESC); a lab-frame convention (V_E > 0) moves the wall down with 1/V_MAX, to 3.80e5 GeV at V_E = 245 km/s.
T_EXPOSURE = float(os.environ.get("LUHDM_T_EXPOSURE", 790_778.0))  # night-selection live-time in s = 219.66 h (the analysis selection; see notebook 00)
# LUHDM_T_EXPOSURE overrides the live-time; the full unvetoed dataset is 1_691_020.0 s = 469.73 h.
V_E = float(os.environ.get("LUHDM_V_EARTH", "0.0")) * 1e3 / C  # Earth's speed through the halo in NU; the env value is in km/s. 0.0 (the default) means the Galactic-rest-frame halo, i.e. exactly the historical luhdm.halo.standard_halo_model -- setting it is a stated change of convention, never a silent one. 245 km/s is the lab-frame boost of Monteiro 2020 (arXiv:2007.12067), the value the two papers this analysis overlays both adopt (Tseng 2025, arXiv:2508.00815, Eq. "shm_f_v", after Lewin & Smith 1996); V0 = 220 km/s and VESC = 544 km/s above are already Monteiro's, so LUHDM_V_EARTH=245 puts the halo entirely in their convention.
V_MAX = VESC + V_E  # Highest speed the halo distribution supports, in NU. In the lab frame a particle sitting at the Galactic escape speed and met head-on by the Earth arrives at VESC + V_E, so this -- not VESC -- is the ceiling of every speed integral, speed grid and rejection-sampler proposal. Equals VESC exactly when V_E = 0, so the Galactic-rest-frame pipeline is untouched.


def set_v_earth_km_s(v_e_km_s):
    """Set the lab-frame boost (km/s) and the derived speed ceiling together.

    The two are one convention, so they are never assigned separately: a stale
    V_MAX would truncate the boosted tail exactly where it matters. Scripts call
    this from their ``--v-earth`` handling before any pool is forked; the env
    knob LUHDM_V_EARTH is the same setting applied at import.
    """
    global V_E, V_MAX
    V_E = float(v_e_km_s) * 1e3 / C
    V_MAX = VESC + V_E
    return V_E
