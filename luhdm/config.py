import os

C = 299792458
V0 = 220e3 / C  # Virial velocity in NU
VESC = 544e3 / C  # Escape velocity in NU
RHO_DM = 2.3e-42  # local DM mass density in Gev^4 (total)
F_X = 0.1  # fraction of local DM in this species (evades self-interaction constraints)
M_MAG = 0.356e-6  # Mass of magnet in kg (Run45 magnet; matches upstream K_n calibration ~3.6e-7 kg)
N_NEUTRONS = M_MAG / 1.67e-27 / 2  # Number of neutrons in magnet, N_n ~ m/(2 m_N) with m_N = 1.67e-27 kg the neutron mass: DM-neutron coupling convention as in the levitated-sphere searches (Monteiro 2020, Tseng 2025)
R_EFF = 0.26e-3  # Effective radius of magnet in m (260 microns): subcomponent (cube/sphere) volumes summed, treated as one equivalent sphere
Q_THRESH = 1e3  # Momentum / analysis threshold, GeV/c (natural units; 1 TeV). Raised from 1e2 for the v7 analysis window so the window edge is a stated physics threshold instead of a place where the limit is carried by a steeply-varying, poorly-constrained efficiency. Under the night-selection marginalised-w efficiency (v8) the mode-1 turn-on reaches 50% at 1.21 TeV, so the edge sits at eps ~ 0.17 (mode 1) / 0.59 (mode 2). All 8 mode-1 candidates (1.52-12.8 TeV) remain inside; the 2 sub-TeV mode-2 impulses (554, 946 GeV) fall outside and are dropped by the optimum-interval support window automatically. Sets the kinematic wall at m_wall = Q_THRESH / VESC = 5.51e5 GeV.
T_EXPOSURE = float(os.environ.get("LUHDM_T_EXPOSURE", 790_778.0))  # night-selection live-time in s = 219.66 h (the analysis selection; see notebook 00)
# LUHDM_T_EXPOSURE overrides the live-time; the full unvetoed dataset is 1_691_020.0 s = 469.73 h.