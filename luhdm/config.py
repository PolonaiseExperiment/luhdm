C = 299792458
V0 = 220e3 / C  # Virial velocity in NU
VESC = 544e3 / C  # Escape velocity in NU
RHO_DM = 2.3e-42  # local DM mass density in Gev^4 (total)
F_X = 0.1  # fraction of local DM in this species (evades self-interaction constraints)
M_MAG = 0.42e-6  # Mass of magnet in kg
N_NEUTRONS = M_MAG / 1.67e-27  # Number of neutrons in magnet
R_EFF = 0.2e-3  # Effective radius of magnet in m (200 microns)
Q_THRESH = 1e2  # Momentum / analysis threshold, GeV/c (natural units; 0.1 TeV)
T_EXPOSURE = 1_691_020.0  # dataset live-time in s (sum of per-segment exposure_s; see notebook 00) = 469.73 h