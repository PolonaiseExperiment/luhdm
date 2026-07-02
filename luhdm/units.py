"""Unit conversions (natural units, GeV based)."""

C_M_S = 299792458.0

# From GeV to /s (hbar = 6.582e-25 GeV s)
CONV2RATE = 1.51928963016523e24

# 1 GeV/c in kg m/s
GEV_TO_KG_M_S = 5.3444e-19


def conv_m2pGeV(meter):
    """Convert a length in meters to GeV^-1."""
    return 5.0679e6 * meter * 1e9
