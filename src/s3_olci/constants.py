"""Constants and lookup tables for Sentinel-3 OLCI processing."""

# Ozone absorption coefficients k_O3 [cm^-1] per OLCI channel
OZONE_COEFFS: dict[int, float] = {
    3: 0.002,  # 442.5 nm
    4: 0.021,  # 490 nm
    5: 0.043,  # 510 nm
    6: 0.105,  # 560 nm
    7: 0.115,  # 620 nm
    8: 0.052,  # 665 nm
    9: 0.043,  # 673.75 nm
    10: 0.035,  # 681.25 nm
    11: 0.020,  # 708.75 nm
    12: 0.009,  # 753.75 nm
    13: 0.007,  # 761.25 nm
    14: 0.008,  # 764.375 nm
    15: 0.008,  # 767.5 nm
    16: 0.006,  # 778.75 nm
}

# Atmospheric pressure in hPa
P0_STANDARD: float = 1013.25

# kg/m^2 total column ozone to atm-cm
OZONE_KG_M2_TO_ATM_CM: float = 46.696
