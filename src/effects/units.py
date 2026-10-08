"""Physical constants and unit conversions. Everything internal is SI."""

KT_JOULES = 4.184e12  # 1 kiloton TNT equivalent, J
KT_TNT_KG = 1.0e6  # 1 kiloton TNT, kg
P0_SEA_LEVEL = 101_325.0  # Pa
PSI = 6_894.757  # Pa per psi
CAL_PER_CM2 = 41_840.0  # J/m^2 per cal/cm^2


def psi_to_pa(x):
    return x * PSI


def pa_to_psi(x):
    return x / PSI


def cal_cm2_to_j_m2(x):
    return x * CAL_PER_CM2


def j_m2_to_cal_cm2(x):
    return x / CAL_PER_CM2
