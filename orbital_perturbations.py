"""
orbital_perturbations.py
=========================

Preliminary engineering models for the principal orbital perturbations
acting on a spacecraft in Low Earth Orbit (LEO, h < 2000 km), specialised
for a Sun-Synchronous Orbit (SSO) scenario.

Perturbations implemented
--------------------------
1. Non-spherical Earth gravity (zonal harmonics J2..J6)
2. Atmospheric drag, using the Harris-Priester density model
3. Solar Radiation Pressure (SRP)
4. Third-body perturbations (Sun, Moon)

All accelerations are returned in [m/s^2] unless otherwise stated.

References (typical of the values/models used here)
-----------------------------------------------------
- Vallado, D.A., "Fundamentals of Astrodynamics and Applications", 4th ed.,
  Microcosm/Springer, 2013 (zonal harmonic potential, low-precision Sun
  ephemeris algorithm, SRP/third body formulations).
- Curtis, H.D., "Orbital Mechanics for Engineering Students", 3rd ed.,
  Elsevier, 2013 (J2 closed-form acceleration, third-body tidal term).
- Montenbruck, O., Gill, E., "Satellite Orbits", Springer, 2000, Sec. 3.5
  and Table 3.3 (Harris-Priester density table and exponential
  interpolation / diurnal-bulge formulation).
- Harris, I., Priester, W., "Time-dependent structure of the upper
  atmosphere", J. Atmos. Sci., 19, 1962 (original model).
- Orekit library, class `org.orekit.forces.drag.HarrisPriester`
  (open-source reference implementation using the same Montenbruck &
  Gill table and algorithm structure, incl. the lag angle and
  configurable cos^n exponent).

NOTE: this module targets *preliminary / Phase-A* order-of-magnitude
estimation. The Harris-Priester table below reproduces the commonly
published Montenbruck & Gill / Orekit values; for mission-critical work
verify the table against the primary source and consider NRLMSISE-00 /
JB2008 for higher fidelity (in particular for solar-cycle, i.e. 11-year,
density variability, which the static Harris-Priester table does not
capture - it only models the diurnal/day-night density bulge at a fixed,
representative level of solar activity). For higher fidelity also
replace the zonal model with a full EGM2008 spherical-harmonics
expansion.
"""

import numpy as np
from numpy.polynomial import legendre as npleg

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
MU_EARTH = 398600.4418          # [km^3/s^2]
R_EARTH = 6378.137              # [km] WGS-84 equatorial radius
OMEGA_EARTH = 7.292115e-5       # [rad/s] Earth rotation rate

# Zonal harmonic coefficients (unnormalized), EGM96/WGS-84 representative values
J_COEFFS = {
    2: 1.0826269e-3,
    3: -2.5323000e-6,
    4: -1.6204000e-6,
    5: -2.2730000e-7,
    6: 5.4068000e-7,
}

MU_MOON = 4902.800               # [km^3/s^2]
D_MOON = 384400.0                # [km] mean Earth-Moon distance

MU_SUN = 1.32712440018e11        # [km^3/s^2]
D_SUN = 149597870.7              # [km] 1 AU

SOLAR_PRESSURE = 4.56e-6         # [N/m^2 = Pa] mean solar radiation pressure at 1 AU
SECONDS_PER_TROPICAL_YEAR = 365.2421897 * 86400.0


# ---------------------------------------------------------------------------
# Sun-synchronous geometry
# ---------------------------------------------------------------------------
def sso_inclination_deg(h_km):
    """
    Inclination [deg] required for a circular orbit at altitude h_km to be
    Sun-synchronous, i.e. nodal regression rate equal to the mean apparent
    motion of the Sun (360 deg / tropical year), using the classical
    J2 secular nodal-rate formula:

        dOmega/dt = -1.5 * n * J2 * (Re/a)^2 * cos(i)

    Returns inclination in degrees (>90 deg, retrograde, as expected for SSO).
    """
    a = R_EARTH + h_km
    n = np.sqrt(MU_EARTH / a**3)                       # mean motion [rad/s]
    omega_dot_target = 2.0 * np.pi / SECONDS_PER_TROPICAL_YEAR  # [rad/s]
    cos_i = -omega_dot_target / (1.5 * n * J_COEFFS[2] * (R_EARTH / a)**2)
    cos_i = np.clip(cos_i, -1.0, 1.0)
    return np.degrees(np.arccos(cos_i))


# ---------------------------------------------------------------------------
# 1) Zonal harmonics (J2..J6) perturbing acceleration
# ---------------------------------------------------------------------------
def _legendre_and_derivative(n, x):
    """Return (Pn(x), dPn/dx) using numpy's Legendre series machinery."""
    coeffs = np.zeros(n + 1)
    coeffs[n] = 1.0
    Pn = npleg.legval(x, coeffs)
    dPn = npleg.legval(x, npleg.legder(coeffs))
    return Pn, dPn


def zonal_accel_components(r_km, lat_rad, n):
    """
    Radial and meridional (tangential) perturbing acceleration components
    [km/s^2] due to a single zonal harmonic Jn, at geocentric radius r_km
    and latitude lat_rad, derived from the perturbing potential

        R_n(r,phi) = mu * Jn * Re^n * Pn(sin(phi)) / r^(n+1)

    via a_r = -dR_n/dr , a_t = -(1/r) dR_n/dphi  (validated against the
    closed-form Cartesian J2 acceleration formula, e.g. Curtis 2013).
    """
    Jn = J_COEFFS[n]
    x = np.sin(lat_rad)
    Pn, dPn = _legendre_and_derivative(n, x)
    a_r = (n + 1) * MU_EARTH * Jn * (R_EARTH / r_km) ** n * Pn / r_km**2
    a_t = -MU_EARTH * Jn * (R_EARTH / r_km) ** n * dPn * np.cos(lat_rad) / r_km**2
    return a_r, a_t


def zonal_max_accel_profile(h_km, n_list=(2, 3, 4, 5, 6), n_lat=361):
    """
    For a given altitude h_km, scan a uniform latitude grid spanning the
    ground-track latitude band actually reachable by the SSO at that
    altitude (+/-(180-i) deg) and return, for every harmonic in n_list,
    the MAXIMUM perturbing acceleration magnitude [m/s^2] over the grid,
    plus the maximum of the combined (summed) zonal perturbation.

    Returns
    -------
    dict: {n: max_accel_m_s2, ..., 'total': max_combined_accel_m_s2,
           'inclination_deg': sso inclination at this altitude}
    """
    r_km = R_EARTH + h_km
    incl = sso_inclination_deg(h_km)
    lat_max_deg = min(incl, 180.0 - incl)
    lat_grid = np.radians(np.linspace(-lat_max_deg, lat_max_deg, n_lat))

    result = {}
    total_vec_r = np.zeros_like(lat_grid)
    total_vec_t = np.zeros_like(lat_grid)
    for n in n_list:
        a_r, a_t = zonal_accel_components(r_km, lat_grid, n)
        mag = np.sqrt(a_r**2 + a_t**2) * 1000.0  # km/s^2 -> m/s^2
        result[n] = np.max(mag)
        total_vec_r += a_r
        total_vec_t += a_t

    total_mag = np.sqrt(total_vec_r**2 + total_vec_t**2) * 1000.0
    result['total'] = np.max(total_mag)
    result['inclination_deg'] = incl
    return result


# ---------------------------------------------------------------------------
# 2) Atmospheric drag -- Harris-Priester density model
# ---------------------------------------------------------------------------
# Table from Montenbruck & Gill, "Satellite Orbits" (2000), Table 3.3,
# reproduced in the same form used by the Orekit `HarrisPriester` class.
#
# IMPORTANT - UNITS: the original table is published in [g/km^3], NOT in
# SI units. The conversion to SI density [kg/m^3] is:
#
#     1 g/km^3 = 1e-3 kg / 1e9 m^3 = 1e-12 kg/m^3
#
# i.e. multiply every tabulated value by 1e-12 to obtain kg/m^3. This
# conversion is applied once, immediately below, so that every other
# function in this module works exclusively in SI units (kg/m^3).
_HP_H_KM = np.array([
    100, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240,
    250, 260, 270, 280, 290, 300, 320, 340, 360, 380, 400, 420, 440, 460,
    480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740,
    760, 780, 800, 840, 880, 920, 960, 1000,
], dtype=float)

_HP_RHO_MIN_G_PER_KM3 = np.array([
    4.974e5, 2.490e4, 8.377e3, 3.899e3, 2.122e3, 1.263e3, 8.008e2, 5.283e2,
    3.617e2, 2.557e2, 1.839e2, 1.341e2, 9.949e1, 7.488e1, 5.709e1, 4.403e1,
    3.430e1, 2.697e1, 2.139e1, 1.708e1, 1.099e1, 7.214e0, 4.824e0, 3.274e0,
    2.249e0, 1.558e0, 1.091e0, 7.701e-1, 5.474e-1, 3.916e-1, 2.819e-1,
    2.042e-1, 1.488e-1, 1.092e-1, 8.070e-2, 6.012e-2, 4.519e-2, 3.430e-2,
    2.632e-2, 2.043e-2, 1.607e-2, 1.281e-2, 1.036e-2, 8.496e-3, 7.069e-3,
    4.680e-3, 3.200e-3, 2.210e-3, 1.560e-3, 1.150e-3,
])

_HP_RHO_MAX_G_PER_KM3 = np.array([
    4.974e5, 2.490e4, 8.710e3, 4.059e3, 2.215e3, 1.344e3, 8.758e2, 6.010e2,
    4.297e2, 3.162e2, 2.396e2, 1.853e2, 1.455e2, 1.157e2, 9.308e1, 7.555e1,
    6.182e1, 5.095e1, 4.226e1, 3.526e1, 2.511e1, 1.819e1, 1.337e1, 9.955e0,
    7.492e0, 5.684e0, 4.355e0, 3.362e0, 2.612e0, 2.042e0, 1.605e0, 1.267e0,
    1.005e0, 7.997e-1, 6.390e-1, 5.123e-1, 4.121e-1, 3.325e-1, 2.691e-1,
    2.185e-1, 1.779e-1, 1.452e-1, 1.190e-1, 9.776e-2, 8.059e-2, 5.741e-2,
    4.210e-2, 3.130e-2, 2.360e-2, 1.810e-2,
])

G_PER_KM3_TO_KG_PER_M3 = 1.0e-12  # <-- explicit unit-conversion factor

_HP_RHO_MIN = _HP_RHO_MIN_G_PER_KM3 * G_PER_KM3_TO_KG_PER_M3  # [kg/m^3]
_HP_RHO_MAX = _HP_RHO_MAX_G_PER_KM3 * G_PER_KM3_TO_KG_PER_M3  # [kg/m^3]

# Official Harris-Priester table validity range; beyond this we extrapolate
# (flagged) using the last segment's exponential scale height.
_HP_H_MAX_VALID_KM = _HP_H_KM[-1]  # 1000 km

# Lag angle between the sub-solar meridian and the diurnal density bulge
# apex (Montenbruck & Gill quote ~30 deg; Orekit default = 30 deg).
HP_LAG_ANGLE_DEG = 0.0


def _hp_table_lookup(h_km, table_rho):
    """
    Exponential interpolation (Montenbruck & Gill, eq. 3.78) of a single
    Harris-Priester density column at altitude h_km [km]; table_rho must
    already be in SI units [kg/m^3]. Scalar in, scalar out.
    """
    if h_km <= _HP_H_KM[0]:
        i = 0
    elif h_km >= _HP_H_MAX_VALID_KM:
        i = len(_HP_H_KM) - 2  # extrapolate with the last (1000 km) segment
    else:
        i = int(np.searchsorted(_HP_H_KM, h_km, side='right') - 1)
        i = min(i, len(_HP_H_KM) - 2)

    h0, h1 = _HP_H_KM[i], _HP_H_KM[i + 1]
    rho0, rho1 = table_rho[i], table_rho[i + 1]
    H = (h0 - h1) / np.log(rho1 / rho0)            # local scale height [km]
    return rho0 * np.exp((h0 - h_km) / H)           # [kg/m^3]


def harris_priester_minmax_density(h_km):
    """
    Return (rho_min, rho_max) [kg/m^3] at altitude h_km [km], i.e. the
    bounding (night-side / day-side bulge) densities from the
    Harris-Priester table, exponentially interpolated.
    """
    h_arr = np.atleast_1d(h_km).astype(float)
    rho_min = np.array([_hp_table_lookup(h, _HP_RHO_MIN) for h in h_arr])
    rho_max = np.array([_hp_table_lookup(h, _HP_RHO_MAX) for h in h_arr])
    if rho_min.size == 1:
        return rho_min[0], rho_max[0]
    return rho_min, rho_max


def sun_ra_dec_deg(jd_ut1):
    """
    Low-precision geocentric Sun right ascension/declination [deg]
    (Vallado low-fidelity Sun algorithm, ~0.01 deg accuracy), used only to
    orient the Harris-Priester diurnal bulge for epoch-resolved use.
    jd_ut1 : Julian Date (UT1, ~interchangeable with TT at this precision).
    """
    T = (jd_ut1 - 2451545.0) / 36525.0
    lam_mean = (280.460 + 36000.771 * T) % 360.0
    M = np.radians((357.528 + 35999.050 * T) % 360.0)
    lam_ecl = np.radians(lam_mean + 1.915 * np.sin(M) + 0.020 * np.sin(2 * M))
    eps = np.radians(23.439 - 0.0130 * T)
    dec = np.arcsin(np.sin(eps) * np.sin(lam_ecl))
    ra = np.arctan2(np.cos(eps) * np.sin(lam_ecl), np.cos(lam_ecl))
    return np.degrees(ra) % 360.0, np.degrees(dec)


def harris_priester_bulge_direction(jd_ut1, lag_deg=HP_LAG_ANGLE_DEG):
    """Unit vector (ECI) towards the diurnal density-bulge apex at epoch jd_ut1."""
    ra_sun, dec_sun = sun_ra_dec_deg(jd_ut1)
    ra_b, dec_b = np.radians(ra_sun + lag_deg), np.radians(dec_sun)
    return np.array([np.cos(dec_b) * np.cos(ra_b),
                      np.cos(dec_b) * np.sin(ra_b),
                      np.sin(dec_b)])


def harris_priester_density(h_km, psi_deg, n_exp=6):
    """
    Harris-Priester density [kg/m^3] at altitude h_km and diurnal-bulge
    angle psi_deg (angle between satellite position vector and bulge
    apex direction, 0 deg = at the bulge / max density, 180 deg =
    opposite the bulge / min density):

        rho(h,psi) = rho_min(h) + [rho_max(h) - rho_min(h)] * cos(psi/2)^n

    n_exp : cosine exponent (Montenbruck & Gill): n=2 for low-inclination
        orbits, n=6 for near-polar orbits. Default n=6, appropriate for
        the ~97-99 deg inclination Sun-synchronous scenario used here.
    """
    rho_min, rho_max = harris_priester_minmax_density(h_km)
    c = np.cos(np.radians(psi_deg) / 2.0)
    c = np.clip(c, 0.0, 1.0)
    return rho_min + (rho_max - rho_min) * c**n_exp


def psi_from_position(r_eci_unit, bulge_unit_vec):
    """Angle [deg] between a satellite ECI unit position vector and the bulge direction."""
    cos_psi = np.clip(np.dot(r_eci_unit, bulge_unit_vec), -1.0, 1.0)
    return np.degrees(np.arccos(cos_psi))


def drag_accel(h_km, area_to_mass, psi_deg, Cd=2.2, n_exp=6, corotating_atm=True):
    """
    Drag acceleration magnitude [m/s^2] for a circular orbit at altitude
    h_km, using the Harris-Priester density at diurnal-bulge angle
    psi_deg:  a_drag = 0.5 * Cd * (A/m) * rho(h,psi) * v_rel^2

    area_to_mass : ballistic area-to-mass ratio [m^2/kg]
    corotating_atm : if True, approximate relative velocity assuming an
        atmosphere co-rotating with Earth (slightly reduces v_rel vs. the
        inertial circular velocity).
    """
    r_km = R_EARTH + h_km
    v_km_s = np.sqrt(MU_EARTH / r_km)            # circular orbital velocity
    if corotating_atm:
        v_rel_km_s = v_km_s - OMEGA_EARTH * r_km  # 1st-order equatorial approx.
    else:
        v_rel_km_s = v_km_s
    v_rel_m_s = v_rel_km_s * 1000.0
    rho = harris_priester_density(h_km, psi_deg, n_exp=n_exp)  # [kg/m^3]
    return 0.5 * Cd * area_to_mass * rho * v_rel_m_s**2


def drag_accel_envelope(h_km, am_grid, psi_grid=None, Cd=2.2, n_exp=6):
    """
    Build a uniform grid over (area-to-mass ratio x diurnal-bulge angle
    psi in [0,180] deg) and return (min, nominal, max) drag acceleration
    [m/s^2] at altitude h_km.

    The min/max bounds reproduce exactly the Harris-Priester rho_min(h)
    and rho_max(h) table values (reached at psi=180 deg and psi=0 deg
    respectively) combined with the A/m extremes; "nominal" is the mean
    drag acceleration over the psi grid at the median A/m, representing
    a simple proxy for the orbit/season-averaged drag level.
    """
    if psi_grid is None:
        psi_grid = np.linspace(0.0, 180.0, 37)  # 5 deg resolution

    vals = []
    for am in am_grid:
        for psi in psi_grid:
            vals.append(drag_accel(h_km, am, psi, Cd=Cd, n_exp=n_exp))
    vals = np.array(vals)

    nominal_vals = np.array([drag_accel(h_km, np.median(am_grid), psi,
                                         Cd=Cd, n_exp=n_exp) for psi in psi_grid])
    nominal = nominal_vals.mean()
    return vals.min(), nominal, vals.max()


# ---------------------------------------------------------------------------
# 3) Solar Radiation Pressure
# ---------------------------------------------------------------------------
def srp_accel(area_to_mass, Cr=1.3, eclipse_fraction=0.0):
    """
    SRP acceleration magnitude [m/s^2]:  a_srp = P * Cr * (A/m) * (1 - f_ecl)
    Essentially independent of LEO altitude (LEO altitude << 1 AU).
    """
    return SOLAR_PRESSURE * Cr * area_to_mass * (1.0 - eclipse_fraction)


def srp_accel_envelope(am_grid, cr_grid, eclipse_fraction=0.0):
    """Uniform grid over (A/m x Cr); returns (min, nominal, max) [m/s^2]."""
    vals = np.array([srp_accel(am, Cr=cr, eclipse_fraction=eclipse_fraction)
                      for am in am_grid for cr in cr_grid])
    nominal = srp_accel(np.median(am_grid), Cr=np.median(cr_grid),
                         eclipse_fraction=eclipse_fraction)
    return vals.min(), nominal, vals.max()


# ---------------------------------------------------------------------------
# 4) Third-body perturbations (Sun, Moon)
# ---------------------------------------------------------------------------
def third_body_max_accel(h_km, mu3, d3_km, n_grid=361):
    """
    Maximum third-body tidal perturbing acceleration [m/s^2] at altitude
    h_km, scanning a uniform grid of orbital phase angles (0-360 deg) for
    the worst-case (coplanar) geometry between the satellite orbit plane
    and the Earth-third-body line:

        a = mu3 * [ (s - r)/|s - r|^3  -  s/|s|^3 ]

    where s is the Earth->third-body vector and r the Earth->satellite
    vector. The maximum over the grid bounds the perturbation regardless
    of the instantaneous relative geometry.
    """
    r_km = R_EARTH + h_km
    thetas = np.linspace(0, 2 * np.pi, n_grid)
    s_vec = np.array([d3_km, 0.0, 0.0])
    s_norm3 = np.linalg.norm(s_vec) ** 3

    r_vecs = r_km * np.column_stack([np.cos(thetas), np.sin(thetas),
                                      np.zeros_like(thetas)])
    rel = s_vec - r_vecs
    rel_norm3 = (np.linalg.norm(rel, axis=1) ** 3)
    a_vecs = mu3 * (rel / rel_norm3[:, None] - s_vec / s_norm3)
    a_mag_km_s2 = np.linalg.norm(a_vecs, axis=1)
    return np.max(a_mag_km_s2) * 1000.0  # km/s^2 -> m/s^2