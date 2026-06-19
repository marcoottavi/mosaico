"""
orbital_perturbations.py
=========================

Preliminary engineering models for the principal orbital perturbations
acting on a spacecraft in Low Earth Orbit (LEO, h < 2000 km), specialised
for a Sun-Synchronous Orbit (SSO) scenario.

Perturbations implemented
--------------------------
1. Non-spherical Earth gravity (zonal harmonics J2..J6)
2. Atmospheric drag — NRLMSISE-00 density model
3. Solar Radiation Pressure (SRP)
4. Third-body perturbations (Sun, Moon)

All accelerations are returned in [m/s^2] unless otherwise stated.

Atmospheric density model
--------------------------
Density is obtained from the NRLMSISE-00 empirical model (Picone et al.,
2002) via the `nrlmsise00` Python package.  Three representative solar
activity levels are supported:

    'low'  : F10.7 =  75 SFU  — solar minimum  (e.g. cycle 23/24 trough)
    'mean' : F10.7 = 150 SFU  — moderate/mean activity
    'high' : F10.7 = 230 SFU  — solar maximum   (e.g. cycle 23/24 peak)

For each level the model is evaluated on a latitude × local-solar-time ×
day-of-year grid and the global minimum and maximum density are extracted;
these play the same role as the rho_min / rho_max columns of the
Harris-Priester table, but they are derived directly from a validated
empirical model rather than from a static lookup table.

The grid used is:
    latitudes  : -90 to +90 deg, 19 points
    local times: 0 to 24 h,      13 points  (2-hour resolution)
    days of year: 80, 172, 264, 355          (equinoxes + solstices)

Results are pre-computed at module import for the standard altitude grid
and cached in _NRLM_CACHE so that subsequent calls to drag functions are
fast.

References
-----------
- Picone, J.M., Hedin, A.E., Drob, D.P., and Aikin, A.C.,
  "NRLMSISE-00 empirical model of the atmosphere: Statistical comparisons
  and scientific issues", Journal of Geophysical Research: Space Physics,
  Vol. 107, No. A12, pp. SIA 15-1 - SIA 15-16, 2002.
  DOI: 10.1029/2002JA009430

- Vallado, D.A., "Fundamentals of Astrodynamics and Applications",
  4th ed., Microcosm/Springer, 2013.

- Curtis, H.D., "Orbital Mechanics for Engineering Students", 3rd ed.,
  Elsevier, 2013.

- Montenbruck, O., Gill, E., "Satellite Orbits: Models, Methods and
  Applications", Springer, 2000.
"""

import numpy as np
from numpy.polynomial import legendre as npleg
from nrlmsise00 import gtd7_flat

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
MU_EARTH    = 398600.4418          # [km^3/s^2]
R_EARTH     = 6378.137             # [km] WGS-84 equatorial radius
OMEGA_EARTH = 7.292115e-5          # [rad/s] Earth rotation rate

J_COEFFS = {
    2:  1.0826269e-3,
    3: -2.5323000e-6,
    4: -1.6204000e-6,
    5: -2.2730000e-7,
    6:  5.4068000e-7,
}

MU_MOON = 4902.800                 # [km^3/s^2]
D_MOON  = 384400.0                 # [km]

MU_SUN  = 1.32712440018e11         # [km^3/s^2]
D_SUN   = 149597870.7              # [km]

SOLAR_PRESSURE = 4.56e-6           # [N/m^2] mean SRP at 1 AU
SECONDS_PER_TROPICAL_YEAR = 365.2421897 * 86400.0

# ---------------------------------------------------------------------------
# NRLMSISE-00 configuration
# ---------------------------------------------------------------------------
# Representative solar activity levels [SFU]
F10P7 = {
    'low':  75.0,
    'mean': 150.0,
    'high': 230.0,
}

# Geomagnetic activity: quiet conditions (Ap = 4), representative of the
# undisturbed background environment used for a preliminary design budget.
_AP_QUIET = 4.0

# NRLMSISE-00 flags: SI units (kg/m^3 and m^-3)
_MSIS_FLAGS = [1] * 24

# Sampling grid for global density min/max extraction
_LAT_GRID  = np.linspace(-90.0,  90.0, 19)   # deg,  19 pts
_LST_GRID  = np.linspace(  0.0,  24.0, 13)   # hours, 13 pts (every 2h)
_DOY_GRID  = [80, 172, 264, 355]              # representative days of year

SOLAR_ACTIVITIES = ('low', 'mean', 'high')

# Standard altitude grid shared with plotting scripts [km]
_CACHE_ALT = np.array([
    100, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240,
    250, 260, 270, 280, 290, 300, 320, 340, 360, 380, 400, 420, 440, 460,
    480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740,
    760, 780, 800, 840, 880, 920, 960, 1000,
], dtype=float)


def _msis_density(alt_km, f107, lat_deg, lst_h, doy):
    """Single NRLMSISE-00 density call [kg/m^3]."""
    out = gtd7_flat(
        2030, int(doy), lst_h * 3600.0,
        alt_km, lat_deg, 0.0, lst_h,
        f107, f107, _AP_QUIET,
        flags=_MSIS_FLAGS,
    )
    return out[5]   # index 5 = total mass density [kg/m^3] in SI mode


def _build_minmax_cache(alt_array):
    """
    For each altitude in alt_array, scan the lat x lst x doy grid and record
    the global minimum and maximum density for each activity level.

    Returns
    -------
    cache : dict  { activity: {'rho_min': ndarray, 'rho_max': ndarray} }
    """
    print("  Pre-computing NRLMSISE-00 density bounds "
          "(lat × LST × DOY grid)...")
    cache = {}
    for act in SOLAR_ACTIVITIES:
        f107  = F10P7[act]
        rho_min_arr = np.empty(len(alt_array))
        rho_max_arr = np.empty(len(alt_array))
        for i, alt in enumerate(alt_array):
            vals = [
                _msis_density(alt, f107, lat, lst, doy)
                for doy in _DOY_GRID
                for lat in _LAT_GRID
                for lst in _LST_GRID
            ]
            rho_min_arr[i] = np.min(vals)
            rho_max_arr[i] = np.max(vals)
        cache[act] = {'rho_min': rho_min_arr, 'rho_max': rho_max_arr}
        print(f"    {act:4s} (F10.7={f107:5.1f}): done")
    print("  Cache complete.")
    return cache


# Build the cache once at import time
print("orbital_perturbations: building NRLMSISE-00 density cache...")
_NRLM_CACHE = _build_minmax_cache(_CACHE_ALT)
print("orbital_perturbations: cache ready.\n")

# Display metadata used by plotting scripts
_ACT_META = {
    'low':  dict(label='Low activity (F10.7=75)',
                 color='#2ca02c', ls=':', lw=1.8, fill='#98df8a'),
    'mean': dict(label='Mean activity (F10.7=150)',
                 color='#1f77b4', ls='-', lw=2.2, fill='#aec7e8'),
    'high': dict(label='High activity (F10.7=230)',
                 color='#d62728', ls='--', lw=1.8, fill='#ffbb78'),
}


# ---------------------------------------------------------------------------
# Density interpolation from cache
# ---------------------------------------------------------------------------
def _interp_minmax(h_km, activity):
    """
    Exponential interpolation of cached rho_min / rho_max at altitude h_km.
    Falls back to nearest endpoint outside the cached range.
    Returns (rho_min, rho_max) in [kg/m^3].
    """
    tbl     = _NRLM_CACHE[activity]
    rho_min = np.interp(h_km, _CACHE_ALT,
                        np.log(tbl['rho_min']))
    rho_max = np.interp(h_km, _CACHE_ALT,
                        np.log(tbl['rho_max']))
    return np.exp(rho_min), np.exp(rho_max)


def nrlmsise_minmax_density(h_km, activity='mean'):
    """
    Return (rho_min, rho_max) [kg/m^3] from the pre-computed NRLMSISE-00
    cache at altitude h_km for the given solar activity level.

    rho_min : global minimum density over the lat × LST × DOY grid
    rho_max : global maximum density over the lat × LST × DOY grid
    """
    h_arr = np.atleast_1d(np.asarray(h_km, dtype=float))
    r_min = np.array([_interp_minmax(h, activity)[0] for h in h_arr])
    r_max = np.array([_interp_minmax(h, activity)[1] for h in h_arr])
    if r_min.size == 1:
        return r_min[0], r_max[0]
    return r_min, r_max


def nrlmsise_density(h_km, psi_deg, n_exp=6, activity='mean'):
    """
    Effective density [kg/m^3] interpolated between the global min and max
    as a function of the diurnal-bulge angle psi_deg, using the same
    cosine-power formula as the Harris-Priester model:

        rho(h, psi) = rho_min(h) + [rho_max(h) - rho_min(h)] * cos(psi/2)^n

    This preserves compatibility with all existing drag functions while
    replacing the static Harris-Priester lookup with NRLMSISE-00 bounds.

    Parameters
    ----------
    psi_deg  : angle between spacecraft position and the density-bulge apex
               (0 = at maximum density, 180 = at minimum density)
    n_exp    : cosine exponent (6 for near-polar/SSO, 2 for low inclination)
    activity : 'low' | 'mean' | 'high'
    """
    rho_min, rho_max = nrlmsise_minmax_density(h_km, activity=activity)
    c = np.clip(np.cos(np.radians(psi_deg) / 2.0), 0.0, 1.0)
    return rho_min + (rho_max - rho_min) * c ** n_exp


# ---------------------------------------------------------------------------
# Sun-synchronous geometry
# ---------------------------------------------------------------------------
def sso_inclination_deg(h_km):
    a = R_EARTH + h_km
    n = np.sqrt(MU_EARTH / a ** 3)
    omega_dot = 2.0 * np.pi / SECONDS_PER_TROPICAL_YEAR
    cos_i = -omega_dot / (1.5 * n * J_COEFFS[2] * (R_EARTH / a) ** 2)
    return np.degrees(np.arccos(np.clip(cos_i, -1.0, 1.0)))


# ---------------------------------------------------------------------------
# 1) Zonal harmonics J2..J6
# ---------------------------------------------------------------------------
def _legendre_and_derivative(n, x):
    coeffs       = np.zeros(n + 1)
    coeffs[n]    = 1.0
    Pn  = npleg.legval(x, coeffs)
    dPn = npleg.legval(x, npleg.legder(coeffs))
    return Pn, dPn


def zonal_accel_components(r_km, lat_rad, n):
    Jn       = J_COEFFS[n]
    x        = np.sin(lat_rad)
    Pn, dPn  = _legendre_and_derivative(n, x)
    a_r = (n + 1) * MU_EARTH * Jn * (R_EARTH / r_km) ** n * Pn / r_km ** 2
    a_t = -MU_EARTH * Jn * (R_EARTH / r_km) ** n * dPn * np.cos(lat_rad) / r_km ** 2
    return a_r, a_t


def zonal_max_accel_profile(h_km, n_list=(2, 3, 4, 5, 6), n_lat=361):
    r_km        = R_EARTH + h_km
    incl        = sso_inclination_deg(h_km)
    lat_max_deg = min(incl, 180.0 - incl)
    lat_grid    = np.radians(np.linspace(-lat_max_deg, lat_max_deg, n_lat))

    result      = {}
    total_r     = np.zeros_like(lat_grid)
    total_t     = np.zeros_like(lat_grid)
    for n in n_list:
        a_r, a_t = zonal_accel_components(r_km, lat_grid, n)
        result[n] = np.max(np.sqrt(a_r ** 2 + a_t ** 2) * 1000.0)
        total_r  += a_r
        total_t  += a_t

    result['total']           = np.max(np.sqrt(total_r ** 2 + total_t ** 2) * 1000.0)
    result['inclination_deg'] = incl
    return result


# ---------------------------------------------------------------------------
# 2) Atmospheric drag  (NRLMSISE-00 backend)
# ---------------------------------------------------------------------------
def drag_accel(h_km, area_to_mass, psi_deg,
               Cd=2.2, n_exp=6, corotating_atm=True, activity='mean'):
    """
    Drag acceleration magnitude [m/s^2] for a circular orbit at altitude
    h_km [km].

    Parameters
    ----------
    area_to_mass  : ballistic area-to-mass ratio [m^2/kg]
    psi_deg       : diurnal-bulge angle [deg]  (0 = max density, 180 = min)
    Cd            : drag coefficient (default 2.2)
    n_exp         : cosine exponent for density interpolation (default 6)
    corotating_atm: account for atmosphere co-rotation (default True)
    activity      : 'low' | 'mean' | 'high'
    """
    r_km      = R_EARTH + h_km
    v_km_s    = np.sqrt(MU_EARTH / r_km)
    v_rel_m_s = (v_km_s - (OMEGA_EARTH * r_km if corotating_atm else 0.0)) * 1000.0
    rho       = nrlmsise_density(h_km, psi_deg, n_exp=n_exp, activity=activity)
    return 0.5 * Cd * area_to_mass * rho * v_rel_m_s ** 2


def drag_accel_envelope(h_km, am_grid, psi_grid=None, Cd=2.2, n_exp=6,
                        activity='mean'):
    """
    Return (min, nominal, max) drag acceleration [m/s^2] over the A/m and
    psi parameter grids for the given solar activity level.

    nominal = psi-averaged drag at the median A/m
    """
    if psi_grid is None:
        psi_grid = np.linspace(0.0, 180.0, 37)

    vals = np.array([
        drag_accel(h_km, am, psi, Cd=Cd, n_exp=n_exp, activity=activity)
        for am  in am_grid
        for psi in psi_grid
    ])
    nominal = np.mean([
        drag_accel(h_km, np.median(am_grid), psi,
                   Cd=Cd, n_exp=n_exp, activity=activity)
        for psi in psi_grid
    ])
    return vals.min(), nominal, vals.max()


# ---------------------------------------------------------------------------
# Convenience: density profile accessors (used by plotting scripts)
# ---------------------------------------------------------------------------
def harris_priester_minmax_density(h_km, activity='mean'):
    """Alias kept for backward compatibility with plotting scripts."""
    return nrlmsise_minmax_density(h_km, activity=activity)


def harris_priester_density(h_km, psi_deg, n_exp=6, activity='mean'):
    """Alias kept for backward compatibility with plotting scripts."""
    return nrlmsise_density(h_km, psi_deg, n_exp=n_exp, activity=activity)


# Expose the validity ceiling used by the density-profile plot
_HP_H_MAX_VALID_KM = _CACHE_ALT[-1]   # 1000 km


# ---------------------------------------------------------------------------
# Sun ephemeris / bulge direction (unchanged)
# ---------------------------------------------------------------------------
def sun_ra_dec_deg(jd_ut1):
    T        = (jd_ut1 - 2451545.0) / 36525.0
    lam_mean = (280.460 + 36000.771 * T) % 360.0
    M        = np.radians((357.528 + 35999.050 * T) % 360.0)
    lam_ecl  = np.radians(lam_mean + 1.915 * np.sin(M) + 0.020 * np.sin(2 * M))
    eps      = np.radians(23.439 - 0.0130 * T)
    dec      = np.arcsin(np.sin(eps) * np.sin(lam_ecl))
    ra       = np.arctan2(np.cos(eps) * np.sin(lam_ecl), np.cos(lam_ecl))
    return np.degrees(ra) % 360.0, np.degrees(dec)


def psi_from_position(r_eci_unit, bulge_unit_vec):
    return np.degrees(np.arccos(np.clip(np.dot(r_eci_unit, bulge_unit_vec),
                                        -1.0, 1.0)))


# ---------------------------------------------------------------------------
# 3) Solar Radiation Pressure
# ---------------------------------------------------------------------------
def srp_accel(area_to_mass, Cr=1.3, eclipse_fraction=0.0):
    return SOLAR_PRESSURE * Cr * area_to_mass * (1.0 - eclipse_fraction)


def srp_accel_envelope(am_grid, cr_grid, eclipse_fraction=0.0):
    vals    = np.array([srp_accel(am, Cr=cr, eclipse_fraction=eclipse_fraction)
                        for am in am_grid for cr in cr_grid])
    nominal = srp_accel(np.median(am_grid), Cr=np.median(cr_grid),
                        eclipse_fraction=eclipse_fraction)
    return vals.min(), nominal, vals.max()


# ---------------------------------------------------------------------------
# 4) Third-body perturbations
# ---------------------------------------------------------------------------
def third_body_max_accel(h_km, mu3, d3_km, n_grid=361):
    r_km    = R_EARTH + h_km
    thetas  = np.linspace(0, 2 * np.pi, n_grid)
    s_vec   = np.array([d3_km, 0.0, 0.0])
    s_norm3 = np.linalg.norm(s_vec) ** 3
    r_vecs  = r_km * np.column_stack([np.cos(thetas), np.sin(thetas),
                                       np.zeros_like(thetas)])
    rel       = s_vec - r_vecs
    rel_norm3 = np.linalg.norm(rel, axis=1) ** 3
    a_vecs    = mu3 * (rel / rel_norm3[:, None] - s_vec / s_norm3)
    return np.max(np.linalg.norm(a_vecs, axis=1)) * 1000.0