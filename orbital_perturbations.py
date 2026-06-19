"""
orbital_perturbations.py
=========================

Preliminary engineering models for the principal orbital perturbations
acting on a spacecraft in Low Earth Orbit (LEO, h < 2000 km), specialised
for a Sun-Synchronous Orbit (SSO) scenario.

Atmospheric density model
--------------------------
Density bounds (rho_min, rho_max) were computed by running NRLMSISE-00
(Picone et al., 2002) over a sampling grid of:
    latitudes   : -90 to +90 deg  (19 points)
    local times : 0 to 24 h       (13 points, 2-hour resolution)
    days of year: 80, 172, 264, 355 (equinoxes + solstices)
at three solar activity levels:
    'low'  : F10.7 =  75 SFU  (solar minimum)
    'mean' : F10.7 = 150 SFU  (moderate / mean activity)
    'high' : F10.7 = 230 SFU  (solar maximum)
Geomagnetic index Ap = 4 (quiet conditions) throughout.

The global minimum and maximum density over that grid are stored as
hardcoded numpy arrays so that the module has no external dependencies
beyond numpy.  The values were generated with the `nrlmsise00` Python
package (v0.1.2, flags=[1]*24) on 2026-06-19 and are exact to the
precision shown.

To regenerate these tables run:
    pip install nrlmsise00
    python3 -c "
    from nrlmsise00 import gtd7_flat
    import numpy as np
    # ... evaluate gtd7_flat with flags=[1]*24 over the grid above
    # and record np.min / np.max at each altitude node
    "

References
-----------
- Picone, J.M., Hedin, A.E., Drob, D.P., and Aikin, A.C.,
  'NRLMSISE-00 empirical model of the atmosphere: Statistical comparisons
  and scientific issues', Journal of Geophysical Research: Space Physics,
  Vol. 107, No. A12, pp. SIA 15-1 - SIA 15-16, 2002.
  DOI: 10.1029/2002JA009430

- Vallado, D.A., 'Fundamentals of Astrodynamics and Applications',
  4th ed., Microcosm/Springer, 2013.

- Curtis, H.D., 'Orbital Mechanics for Engineering Students', 3rd ed.,
  Elsevier, 2013.
"""

import numpy as np
from numpy.polynomial import legendre as npleg

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

SOLAR_PRESSURE = 4.56e-6           # [N/m^2]
SECONDS_PER_TROPICAL_YEAR = 365.2421897 * 86400.0

# ---------------------------------------------------------------------------
# NRLMSISE-00 density tables  [kg/m^3]
#
# Generated with nrlmsise00 v0.1.2, Ap=4 (quiet), sampling grid:
#   lat: -90..+90 deg (19 pts), LST: 0..24 h (13 pts), DOY: 80/172/264/355
# rho_min = global minimum over grid; rho_max = global maximum over grid.
# ---------------------------------------------------------------------------

# Altitude nodes [km] — same for all tables
_CACHE_ALT = np.array([
    100, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240,
    250, 260, 270, 280, 290, 300, 320, 340, 360, 380, 400, 420, 440, 460,
    480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740,
    760, 780, 800, 840, 880, 920, 960, 1000,
], dtype=float)

# --- LOW activity  F10.7 = 75 SFU ---
_RHO_MIN_LOW = np.array([
    2.896618e-07, 1.240332e-08, 4.830609e-09, 2.244753e-09, 1.215097e-09,
    7.191184e-10, 4.505209e-10, 2.888921e-10, 1.892183e-10, 1.271152e-10,
    8.718448e-11, 6.085301e-11, 4.311986e-11, 3.095952e-11, 2.229567e-11,
    1.604160e-11, 1.162736e-11, 8.455285e-12, 6.188713e-12, 4.556601e-12,
    2.500718e-12, 1.399786e-12, 8.011659e-13, 4.705318e-13, 2.851264e-13,
    1.794322e-13, 1.179884e-13, 7.611352e-14, 4.790091e-14, 3.077755e-14,
    2.028406e-14, 1.377951e-14, 9.616067e-15, 6.893859e-15, 4.869140e-15,
    3.551799e-15, 2.681301e-15, 2.095143e-15, 1.691499e-15, 1.384267e-15,
    1.139621e-15, 9.627853e-16, 8.316011e-16, 7.229848e-16, 6.237331e-16,
    4.888616e-16, 3.895813e-16, 3.214204e-16, 2.678955e-16, 2.296837e-16,
])  # [kg/m^3]  F10.7=75.0

_RHO_MAX_LOW = np.array([
    9.058144e-07, 2.909548e-08, 8.593027e-09, 3.888978e-09, 2.071521e-09,
    1.203145e-09, 7.421805e-10, 4.838856e-10, 3.337863e-10, 2.354675e-10,
    1.691708e-10, 1.236673e-10, 9.157373e-11, 6.857696e-11, 5.187307e-11,
    3.964864e-11, 3.069949e-11, 2.394350e-11, 1.879594e-11, 1.484098e-11,
    9.394096e-12, 6.063229e-12, 3.967721e-12, 2.626055e-12, 1.759636e-12,
    1.192869e-12, 8.146934e-13, 5.602057e-13, 3.877511e-13, 2.701465e-13,
    1.895012e-13, 1.339157e-13, 9.541337e-14, 6.861229e-14, 4.986090e-14,
    3.666975e-14, 2.733442e-14, 2.068349e-14, 1.630847e-14, 1.438180e-14,
    1.274726e-14, 1.134248e-14, 1.012298e-14, 9.056158e-15, 8.117410e-15,
    6.551710e-15, 5.313917e-15, 4.327628e-15, 3.537514e-15, 2.901985e-15,
])  # [kg/m^3]  F10.7=75.0

# --- MEAN activity  F10.7 = 150 SFU ---
_RHO_MIN_MEAN = np.array([
    2.828443e-07, 1.279224e-08, 5.004602e-09, 2.388226e-09, 1.347008e-09,
    8.374575e-10, 5.539505e-10, 3.828389e-10, 2.729415e-10, 1.992240e-10,
    1.481244e-10, 1.117874e-10, 8.505807e-11, 6.526153e-11, 5.056572e-11,
    3.951450e-11, 3.111041e-11, 2.432375e-11, 1.909356e-11, 1.505940e-11,
    9.484879e-12, 6.058916e-12, 3.895538e-12, 2.524521e-12, 1.652966e-12,
    1.093601e-12, 7.305694e-13, 4.932216e-13, 3.375418e-13, 2.346528e-13,
    1.660954e-13, 1.199928e-13, 8.866053e-14, 6.653438e-14, 4.767245e-14,
    3.451579e-14, 2.529013e-14, 1.878301e-14, 1.416299e-14, 1.085803e-14,
    8.473279e-15, 6.735362e-15, 5.454399e-15, 4.480712e-15, 3.647935e-15,
    2.415846e-15, 1.702038e-15, 1.268585e-15, 9.667598e-16, 7.570368e-16,
])  # [kg/m^3]  F10.7=150.0

_RHO_MAX_MEAN = np.array([
    8.843050e-07, 3.016573e-08, 9.008613e-09, 4.182318e-09, 2.337016e-09,
    1.438035e-09, 9.424695e-10, 6.456901e-10, 4.569785e-10, 3.363365e-10,
    2.551397e-10, 1.974532e-10, 1.550678e-10, 1.233662e-10, 9.909217e-11,
    8.025660e-11, 6.547278e-11, 5.375333e-11, 4.438169e-11, 3.688767e-11,
    2.582071e-11, 1.834693e-11, 1.319783e-11, 9.592057e-12, 7.032300e-12,
    5.194028e-12, 3.865620e-12, 2.894634e-12, 2.178419e-12, 1.646751e-12,
    1.249940e-12, 9.554460e-13, 7.344617e-13, 5.665198e-13, 4.384330e-13,
    3.404240e-13, 2.652026e-13, 2.073059e-13, 1.626224e-13, 1.280455e-13,
    1.012199e-13, 8.035393e-14, 6.408115e-14, 5.135644e-14, 4.137855e-14,
    2.734208e-14, 1.855319e-14, 1.296977e-14, 9.434265e-15, 7.179804e-15,
])  # [kg/m^3]  F10.7=150.0

# --- HIGH activity  F10.7 = 230 SFU ---
_RHO_MIN_HIGH = np.array([
    2.761258e-07, 1.318467e-08, 5.214604e-09, 2.565928e-09, 1.502313e-09,
    9.714371e-10, 6.679601e-10, 4.794015e-10, 3.544374e-10, 2.679025e-10,
    2.059766e-10, 1.605256e-10, 1.264953e-10, 1.006051e-10, 8.064807e-11,
    6.509519e-11, 5.255964e-11, 4.241376e-11, 3.438795e-11, 2.799971e-11,
    1.865323e-11, 1.254054e-11, 8.493972e-12, 5.778211e-12, 3.961649e-12,
    2.735443e-12, 1.901327e-12, 1.330134e-12, 9.366921e-13, 6.642307e-13,
    4.745993e-13, 3.419697e-13, 2.487479e-13, 1.828856e-13, 1.360938e-13,
    1.026469e-13, 7.857453e-14, 6.111438e-14, 4.833800e-14, 3.889481e-14,
    3.183586e-14, 2.484951e-14, 1.941225e-14, 1.532677e-14, 1.224024e-14,
    8.098907e-15, 5.638454e-15, 4.124839e-15, 3.154738e-15, 2.421168e-15,
])  # [kg/m^3]  F10.7=230.0

_RHO_MAX_HIGH = np.array([
    8.630636e-07, 3.130290e-08, 9.495073e-09, 4.532237e-09, 2.640261e-09,
    1.697875e-09, 1.162421e-09, 8.308273e-10, 6.124800e-10, 4.621730e-10,
    3.551465e-10, 2.817785e-10, 2.282552e-10, 1.870810e-10, 1.549413e-10,
    1.292901e-10, 1.085878e-10, 9.171882e-11, 7.785846e-11, 6.638706e-11,
    4.882655e-11, 3.648370e-11, 2.755748e-11, 2.100587e-11, 1.613700e-11,
    1.248019e-11, 9.708635e-12, 7.591238e-12, 5.966168e-12, 4.712306e-12,
    3.735650e-12, 2.971317e-12, 2.370628e-12, 1.896762e-12, 1.521670e-12,
    1.227419e-12, 9.945769e-13, 8.077526e-13, 6.574758e-13, 5.363146e-13,
    4.384141e-13, 3.591449e-13, 2.948348e-13, 2.425620e-13, 1.999957e-13,
    1.368955e-13, 9.461873e-14, 6.609758e-14, 4.672127e-14, 3.346164e-14,
])  # [kg/m^3]  F10.7=230.0

# Master lookup dict
_NRLM_TABLES = {
    'low':  {'rho_min': _RHO_MIN_LOW,  'rho_max': _RHO_MAX_LOW},
    'mean': {'rho_min': _RHO_MIN_MEAN, 'rho_max': _RHO_MAX_MEAN},
    'high': {'rho_min': _RHO_MIN_HIGH, 'rho_max': _RHO_MAX_HIGH},
}

# F10.7 values associated with each level (for labelling / reference)
F10P7 = {'low': 75.0, 'mean': 150.0, 'high': 230.0}

SOLAR_ACTIVITIES = ('low', 'mean', 'high')

# Display metadata used by plotting scripts
_ACT_META = {
    'low':  dict(label='Low activity (F10.7=75)',
                 color='#2ca02c', ls=':', lw=1.8, fill='#98df8a'),
    'mean': dict(label='Mean activity (F10.7=150)',
                 color='#1f77b4', ls='-', lw=2.2, fill='#aec7e8'),
    'high': dict(label='High activity (F10.7=230)',
                 color='#d62728', ls='--', lw=1.8, fill='#ffbb78'),
}

# Ceiling of the pre-computed altitude grid
_CACHE_ALT_MAX_KM = _CACHE_ALT[-1]   # 1000 km


# ---------------------------------------------------------------------------
# Density interpolation
# ---------------------------------------------------------------------------
def _interp_log(h_km, table):
    """Log-linear interpolation into a density table at altitude h_km [km]."""
    log_rho = np.interp(h_km, _CACHE_ALT, np.log(table))
    return np.exp(log_rho)


def nrlmsise_minmax_density(h_km, activity='mean'):
    """
    Return (rho_min, rho_max) [kg/m^3] from the pre-computed NRLMSISE-00
    tables at altitude h_km for the given solar activity level.

    rho_min : global minimum density over the lat x LST x DOY sampling grid
    rho_max : global maximum density over the lat x LST x DOY sampling grid
    """
    tbl   = _NRLM_TABLES[activity]
    h_arr = np.atleast_1d(np.asarray(h_km, dtype=float))
    r_min = np.array([_interp_log(h, tbl['rho_min']) for h in h_arr])
    r_max = np.array([_interp_log(h, tbl['rho_max']) for h in h_arr])
    if r_min.size == 1:
        return r_min[0], r_max[0]
    return r_min, r_max


def nrlmsise_density(h_km, psi_deg, n_exp=6, activity='mean'):
    """
    Effective density [kg/m^3] interpolated between rho_min and rho_max as a
    function of the diurnal-bulge angle psi_deg:

        rho(h, psi) = rho_min(h) + [rho_max(h) - rho_min(h)] * cos(psi/2)^n

    psi_deg = 0   -> maximum density (bulge apex / noon side)
    psi_deg = 180 -> minimum density (anti-bulge / midnight side)
    n_exp   = 6 for near-polar / SSO (default); 2 for low inclination
    """
    rho_min, rho_max = nrlmsise_minmax_density(h_km, activity=activity)
    c = np.clip(np.cos(np.radians(psi_deg) / 2.0), 0.0, 1.0)
    return rho_min + (rho_max - rho_min) * c ** n_exp


# ---------------------------------------------------------------------------
# Sun-synchronous geometry
# ---------------------------------------------------------------------------
def sso_inclination_deg(h_km):
    a      = R_EARTH + h_km
    n      = np.sqrt(MU_EARTH / a ** 3)
    omega_dot = 2.0 * np.pi / SECONDS_PER_TROPICAL_YEAR
    cos_i  = -omega_dot / (1.5 * n * J_COEFFS[2] * (R_EARTH / a) ** 2)
    return np.degrees(np.arccos(np.clip(cos_i, -1.0, 1.0)))


# ---------------------------------------------------------------------------
# 1) Zonal harmonics J2..J6
# ---------------------------------------------------------------------------
def _legendre_and_derivative(n, x):
    coeffs    = np.zeros(n + 1)
    coeffs[n] = 1.0
    Pn  = npleg.legval(x, coeffs)
    dPn = npleg.legval(x, npleg.legder(coeffs))
    return Pn, dPn


def zonal_accel_components(r_km, lat_rad, n):
    Jn      = J_COEFFS[n]
    x       = np.sin(lat_rad)
    Pn, dPn = _legendre_and_derivative(n, x)
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
        a_r, a_t  = zonal_accel_components(r_km, lat_grid, n)
        result[n] = np.max(np.sqrt(a_r ** 2 + a_t ** 2) * 1000.0)
        total_r  += a_r
        total_t  += a_t
    result['total']           = np.max(np.sqrt(total_r ** 2 + total_t ** 2) * 1000.0)
    result['inclination_deg'] = incl
    return result


# ---------------------------------------------------------------------------
# 2) Atmospheric drag
# ---------------------------------------------------------------------------
def drag_accel(h_km, area_to_mass, psi_deg,
               Cd=2.2, n_exp=6, corotating_atm=True, activity='mean'):
    """
    Drag acceleration magnitude [m/s^2] for a circular orbit at h_km [km].

    Density from NRLMSISE-00 pre-computed tables (Picone et al., 2002).
    activity : 'low' | 'mean' | 'high'
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
    psi grids for the given solar activity level.
    nominal = psi-averaged drag at the median A/m.
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
# Sun ephemeris
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
    return np.degrees(np.arccos(
        np.clip(np.dot(r_eci_unit, bulge_unit_vec), -1.0, 1.0)))


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
    r_km      = R_EARTH + h_km
    thetas    = np.linspace(0, 2 * np.pi, n_grid)
    s_vec     = np.array([d3_km, 0.0, 0.0])
    s_norm3   = np.linalg.norm(s_vec) ** 3
    r_vecs    = r_km * np.column_stack([np.cos(thetas), np.sin(thetas),
                                         np.zeros_like(thetas)])
    rel       = s_vec - r_vecs
    rel_norm3 = np.linalg.norm(rel, axis=1) ** 3
    a_vecs    = mu3 * (rel / rel_norm3[:, None] - s_vec / s_norm3)
    return np.max(np.linalg.norm(a_vecs, axis=1)) * 1000.0