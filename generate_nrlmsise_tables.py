"""
generate_nrlmsise_tables.py
============================

Regenerates the NRLMSISE-00 density tables embedded in orbital_perturbations.py.

Requires:  pip install nrlmsise00

Run:
    python generate_nrlmsise_tables.py

Prints the three pairs of rho_min / rho_max arrays to stdout in copy-paste
format.  Paste the output into the density-table section of
orbital_perturbations.py to update the cached values.

Reference:
    Picone, J.M., Hedin, A.E., Drob, D.P., and Aikin, A.C.,
    "NRLMSISE-00 empirical model of the atmosphere: Statistical comparisons
    and scientific issues", Journal of Geophysical Research: Space Physics,
    Vol. 107, No. A12, pp. SIA 15-1 - SIA 15-16, 2002.
    DOI: 10.1029/2002JA009430
"""

import numpy as np
from nrlmsise00 import gtd7_flat

CACHE_ALT = np.array([
    100, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240,
    250, 260, 270, 280, 290, 300, 320, 340, 360, 380, 400, 420, 440, 460,
    480, 500, 520, 540, 560, 580, 600, 620, 640, 660, 680, 700, 720, 740,
    760, 780, 800, 840, 880, 920, 960, 1000,
], dtype=float)

F10P7   = {'low': 75.0, 'mean': 150.0, 'high': 230.0}
AP      = 4.0                     # quiet geomagnetic conditions
FLAGS   = [1] * 24          # flag[0]=0 -> SI output (kg/m^3)
LAT_PTS = np.linspace(-90.0, 90.0, 19)
LST_PTS = np.linspace(0.0, 24.0, 13)
DOYS    = [80, 172, 264, 355]

VARNAMES = {'low': 'LOW', 'mean': 'MEAN', 'high': 'HIGH'}

for act, f107 in F10P7.items():
    rho_min_arr, rho_max_arr = [], []
    for alt in CACHE_ALT:
        vals = [
            gtd7_flat(2003, doy, lst * 3600, alt, lat, 0.0, lst,
                      f107, f107, AP, flags=FLAGS)[5]
            for doy in DOYS
            for lat in LAT_PTS
            for lst in LST_PTS
        ]
        rho_min_arr.append(np.min(vals))
        rho_max_arr.append(np.max(vals))
        print(f"  {act}  h={alt:.0f} km  "
              f"min={rho_min_arr[-1]:.3e}  max={rho_max_arr[-1]:.3e}")

    vn = VARNAMES[act]
    print(f"\n# --- {act.upper()} activity  F10.7 = {f107} SFU ---")
    for prefix, arr in [('MIN', rho_min_arr), ('MAX', rho_max_arr)]:
        print(f"_RHO_{prefix}_{vn} = np.array([")
        for i, v in enumerate(arr):
            sep = ',' if i % 5 < 4 else ','
            end = '\n' if (i + 1) % 5 == 0 else ' '
            print(f"    {v:.6e}{sep}", end=end)
        print(f"\n])  # [kg/m^3]\n")