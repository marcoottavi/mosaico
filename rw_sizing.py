# Sizing of the Reaction Wheels for GG Torque Control

"""
gg_multi_beta.py
================
GG torque components and angular momentum accumulation for multiple
beta values, held on the same axes.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from orbital_perturbations import R_EARTH, MU_EARTH
from torque_disturbances import N_ATOMS, NOMINAL_EDGE, NOMINAL_WIDTH, NOMINAL_MASS, COM_COP
from assembler import compute_mass_properties
import scipy as sp

os.system('clear')


# ── Configuration ─────────────────────────────────────────────────────────────
OUT_DIR = 'figures'
INC = np.radians(97.79) # Inclination of the orbit
EPS = np.radians(23.44) # Earth's obliquity
ALTITUDE = 600   # km
B1_B2 = np.array([[0,1,0],
                  [0,0,1],
                  [1,0,0]]) # Rotation matrix from body_2 to body_1 frame


# - Script--------------------------------------------------------------------
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

t_grid = np.arange(0, 86400*365.2425, 10) # 1 year in seconds, with 10 seconds increments
n_orbit = np.sqrt(MU_EARTH/(R_EARTH + ALTITUDE)**3) # rad/s
n_earth = 2*np.pi/(365.2425*86400) # rad/s

theta = n_orbit*t_grid
lambda_s = n_earth*t_grid

# Precomputations
cos_theta = np.cos(theta)
sin_theta = np.sin(theta)
cos_lambda_s = np.cos(lambda_s)
sin_lambda_s = np.sin(lambda_s)
cos_inc = np.cos(INC)
sin_inc = np.sin(INC)
cos_eps = np.cos(EPS)
sin_eps = np.sin(EPS)

rx = (cos_eps-1)*cos_lambda_s*sin_lambda_s*cos_theta + \
     (sin_eps*sin_inc*sin_lambda_s-cos_inc*(cos_lambda_s**2+cos_eps*sin_lambda_s**2))*sin_theta

ry = (cos_eps*cos_lambda_s**2+sin_lambda_s**2)*cos_theta \
      +cos_lambda_s*(sin_eps*sin_inc+(1-cos_eps)*cos_inc*sin_lambda_s)*sin_theta

rz = -sin_eps*cos_lambda_s*cos_theta + \
     (cos_eps*sin_inc+cos_inc*sin_eps*sin_lambda_s)*sin_theta

dict_atom = compute_mass_properties(NOMINAL_EDGE, NOMINAL_WIDTH, NOMINAL_MASS/7, N_ATOMS)
inertia_b1 = dict_atom["inertia_about_origin"]
inertia_b2 = B1_B2.T @ inertia_b1 @ B1_B2
print("Inertia Matrix in rotated frame:\n", inertia_b2)

r = np.vstack((rx, ry, rz))
Ir = inertia_b2 @ r
Tg = np.cross(r, inertia_b2 @ r, axis = 0) * 3 * MU_EARTH/(R_EARTH + ALTITUDE)**3
H_accumulation = sp.integrate.simpson(Tg, t_grid, axis = 1)
Tg_mean = H_accumulation/(t_grid[-1]-t_grid[0])
print("Accumulated angular momentum over one year [Nms]:\n", H_accumulation)
print("Mean torque over one year [Nm]:\n", Tg_mean)

# Plotting
plt.figure(figsize=(16,6))
plt.rcParams.update({'font.size': 15,
                     'font.family': 'serif'})
grid_plot = t_grid/86400
xmax = grid_plot.max()
xmin = grid_plot.min()
plt.plot(grid_plot, Tg[0,:], label=r'$Tg_x$', color='r', alpha=0.4)
plt.plot(grid_plot, Tg[1,:], label=r'$Tg_y$', color='y', alpha=0.4)
plt.plot(grid_plot, Tg[2,:], label=r'$Tg_z$', color='b', alpha = 0.4)
plt.hlines(y=Tg_mean[0], xmin=xmin, xmax=xmax, color='r', linestyle='--', label=r'$\overline{Tg_x}$')
plt.hlines(y=Tg_mean[1], xmin=xmin, xmax=xmax, color='y', linestyle='--', label=r'$\overline{Tg_y}$')
plt.hlines(y=Tg_mean[2], xmin=xmin, xmax=xmax, color='b', linestyle='--', label=r'$\overline{Tg_z}$')
plt.grid(which = 'both', linestyle = '--', linewidth = 0.5, color = 'gray')
plt.ylabel('Torque [Nm]')
plt.xlabel('Time [days]')
plt.title('Gravity Gradient Torque Components')
plt.legend(bbox_to_anchor=(-0.03, -0.2), loc='center left', ncols = 6)
plt.savefig(os.path.join(OUT_DIR, 'gg_torque_components.svg'), dpi=300)